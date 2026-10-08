"""Three spatial regions, one material shift, and joint-only physical actuation.

The route is registered to a saved, supported posture. Captured bends join the
two known pipe axes; helical extensions provide space for the complete body.
This first implementation explicitly uses simulated world poses/contact points
in its shape tracking and traction solve. Encoder errors and actuator loads
are separate inputs, not a claim of hardware-ready global observation.
Units: metres, radians, seconds.
"""
import copy
import json
import math

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import lsq_linear, minimize
from scipy.spatial.transform import Rotation

from snake_control.spatial_bishop_backbone import SpatialBishopBackbone
from snake_control.pipe_clearance import PipeClearance


DT = .05
LINK = .0645
BODY_RADIUS = .029
CORNER = np.array([1.1, 0., 1.])


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


def physical_wrap_masks(centers):
    """Observed cylinder bands, independent of commanded material regions.

    The posterior-most connected source band must span at least three modules
    and one radian around the cylinder. An isolated elbow touch cannot replace
    that wrap. A band can shrink/move with the body; no fixed module ID is used.
    Contact-side surface tests are applied separately to its normal forces.
    """
    centers = np.asarray(centers)
    radial = centers[:, [1, 2]]-[0., 1.]
    source_radius = np.linalg.norm(radial, axis=1)
    candidate = (centers[:, 0] < 1.075) & (source_radius > .07) & (source_radius < .13)
    # A free connecting rod can straddle the arbitrary 25mm elbow buffer.
    # Do not cut one physically continuous source wrap at that single centre
    # when both adjacent rods qualify and this rod stays before the corner.
    bridge = np.zeros(21, dtype=bool)
    bridge[1:-1] = candidate[:-2] & candidate[2:]
    candidate |= bridge & (centers[:, 0] < CORNER[0]) & (source_radius > .07) & (source_radius < .13)
    breaks = np.flatnonzero(np.diff(np.r_[False, candidate, False]))
    source = np.zeros(21, dtype=bool)
    if len(breaks):
        start, end = breaks[-2:]
        angle = np.unwrap(np.arctan2(radial[start:end, 1], radial[start:end, 0]))
        if end-start >= 3 and np.ptp(angle) >= 1.:
            source[start:end] = True
    target_radius = np.linalg.norm(centers[:, [0, 2]]-[1.1, 1.], axis=1)
    target = (centers[:, 1] > .075) & (target_radius > .07) & (target_radius < .13)
    return source, target


def cylindrical_contact(side, point, normal):
    """Known source/target side surface, excluding end/miter face contact."""
    axis_component = abs(float(normal[side]))
    away_from_corner = point[0] < 1.075 if side == 0 else point[1] > .075
    return bool(away_from_corner and axis_component < .35)


class RegisteredThreeRegionCurve:
    """A single increasing tail-to-head route; all material locations share shift.

    After each deformation the CURRENT curve is reparameterized by arclength.
    One shift advances the head; all remaining nodes follow with exactly one
    fixed-length link chord. Region/feedback coordinates use the same inverse
    arclength map. The underlying spline parameter is not called arclength.
    """
    def __init__(self, points, source_node=13, target_node=8):
        points = np.asarray(points, float)
        self.length = 21*LINK
        self.node_s0 = self.length-np.arange(22)*LINK
        self.center_s0 = self.length-(np.arange(21)+.5)*LINK
        self.source_bound = self.node_s0[source_node]
        self.target_bound = self.node_s0[target_node]
        self.radii, self.axial_sines, self.theta_rates = [], [], []
        additions = []
        for side, ids, endpoint, distances in [
                (0, np.arange(source_node, 22), 21, -np.arange(32, 0, -1)*LINK),
                (1, np.arange(0, target_node+1), 0, np.arange(1, 49)*LINK)]:
            axis_index = side
            radial_indices = [1, 2] if side == 0 else [0, 2]
            origin = np.array([0., 1.]) if side == 0 else np.array([1.1, 1.])
            radial = points[ids][:, radial_indices]-origin
            angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
            rate = float(np.polyfit(self.node_s0[ids], angles, 1)[0])
            axial = float(np.clip(np.polyfit(self.node_s0[ids], points[ids, axis_index], 1)[0], .12, .55))
            radius = float(np.median(np.linalg.norm(radial, axis=1)))
            # Same winding as the actual supported shape; no free-tail activation.
            rate = math.copysign(math.sqrt(1-axial*axial)/radius, rate)
            self.radii.append(radius)
            self.axial_sines.append(axial)
            self.theta_rates.append(rate)
            end = points[endpoint].copy()
            r = end[radial_indices]-origin
            theta = math.atan2(r[1], r[0])
            # Match the end radius exactly, then gradually approach fitted radius.
            end_radius = np.linalg.norm(r)
            a = theta+rate*distances
            rr = end_radius+(radius-end_radius)*smooth(abs(distances)/(3*LINK))
            extra = np.tile(end, (len(distances), 1))
            extra[:, axis_index] += axial*distances
            extra[:, radial_indices] = origin+rr[:, None]*np.column_stack([np.cos(a), np.sin(a)])
            additions.append((self.node_s0[endpoint]+distances, extra))
        locations = np.r_[additions[0][0], self.node_s0[::-1], additions[1][0]]
        values = np.vstack([additions[0][1], points[::-1], additions[1][1]])
        self.base = CubicSpline(locations, values, axis=0)
        self.domain = [float(locations[0]), float(locations[-1])]
        self.phase = np.zeros(2)
        self.phase_correction = np.zeros(2)
        self.radius_delta = np.zeros(2)
        self.pipe_smoothing = np.zeros(2)
        self.phase_origin_shift = 0.
        self.deformation_anchor_parameter = None
        self.deformation_anchor_shift = 0.
        self.clearance = None
        self.clearance_enabled = False
        self.clearance_margin = .036
        self.grip_clearance_profile = None
        self.grip_clearance_parameters = None
        self.reference_clearance_deficit_mm = None
        self.soft_clearance_budget = .001
        self.soft_positive_clearance = False
        self.bridge_offset = np.zeros(3)
        self.shape_grid = np.arange(self.domain[0], self.domain[1]+LINK/2, LINK/2)
        self.shape_offsets = np.zeros((len(self.shape_grid), 3))
        self.clearance_offsets = np.zeros(len(self.shape_grid))
        self.reference_pipe_clearance_mm = None
        self.arc_grid_u = np.unique(np.r_[locations, np.linspace(self.domain[0], self.domain[1],
            math.ceil((self.domain[1]-self.domain[0])/.0025)+1)])
        self.contact_transport_offsets = np.zeros((len(self.arc_grid_u), 3))
        self.persisted_reference_offsets = np.zeros((len(self.arc_grid_u), 3))
        self.tail_recovery_boundary = None
        self.shape_revision = 0
        self.arc_cache = None
        arc, _ = self.arc_map(np.zeros(2))
        self.initial_node_arc = np.interp(self.node_s0, self.arc_grid_u, arc)
        self.initial_head_arc = float(self.initial_node_arc[0])
        self.reference_node_arc = self.initial_node_arc.copy()
        self.reference_center_arc = (self.initial_node_arc[:-1]+self.initial_node_arc[1:])/2
        self.current_target_arc_bound = float(np.interp(self.target_bound, self.arc_grid_u, arc))

    def weights(self, s):
        f = smooth((np.asarray(s)-self.source_bound)/(self.target_bound-self.source_bound))
        return np.stack([1-f, f], axis=-1)

    def regions(self, s):
        s = np.asarray(s)
        return np.where(s <= self.source_bound, 0, np.where(s >= self.target_bound, 2, 1))

    def increment(self, shift_increment):
        # For a helical centreline T=sin(alpha)*axis +/- cos(alpha)*azimuth:
        # centre axial speed = sin(alpha)*shift_dot. Rotate the spatial helix
        # as well, so centre velocity is orthogonal to T and consistent with
        # rolling of a shell about the local material tangent. Merely adding a
        # global roll phase to a moving curvature window can cancel one side.
        self.phase += -np.sign(self.theta_rates)*shift_increment/(
            np.asarray(self.radii)*np.sqrt(1-np.asarray(self.axial_sines)**2))

    def transport_contacts(self, shift, reference_centers, locations, motions):
        """Write the actual-contact task field into this same spatial route.

        The common shift has already advanced once. Replace its provisional
        captured-wrap feed displacement by the SAME nominal contact field
        used by the joint solve. Pose recovery stays in shape feedback, never
        adding observed-minus-reference again. Extensions retain their phase.
        Fixed chords can
        redistribute that field, so return their residual instead of asserting
        exact equality of independently constrained module centre velocities.
        """
        if not motions:
            return self.reference(shift)
        _, current_centers, _ = self.reference(shift)
        numerator = np.zeros_like(self.contact_transport_offsets)
        denominator = np.zeros(len(self.arc_grid_u))
        for side in range(2):
            modules = [module for module, value in motions.items() if value[0] == side]
            if not modules:
                continue
            order = np.argsort(locations[modules])
            modules = np.asarray(modules)[order]
            parameters = locations[modules]
            changes = np.asarray([reference_centers[module]+DT*motions[module][1]
                -current_centers[module] for module in modules])
            outside = np.maximum(parameters[0]-self.arc_grid_u, self.arc_grid_u-parameters[-1])
            window = np.exp(-.5*(np.maximum(0., outside)/(.75*LINK))**2)
            field = np.column_stack([np.interp(self.arc_grid_u, parameters, changes[:, j]) for j in range(3)])
            numerator += window[:, None]*field
            denominator += window
        field = numerator/np.maximum(denominator[:, None], 1.)
        magnitude = np.linalg.norm(field, axis=1)
        field *= np.minimum(1., .008*DT/np.maximum(magnitude, 1e-9))[:, None]
        offsets, clearance = self.contact_transport_offsets.copy(), self.clearance_offsets.copy()
        revision, head_arc = self.shape_revision, self.initial_head_arc
        anchor = (self.deformation_anchor_parameter, self.deformation_anchor_shift)
        head_parameter = float(self.reference_node_parameters[0])
        budget = self.soft_clearance_budget
        for fraction in (1., .5, .25, .125, .0625, .03125, .015625):
            self.contact_transport_offsets = offsets+fraction*field
            self.clearance_offsets = clearance.copy()
            self.soft_clearance_budget = budget
            self.initial_head_arc = head_arc
            self.deformation_anchor_parameter = head_parameter
            self.deformation_anchor_shift = shift
            self.shape_revision = revision+1
            self.arc_cache = None
            try:
                reference = self.reference(shift)
            except ValueError:
                continue
            modules = np.asarray(list(motions))
            displacement = np.linalg.norm(reference[1][modules]-current_centers[modules], axis=1)
            if displacement.max() > .008*DT+1e-8:
                continue
            self.contact_transport_fraction = fraction
            return reference
        # No physical command has been sent. Keep the pre-transport valid
        # curve when this contact field cannot satisfy the original geometry.
        self.contact_transport_offsets, self.clearance_offsets = offsets, clearance
        self.soft_clearance_budget, self.initial_head_arc = budget, head_arc
        self.deformation_anchor_parameter, self.deformation_anchor_shift = anchor
        self.shape_revision, self.arc_cache = revision, None
        self.contact_transport_fraction = 0.
        return self.reference(shift)

    def phase_at(self, shift):
        return self.phase_correction+(np.asarray(shift)-self.phase_origin_shift)[..., None]*(-np.sign(self.theta_rates)/(
            np.asarray(self.radii)*np.sqrt(1-np.asarray(self.axial_sines)**2)))

    def sample(self, s, phase=None):
        s = np.asarray(s)
        base = self.base(s)
        for side in range(2):
            if self.pipe_smoothing[side] <= 0.:
                continue
            # The captured physical polygon includes alternating-joint zigzags
            # and tight local bends. Propagating that spline unchanged moves
            # those peaks into successive joint IDs. Replace only the pipe span
            # progressively by the SAME fitted radius/pitch/winding helix used
            # for its extension; keep the connector endpoint and tangent.
            endpoint = 21 if side == 0 else 0
            anchor_u = self.node_s0[endpoint]
            anchor = self.base(anchor_u)
            xy = [1, 2] if side == 0 else [0, 2]
            origin = np.array([0., 1.]) if side == 0 else np.array([1.1, 1.])
            radial = anchor[xy]-origin
            distance = s-anchor_u
            theta = math.atan2(radial[1], radial[0])+self.theta_rates[side]*distance
            radius = np.linalg.norm(radial)+(self.radii[side]-np.linalg.norm(radial))*smooth(abs(distance)/(3*LINK))
            template = np.broadcast_to(anchor, (*s.shape, 3)).copy()
            template[..., side] += self.axial_sines[side]*distance
            template[..., xy] = origin+radius[..., None]*np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            window = smooth((self.source_bound-s)/(2*LINK)) if side == 0 else smooth((s-self.target_bound)/(2*LINK))
            base += (self.pipe_smoothing[side]*window)[..., None]*(template-base)
        transformed = []
        for side in range(2):
            axis = np.array([1., 0., 0.]) if side == 0 else np.array([0., 1., 0.])
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            relative = base-origin
            radial = relative-(relative@axis)[..., None]*axis
            normal = radial/np.maximum(np.linalg.norm(radial, axis=-1)[..., None], 1e-9)
            # Positive polar angle is +X rotation at source, -Y at target.
            angle = (self.phase[side] if phase is None else np.asarray(phase)[..., side])*(1 if side == 0 else -1)
            transformed.append(origin+Rotation.from_rotvec(np.asarray(angle)[..., None]*axis).apply(
                relative+self.radius_delta[side]*normal))
        weights = self.weights(s)
        point = weights[..., 0, None]*transformed[0]+weights[..., 1, None]*transformed[1]
        f = weights[..., 1]
        point += (4*f*(1-f))[..., None]*self.bridge_offset
        # Corrections live on the spatial route, not permanently on joint IDs.
        point += np.stack([np.interp(s, self.shape_grid, self.shape_offsets[:, i]) for i in range(3)], axis=-1)
        point += np.stack([np.interp(s, self.arc_grid_u, self.contact_transport_offsets[:, i]) for i in range(3)], axis=-1)
        if self.clearance_enabled:
            margin = self.clearance_margins(s)+np.interp(s, self.shape_grid, self.clearance_offsets)
            projected = self.clearance.project(point, margin)
            if self.soft_positive_clearance:
                correction = projected-point
                length = np.linalg.norm(correction, axis=-1)
                outside = (self.clearance.signed(point) > 0.) & (self.regions(s) != 1)
                bounded = point+correction*np.minimum(1., .001/np.maximum(length, 1e-9))[..., None]
                safe = outside & (self.clearance.signed(bounded) > 0.)
                projected = np.where(safe[..., None], bounded, projected)
            point = projected
        # A handover writes the already accepted finite-link reference back
        # into the lasting route. Its polygon can have positive conservative
        # envelope deficits just like any other existing reference. The final
        # original solid/interference checks below remain authoritative.
        point += np.stack([np.interp(s, self.arc_grid_u, self.persisted_reference_offsets[:, i]) for i in range(3)], axis=-1)
        return point

    def capture_grip_clearance(self, physical_nodes):
        """Retain observed usable mesh wrapping, without a uniform tube claim.

        This local centreline allowance comes from the restored physical links,
        whose real collision shells are meshes. It is not an exact mesh swept
        volume. The bridge keeps its original conservative envelope; dynamics
        and original collision contacts still decide whether squeezing works.
        """
        fractions = np.linspace(0., 1., 65)
        points = physical_nodes[:-1, None]*(1-fractions[None, :, None])+physical_nodes[1:, None]*fractions[None, :, None]
        parameters = self.node_s0[:-1, None]*(1-fractions)+self.node_s0[1:, None]*fractions
        self.grip_clearance_parameters, first = np.unique(parameters.ravel(), return_index=True)
        gaps = self.clearance.signed(points).ravel()[first]
        self.grip_clearance_profile = np.clip(gaps-.0015, .004, BODY_RADIUS+.00075)
        self.shape_revision += 1

    def clearance_margins(self, parameters, links=False):
        parameters = np.asarray(parameters)
        base = BODY_RADIUS+.00075 if links else self.clearance_margin
        margin = np.full(parameters.shape, base)
        if self.grip_clearance_profile is not None:
            observed = np.interp(parameters, self.grip_clearance_parameters, self.grip_clearance_profile)
            # Keep the margin continuous at each grasp/bridge handover. The
            # central bridge has the old envelope; only its adjoining half-link
            # blends to the measured wrapping allowance.
            into_bridge = np.minimum(parameters-self.source_bound, self.target_bound-parameters)
            margin = observed+smooth(into_bridge/(.5*LINK))*(base-observed)
        return margin

    def tangent(self, s):
        tangent = self.sample(np.asarray(s)+1e-5)-self.sample(np.asarray(s)-1e-5)
        return tangent/np.maximum(np.linalg.norm(tangent, axis=-1)[..., None], 1e-9)

    def arc_map(self, phase=None):
        phase = self.phase if phase is None else np.asarray(phase)
        key = (*phase, *self.radius_delta, *self.pipe_smoothing, *self.bridge_offset,
               self.clearance_enabled, self.clearance_margin, self.soft_positive_clearance, self.shape_revision)
        if self.arc_cache is not None and self.arc_cache[0] == key:
            return self.arc_cache[1], self.arc_cache[2]
        xyz = self.sample(self.arc_grid_u, np.broadcast_to(phase, (len(self.arc_grid_u), 2)))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        arc -= np.interp(self.source_bound, self.arc_grid_u, arc)
        if np.any(np.diff(arc) <= 1e-10):
            raise ValueError('Degenerate current curve cannot define arclength')
        if np.array_equal(phase, self.phase):
            self.arc_cache = key, arc, xyz
        return arc, xyz

    def reference(self, shift, phase=None, _clearance_attempt=0):
        arc, xyz = self.arc_map(phase)
        def point(s):
            return np.array([np.interp(s, arc, xyz[:, j]) for j in range(3)])
        if self.deformation_anchor_parameter is not None:
            # Local deformation changes the current arc map, including during
            # clearance correction. Keep the head's existing route point as
            # that deformation's origin; only common shift moves correspondence.
            # This changes no physical/progress state and no common-shift value.
            self.initial_head_arc = float(np.interp(self.deformation_anchor_parameter,
                self.arc_grid_u, arc))-self.deformation_anchor_shift
        boundary = getattr(self, 'tail_recovery_boundary', None)
        preserve_tail_boundary = (boundary is not None and
            abs(shift-boundary['shift']) < 1e-12)
        node_arc = [self.initial_head_arc+shift]
        if node_arc[0] >= arc[-1]-.3:
            raise ValueError('Head has exhausted the target helix extension')
        nodes = [point(node_arc[0])]
        first_module = 0
        if preserve_tail_boundary:
            # Tail recovery has zero common feed. Keep the entire reference
            # through the carrying rod's distal end, including both ends of
            # that rod, before taking the remaining equal-length chords.
            # These are existing reference points, not a physical pose reset.
            nodes = list(np.asarray(boundary['prefix_points']))
            node_arc = list(np.interp(boundary['prefix_parameters'], self.arc_grid_u, arc))
            first_module = int(boundary['module'])+1
        for module in range(first_module, 21):
            right = node_arc[-1]
            span = LINK*1.25
            left = right-span
            while np.linalg.norm(point(left)-nodes[-1]) < LINK and span < .35:
                span *= 1.35
                left = right-span
            if left <= arc[0] or np.linalg.norm(point(left)-nodes[-1]) < LINK:
                raise ValueError('Cannot place one fixed-length chord on current route')
            for _ in range(19):
                middle = (left+right)/2
                if np.linalg.norm(point(middle)-nodes[-1]) >= LINK:
                    left = middle
                else:
                    right = middle
            coordinate = (left+right)/2
            node_arc.append(coordinate)
            nodes.append(point(coordinate))
        node_arc = np.asarray(node_arc)
        nodes = np.asarray(nodes)
        if preserve_tail_boundary and 'stop_module' in boundary:
            low, high = int(boundary['module'])+1, int(boundary['stop_module'])
            retained = np.asarray(boundary['retained_points'])
            piece = nodes[low:high+1].copy()
            start, end = retained[low], retained[high]
            if np.linalg.norm(end-start) >= (high-low)*LINK:
                raise ValueError('Carrying-rod endpoints leave no local bending room')
            # Equal-length finite-link sampling with both carrying rods fixed.
            # Iterate only the free intermediate nodes of this same reference;
            # the hinge objectives are still generated from its final geometry.
            for _ in range(80):
                piece[-1] = end
                for i in range(len(piece)-2, -1, -1):
                    v = piece[i]-piece[i+1]
                    piece[i] = piece[i+1]+LINK*v/max(np.linalg.norm(v), 1e-12)
                piece[0] = start
                for i in range(1, len(piece)):
                    v = piece[i]-piece[i-1]
                    piece[i] = piece[i-1]+LINK*v/max(np.linalg.norm(v), 1e-12)
                if np.linalg.norm(piece[-1]-end) < 1e-8:
                    break
            if np.linalg.norm(piece[-1]-end) >= 1e-8:
                raise ValueError('Local equal-link sampling did not close between carrying rods')
            piece[-1] = end
            nodes[low:high+1] = piece
            nodes[high:] = retained[high:]
            node_arc[high:] = np.interp(boundary['retained_parameters'][high:], self.arc_grid_u, arc)
        center_arc = (node_arc[:-1]+node_arc[1:])/2
        locations = np.interp(center_arc, arc, self.arc_grid_u)
        if phase is None and self.clearance_enabled:
            margins = self.clearance.link_minimum(nodes, samples=65)
            fractions = np.linspace(0., 1., 65)
            lines = nodes[:-1, None, :]*(1-fractions[None, :, None])+nodes[1:, None, :]*fractions[None, :, None]
            node_parameters = np.interp(node_arc, arc, self.arc_grid_u)
            parameters = node_parameters[:-1, None]*(1-fractions)+node_parameters[1:, None]*fractions
            deficits = (self.clearance_margins(parameters, links=True)-self.clearance.signed(lines)).max(1)
            invalid = deficits > 1e-8
            if invalid.any():
                intersects_solid = margins.min() <= 0.
                if intersects_solid and _clearance_attempt >= 10:
                    raise ValueError(f'Reference link crosses original pipe solid: {margins.min()*1000:.3f} mm')
                budget = float(getattr(self, 'soft_clearance_budget', .001))
                if intersects_solid or budget > 1e-8:
                    bumps = np.exp(-.5*((self.shape_grid[:, None]-locations[None, invalid])/(.8*LINK))**2)
                    increments = deficits[invalid]+.00025
                    if not intersects_solid:
                        increments = np.minimum(increments, budget)
                        self.soft_clearance_budget = max(0., budget-float(increments.max()))
                    self.clearance_offsets += np.max(bumps*increments[None, :], axis=1)
                    self.clearance_offsets = np.minimum(self.clearance_offsets, .04)
                    self.shape_revision += 1
                    return self.reference(shift, phase, _clearance_attempt+1)
                # Positive distance is a valid soft guide even if a conservative
                # mesh surrogate is unmet. Do not abort the physical experiment.
            self.reference_pipe_clearance_mm = float(margins.min()*1000)
            self.reference_clearance_deficit_mm = float(max(0., deficits.max())*1000)
        if phase is None:
            self.reference_node_arc = node_arc
            self.reference_center_arc = center_arc
            self.reference_node_parameters = np.interp(node_arc, arc, self.arc_grid_u)
            self.current_target_arc_bound = float(np.interp(self.target_bound, self.arc_grid_u, arc))
        return nodes, (nodes[:-1]+nodes[1:])/2, locations

    def deposit(self, locations, changes, rate, dt):
        weights = np.exp(-.5*((self.shape_grid[:, None]-locations[None, :])/(.8*LINK))**2)
        averaged = weights@changes/np.maximum(weights.sum(1)[:, None], 1.)
        boundary = getattr(self, 'tail_recovery_boundary', None)
        if boundary is not None:
            # Gaussian blending must not reshape either end of the holder.
            averaged[self.shape_grid >= boundary['after_parameter']] = 0.
            if 'before_parameter' in boundary:
                averaged[self.shape_grid <= boundary['before_parameter']] = 0.
        self.shape_offsets += rate*dt*averaged
        norm = np.linalg.norm(self.shape_offsets, axis=1)
        self.shape_offsets *= np.minimum(1., .012/np.maximum(norm, 1e-9))[:, None]
        self.shape_revision += 1

    def persist_tail_recovery_reference(self, shift):
        """Put the final protected reference into the lasting spatial curve."""
        if getattr(self, 'tail_recovery_boundary', None) is None:
            return 0.
        previous = copy.deepcopy(self.__dict__)
        try:
            return self._persist_tail_recovery_reference(shift)
        except Exception:
            # A rejected representation update must leave the last valid
            # reference exportable; no physical state is changed here.
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

    def _persist_tail_recovery_reference(self, shift):
        protected_shift = self.tail_recovery_boundary['shift']
        saved_phase = self.phase.copy()
        if abs(shift-protected_shift) > 1e-12:
            # A legacy handover checkpoint already applied one tiny feed before
            # persisting its protected shape. Recover that reference only;
            # neither the real body, shared shift nor progress is rewound.
            self.increment(protected_shift-shift)
        nodes, _, _ = self.reference(protected_shift)
        parameters = self.reference_node_parameters.copy()
        self.phase = saved_phase
        # Include each final material point in the route grid. Baking its
        # world-space correction preserves the existing phase/clearance model
        # and never registers the entire curve to a new physical observation.
        previous_grid = self.arc_grid_u.copy()
        previous_offsets = self.contact_transport_offsets.copy()
        previous_persisted = self.persisted_reference_offsets.copy()
        # Repeated handovers recompute material coordinates with roundoff.
        # np.unique retains almost identical knots, whose sampled chord can
        # fall below the arclength map's numerical nondegeneracy threshold.
        # Keep existing knots and add only materially separated coordinates.
        merged_grid = previous_grid.tolist()
        for parameter in parameters:
            if np.min(np.abs(np.asarray(merged_grid)-parameter)) > 1e-8:
                merged_grid.append(float(parameter))
        self.arc_grid_u = np.sort(np.asarray(merged_grid))
        self.contact_transport_offsets = np.column_stack([
            np.interp(self.arc_grid_u, previous_grid, previous_offsets[:, j]) for j in range(3)])
        self.persisted_reference_offsets = np.column_stack([
            np.interp(self.arc_grid_u, previous_grid, previous_persisted[:, j]) for j in range(3)])
        self.tail_recovery_boundary = None
        self.deformation_anchor_parameter = float(parameters[0])
        self.deformation_anchor_shift = shift
        self.shape_revision += 1
        self.arc_cache = None
        for _ in range(12):
            correction = nodes-self.sample(parameters)
            order = np.argsort(parameters)
            field = np.column_stack([np.interp(self.arc_grid_u, parameters[order], correction[order, j]) for j in range(3)])
            outside = np.maximum(parameters[-1]-self.arc_grid_u, self.arc_grid_u-parameters[0])
            field *= np.exp(-.5*(np.maximum(outside, 0.)/(2*LINK))**2)[:, None]
            self.persisted_reference_offsets += field
            self.shape_revision += 1
            self.arc_cache = None
            persisted, _, _ = self.reference(shift)
            error = float(np.linalg.norm(persisted-nodes, axis=1).max())
            if error < 1e-6:
                return error*1000
        raise ValueError(f'Protected-reference persistence did not converge: {error*1000:.6f} mm')

    def parameters(self):
        return dict(source_phase_rad=float(self.phase[0]), target_phase_rad=float(self.phase[1]),
                    source_feedback_phase_rad=float(self.phase_correction[0]),
                    target_feedback_phase_rad=float(self.phase_correction[1]),
                    source_radius_delta_mm=float(self.radius_delta[0]*1000),
                    target_radius_delta_mm=float(self.radius_delta[1]*1000),
                    source_captured_bend_smoothing=float(self.pipe_smoothing[0]),
                    target_captured_bend_smoothing=float(self.pipe_smoothing[1]),
                    phase_origin_shift_mm=float(self.phase_origin_shift*1000),
                    reference_pipe_clearance_mm=self.reference_pipe_clearance_mm,
                    reference_clearance_deficit_mm=self.reference_clearance_deficit_mm,
                    observed_grip_clearance_range_mm=None if self.grip_clearance_profile is None else [float(self.grip_clearance_profile.min()*1000), float(self.grip_clearance_profile.max()*1000)],
                    maximum_route_clearance_correction_mm=float(self.clearance_offsets.max()*1000),
                    bridge_offset_mm=(self.bridge_offset*1000).tolist(),
                    maximum_spatial_shape_offset_mm=float(np.linalg.norm(self.shape_offsets, axis=1).max()*1000))


class UnifiedCurveTransfer:
    def __init__(self, model, data):
        self.m, self.d = model, mujoco.MjData(model)
        self.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.va = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id
        self.last = data.ctrl.copy()
        self.physics_execution_prediction_enabled = True
        p, c = self.observe(data)
        self.initial_centers = c.copy()
        # Save only for reporting. The initial target is continuous at t=0;
        # its target-minus-encoder bias must not become a permanent sign lock.
        self.initial_preload = data.ctrl-data.qpos[self.qa]
        self.curve_grip_strength = np.zeros(21)
        self.curve_curvature_release_strength = np.zeros(20)
        self.support_force_goals = None
        self.curve = RegisteredThreeRegionCurve(p)
        self.pipe_geometry = PipeClearance(model, self.d)
        self.shift = 0.
        self.speed = 0.
        self.events = []
        self.filtered_load = np.zeros(20)
        self.last_mode = None
        self.last_shape_time = -.5
        self.progress_history = []
        self.last_body_progress = 0.
        self.n_adaptations = 0
        self.projected_locations = self.curve.center_s0.copy()
        self.last_observer_time = -1.
        self.observed_shift = 0.
        self.first_dispatch = True
        self.last_route_registration_time = -1e6
        self.observer_rebased = False
        self.grip_clearance_enabled = False
        self.physical_observer_ready = False
        self.physical_observer_just_initialized = False
        self.physical_progress_origin_shift = 0.
        self.physical_pipe_progress = np.zeros(2)
        self.physical_pipe_speed = np.zeros(2)
        self.physical_axial_sines = np.asarray(self.curve.axial_sines)
        self.physical_last_centers = c.copy()
        self.physical_continuous_groups = np.zeros((2, 21), dtype=bool)
        self.physical_cohort_modules = [[], []]
        self.local_reference_meter_ready = False
        self.local_reference_shift_origin = 0.
        self.local_reference_pipe_progress_origin = np.zeros(2)
        self.localized_yield_enabled = False
        self.alternating_yield_enabled = False
        self.soft_positive_clearance_enabled = False
        self.yield_state = None
        self.yield_release_modules = []
        self.transferred_material_modules = []

    def save_state(self, path, elapsed_s):
        # Numeric route/controller history; MuJoCo physics is saved separately.
        from validation.recover_unified_curve_state import save_state
        save_state(path, self, elapsed_s)

    def load_state(self, path):
        state = json.loads(path.read_text(encoding='utf8'))
        if state['version'] != 1:
            raise ValueError('Unsupported controller checkpoint version')
        arrays = {'last', 'initial_centers', 'initial_preload', 'curve_grip_strength', 'curve_curvature_release_strength',
                  'support_force_goals', 'filtered_load', 'projected_locations', 'physical_pipe_progress',
                  'physical_pipe_speed', 'physical_axial_sines', 'physical_last_centers', 'physical_continuous_groups',
                  'local_reference_pipe_progress_origin'}
        for key, value in state['controller'].items():
            setattr(self, key, np.asarray(value) if key in arrays else value)
        curve_arrays = {'node_s0', 'center_s0', 'phase', 'phase_correction', 'radius_delta', 'pipe_smoothing',
            'bridge_offset', 'shape_grid', 'shape_offsets', 'clearance_offsets', 'arc_grid_u', 'initial_node_arc',
            'reference_node_arc', 'reference_center_arc', 'reference_node_parameters', 'grip_clearance_profile', 'grip_clearance_parameters',
            'contact_transport_offsets', 'persisted_reference_offsets'}
        for key, value in state['curve'].items():
            if key not in {'base_x', 'base_c'}:
                setattr(self.curve, key, np.asarray(value) if key in curve_arrays and value is not None else value)
        self.curve.base = PPoly(np.asarray(state['curve']['base_c']), np.asarray(state['curve']['base_x']))
        if self.yield_state is not None and 'released_material_modules' not in self.yield_state:
            # The saved action's actual released rods are material identities,
            # not nearest neighbours of an old spatial curve parameter.
            if self.yield_state.get('released_center_parameters'):
                self.yield_state['released_material_modules'] = list(self.yield_release_modules)
        if self.curve.clearance_enabled:
            self.curve.clearance = self.pipe_geometry
        self.curve.arc_cache = None
        self.first_dispatch = True
        reference, _, _ = self.curve.reference(self.shift)
        if 'reference_points_mm' in state:
            self.restored_reference_error_mm = float(np.max(np.abs(reference*1000-state['reference_points_mm'])))
            if self.restored_reference_error_mm > 1e-6:
                raise ValueError('Restored curve differs from checkpoint reference')
        if (getattr(self.curve, 'tail_recovery_boundary', None) is not None and
                self.yield_state is not None and self.yield_state['stage'] == 'completed'):
            old_reference = reference.copy()
            error = self.curve.persist_tail_recovery_reference(self.shift)
            repaired, _, _ = self.curve.reference(self.shift)
            self.events.append(dict(t_s=float(state['elapsed_s']),
                mode='resume_protected_handover_reference_continuously',
                maximum_baking_error_mm=error,
                maximum_legacy_reference_correction_mm=float(np.linalg.norm(repaired-old_reference, axis=1).max()*1000),
                meaning='correct saved reference transition only; full physical integration state, ctrl, shift and actual progress unchanged'))
        if (self.alternating_yield_enabled and self.yield_state is not None and
                self.yield_state.get('failure') == 'no_supported_rear_release_available_after_feed_stall'):
            # Resume the stopped action by restoring its actual released span.
            # No physical/reference/progress history is rebased here.
            self.begin_local_regrip(float(state['elapsed_s']), False,
                self.yield_state.get('achieved_material_mm'))
        if (self.alternating_yield_enabled and self.yield_state is not None and
                self.yield_state.get('failure') == 'local_release_not_observed_12s' and
                self.transferred_material_modules):
            # A corrected role assignment resumes by physically restoring the
            # current grasp, then selecting a fresh supported release. The old
            # 12s action remains failed; no timer extension or feed credit.
            self.begin_local_regrip(float(state['elapsed_s']), False,
                self.yield_state.get('achieved_material_mm'))
            self.events.append(dict(t_s=float(state['elapsed_s']),
                mode='resume_failed_release_with_transferred_material_roles',
                meaning='restore and confirm the actual current grasp before a fresh action; previous release remains failed'))
        return float(state['elapsed_s'])

    def repair_registered_route(self, p, centers, t, forces):
        """Use observed wrap endpoints to repair a wrong-side/colliding bridge.

        Only reference history is rebased. The shared shift and every physical
        state/target remain unchanged. Actual contact/position scores stay the
        transfer evidence; the observer rebase is explicitly not body progress.
        """
        source, target = physical_wrap_masks(centers)
        if np.count_nonzero(source & (forces[0] > .1)) < 2 or np.count_nonzero(target & (forces[1] > .1)) < 4:
            return False
        source_ids = np.flatnonzero(source)
        if not len(source_ids):
            return False
        source_node = int(source_ids[0]+1)
        radial = p[:, [0, 2]]-[1.1, 1.]
        radii = np.linalg.norm(radial, axis=1)
        angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
        winding = -np.sign(self.curve.theta_rates[1])
        target_node = 0
        for j in range(1, min(source_node-1, 15)):
            if p[j, 1] < .075 or not .07 < radii[j] < .14 or winding*(angles[j]-angles[j-1]) < .02:
                break
            target_node = j
        if target_node < 3 or source_node-target_node < 2:
            return False
        old_head = self.curve.initial_head_arc
        self.curve = RegisteredThreeRegionCurve(p, source_node, target_node)
        # Preserve the observed gripping bends. A newly fitted smooth helix
        # must not instantly replace them at registration; limit feedback can
        # gradually smooth this span later if the real joints need it.
        self.curve.phase_origin_shift = self.shift
        self.curve.clearance_enabled = True
        self.curve.clearance = self.pipe_geometry
        self.curve.soft_positive_clearance = self.soft_positive_clearance_enabled
        if self.grip_clearance_enabled:
            self.curve.capture_grip_clearance(p)
        arc, _ = self.curve.arc_map()
        self.curve.initial_node_arc = np.interp(self.curve.node_s0, self.curve.arc_grid_u, arc)
        self.curve.initial_head_arc = float(self.curve.initial_node_arc[0])-self.shift
        _, registered_centers, locations = self.curve.reference(self.shift)
        self.projected_locations = locations.copy()
        registration_error = float(np.linalg.norm(registered_centers-centers, axis=1).max())
        if registration_error <= LINK/2 and self.physical_observer_ready:
            # Local curve lead changes its origin when the reference actually
            # fits the observed body. Historical physical progress is untouched.
            self.local_reference_meter_ready = True
            self.local_reference_shift_origin = self.shift
            self.local_reference_pipe_progress_origin = self.physical_pipe_progress.copy()
        self.last_route_registration_time = t
        self.last_shape_time = t
        self.observer_rebased = True
        self.events.append(dict(t_s=t, mode='repair_external_supported_route', source_node=source_node,
            target_node=target_node, shared_shift_mm=self.shift*1000,
            physical_progress_unchanged_mm=self.observed_shift*1000,
            observed_grip_clearance_enabled=self.grip_clearance_enabled,
            registered_max_center_error_mm=registration_error*1000,
            local_reference_shift_origin_mm=self.local_reference_shift_origin*1000,
            local_pipe_progress_origin_mm=(self.local_reference_pipe_progress_origin*1000).tolist(),
            old_reference_head_arc_mm=old_head*1000,
            meaning='reference correspondence repair only; physical progress meter unchanged; captured gripping shape retained'))
        return True

    def retain_observed_grip_clearance(self, data, t, forces):
        self.grip_clearance_enabled = True
        p, centers = self.observe(data)
        if not self.repair_registered_route(p, centers, t, forces):
            raise ValueError('Observed grip clearance requires the saved dual-pipe support')

    def source_span(self, locations):
        # Include one link at the source/bridge handover. A connector touching
        # the elbow is not counted as the remaining posterior wrap.
        return np.asarray(locations) <= self.curve.source_bound+LINK

    def supported_curve_regions(self, locations, centers):
        return physical_wrap_masks(centers)

    def observe_progress(self, centers, forces, t):
        """Measure same-module axial displacement on continuously occupied pipes.

        The curve and its registration do not enter this position difference.
        Cohorts must remain in the observed cylinder bands over the whole
        interval. Fixed entry pitch factors convert each physical axial change
        to an approximate material change; the slower occupied side wins.
        """
        groups = np.asarray(physical_wrap_masks(centers))
        if not self.physical_observer_ready:
            self.physical_observer_ready = self.physical_observer_just_initialized = True
            self.physical_progress_origin_shift = self.shift
            self.physical_pipe_progress[:] = 0.
            self.physical_pipe_speed[:] = 0.
            self.physical_axial_sines = np.maximum(self.curve.axial_sines, .12)
            self.physical_last_centers = centers.copy()
            self.physical_continuous_groups = groups.copy()
            self.last_observer_time = t
            self.observed_shift = self.last_body_progress = 0.
            self.progress_history = []
            return 0.
        self.physical_continuous_groups &= groups
        if t-self.last_observer_time < .5:
            return self.observed_shift
        elapsed = t-self.last_observer_time
        for side in range(2):
            mask = self.physical_continuous_groups[side]
            self.physical_cohort_modules[side] = np.flatnonzero(mask).tolist()
            if mask.sum() >= 2:
                change = float(np.median(centers[mask, side]-self.physical_last_centers[mask, side]))
                self.physical_pipe_progress[side] += change
                self.physical_pipe_speed[side] = .6*self.physical_pipe_speed[side]+.4*change/elapsed
            else:
                self.physical_pipe_speed[side] = 0.
        occupied = [side for side in range(2) if groups[side].sum() >= 2 or np.count_nonzero(forces[side] > .1) >= 2]
        material = self.physical_pipe_progress/self.physical_axial_sines
        self.observed_shift = float(min(material[occupied])) if occupied else self.observed_shift
        self.physical_last_centers = centers.copy()
        self.physical_continuous_groups = groups.copy()
        self.last_observer_time = t
        return self.observed_shift

    def observe(self, data):
        self.d.qpos[:] = data.qpos
        mujoco.mj_fwdPosition(self.m, self.d)
        p = np.vstack([self.d.site('s_head').xpos,
                       *[self.d.joint('joint'+str(i)).xanchor for i in range(1, 21)],
                       self.d.site('s_tail').xpos])
        return p, (p[:-1]+p[1:])/2

    def feedback_shape(self, p, q, reference, s, tracking, forces, dt):
        coordinated_yielding = (self.yield_state is not None
            and self.yield_state['stage']=='feed'
            and self.yield_state.get('coordinated_feed_active',False)
            and not self.yield_state.get('contact_loss_after_partial_feed',False))
        """Measured bends and load change the route before target generation."""
        if self.yield_state is not None and self.yield_state.get('target_handover_pending', False):
            # Verify the actual target grasp without simultaneously pulling
            # those same material rods back toward the old source normal.
            return
        if self.yield_state is not None:
            self.curve.deformation_anchor_parameter = float(self.curve.reference_node_parameters[0])
            self.curve.deformation_anchor_shift = self.shift
        residual = (p[:-1]+p[1:])/2-reference
        regions = self.curve.regions(s)
        stressed = (self.filtered_load > 1.9) & (tracking > np.radians(3.))
        module_stress = np.r_[stressed[0], stressed] | np.r_[stressed, stressed[-1]]
        centers = (p[:-1]+p[1:])/2
        for side, code in [(0, 0), (1, 2)]:
            if self.yield_state is not None:
                # Hold the other captured wraps while this small span makes
                # space. Pressure/contact tasks maintain their observed loads.
                continue
            ids = np.flatnonzero(regions == code)
            if len(ids) < 3:
                continue
            axis = np.array([1., 0., 0.]) if side == 0 else np.array([0., 1., 0.])
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            radial = centers[ids]-origin
            radial -= (radial@axis)[:, None]*axis
            normal = radial/np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-8)
            mismatch = float(np.median(np.sum(residual[ids]*normal, axis=1)))
            # Opening a loaded source wrap is bounded; no progressive unwrapping.
            # Target capture retains a small inward bias if position tracks.
            bias = .001 if module_stress[ids].any() else (-.0006 if side == 1 else -.0003)
            wanted = np.clip(self.curve.radius_delta[side]+.15*mismatch+bias, -.004, .004)
            pressure_deficit = (self.support_force_goals[side]-forces[side].sum())/self.support_force_goals[side]
            wanted = np.clip(wanted-.001*np.clip(pressure_deficit, -1., 1.), -.004, .004)
            if side == 1:
                # U01 opened the receiving wrap under tracking stress, then lost
                # M3 and M0. Receiving support takes priority over unloading it.
                count = np.count_nonzero(forces[1] > .1)
                upper = 0. if count >= 3 else -.0015
                wanted = min(wanted, upper)
            self.curve.radius_delta[side] += np.clip(wanted-self.curve.radius_delta[side], -.00025*dt, .00025*dt)
            # Register the reference wrap to supported material observations,
            # rather than keeping a fictional pipe phase after actual slipping.
            supported = ids[forces[side, ids] > .1]
            if len(supported) >= 2:
                xy = [1, 2] if side == 0 else [0, 2]
                o = np.array([0., 1.]) if side == 0 else np.array([1.1, 1.])
                measured_r = centers[supported][:, xy]-o
                reference_r = reference[supported][:, xy]-o
                errors = np.arctan2(measured_r[:, 1], measured_r[:, 0])-np.arctan2(reference_r[:, 1], reference_r[:, 0])
                angle = float(np.angle(np.sum(forces[side, supported]*np.exp(1j*errors))))
                change = float(np.clip(.15*angle, -.04*dt, .04*dt))
                self.curve.phase_correction[side] += change
                self.curve.phase[side] += change

        # The squeeze is now a change of the common 3D route. Integrate the
        # measured Bishop field with pressure-based curvature gains, compare it
        # to the SAME field without those gains, and deposit only that shape
        # increment at current route coordinates. No joint clamp target is added
        # after motion generation, so tracking cannot undo a separate objective.
        measured_bone = SpatialBishopBackbone(p*1000, np.full(21, LINK*1000))
        strength = np.zeros(21)
        for side, code in [(0, 0), (1, 2)]:
            mask = regions == code
            count = max(1, np.count_nonzero(mask & (forces[side] > .1)))
            goal = self.support_force_goals[side]/count
            strength[mask] = np.clip(.018*(goal-forces[side, mask])/max(goal, 1.), -.006, .018)
        gain = np.zeros(len(measured_bone.field))
        gain[1:-1] = (strength[:-1]+strength[1:])/2
        source_support, target_support = physical_wrap_masks(centers)
        supported = (np.count_nonzero(source_support & (forces[0] > .1)) >= 2 and
                     np.count_nonzero(target_support & (forces[1] > .1)) >= 4)
        limit_weight = np.clip((np.abs(q)-np.radians(82.))/np.radians(6.), 0., 1.)
        driven_into_limit = q*(self.last-q) > 0.
        # This is a CURVE curvature release, never a post-solve straightening
        # command. It follows the current material location, not J10/J11 IDs.
        receiving_bend = self.curve.reference_node_parameters[1:-1] >= self.curve.target_bound-LINK
        release = -.02*limit_weight*driven_into_limit*receiving_bend if supported else np.zeros(20)
        gain[1:-1] = np.where(release < 0., np.minimum(gain[1:-1], release), gain[1:-1])
        self.curve_curvature_release_strength = release.copy()
        captured_target_limit = receiving_bend & driven_into_limit & (limit_weight > 0.)
        if supported and captured_target_limit.any() and self.yield_state is None:
            previous = self.curve.pipe_smoothing[1]
            self.curve.pipe_smoothing[1] = min(1., previous+.015*dt)
            self.curve.shape_revision += 1
            if previous == 0.:
                self.events.append(dict(t_s=self.current_control_time, mode='smooth_captured_receiving_bends',
                    joints=(np.flatnonzero(captured_target_limit)+1).tolist(),
                    action='same fitted target helix gradually replaces captured pipe-span curvature peaks; local limit feedback relaxes curvature'))
        self.curve_grip_strength = strength.copy()
        local_joint = None
        local_remaining_load = 0.
        if self.yield_state is not None:
            local_joint = int(self.yield_state['selected_joint'])
            gain[:] = 0.
            strength[:] = 0.
            if (self.yield_state['stage'] == 'unload' or coordinated_yielding) and not self.yield_state.get('anchor_recovery_active', False):
                side = self.yield_state['side']
                force_ratio = forces[side, self.yield_release_modules].sum()/max(self.yield_state['initial_force_N'], .1)
                load_ratio = self.filtered_load[local_joint-1]/max(self.yield_state.get('initial_load_Nm', 2.355), .1)
                local_remaining_load = float(np.clip(max(force_ratio, load_ratio), 0., 1.))
                gain[local_joint] = -.06*local_remaining_load
            elif self.yield_state['stage'] == 'regrip':
                side = self.yield_state['side']
                desired = self.yield_state.get('recovery_force_goal_N', .5*self.yield_state['initial_force_N'])
                current = forces[side, self.yield_release_modules].sum()
                gain[local_joint] = .015*float(np.clip((desired-current)/max(desired, 1.), 0., 1.))
            self.curve_curvature_release_strength[:] = 0.
            self.curve_curvature_release_strength[local_joint-1] = min(0., gain[local_joint])
            self.curve_grip_strength = strength.copy()
        ordinary = measured_bone.sample(step_mm=3.)
        tightened = measured_bone.sample(field=measured_bone.field*(1+gain[:, None]), step_mm=3.)
        material_nodes = np.arange(22)*LINK*1000
        delta = np.column_stack([np.interp(material_nodes, tightened['s_mm'], np.array(tightened['points_mm'])[:, j])-
                                 np.interp(material_nodes, ordinary['s_mm'], np.array(ordinary['points_mm'])[:, j]) for j in range(3)])/1000
        if local_joint is not None:
            low, high = max(0, local_joint-2), min(21, local_joint+2)
            if (self.yield_state['stage'] in ['unload', 'regrip'] or coordinated_yielding) and not self.yield_state.get('anchor_recovery_active', False):
                current_reference_p, _, _ = self.curve.reference(self.shift)
                # Continue making room from the CURRENT measured shape while
                # load remains, rather than exhausting a fixed displacement
                # from the original captured pose. This remains a curve update.
                delta += p-current_reference_p
            fraction = np.linspace(0., 1., high-low+1)[:, None]
            local = np.zeros_like(delta)
            boundary = getattr(self.curve, 'tail_recovery_boundary', None)
            if boundary is not None:
                low = int(boundary['module'])+1
                high = int(boundary.get('stop_module', 21))
                if 'stop_module' in boundary:
                    fraction = np.linspace(0., 1., high-low+1)[:, None]
                    local[low:high+1] = delta[low:high+1]-(1-fraction)*delta[low]-fraction*delta[high]
                else:
                    # The holder's distal endpoint anchors the free tail.
                    local[low+1:high+1] = delta[low+1:high+1]-delta[low]
            else:
                local[low:high+1] = delta[low:high+1]-(1-fraction)*delta[low]-fraction*delta[high]
            # A small outward plateau makes actual room; its central three
            # nodes translate together, preserving the curvature relief there.
            if (self.yield_state['stage'] == 'unload' or coordinated_yielding) and not self.yield_state.get('anchor_recovery_active', False):
                side = self.yield_state['side']
                axis = np.eye(3)[side]
                origin = np.array([0., 0., 1.]) if side == 0 else CORNER
                normal = p[local_joint]-origin
                normal -= (normal@axis)*axis
                normal /= max(np.linalg.norm(normal), 1e-9)
                room_error = .005*local_remaining_load
                self.yield_state['curve_unload_feedback'] = dict(remaining_load_fraction=local_remaining_load,
                    curvature_fraction=float(gain[local_joint]), requested_outward_step_mm=room_error*1000,
                    basis='current measured local shape and remaining contact/actuator load; existing 12mm curve offset cap')
            else:
                normal = np.asarray(self.yield_state['outward_normal'])
                origin = np.asarray(self.yield_state['reference_origin'])
                current = self.curve.sample(self.yield_state['parameter'])
                if self.yield_state['stage'] == 'regrip':
                    side = self.yield_state['side']
                    deficit = self.yield_state.get('recovery_force_goal_N',
                        .5*self.yield_state['initial_force_N'])-forces[side, self.yield_release_modules].sum()
                    # Close toward the CURRENT body only while measured local
                    # pressure is deficient. Do not keep adding the old outward
                    # release plateau during restoration or restore an old q.
                    room_error = -float(np.clip(.00002*deficit, 0., .0006))*dt
                else:
                    room_error = float(np.clip(.005-(current-origin)@normal, 0., .006))
            local[low+1:high] += room_error*normal
            delta = local
        deposit_rate = .18
        if self.yield_state is not None and self.yield_state.get('anchor_recovery_active', False):
            delta[:] = 0.
        elif local_joint is not None and (self.yield_state['stage'] in ['unload', 'regrip'] or coordinated_yielding):
            # A measured-shape feedback step must reach the local reference,
            # rather than remain behind the actual opening for the whole 12s
            # action. Bound each node by the existing 8mm/s motion cap and keep
            # deposit's Gaussian spatial blend and 12mm total offset cap.
            lengths = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(1., .008*dt/np.maximum(lengths, 1e-9))[:, None]
            deposit_rate = 1./dt
        self.curve.deposit(self.curve.reference_node_parameters, delta, deposit_rate, dt)
        transition = regions == 1
        if transition.any():
            correction = residual.copy()
            correction[~transition] = 0.
            # Remove path-tangent mismatch: shape yielding must not silently
            # replace a stalled shift with a different feed for each module.
            tangent = self.curve.tangent(s)
            correction -= np.sum(correction*tangent, axis=1)[:, None]*tangent
            lengths = np.linalg.norm(correction, axis=1)
            correction *= np.minimum(1., .008/np.maximum(lengths, 1e-9))[:, None]
            self.curve.deposit(s, correction, .18, dt)
        self.n_adaptations += 1
        self.curve.soft_clearance_budget = .001

    def protect_local_carrying_rods(self, p, held, forces, t, joint, modules):
        """One entry for release, tail restoration and its support wait."""
        current = getattr(self.curve, 'tail_recovery_boundary', None)
        if current is not None and current.get('action_joint') == joint:
            return
        if current is not None:
            self.curve.persist_tail_recovery_reference(self.shift)
        holders = np.any(held, axis=0)
        before = np.flatnonzero(holders & (np.arange(21) < min(modules)))
        if not len(before):
            return
        holder = int(before[-1])
        # Preserve the complete proximal carrying rod, while the posterior
        # holders can move under their same contact/pressure tasks. A middle
        # lift has too few alternating one-axis hinges to fix both world poses.
        rear_holder = None
        reference_p, _, _ = self.curve.reference(self.shift)
        parameters = self.curve.reference_node_parameters.copy()
        boundary = dict(module=holder, shift=self.shift, action_joint=joint,
            after_parameter=float(parameters[holder+1]),
            prefix_points=reference_p[:holder+2].tolist(),
            prefix_parameters=parameters[:holder+2].tolist())
        if rear_holder is not None:
            boundary.update(stop_module=rear_holder, before_parameter=float(parameters[rear_holder]),
                retained_points=reference_p.tolist(), retained_parameters=parameters.tolist())
        self.curve.tail_recovery_boundary = boundary
        self.events.append(dict(t_s=t, mode='local_action_preserves_complete_carrying_rods',
            action_joint=joint, proximal_holder=holder, distal_holder=rear_holder,
            free_modules=list(range(holder+1, rear_holder if rear_holder is not None else 21)),
            meaning='same local boundary for unload, recovery and next-support wait; final equal-link sampling retains carrying rods; physical pose unchanged'))

    def start_rear_release(self, p, centers, q, tracking, forces, t, locations):
        """Receiving grasps carry while one measured posterior span lifts."""
        source, target = physical_wrap_masks(centers)
        held = np.asarray([source, target]) & (forces > .1)
        held[0, self.transferred_material_modules] = False
        old = self.yield_state
        old_spans = list(old.get('freed_source_parameters', []))
        # A restored contact is a holder again. Retire old free-space labels
        # whenever that spatial neighbourhood is actually carrying load.
        old_spans = [value for value in old_spans
                     if not np.any(held[0] & (abs(locations-value) < .75*LINK))]
        if old.get('role') == 'rear_release' and not old.get('handover_verified', False):
            old_spans.extend(old.get('released_center_parameters', [float(old['parameter'])]))
        scores = self.filtered_load*tracking
        for index in np.argsort(scores)[::-1]:
            joint = int(index)+1
            if not 2 <= joint <= 20:
                continue
            # This hinge directly changes its distal module. An already free
            # distal neighbour cannot be reused as a fresh lifting success for
            # the loaded proximal neighbour on the other side of the hinge.
            if not held[0, joint]:
                continue
            parameter = float(self.curve.reference_node_parameters[joint])
            if old.get('handover_verified', False) and abs(parameter-old['parameter']) < LINK:
                # The restored span carries the next action; immediately
                # releasing it again is not a distinct body-transfer step.
                continue
            modules = np.flatnonzero(abs(locations-parameter) < .75*LINK)
            modules = modules[held[0, modules]]
            remaining = held.copy()
            remaining[0, modules] = False
            for other in old_spans:
                remaining[0, abs(locations-other) < .75*LINK] = False
            if (not len(modules) or forces[0, modules].sum() <= .1 or
                    remaining[0].sum() < 2 or remaining[1].sum() < 3):
                continue
            if any(abs(parameter-other) < LINK for other in old_spans):
                continue
            cohort = np.flatnonzero(remaining[0]).tolist()
            same_candidate = old.get('next_release_joint') == joint
            continuous = sorted(set(old.get('next_release_continuous_modules', cohort)) & set(cohort)) if same_candidate else cohort
            if not same_candidate or len(continuous) < 2:
                continuous = cohort
                old['next_release_seen_t'] = t
            old.update(stage='regrip', next_release_pending=True,
                next_release_joint=joint, next_release_continuous_modules=continuous)
            old.setdefault('next_release_seen_t', t)
            if t-old['next_release_seen_t'] < 1.:
                # Continue the existing tail restoration and keep this loaded
                # candidate until two OTHER source rods have actually carried
                # continuously for the original handover interval.
                self.protect_local_carrying_rods(p, held, forces, t,
                    old['selected_joint'], self.yield_release_modules)
                return True
            self.curve.persist_tail_recovery_reference(self.shift)
            normal = p[joint]-np.array([0., 0., 1.])
            normal[0] = 0.
            normal /= max(np.linalg.norm(normal), 1e-9)
            receiver = old.get('receiving_unload', dict(parameter=old['parameter'],
                selected_joint=old['selected_joint'], unloaded_t=t,
                actual_force_N=float(forces[1, self.yield_release_modules].sum())))
            self.yield_state = dict(stage='unload', role='rear_release', side=0,
                parameter=parameter, started_t=t, selected_joint=joint,
                initial_angle_deg=float(abs(np.degrees(q[index]))),
                initial_force_N=float(forces[0, modules].sum()),
                initial_load_Nm=float(self.filtered_load[index]),
                anchor_goals_N=forces.tolist(), outward_normal=normal.tolist(),
                reference_origin=self.curve.sample(parameter).tolist(),
                receiving_unload=receiver, freed_source_parameters=old_spans,
                intended_center_parameters=locations[modules].tolist(), intended_material_modules=modules.tolist(),
                feed_total_pipe_origin=old.get('feed_total_pipe_origin', self.physical_pipe_progress.tolist()))
            self.yield_release_modules = modules.tolist()
            self.protect_local_carrying_rods(p, remaining, forces, t, joint, modules.tolist())
            self.events.append(dict(t_s=t, mode='front_carries_rear_span_unload',
                joint=joint, modules=modules.tolist(), current_source_force_N=float(forces[0, modules].sum()),
                remaining_source=np.flatnonzero(remaining[0]).tolist(),
                receiving_grasps=np.flatnonzero(remaining[1]).tolist(),
                meaning='curve lifts a measured posterior span; source contact must actually disappear before feed'))
            return True
        if self.transferred_material_modules and held[0].sum()>=2 and held[1].sum()>=3 and old['selected_joint']!=20:
            free_source = source & ~target & ~held[0]
            free_source[self.transferred_material_modules] = False
            if np.any(free_source[2:]):
                self.events.append(dict(t_s=t,mode='restore_replacement_after_successful_feed',
                    previous_feed_succeeded=old.get('feed_succeeded',False),
                    previous_material_mm=old.get('achieved_material_mm'),
                    meaning='keep actual loaded rods and completed progress; restore a free source carrier before next distinct action'))
                self.restore_posterior_after_transfer(p,centers,q,forces,held,t,locations,
                    list(self.transferred_material_modules),verify_target=False)
                return True
        if self.transferred_material_modules and old['selected_joint'] == 20:
            old.update(stage='regrip', next_release_pending=True)
            old.pop('next_release_joint', None)
            old.pop('next_release_seen_t', None)
            old.pop('next_release_continuous_modules', None)
            self.protect_local_carrying_rods(p, held, forces, t,
                old['selected_joint'], self.yield_release_modules)
            return True
        return False

    def begin_local_regrip(self, t, feed_succeeded, material_mm=None, held=None):
        state = self.yield_state
        state.update(stage='regrip', regrip_t=t, feed_succeeded=bool(feed_succeeded),
            achieved_material_mm=material_mm, handover_verified=False, refresh_anchor_goals=True)
        state.pop('failure', None)
        state.pop('handover_contact_seen_t', None)
        state.pop('handover_continuous_modules', None)
        state['handover_baseline_modules'] = np.flatnonzero(held[state['side']]).tolist() if held is not None else []
        self.events.append(dict(t_s=t, mode='local_yield_regrip' if feed_succeeded else 'feed_stalled_restore_current_grasp',
            actual_material_mm=material_mm,
            meaning='restore current released span before another release; restoring support alone is no successful feed'))

    def restore_posterior_after_transfer(self, p, centers, q, forces, held, t, locations, transferred, verify_target=True):
        """Keep the verified receiving rods; restore a rod still on source."""
        old = self.yield_state
        self.transferred_material_modules = sorted(set(self.transferred_material_modules+transferred))
        if verify_target:
            self.events.append(dict(t_s=t, mode='material_transfer_target_handover_verified',
                modules=transferred, carrier_side=1, original_action_side=old['side'],
                preceding_feed_succeeded=old.get('feed_succeeded', False),
                preceding_material_mm=old.get('achieved_material_mm'),
                meaning='same material holds actual target cylinder for 1s; keep it there; no feed-threshold override'))
        source, target = physical_wrap_masks(centers)
        candidates = [module for module in range(20, 1, -1)
            if source[module] and not target[module] and not held[0, module]
            and module not in transferred]
        if not candidates:
            old.update(stage='completed', handover_verified=True, handover_side=1,
                handover_observed_modules=transferred)
            return
        module = candidates[0]
        parameter = float(self.curve.reference_node_parameters[module])
        normal = p[module]-np.array([0., 0., 1.])
        normal[0] = 0.
        normal /= max(np.linalg.norm(normal), 1e-9)
        source_count = int(held[0].sum())
        carrying_share = float(forces[0, held[0]].sum())/max(1, source_count+1)
        previous_goal = float(old['anchor_goals_N'][0][module])
        goal = previous_goal if previous_goal>.1 else carrying_share
        anchor_goals = forces.copy()
        if not verify_target:
            for side in range(2):
                anchor_goals[side, held[side]] = np.maximum(anchor_goals[side, held[side]],
                    np.asarray(old['anchor_goals_N'])[side, held[side]])
        self.yield_state = dict(stage='regrip', role='rear_release', side=0,
            parameter=parameter, started_t=t, regrip_t=t, selected_joint=module,
            initial_angle_deg=float(abs(np.degrees(q[module-1]))),
            initial_force_N=float(forces[0, module]), initial_load_Nm=float(self.filtered_load[module-1]),
            recovery_force_goal_N=goal, anchor_goals_N=anchor_goals.tolist(),
            outward_normal=normal.tolist(), reference_origin=self.curve.sample(parameter).tolist(),
            intended_material_modules=[module], released_material_modules=[module],
            released_center_parameters=[float(locations[module])],
            freed_source_parameters=list(old.get('freed_source_parameters', [])),
            feed_total_pipe_origin=old.get('feed_total_pipe_origin', self.physical_pipe_progress.tolist()),
            feed_succeeded=False, handover_verified=False, handover_baseline_modules=np.flatnonzero(held[0]).tolist(),
            anchor_recovery_active=False, transferred_target_modules=transferred,
            preceding_action_joint=old['selected_joint'],
            preceding_feed_succeeded=old.get('feed_succeeded', False),
            preceding_material_mm=old.get('achieved_material_mm'))
        self.yield_release_modules = [module]
        self.protect_local_carrying_rods(p, held, forces, t, module, [module])
        self.events.append(dict(t_s=t, mode='restore_remaining_source_span_after_material_transfer',
            selected_joint=module, modules=[module], preserved_target_modules=transferred,
            actual_source_force_N=float(forces[0, module]), recovery_goal_N=goal,
            remaining_source=np.flatnonzero(held[0]).tolist(),
            meaning='fresh source normal and current load share; recovery alone is not successful feed'))

    def update_localized_yield(self, p, centers, q, tracking, forces, t, locations):
        if not self.localized_yield_enabled:
            return
        source, target = physical_wrap_masks(centers)
        held = np.asarray([source, target]) & (forces > .1)
        held[0, self.transferred_material_modules] = False
        if self.yield_state is None:
            loaded = (self.filtered_load > 2.1) & (tracking > np.radians(2.5)) & (q*(self.last-q) > 0.)
            scores = self.filtered_load*tracking*loaded
            for index in np.argsort(scores)[::-1]:
                joint = int(index)+1
                if scores[index] <= 0. or not 2 <= joint <= 19:
                    continue
                parameter = float(self.curve.reference_node_parameters[joint])
                modules = np.flatnonzero(abs(locations-parameter) < .75*LINK)
                side = int(np.argmax(forces[:, modules].sum(1)))
                remaining = held.copy()
                remaining[side, modules] = False
                if forces[side, modules].sum() < .1 or remaining[0].sum() < 2 or remaining[1].sum() < 3:
                    continue
                axis = np.eye(3)[side]
                origin = np.array([0., 0., 1.]) if side == 0 else CORNER
                normal = p[joint]-origin
                normal -= (normal@axis)*axis
                normal /= max(np.linalg.norm(normal), 1e-9)
                self.yield_state = dict(stage='unload', side=side, parameter=parameter, started_t=t,
                    selected_joint=joint, initial_angle_deg=float(abs(np.degrees(q[index]))),
                    initial_force_N=float(forces[side, modules].sum()),
                    initial_load_Nm=float(self.filtered_load[index]),
                    anchor_goals_N=forces.tolist(), outward_normal=normal.tolist(),
                    reference_origin=self.curve.sample(parameter).tolist(), intended_material_modules=modules.tolist())
                self.events.append(dict(t_s=t, mode='local_yield_unload', joint=joint,
                    modules=modules.tolist(), side=side, load_Nm=float(self.filtered_load[index]),
                    tracking_deg=float(np.degrees(tracking[index])),
                    remaining_source=np.flatnonzero(remaining[0]).tolist(),
                    remaining_target=np.flatnonzero(remaining[1]).tolist(),
                    action='outward local plateau and endpoint-anchored curvature relief; other observed grasps retain their loads'))
                break
        if self.yield_state is None:
            return
        state = self.yield_state
        # Keep this selected physical hinge and its released rods for the
        # duration of this action. Its location follows the shared curve map;
        # the next action is selected afresh from current measured contacts.
        state['parameter'] = float(self.curve.reference_node_parameters[state['selected_joint']])
        state['freed_source_parameters'] = [value for value in state.get('freed_source_parameters', [])
            if not np.any(held[0] & (abs(locations-value) < .75*LINK))]
        modules = np.flatnonzero(abs(locations-state['parameter']) < .75*LINK)
        if state.get('intended_material_modules'):
            modules = np.asarray(state['intended_material_modules'], dtype=int)
        elif state.get('intended_center_parameters'):
            modules = np.unique([int(np.argmin(abs(locations-value))) for value in state['intended_center_parameters']])
        if state.get('released_material_modules'):
            modules = np.asarray(state['released_material_modules'], dtype=int)
        elif state.get('released_center_parameters'):
            modules = np.unique([int(np.argmin(abs(locations-value))) for value in state['released_center_parameters']])
        self.yield_release_modules = modules.tolist()
        if state.pop('refresh_anchor_goals', False):
            # Carrying roles changed during feed. Old low-pressure goals must
            # not open a newly loaded neighbour while restoring the free span.
            for side in range(2):
                for module in np.flatnonzero(held[side]):
                    if side != state['side'] or module not in modules:
                        state['anchor_goals_N'][side][module] = float(forces[side, module])
            self.events.append(dict(t_s=t, mode='regrip_capture_current_carrying_loads',
                source_holders=np.flatnonzero(held[0]).tolist(), target_holders=np.flatnonzero(held[1]).tolist()))
        if state['stage'] == 'regrip' and state['side'] == 0:
            transferred = modules[held[1, modules] & ~held[0, modules]].tolist()
            if transferred and held[0].sum() >= 2 and held[1].sum() >= 3:
                state['target_handover_pending'] = True
                continuous = sorted(set(state.get('target_handover_continuous_modules', transferred)) & set(transferred))
                if not continuous:
                    continuous = transferred
                    state['target_handover_seen_t'] = t
                state['target_handover_continuous_modules'] = continuous
                state.setdefault('target_handover_seen_t', t)
                if t-state['target_handover_seen_t'] >= 1.:
                    self.restore_posterior_after_transfer(p, centers, q, forces, held, t, locations, continuous)
                    state = self.yield_state
                    modules = np.asarray(self.yield_release_modules, dtype=int)
            else:
                state['target_handover_pending'] = False
                state.pop('target_handover_seen_t', None)
                state.pop('target_handover_continuous_modules', None)
        joint = int(state['selected_joint'])
        current_force = float(forces[state['side'], modules].sum())
        if state['stage'] == 'unload' and t-state['started_t'] >= 12.:
            state.update(stage='failed', failure='local_release_not_observed_12s')
            return
        if (state['stage'] == 'feed' and state.get('anchor_recovery_active', False)
                and np.any(held[state['side'], modules])
                and (not state.get('contact_retained_feed',False)
                     or state.get('contact_loss_after_partial_feed',False))):
            material = (self.physical_pipe_progress-np.asarray(state['feed_pipe_origin']))/self.physical_axial_sines
            returned = modules[held[state['side'], modules]].tolist()
            self.begin_local_regrip(t, min(material) >= .008, float(min(material)*1000), held)
            self.events.append(dict(t_s=t, mode='released_span_returns_during_anchor_recovery',
                modules=returned, meaning='contact returned; enter regrip and apply original 1s confirmation, no immediate handover'))
        remaining = held.copy()
        if state.get('role') == 'rear_release' and state['stage'] == 'unload':
            # The candidate remains a real holder until lifted, but cannot
            # replace the two OTHER rods that must carry its release. Detect
            # their loss immediately even while this candidate is still loaded.
            remaining[0, modules] = False
        elif state['stage'] not in ['regrip', 'completed']:
            remaining[state['side'], modules] = False
        for other in state.get('freed_source_parameters', []):
            remaining[0, abs(locations-other) < .75*LINK] = False
        if remaining[0].sum() < 2 or remaining[1].sum() < 3:
            if not state.get('anchor_recovery_active', False):
                state['anchor_loss_t'] = t
                self.events.append(dict(t_s=t, mode='local_yield_pause_restore_anchors',
                    remaining_source=np.flatnonzero(remaining[0]).tolist(),
                    remaining_target=np.flatnonzero(remaining[1]).tolist()))
            state['anchor_recovery_active'] = True
            safely_holding_coordinated_action = (state['stage']=='feed'
                and state.get('coordinated_feed_active',False)
                and held[0].sum()>=2 and held[1].sum()>=3)
            if t-state['anchor_loss_t'] >= 1. and not safely_holding_coordinated_action:
                state['stage'] = 'failed'
                state['failure'] = 'remaining_observed_anchors_insufficient_1s'
        else:
            if state.get('anchor_recovery_active', False):
                # The loss timer measures continuous actual insufficiency.
                # Restored holders still need their original one-second
                # confirmation, but that interval is not continuing loss.
                state['anchor_loss_t'] = t
                if state.get('role') == 'rear_release' and state['stage'] in ['unload','feed']:
                    carrying = held[0].copy()
                    carrying[modules] = False
                    cohort = np.flatnonzero(carrying).tolist()
                    continuous = sorted(set(state.get('anchor_handover_continuous_modules', cohort)) & set(cohort))
                    if len(continuous) < 2 or held[1].sum() < 3:
                        state.pop('anchor_handover_seen_t', None)
                        state.pop('anchor_handover_continuous_modules', None)
                        if (held[0].sum() >= 2 and held[1].sum() >= 3
                                and not state.get('coordinated_feed_active',False)):
                            transferred = list(self.transferred_material_modules)
                            self.events.append(dict(t_s=t, mode='keep_loaded_release_candidate_restore_tail',
                                kept_modules=modules.tolist(), remaining_source=cohort,
                                meaning='do not count the intended released rod as its own replacement'))
                            self.restore_posterior_after_transfer(p, centers, q, forces, held,
                                t, locations, transferred, verify_target=False)
                        return
                    state['anchor_handover_continuous_modules'] = continuous
                    state.setdefault('anchor_handover_seen_t', t)
                    if t-state['anchor_handover_seen_t'] < 1.:
                        return
                self.events.append(dict(t_s=t, mode='local_yield_anchors_recovered'))
            state['anchor_recovery_active'] = False
            state.pop('anchor_loss_t', None)
        if state['stage'] == 'failed':
            return
        if (state['stage'] == 'completed' or state.get('next_release_pending', False)) and self.alternating_yield_enabled:
            if state.get('handover_verified', False):
                if self.start_rear_release(p, centers, q, tracking, forces, t, locations):
                    return
                state.update(stage='failed', failure='no_supported_distinct_rear_release_after_regrip')
                return
            else:
                # A legacy elapsed regrip stage is not observed grasping.
                # Preserve its existing timer and wait for actual holders.
                state['stage'] = 'regrip'
                state.setdefault('regrip_t', t)
        if state['stage'] == 'unload':
            relieved = (current_force < .3*state['initial_force_N'] and self.filtered_load[joint-1] < 1.9)
            if state.get('role') == 'rear_release':
                free_modules = []
                for module in modules[forces[0, modules] <= .1]:
                    intersecting = False
                    for contact in self.d.contact:
                        for robot, pipe in [(contact.geom1, contact.geom2), (contact.geom2, contact.geom1)]:
                            if (int(self.m.geom_bodyid[robot]) == self.head+module and contact.dist <= 0. and
                                    (self.m.geom(pipe).name or '') in ['pipe_straight_01', 'pipe_sharp_in_02']):
                                intersecting = True
                    if not intersecting:
                        free_modules.append(int(module))
                relieved = (bool(free_modules) and self.filtered_load[joint-1] < 1.9)
                # Contact loss remains a recorded event, not the only way to
                # try transport. A measured reduction in load can admit one
                # small COMMON shift while the other observed rods carry.
                reduced = (current_force < .9*state['initial_force_N']
                    and self.filtered_load[joint-1] < 1.9
                    and remaining[0].sum() >= 2 and remaining[1].sum() >= 3
                    and not state['anchor_recovery_active'])
                if reduced and not state.get('coordinated_feed_active', False):
                    error = self.curve.persist_tail_recovery_reference(self.shift)
                    state.update(coordinated_feed_active=True,
                        coordination_shift_origin=self.shift,
                        coordination_pipe_origin=self.physical_pipe_progress.tolist(),
                        stage='feed', feed_t=t, feed_shift=self.shift,
                        feed_pipe_origin=self.physical_pipe_progress.tolist(),
                        feed_reference_registered=True, contact_retained_feed=bool(held[0,joint]))
                    self.events.append(dict(t_s=t, mode='partial_unload_common_feed_admitted',
                        module=joint, actual_force_N=current_force,
                        initial_force_N=state['initial_force_N'], still_contacting=bool(held[0, joint]),
                        maximum_reference_point_change_mm=error,
                        meaning='measured unloading admits bounded common transport; contact loss and actual progress remain separate'))
                    return
            if t-state['started_t'] >= 2. and relieved and not state['anchor_recovery_active']:
                if self.alternating_yield_enabled and state.get('role') != 'rear_release':
                    self.events.append(dict(t_s=t, mode='receiving_local_unload_observed',
                        current_joint=joint, actual_force_N=current_force))
                    if self.start_rear_release(p, centers, q, tracking, forces, t, locations):
                        return
                self.curve.persist_tail_recovery_reference(self.shift)
                state.update(stage='feed', feed_t=t, feed_shift=self.shift,
                    feed_pipe_origin=self.physical_pipe_progress.tolist())
                if state.get('role') == 'rear_release':
                    state['released_center_parameters'] = locations[free_modules].tolist()
                    state['released_material_modules'] = free_modules
                    retained = [int(i) for i in modules if i not in free_modules]
                    for module in retained:
                        state['anchor_goals_N'][0][module] = float(forces[0, module])
                    self.yield_release_modules = free_modules
                    self.events.append(dict(t_s=t, mode='rear_contact_lift_observed_front_carries',
                        lifted_modules=free_modules, retained_loaded_neighbors=retained,
                        remaining_source=np.flatnonzero(held[0]).tolist(), receiving_grasps=np.flatnonzero(held[1]).tolist(),
                        meaning='lifted span has no intersecting source contact; loaded neighbour stays a holder'))
                self.events.append(dict(t_s=t, mode='local_yield_feed',
                    released_force_N=float(forces[state['side'], self.yield_release_modules].sum()),
                    initially_considered_span_force_N=current_force,
                    current_joint=joint, angle_deg=float(np.degrees(q[joint-1])),
                    meaning='local unloading observed; now resume the existing bounded common shift'))
            elif t-state['started_t'] >= 12.:
                state.update(stage='failed', failure='local_release_not_observed_12s')
        elif state['stage'] == 'feed':
            if (state.get('contact_retained_feed', False)
                    and not state.get('contact_loss_after_partial_feed', False)
                    and current_force <= .1):
                intersecting = any(contact.dist <= 0. and any(
                    int(self.m.geom_bodyid[robot]) == self.head+joint and
                    (self.m.geom(pipe).name or '') in ['pipe_straight_01','pipe_sharp_in_02']
                    for robot,pipe in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)])
                    for contact in self.d.contact)
                if not intersecting:
                    state['contact_loss_after_partial_feed'] = True
                    self.events.append(dict(t_s=t, mode='actual_contact_loss_after_partial_feed',
                        modules=modules.tolist(), actual_force_N=current_force,
                        meaning='physical lift recorded separately from earlier load-reduced feed admission'))
            material = (self.physical_pipe_progress-np.asarray(state['feed_pipe_origin']))/self.physical_axial_sines
            if min(material) >= .008 or (not self.alternating_yield_enabled and t-state['feed_t'] >= 35.):
                self.begin_local_regrip(t, min(material) >= .008, float(min(material)*1000), held)
            elif self.alternating_yield_enabled and t-state['feed_t'] >= 35.:
                self.begin_local_regrip(t, False, float(min(material)*1000), held)
        elif state['stage'] == 'regrip':
            side = state['side']
            actual_holders = np.flatnonzero(held[side]).tolist()
            baseline = set(state.get('handover_baseline_modules', []))
            region_holders = [int(i) for i in modules if held[side, i]]
            new_holders = [int(i) for i in actual_holders if i not in baseline]
            observed = sorted(set(region_holders+(new_holders if state.get('feed_succeeded', True) else [])))
            if observed and not state['anchor_recovery_active']:
                continuous = sorted(set(state.get('handover_continuous_modules', observed)) & set(observed))
                if not continuous:
                    state['handover_contact_seen_t'] = t
                    continuous = observed
                state['handover_continuous_modules'] = continuous
                state.setdefault('handover_contact_seen_t', t)
                if t-state['handover_contact_seen_t'] >= 1.:
                    had_boundary = getattr(self.curve, 'tail_recovery_boundary', None) is not None
                    persistence_error = self.curve.persist_tail_recovery_reference(self.shift)
                    if had_boundary:
                        self.events.append(dict(t_s=t, mode='local_reference_persisted_before_handover',
                            maximum_reference_point_change_mm=persistence_error,
                            shift_mm=self.shift*1000,
                            meaning='final reference baked into the same lasting route before removing protection and allowing common feed'))
                    state.update(stage='completed', handover_verified=True,
                        handover_observed_modules=continuous)
                    self.events.append(dict(t_s=t,
                        mode='actual_grasp_handover_observed' if state.get('feed_succeeded', True) else 'actual_grasp_restored_after_feed_stall',
                        side=side, holder_modules=continuous, current_local_force_N=current_force,
                        feed_succeeded=state.get('feed_succeeded', True),
                        meaning='same actual holder remains for 1s; restoration alone is no body-transfer success'))
            else:
                state.pop('handover_contact_seen_t', None)
                state.pop('handover_continuous_modules', None)

    def command(self, data, t, forces, torque, support_forces=None):
        self.current_control_time = t
        self.observer_rebased = False
        self.physical_observer_just_initialized = False
        p, centers = self.observe(data)
        q = data.qpos[self.qa]
        actual_material = self.observe_progress(centers, forces, t)
        reference_p, reference_c, locations = self.curve.reference(self.shift)
        if t-self.last_route_registration_time > 20. and self.yield_state is None:
            bad_chord = self.pipe_geometry.link_minimum(reference_p).min() < .0145
            wrong_side = np.linalg.norm(reference_c-centers, axis=1).max() > .12
            if (bad_chord or wrong_side) and self.repair_registered_route(p, centers, t, forces):
                reference_p, reference_c, locations = self.curve.reference(self.shift)
        self.filtered_load += .1*(np.abs(torque)-self.filtered_load)
        if self.support_force_goals is None:
            # Preserve the measured supported loading scale, not captured joint
            # target signs. Distribution changes with current contact modules.
            self.support_force_goals = np.maximum(forces.sum(1), 1.)
        tracking = np.abs(self.last-q)
        support_forces = forces if support_forces is None else support_forces
        self.update_localized_yield(p, centers, q, tracking, support_forces, t, locations)
        if (self.alternating_yield_enabled and self.yield_state is not None
                and self.yield_state['stage'] == 'feed'
                and not self.yield_state.get('feed_reference_registered', False)
                and t >= self.yield_state.get('next_feed_registration_t', -1.)):
            # The actual grasp/release cycle changed the supported shape. Reuse
            # the existing ONLINE observed-route registration once per feed;
            # never keep rolling back toward the previous action's pose.
            # This preserves shared shift and the historical physical meter.
            state = self.yield_state
            state['next_feed_registration_t'] = t+.5
            if self.repair_registered_route(p, centers, t, support_forces):
                reference_p, reference_c, locations = self.curve.reference(self.shift)
                state['parameter'] = float(self.curve.reference_node_parameters[state['selected_joint']])
                state['reference_origin'] = self.curve.sample(state['parameter']).tolist()
                state['feed_reference_registered'] = True
                self.events.append(dict(t_s=t, mode='feed_reference_matches_current_supported_shape',
                    shared_shift_mm=self.shift*1000, physical_pipe_progress_mm=(self.physical_pipe_progress*1000).tolist(),
                    meaning='reference correspondence only; no physical/root/control/progress reset or feed credit'))
        maximum_error = float(tracking.max())
        front_progress = float(np.median(centers[:7, 1]-self.initial_centers[:7, 1]))
        rear_progress = float(np.median(centers[14:, 0]-self.initial_centers[14:, 0]))
        self.last_body_progress = actual_material
        if not self.local_reference_meter_ready:
            self.local_reference_meter_ready = True
            self.local_reference_shift_origin = self.physical_progress_origin_shift
            self.local_reference_pipe_progress_origin = np.zeros(2)
        historical_lag = self.shift-self.physical_progress_origin_shift-actual_material
        local_pipe_material = (self.physical_pipe_progress-self.local_reference_pipe_progress_origin)/self.physical_axial_sines
        self.progress_history.append((t, actual_material))
        self.progress_history = [(a, b) for a, b in self.progress_history if a >= t-10.]
        actual_speed = ((actual_material-self.progress_history[0][1])/
                        max(t-self.progress_history[0][0], DT))
        error_scale = np.clip((np.radians(18)-maximum_error)/np.radians(14), 0., 1.)
        source_count, target_count = [int(np.count_nonzero(f > .1)) for f in forces]
        support_forces = forces if support_forces is None else support_forces
        source_grip_mask, target_grip_mask = physical_wrap_masks(centers)
        posterior_count = int(np.count_nonzero((support_forces[0] > .1) & source_grip_mask))
        receiver_count = int(np.count_nonzero((support_forces[1] > .1) & target_grip_mask))
        local_material = float(local_pipe_material[1] if target_grip_mask[-1] else min(local_pipe_material))
        lag = self.shift-self.local_reference_shift_origin-local_material
        lag_scale = np.clip((.075-lag)/.05, 0., 1.)
        support_scale = 1. if receiver_count >= 2 and (posterior_count >= 2 or target_grip_mask[-1]) else 0.
        requested = .004*float(error_scale)*float(lag_scale)*support_scale*smooth(t/2.)
        physical_speeds = self.physical_pipe_speed/self.physical_axial_sines
        occupied_speeds = [physical_speeds[1]] if target_grip_mask[-1] else list(physical_speeds)
        slower = float(min(occupied_speeds))
        # A small, bounded lead starts rolling from rest. It disappears at
        # 15mm of unmatched feed; actual retreat brakes both pipe regions.
        bootstrap = .0005*float(np.clip((.015-lag)/.015, 0., 1.))
        physical_limit = 0. if slower < -.0002 else max(0., slower)+bootstrap
        requested = min(requested, physical_limit)
        if self.yield_state is not None and self.yield_state['stage'] in ['unload', 'failed']:
            requested = 0.
        if self.yield_state is not None:
            state = self.yield_state
            if state['stage'] in ['unload','feed'] and state.get('coordinated_feed_active',False):
                action_material = (self.physical_pipe_progress-np.asarray(state['coordination_pipe_origin']))/self.physical_axial_sines
                action_lag = self.shift-state['coordination_shift_origin']-float(min(action_material))
                action_bootstrap = .0005*float(np.clip((.015-action_lag)/.015,0.,1.))
                probe_remaining = max(0., .001-(self.shift-state['coordination_shift_origin']))
                # A finite 1mm shared-reference probe tests coordinated feed
                # even during the preceding reshape retreat. Once consumed,
                # the original measured-retreat brake controls continuation.
                requested = (min(.004*float(error_scale)*float(np.clip((.075-action_lag)/.05,0.,1.)),
                    max(0.,slower)+action_bootstrap)
                    if slower >= -.0002 else min(.0005,probe_remaining/DT))
        if self.yield_state is not None and self.yield_state.get('anchor_recovery_active', False):
            requested = 0.
        restoring_stalled_grasp = (self.yield_state is not None and self.yield_state['stage'] == 'regrip')
        if restoring_stalled_grasp:
            requested = 0.
        self.speed += np.clip(requested-self.speed, -.002*DT, .001*DT)
        mode = 'advance' if self.speed > .00015 else 'reshape_with_shift_braked'
        if self.yield_state is not None:
            mode = 'local_yield_'+self.yield_state['stage']+'_'+mode
        if mode != self.last_mode:
            self.events.append(dict(t_s=t, mode=mode, shift_mm=self.shift*1000,
                maximum_tracking_error_deg=math.degrees(maximum_error), lag_mm=lag*1000))
            self.last_mode = mode
        if t-self.last_shape_time >= .5:
            self.feedback_shape(p, q, reference_c, locations, tracking, forces, .5)
            self.last_shape_time = t
        # Separate deformation at the SAME shift from feed caused by the one
        # shared shift. Reshaping still contributes to pose/tangent tracking,
        # but is never reported as nominal material advance or nominal spin.
        reshaped_p, reshaped_c, _ = self.curve.reference(self.shift)
        shape_feedforward = (reshaped_c-reference_c)/DT
        ds = DT*self.speed
        if self.yield_state is not None and self.yield_state.get('coordinated_feed_active',False) and slower < -.0002:
            ds = min(ds,max(0.,.001-(self.shift-self.yield_state['coordination_shift_origin'])))
            self.speed = ds/DT
        self.curve.increment(ds)
        self.shift += ds
        wanted_p, wanted_c, locations = self.curve.reference(self.shift)
        feedforward = (wanted_c-reshaped_c)/DT
        source_grip_mask, target_grip_mask = self.supported_curve_regions(locations, centers)

        n = self.m.nv
        A, b, physical_A, physical_b, task_labels = [], [], [], [], []
        jp, jr = np.zeros((3, n)), np.zeros((3, n))
        def add(j, value, weight, label, physical_j=None, physical_value=None):
            A.append(weight*np.atleast_2d(j).copy())
            b.append(weight*np.atleast_1d(value).copy())
            physical_A.append(weight*np.atleast_2d(j if physical_j is None else physical_j).copy())
            physical_b.append(weight*np.atleast_1d(value if physical_value is None else physical_value).copy())
            task_labels.extend([label]*len(np.atleast_1d(value)))
        weights = self.curve.weights(locations)
        current_regions = self.curve.regions(locations)
        source_axial = float(np.median(feedforward[current_regions == 0, 0])) if np.any(current_regions == 0) else 0.
        target_axial = float(np.median(feedforward[current_regions == 2, 1])) if np.any(current_regions == 2) else 0.
        source_roll = float(np.clip(-np.sign(self.curve.theta_rates[0])*source_axial/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[0]**2)), -.12, .12))
        target_roll = float(np.clip(np.sign(self.curve.theta_rates[1])*target_axial/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[1]**2)), -.12, .12))
        nominal_roll = np.array([source_roll, target_roll])
        recovery_roll = np.zeros(2)
        correction = .35*(wanted_c-centers)
        if self.yield_state is not None and (self.yield_state['stage'] in ['unload', 'failed'] or
                (self.yield_state.get('role') == 'rear_release' and self.yield_state['stage']=='feed')):
            # Switching from restoration to unload must not reactivate an old
            # whole-side pose error as backward rolling. Coordinated rolling
            # here comes only from THIS step's curve deformation; local normal
            # pressure and the actual-lever angular/contact field remain.
            correction = shape_feedforward.copy()
        correction *= np.minimum(1., .008/np.maximum(np.linalg.norm(correction, axis=1), 1e-9))[:, None]
        for side, code, supported_mask in [(0, 0, source_grip_mask), (1, 2, target_grip_mask)]:
            mask = supported_mask.copy() if self.alternating_yield_enabled else (current_regions == code) & supported_mask
            if self.yield_state is not None and side == self.yield_state['side'] and self.yield_state['stage'] in ['unload', 'feed']:
                mask[self.yield_release_modules] = False
            if not mask.any():
                continue
            axis = np.eye(3)[side]
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            radial = centers[mask]-origin
            radial -= (radial@axis)[:, None]*axis
            radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-9)
            azimuth = (1 if side == 0 else -1)*np.cross(axis, radial)
            alpha = self.curve.axial_sines[side]
            # Unit direction of rolling helix translation. Its radial
            # component is zero; bridge deformation is excluded by the mask.
            direction = math.sqrt(1-alpha*alpha)*axis-np.sign(self.curve.theta_rates[side])*alpha*azimuth
            along_helix = float(np.median(np.sum(correction[mask]*direction, axis=1)))
            sign = -np.sign(self.curve.theta_rates[side]) if side == 0 else np.sign(self.curve.theta_rates[side])
            recovery_roll[side] = sign*along_helix/BODY_RADIUS
        if restoring_stalled_grasp:
            # This phase closes the curve/pressure deficit while current
            # holders carry. Axial rolling resumes after actual restoration.
            recovery_roll[:] = 0.
        source_roll, target_roll = np.clip(nominal_roll+recovery_roll, -.12, .12)
        selected = {}
        for contact in self.d.contact:
            for robot, pipe in [(contact.geom1, contact.geom2), (contact.geom2, contact.geom1)]:
                body = int(self.m.geom_bodyid[robot])
                name = self.m.geom(pipe).name or ''
                if self.head <= body < self.head+21 and name.startswith('pipe_'):
                    module = body-self.head
                    side = 1 if name in ['pipe_sharp_out_03', 'pipe_straight_04'] else 0
                    key = (module, side)
                    if key not in selected or contact.dist < selected[key][0]:
                        selected[key] = (contact.dist, contact.pos.copy(), contact.frame[:3].copy())
        pressure_targets = {}
        for (module, side), (_, point, contact_normal) in selected.items():
            axis = np.eye(3)[side]
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            normal = point-origin
            normal -= normal@axis*axis
            normal /= max(np.linalg.norm(normal), 1e-9)
            mask = source_grip_mask if side == 0 else target_grip_mask
            is_support = mask[module] and cylindrical_contact(side, point, contact_normal)
            count = max(1, int(np.count_nonzero(mask & (support_forces[side] > .1))))
            goal = self.support_force_goals[side]/count if is_support else 0.
            peeling = False
            if self.yield_state is not None:
                state = self.yield_state
                released = side == state['side'] and module in self.yield_release_modules
                released = released or (side == 0 and any(abs(locations[module]-other) < .75*LINK
                    for other in state.get('freed_source_parameters', [])))
                goals = np.asarray(state['anchor_goals_N'])
                if released:
                    if state.get('anchor_recovery_active', False):
                        factor = 1.
                    elif state['stage'] == 'regrip':
                        factor = float(np.clip((t-state['regrip_t'])/8., 0., 1.))
                    else:
                        factor = 1. if state['stage'] == 'completed' else 0.
                    goal = state.get('recovery_force_goal_N', .5*state['initial_force_N'])/max(1, len(self.yield_release_modules))*factor
                    if state.get('target_handover_pending', False) and side == 0:
                        goal = float(forces[side, module])
                        is_support = False
                    peeling = state['stage'] in ['unload', 'feed'] and not state.get('anchor_recovery_active', False)
                    is_support = is_support and not peeling
                else:
                    if goals[side, module] <= .1 and forces[side, module] > .1:
                        state['anchor_goals_N'][side][module] = float(forces[side, module])
                    goal = float(state['anchor_goals_N'][side][module]) if is_support else 0.
            if side == 0 and module in self.transferred_material_modules:
                # An incidental source-cylinder recontact does not move a
                # verified target carrier back into a source-clamping role.
                goal, is_support, peeling = 0., False, True
            vn = float(np.clip(.00002*(forces[side, module]-goal), -.0006, .0006))
            pressure_targets[module, side] = (normal, vn, is_support, peeling)

        axes = self.d.xmat[self.head:self.head+21].reshape(21, 3, 3)[:, :, 2]
        tangent_feedforward, perpendicular_omega, perpendicular_recovery, nominal_perpendicular_omega = [], [], [], []
        for module, u in enumerate(axes):
            wanted_u = wanted_p[module]-wanted_p[module+1]
            wanted_u /= max(np.linalg.norm(wanted_u), 1e-9)
            previous_u = reference_p[module]-reference_p[module+1]
            previous_u /= max(np.linalg.norm(previous_u), 1e-9)
            rotation_axis = np.cross(previous_u, wanted_u)
            sine = np.linalg.norm(rotation_axis)
            angle = math.atan2(sine, float(previous_u@wanted_u))
            omega_ff = rotation_axis*(angle/(max(sine, 1e-12)*DT))
            tangent_feedforward.append(omega_ff)
            perpendicular = np.eye(3)-np.outer(u, u)
            pose_omega = perpendicular@(.35*np.cross(u, wanted_u))
            perpendicular_recovery.append(pose_omega)
            perpendicular_omega.append(perpendicular@omega_ff+pose_omega)
            reshaped_u = reshaped_p[module]-reshaped_p[module+1]
            reshaped_u /= max(np.linalg.norm(reshaped_u), 1e-9)
            common_axis = np.cross(reshaped_u, wanted_u)
            common_sine = np.linalg.norm(common_axis)
            common_angle = math.atan2(common_sine, float(reshaped_u@wanted_u))
            nominal_perpendicular_omega.append(perpendicular@(
                common_axis*common_angle/(max(common_sine, 1e-12)*DT)))
        perpendicular_omega = np.asarray(perpendicular_omega)
        nominal_perpendicular_omega = np.asarray(nominal_perpendicular_omega)
        rolling_contacts, roll_scale, axial_recovery_removed = {}, np.ones(2), []
        contact_material_roll, contact_nominal_roll, contact_roll_geometry = {}, {}, []
        for module, center in enumerate(centers):
            region = current_regions[module]
            side = 0 if region == 0 else 1 if region == 2 else None
            if (module in self.transferred_material_modules and
                    (module, 1) in pressure_targets and pressure_targets[module, 1][2]
                    and support_forces[1, module] > .1):
                side = 1
            if self.alternating_yield_enabled:
                actual_sides = [value for value in range(2) if (module, value) in pressure_targets
                    and pressure_targets[module, value][2] and support_forces[value, module] > .1]
                if actual_sides:
                    side = max(actual_sides, key=lambda value: support_forces[value, module])
            key = (module, side)
            if key not in pressure_targets or not pressure_targets[key][2] or support_forces[side, module] <= .1:
                continue
            point = selected[key][1]
            normal, vn, _, _ = pressure_targets[key]
            if self.alternating_yield_enabled and self.yield_state is not None and self.yield_state.get('role') == 'rear_release':
                # Keep normal-plane pose recovery, but remove its accidental
                # axial centre motion at this actual supporting contact. Common
                # feed and material rolling retain their signs and authority.
                lever = point-center
                perpendicular = np.eye(3)-np.outer(axes[module], axes[module])
                axial_row = perpendicular@np.cross(np.eye(3)[side], lever)
                norm = float(axial_row@axial_row)
                if norm > 1e-12:
                    removed = axial_row*float(axial_row@perpendicular_recovery[module])/norm
                    perpendicular_omega[module] -= removed
                    axial_recovery_removed.append(module)
            spin = source_roll if side == 0 else target_roll
            if self.alternating_yield_enabled:
                lever = point-center
                coefficient = float(-np.cross(axes[module], lever)[side])
                fitted_coefficient = BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[side]**2)*(
                    -np.sign(self.curve.theta_rates[side]) if side == 0 else np.sign(self.curve.theta_rates[side]))
                axial_goal = spin*fitted_coefficient
                required = axial_goal+float(np.cross(perpendicular_omega[module], lever)[side])
                spin = float(np.clip(required/coefficient, -.12, .12)) if abs(coefficient) > 1e-6 else 0.
                nominal_goal = nominal_roll[side]*fitted_coefficient
                nominal_required = nominal_goal+float(np.cross(nominal_perpendicular_omega[module], lever)[side])
                contact_nominal_roll[module] = (float(np.clip(nominal_required/coefficient, -.12, .12))
                    if abs(coefficient) > 1e-6 else 0.)
                contact_roll_geometry.append(dict(module=module, side=side,
                    actual_axial_per_spin_mm=coefficient*1000,
                    common_axial_goal_before_scale_mm_s=axial_goal*1000,
                    bounded_material_roll_rad_s=spin))
            contact_material_roll[module] = spin
            omega = perpendicular_omega[module]+spin*axes[module]
            motion = -np.cross(omega, point-center)
            # The existing centre cap scales the whole side's angular/linear
            # rolling motion together, never translation independently of spin.
            roll_scale[side] = min(roll_scale[side], (.008-abs(vn))/max(np.linalg.norm(motion), 1e-9))
            rolling_contacts[module] = (side, point, normal, vn)
        source_roll *= roll_scale[0]
        target_roll *= roll_scale[1]
        nominal_roll *= roll_scale
        recovery_roll *= roll_scale
        for module, (side, _, _, _) in rolling_contacts.items():
            perpendicular_omega[module] *= roll_scale[side]
            contact_material_roll[module] *= roll_scale[side]
            nominal_perpendicular_omega[module] *= roll_scale[side]
            if module in contact_nominal_roll:
                contact_nominal_roll[module] *= roll_scale[side]

        shared_contact_motions, shared_curve_field_error = {}, []
        if self.alternating_yield_enabled:
            nominal_contact_motions = {}
            for module, (side, point, normal, vn) in rolling_contacts.items():
                omega = perpendicular_omega[module]+contact_material_roll[module]*axes[module]
                shared_contact_motions[module] = (side, vn*normal-np.cross(omega, point-centers[module]))
                nominal_omega = nominal_perpendicular_omega[module]+contact_nominal_roll[module]*axes[module]
                nominal_contact_motions[module] = (side, -np.cross(nominal_omega, point-centers[module]))
            if ds > 0.:
                wanted_p, wanted_c, locations = self.curve.transport_contacts(
                    self.shift, reshaped_c, locations, nominal_contact_motions)
            else:
                self.curve.contact_transport_fraction = 0.
            feedforward = (wanted_c-reshaped_c)/DT
            weights = self.curve.weights(locations)
            current_regions = self.curve.regions(locations)
            for module, (_, velocity) in nominal_contact_motions.items():
                curve_velocity = (wanted_c[module]-reshaped_c[module])/DT
                old_u = reshaped_p[module]-reshaped_p[module+1]
                new_u = wanted_p[module]-wanted_p[module+1]
                old_u /= max(np.linalg.norm(old_u), 1e-9)
                new_u /= max(np.linalg.norm(new_u), 1e-9)
                axis_change = np.cross(old_u, new_u)
                sine = np.linalg.norm(axis_change)
                curve_omega = axis_change*math.atan2(sine, float(old_u@new_u))/(max(sine, 1e-12)*DT)
                perpendicular = np.eye(3)-np.outer(axes[module], axes[module])
                shared_curve_field_error.append(dict(module=module,
                    curve_velocity_mm_s=(curve_velocity*1000).tolist(),
                    nominal_contact_velocity_mm_s=(velocity*1000).tolist(),
                    full_contact_task_velocity_mm_s=(shared_contact_motions[module][1]*1000).tolist(),
                    shape_curve_velocity_mm_s=(shape_feedforward[module]*1000).tolist(),
                    recovery_contact_velocity_mm_s=((shared_contact_motions[module][1]-velocity)*1000).tolist(),
                    recorded_reference_velocity_mm_s=((wanted_c[module]-reference_c[module])/DT*1000).tolist(),
                    fixed_chord_residual_mm_s=float(np.linalg.norm(curve_velocity-velocity)*1000),
                    curve_perpendicular_omega_rad_s=(perpendicular@curve_omega).tolist(),
                    contact_perpendicular_omega_rad_s=nominal_perpendicular_omega[module].tolist()))
            for module, u in enumerate(axes):
                if module in rolling_contacts:
                    continue
                old_u = reference_p[module]-reference_p[module+1]
                new_u = wanted_p[module]-wanted_p[module+1]
                old_u /= max(np.linalg.norm(old_u), 1e-9)
                new_u /= max(np.linalg.norm(new_u), 1e-9)
                axis_change = np.cross(old_u, new_u)
                sine = np.linalg.norm(axis_change)
                omega_ff = axis_change*math.atan2(sine, float(old_u@new_u))/(max(sine, 1e-12)*DT)
                perpendicular_omega[module] = (np.eye(3)-np.outer(u, u))@(
                    omega_ff+.35*np.cross(u, new_u))
        bone = SpatialBishopBackbone(wanted_p*1000, np.full(21, LINK*1000))
        material_field, _ = bone.material_field(q)
        bone_angles = bone.joint_angles(field=material_field)

        solved_spin_rows, avoided_corrections = [], []
        center_task_velocities, rolling_consistency_errors, uncontacted_axial_recovery_removed = [], [], []
        small_step_grasp_approaches = []
        tail_normal_execution = []
        tail_normal_rows = []
        for module, center in enumerate(centers):
            mujoco.mj_jac(self.m, self.d, jp, jr, center, self.head+module)
            u = axes[module]
            region = current_regions[module]
            supported_side = (0 if region == 0 and source_grip_mask[module] else
                              1 if region == 2 and target_grip_mask[module] else None)
            retained_margin = (float(self.curve.clearance_margins(locations[module], links=True))
                               if self.grip_clearance_enabled and supported_side is not None else None)
            spin = weights[module, 0]*source_roll+weights[module, 1]*target_roll
            pose_velocity = .35*(wanted_c[module]-center)
            v = feedforward[module]+shape_feedforward[module]+pose_velocity
            guard_goal = wanted_c[module]
            if module in rolling_contacts:
                side, point, normal, vn = rolling_contacts[module]
                supported_side = side
                if self.alternating_yield_enabled:
                    spin = contact_material_roll[module]
                retained_margin = (float(self.curve.clearance_margins(locations[module], links=True))
                    if self.grip_clearance_enabled else None)
                omega = perpendicular_omega[module]+spin*u
                # ONE motion: contact velocity = centre velocity + omega x
                # (contact-centre) = pressure normal velocity. The full XYZ
                # recovery error no longer contradicts its requested spin.
                v = (shared_contact_motions[module][1] if module in shared_contact_motions
                    else vn*normal-np.cross(omega, point-center))
                guard_goal = center+DT*v
            elif (self.alternating_yield_enabled and self.yield_state is not None
                    and self.yield_state.get('role') == 'rear_release'
                    and target_grip_mask[module] and region == 2):
                # A temporarily detached member of the receiving wrap follows
                # the same shared feed and curve deformation as its holders.
                # Its individual old axial position is not another material
                # feed command. Remove BOTH signs of this pose-recovery row;
                # common feed/shape feedforward may still legitimately be -Y.
                v -= pose_velocity[1]*np.array([0., 1., 0.])
                guard_goal = center+DT*v
                uncontacted_axial_recovery_removed.append(dict(module=module,
                    removed_pose_axial_mm_s=float(pose_velocity[1]*1000),
                    common_feed_axial_mm_s=float(feedforward[module, 1]*1000),
                    shape_feed_axial_mm_s=float(shape_feedforward[module, 1]*1000)))
            approaching_grasp = (self.yield_state is not None and
                self.yield_state['stage'] == 'regrip' and module in self.yield_release_modules and
                support_forces[self.yield_state['side'], module] <= .1 and
                module not in rolling_contacts)
            if self.yield_state is not None and self.yield_state['stage'] in ['unload', 'feed']:
                carrying_side = supported_side
                if (carrying_side is not None and module not in self.yield_release_modules
                        and self.yield_state['anchor_goals_N'][carrying_side][module] > .1
                        and support_forces[carrying_side, module] <= .1
                        and module not in rolling_contacts):
                    # A temporarily detached carrying rod needs the same real
                    # bounded-step approach as the named regrip object. Its
                    # distant shape goal must not erase safe contact recovery.
                    approaching_grasp = True
            if approaching_grasp:
                # The distant reference is a shape objective, not the path
                # traversed by this control tick. Check the bounded real step
                # to reach contact, then let the original pressure task take
                # over. Every original pipe face is still checked by the guard.
                v *= min(1., .008/max(np.linalg.norm(v), 1e-9))
                requested_v = v.copy()
                full_goal = guard_goal.copy()
                guard_goal = center+DT*v
            v, avoided = self.pipe_geometry.guard_velocity(center, guard_goal, v, BODY_RADIUS,
                supported_side=supported_side, retained_margin=retained_margin)
            if approaching_grasp:
                small_step_grasp_approaches.append(dict(module=module, side=supported_side,
                    requested_velocity_mm_s=(requested_v*1000).tolist(),
                    applied_velocity_mm_s=(v*1000).tolist(),
                    requested_step_mm=float(np.linalg.norm(DT*requested_v)*1000),
                    current_pipe_distance_mm=float(self.pipe_geometry.signed(center)*1000),
                    full_reference_pipe_distance_mm=float(self.pipe_geometry.signed(full_goal)*1000),
                    step_pipe_distance_mm=float(self.pipe_geometry.signed(guard_goal)*1000),
                    guard_removed_component=bool(avoided)))
            if avoided:
                avoided_corrections.append(module)
            v *= min(1., .008/max(np.linalg.norm(v), 1e-9))
            if module in rolling_contacts:
                rolling_consistency_errors.append(float(np.linalg.norm(v+np.cross(omega, point-center)-vn*normal)))
            center_task_velocities.append(v)
            center_jac, execution_v = jp, v
            if (self.localized_yield_enabled and self.yield_state is not None and
                    module == len(centers)-1 and supported_side is not None):
                origin = np.array([0., 0., 1.]) if supported_side == 0 else CORNER
                radial_normal = center-origin
                radial_normal[supported_side] = 0.
                radial_normal /= max(np.linalg.norm(radial_normal), 1e-9)
                root_normal = float(radial_normal@jp[:, :6]@data.qvel[:6])
                center_jac = jp.copy()
                center_jac[:, :6] -= np.outer(radial_normal, radial_normal@jp[:, :6])
                execution_v = v-root_normal*radial_normal
                row = radial_normal@center_jac
                tail_normal_rows.append(row.copy())
                tail_normal_execution.append(dict(module=module, side=supported_side,
                    requested_world_radial_mm_s=float(radial_normal@v*1000),
                    measured_root_radial_mm_s=root_normal*1000,
                    requested_actuated_radial_mm_s=float(radial_normal@execution_v*1000),
                    basis='same guarded centre motion; subtract measured root velocity, never borrow predicted unactuated radial root motion'))
            add(center_jac, execution_v, 2. if region != 1 else 2.5,
                'coherent_rolling_center' if module in rolling_contacts else 'center_motion_other',
                physical_j=jp, physical_value=v)
            perpendicular = np.eye(3)-np.outer(u, u)
            add(perpendicular@jr, perpendicular_omega[module], .055, 'tangent_motion')
            add(u@jr, spin, .12, 'material_spin')
            solved_spin_rows.append((u@jr).copy())

        # Supporting contact points constrain the predicted free-root motion.
        # Only joint increments are dispatched; physical root is never assigned.
        for (module, side), (_, point, _) in selected.items():
            mujoco.mj_jac(self.m, self.d, jp, jr, point, self.head+module)
            normal, normal_velocity, is_support, peeling = pressure_targets[module, side]
            pressure_jac = normal@jp
            physical_pressure_jac, physical_normal_velocity = pressure_jac.copy(), normal_velocity
            if self.localized_yield_enabled and self.yield_state is not None and module == len(centers)-1:
                normal_velocity -= float(pressure_jac[:6]@data.qvel[:6])
                pressure_jac[:6] = 0.
            add(pressure_jac, normal_velocity, 8. if is_support or peeling else 2., 'contact_pressure',
                physical_j=physical_pressure_jac, physical_value=physical_normal_velocity)
            if not peeling:
                add((np.eye(3)-np.outer(normal, normal))@jp, np.zeros(3), 3., 'contact_tangent_stationary')
        joint_jac = np.zeros((20, n))
        joint_jac[np.arange(20), self.va] = 1.
        inward = -.2*np.sign(q)*np.maximum(np.abs(q)-np.radians(80.), 0.)
        add(joint_jac, inward, .025, 'joint_limit_release')
        low, high = np.r_[np.full(3, -.025), np.full(3, -.15), np.full(20, -np.radians(10))], np.r_[np.full(3, .025), np.full(3, .15), np.full(20, np.radians(10))]
        speed = np.radians(10.)
        low[self.va] = np.clip((np.radians(-88)-self.last)/DT, -speed, speed-1e-7)
        high[self.va] = np.clip((np.radians(88)-self.last)/DT, -speed+1e-7, speed)
        matrix, vector = np.vstack(A), np.concatenate(b)
        measured_root_execution = (self.localized_yield_enabled and self.yield_state is not None
            and self.yield_state.get('role') == 'rear_release')
        if measured_root_execution:
            # Every world/contact task must use the root motion that actually
            # exists. Solve the remaining SAME motion with actuated joints;
            # six unactuated prediction variables cannot promise propulsion.
            # Tail-normal rows already removed their root component, so this
            # also subtracts only their remaining tangential root contribution.
            vector -= matrix[:,:6]@data.qvel[:6]
            matrix[:,:6] = 0.
        local_bend_constraint = None
        if (self.yield_state is not None and self.yield_state['stage'] in ['unload', 'feed', 'regrip']
                and not self.yield_state.get('anchor_recovery_active', False)
                and not self.yield_state.get('target_handover_pending', False)):
            joint = int(self.yield_state['selected_joint'])
            before = wanted_p[joint-1]-wanted_p[joint]
            after = wanted_p[joint]-wanted_p[joint+1]
            # A hinge's adjacent-rod angle is |q|. Preserve the current bending
            # branch and take its magnitude from the SAME finite-link curve.
            # Fix this one target increment before solving the remaining DOFs:
            # centre/contact soft compromises must not reverse local relief.
            angle = math.atan2(np.linalg.norm(np.cross(before, after)), float(before@after))
            curve_q = math.copysign(angle, q[joint-1])
            previous_before = reference_p[joint-1]-reference_p[joint]
            previous_after = reference_p[joint]-reference_p[joint+1]
            previous_angle = math.atan2(np.linalg.norm(np.cross(previous_before, previous_after)),
                float(previous_before@previous_after))
            previous_curve_q = math.copysign(previous_angle, q[joint-1])
            column = int(self.va[joint-1])
            # The Jacobian variable describes actual geometric motion. Unwinding
            # a saved servo preload is a ctrl change, not that motion: feeding
            # its full -10deg/s here made neighbouring grasps compensate for
            # an artificial bend. Use the same curve feedforward and .35 pose
            # recovery as the existing tangent tasks, then dispatch its curve
            # position separately through the unchanged actuator limits.
            geometric_rate = (curve_q-previous_curve_q)/DT+.35*(curve_q-q[joint-1])
            if self.yield_state.get('role') == 'rear_release' and self.yield_state['stage'] in ['unload', 'feed']:
                # Local yielding follows THIS accepted finite-link curve
                # increment. An old absolute angular mismatch is not another
                # opening command that may undo the carrying-contact motion.
                geometric_rate = (curve_q-previous_curve_q)/DT
            fixed_rate = float(np.clip(geometric_rate, low[column], high[column]))
            free = np.arange(n) != column
            solution = lsq_linear(matrix[:, free], vector-matrix[:, column]*fixed_rate,
                bounds=(low[free], high[free]), tol=1e-6, max_iter=60)
            full_solution = np.empty(n)
            full_solution[free] = solution.x
            full_solution[column] = fixed_rate
            solution.x = full_solution
            local_bend_constraint = dict(joint=joint, curve_angle_deg=math.degrees(curve_q),
                actual_angle_deg=math.degrees(q[joint-1]),
                geometric_rate_deg_s=math.degrees(fixed_rate),
                previous_target_deg=math.degrees(self.last[joint-1]),
                target_rate_deg_s=math.degrees(float(np.clip((curve_q-self.last[joint-1])/DT, -speed, speed))),
                source='same curve adjacent-rod bend; one fixed variable in the existing support solve')
        else:
            solution = lsq_linear(matrix, vector, bounds=(low, high), tol=1e-6, max_iter=60)
        if measured_root_execution:
            solution.x[:6] = data.qvel[:6]
        for record, row in zip(tail_normal_execution, tail_normal_rows):
            record['predicted_actuated_radial_mm_s'] = float(row@solution.x*1000)
            record['predicted_world_radial_mm_s'] = record['predicted_actuated_radial_mm_s']+record['measured_root_radial_mm_s']
        if getattr(self, 'capture_execution_tasks', False):
            self.execution_tasks = dict(matrix=matrix, vector=vector, low=low, high=high,
                labels=task_labels, solution=solution.x, spin_rows=np.asarray(solved_spin_rows),
                nominal_roll=nominal_roll, recovery_roll=recovery_roll,
                requested_roll=[float(source_roll), float(target_roll)],
                source_modules=np.flatnonzero(source_grip_mask & (support_forces[0] > .1)),
                target_modules=np.flatnonzero(target_grip_mask & (support_forces[1] > .1)))
        # The single updated route and its support objectives generate execution;
        # there is no separate additive joint-grip target for tracking to undo.
        raw = self.last+DT*solution.x[self.va]
        coordinated_rear_release = (self.yield_state is not None
            and self.yield_state.get('role') == 'rear_release'
            and self.yield_state['stage'] in ['unload', 'feed'])
        if local_bend_constraint is not None and not coordinated_rear_release:
            raw[local_bend_constraint['joint']-1] = curve_q
        if local_bend_constraint is not None and coordinated_rear_release:
            local_bend_constraint['dispatch'] = 'same solved increment as carrying-contact compensation; no curve-position overwrite'
        # Symmetric anti-windup around the CURRENT encoders. Dynamic gripping
        # does not shift this window, so every joint can receive either sign of
        # drive. Rate limiting unwinds the first saved clamp continuously.
        raw = np.clip(raw, q-np.radians(3.), q+np.radians(3.))
        target = np.clip(raw, np.maximum(-np.radians(88), self.last-speed*DT),
                        np.minimum(np.radians(88), self.last+speed*DT))
        physics_execution = None
        if (self.physics_execution_prediction_enabled and not self.first_dispatch
                and self.yield_state is not None and self.yield_state['stage'] in ['unload','feed','regrip']):
            target, physics_execution = self.refine_physical_execution(data,target,
                np.vstack(physical_A),np.concatenate(physical_b),local_bend_constraint,support_forces)
            if physics_execution.get('forward_probe_rejected',False):
                self.speed = 0.
        if self.first_dispatch:
            target = self.last.copy()
            self.first_dispatch = False
        if local_bend_constraint is not None:
            local_bend_constraint['final_dispatch_rate_deg_s'] = math.degrees((target[local_bend_constraint['joint']-1]-self.last[local_bend_constraint['joint']-1])/DT)
        self.last = target.copy()
        info = dict(stage=mode, shift_mm=float(self.shift*1000), shift_speed_mm_s=float(ds/DT*1000),
            actual_material_estimate_mm=actual_material*1000, actual_material_speed_mm_s=actual_speed*1000,
            reference_lead_mm=lag*1000, fixed_front_y_mm=front_progress*1000,
            historical_shift_lead_mm=historical_lag*1000,
            local_reference_shift_origin_mm=self.local_reference_shift_origin*1000,
            local_actual_material_progress_mm=local_material*1000,
            local_pipe_material_progress_mm=(local_pipe_material*1000).tolist(),
            bootstrap_feed_limit_mm_s=bootstrap*1000,
            common_shift_feedforward_axial_mm_s=[source_axial*1000, target_axial*1000],
            shape_feedforward_axial_mm_s=[float(np.median(shape_feedforward[current_regions == 0, 0])*1000) if np.any(current_regions == 0) else 0.,
                float(np.median(shape_feedforward[current_regions == 2, 1])*1000) if np.any(current_regions == 2) else 0.],
            observer_correspondence_rebased=self.observer_rebased,
            physical_progress_observer_initialized=self.physical_observer_just_initialized,
            physical_progress_origin_shift_mm=self.physical_progress_origin_shift*1000,
            physical_pipe_axial_progress_mm=(self.physical_pipe_progress*1000).tolist(),
            physical_pipe_axial_speed_mm_s=(self.physical_pipe_speed*1000).tolist(),
            physical_pipe_material_speed_mm_s=(physical_speeds*1000).tolist(),
            physical_progress_cohort_modules=self.physical_cohort_modules,
            physical_common_feed_limit_mm_s=physical_limit*1000,
            straight_correction_avoidance_modules=avoided_corrections,
            reference_pipe_clearance_mm=self.curve.reference_pipe_clearance_mm,
            fixed_rear_x_mm=rear_progress*1000, tracking_error_deg=math.degrees(maximum_error),
            source_modules=np.flatnonzero(forces[0] > .1).tolist(), target_modules=np.flatnonzero(forces[1] > .1).tolist(),
            source_grip_region_mask=source_grip_mask.tolist(),
            reference_source_region_mask=self.source_span(locations).tolist(),
            actual_source_grip_modules=np.flatnonzero((support_forces[0] > .1) & source_grip_mask).tolist(),
            actual_target_grip_modules=np.flatnonzero((support_forces[1] > .1) & target_grip_mask).tolist(),
            initial_preload_deg=np.degrees(self.initial_preload).tolist(), dynamic_grip_bias_deg=np.zeros(20).tolist(),
            curve_feedback_curvature_strength=self.curve_grip_strength.tolist(), anti_windup_error_window_deg=3.,
            curve_curvature_release_strength=self.curve_curvature_release_strength.tolist(),
            execution_grip='pressure feedback in the common spatial curve and current support solve; no additive joint bias',
            localized_yield=json.loads(json.dumps(self.yield_state)),
            local_curve_bend_constraint=local_bend_constraint,
            local_yield_release_modules=self.yield_release_modules,
            support_force_goals_N=self.support_force_goals.tolist(), actual_pipe_normal_force_N=forces.sum(1).tolist(),
            current_target_minus_encoder_deg=np.degrees(target-q).tolist(),
            unwinding_or_transient_joints=(np.flatnonzero(abs(target-q) > np.radians(3.01))+1).tolist(),
            material_locations_mm=(locations*1000).tolist(), material_region_codes=self.curve.regions(locations).tolist(),
            reference_points_mm=(wanted_p*1000).tolist(), reference_centers_mm=(wanted_c*1000).tolist(),
            reference_node_arclength_mm=(self.curve.reference_node_arc*1000).tolist(),
            reference_center_arclength_mm=(self.curve.reference_center_arc*1000).tolist(),
            current_source_arc_bound_mm=0., current_target_arc_bound_mm=self.curve.current_target_arc_bound*1000,
            maximum_reference_link_error_mm=float(np.abs(np.linalg.norm(np.diff(wanted_p, axis=0), axis=1)-LINK).max()*1000),
            body_curve_rms_error_mm=float(np.sqrt(np.mean((wanted_c-centers)**2))*1000),
            source_material_roll_rad_s=source_roll, target_material_roll_rad_s=target_roll,
            coherent_rolling_modules=sorted(rolling_contacts),
            axial_pose_recovery_removed_modules=axial_recovery_removed,
            actual_contact_roll_geometry=contact_roll_geometry,
            contact_material_roll_rad_s={str(module):value for module, value in contact_material_roll.items()},
            shared_curve_contact_field=shared_curve_field_error,
            contact_transport_fraction=getattr(self.curve, 'contact_transport_fraction', 0.),
            uncontacted_axial_pose_recovery_removed=uncontacted_axial_recovery_removed,
            small_step_grasp_approaches=small_step_grasp_approaches,
            tail_normal_execution=tail_normal_execution,
            rolling_motion_scale=roll_scale.tolist(),
            maximum_requested_rolling_contact_error_mm_s=max(rolling_consistency_errors, default=0.)*1000,
            center_task_velocity_mm_s=(np.asarray(center_task_velocities)*1000).tolist(),
            nominal_material_roll_rad_s=nominal_roll.tolist(), recovery_material_roll_rad_s=recovery_roll.tolist(),
            reference_tangent_feedforward_rad_s=np.asarray(tangent_feedforward).tolist(),
            solved_material_spin_rad_s=(np.asarray(solved_spin_rows)@solution.x).tolist(),
            tangent_task='perpendicular angular velocity plus reference tangent change; axial spin has a separate row',
            curve_parameters=self.curve.parameters(), shape_updates=self.n_adaptations,
            bishop_joint_fit_max_error_deg=float(np.degrees(np.abs(bone_angles-q)).max()),
            solve_residual=float(np.linalg.norm(matrix@solution.x-vector)), solve_status=int(solution.status),
            root_execution_model='measured root velocity; actuated remainder of same tasks' if measured_root_execution else 'free-root kinematic prediction',
            physics_execution_prediction=physics_execution,
            feedback_fields=dict(hardware_candidates=['joint_angles', 'actuator_load', 'last_sent_targets'],
                simulation_truth_used=['world_body_pose', 'pipe_contact_points', 'pipe_contact_force_for_support_gate']),
            limitation='physical cohort axial progress requires world-pose truth; pitch-normalized progress is approximate; contacts/positions score completion')
        return target, info

    def refine_physical_execution(self,data,target,matrix,vector,local_bend,forces):
        """Predict joint-only servo/contact response to the same curve tasks."""
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.m,specification))
        mujoco.mj_getState(self.m,data,integration,specification)
        shadow = mujoco.MjData(self.m)
        steps = round(DT/self.m.opt.timestep)
        points,centers = self.observe(data)
        masks = np.asarray(physical_wrap_masks(centers)) & (forces>.1)
        masks[0,self.transferred_material_modules] = False
        opening = (self.yield_state['stage'] in ['unload','feed']
            and not self.yield_state.get('anchor_recovery_active',False))
        if opening:
            masks[self.yield_state['side'],self.yield_release_modules] = False
        required = [(side,int(module)) for side,count in [(0,2 if opening else 21),(1,3)]
            for module in sorted(np.flatnonzero(masks[side]),key=lambda i:forces[side,i],reverse=True)[:count]]
        forward = self.yield_state['stage']=='feed' and self.speed>0.
        feature_floor = np.full(len(required),.100001)
        # Forward motion is a soft part of the SAME action objective. Do not
        # require both occupied sides to advance in every 50ms substep.
        nominal_axis_goal = self.speed*self.physical_axial_sines if forward else np.zeros(2)
        def predict(candidate):
            mujoco.mj_setState(self.m,shadow,integration,specification)
            shadow.ctrl[:] = candidate
            for _ in range(steps):
                mujoco.mj_step(self.m,shadow)
            velocity = np.empty(self.m.nv)
            mujoco.mj_differentiatePos(self.m,velocity,DT,data.qpos,shadow.qpos)
            residual = matrix@velocity-vector
            mujoco.mj_forward(self.m,shadow)
            predicted_force = np.zeros((2,21))
            for index,contact in enumerate(shadow.contact):
                for robot,pipe in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
                    module = int(self.m.geom_bodyid[robot])-self.head
                    name = self.m.geom(pipe).name or ''
                    side = (0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else
                        1 if name in ['pipe_sharp_out_03','pipe_straight_04'] else None)
                    if 0<=module<21 and side is not None and cylindrical_contact(side,contact.pos,contact.frame[:3]):
                        force = np.zeros(6);mujoco.mj_contactForce(self.m,shadow,index,force)
                        predicted_force[side,module] += max(0.,force[0])
            feature = [predicted_force[side,module] for side,module in required]
            if forward:
                predicted_points = np.vstack([shadow.site('s_head').xpos,
                    *[shadow.joint('joint'+str(i)).xanchor for i in range(1,21)],shadow.site('s_tail').xpos])
                predicted_centers = .5*(predicted_points[:-1]+predicted_points[1:])
                feature.extend([float(np.mean((predicted_centers[masks[side],side]-centers[masks[side],side])/DT))
                    for side in range(2)])
            return residual,velocity,np.asarray(feature)
        def cost_residual(residual,feature):
            return np.r_[residual,2.*(feature[len(required):]-nominal_axis_goal)] if forward else residual
        def force_features(feature):
            return feature[:len(required)]
        baseline,baseline_velocity,baseline_feature = predict(target)
        selected = None if local_bend is None else local_bend['joint']-1
        q = data.qpos[self.qa]
        lower = np.maximum(np.maximum(-np.radians(88),self.last-np.radians(10)*DT),q-np.radians(3))
        upper = np.minimum(np.minimum(np.radians(88),self.last+np.radians(10)*DT),q+np.radians(3))
        if selected is not None:
            # The final physical support solve may reduce this same curve
            # increment, but cannot add an opposite yielding motion afterward.
            lower[selected] = max(lower[selected],min(self.last[selected],target[selected]))
            upper[selected] = min(upper[selected],max(self.last[selected],target[selected]))
        active = [i for i in range(20) if upper[i]-lower[i]>1e-8]
        center = target.copy()
        current_residual,current_feature = cost_residual(baseline,baseline_feature),baseline_feature
        options = [(target.copy(),baseline,baseline_velocity,baseline_feature)]
        predictions = 1
        for iteration in range(3):
            jacobian = np.empty((len(current_residual),len(active)))
            feature_jacobian = np.empty((len(feature_floor),len(active)))
            for column,joint in enumerate(active):
                positive_step = min(.002,max(0.,upper[joint]-center[joint]))
                negative_step = min(.002,max(0.,center[joint]-lower[joint]))
                step = positive_step if positive_step>=negative_step else -negative_step
                perturbation = center.copy();perturbation[joint] += step
                residual,_,feature = predict(perturbation)
                predictions += 1
                jacobian[:,column] = (cost_residual(residual,feature)-current_residual)/step
                feature_jacobian[:,column] = (force_features(feature)-force_features(current_feature))/step
            low,high = lower[active]-center[active],upper[active]-center[active]
            fit = lsq_linear(np.vstack([jacobian,.5*np.eye(len(active))]),
                np.r_[-current_residual,np.zeros(len(active))],bounds=(low,high),tol=1e-7,max_iter=60)
            def objective(change):
                error = current_residual+jacobian@change
                return .5*(error@error+.25*(change@change))
            def gradient(change):
                return jacobian.T@(current_residual+jacobian@change)+.25*change
            constrained = minimize(objective,fit.x,jac=gradient,method='SLSQP',bounds=list(zip(low,high)),
                constraints=[dict(type='ineq',fun=lambda change:force_features(current_feature)+feature_jacobian@change-feature_floor,
                    jac=lambda change:feature_jacobian)],options=dict(ftol=1e-10,maxiter=80))
            candidate = center.copy();candidate[active] += constrained.x
            candidate = np.clip(candidate,lower,upper)
            corrected,corrected_velocity,corrected_feature = predict(candidate)
            predictions += 1
            options.append((candidate.copy(),corrected,corrected_velocity,corrected_feature))
            feasible = bool(np.all(force_features(corrected_feature)>=feature_floor-1e-7))
            if feasible:
                break
            center,current_residual,current_feature = candidate,cost_residual(corrected,corrected_feature),corrected_feature
        baseline_feasible = bool(np.all(force_features(baseline_feature)>=feature_floor-1e-7))
        accepted = feasible and (np.linalg.norm(cost_residual(corrected,corrected_feature))<np.linalg.norm(cost_residual(baseline,baseline_feature)) or not baseline_feasible)
        # A nonlinear contact change may invalidate the local differential.
        # Only a model-confirmed improvement replaces the original increment.
        final = candidate if accepted else target
        support_fallback = False
        if not feasible:
            hold = np.clip(self.last,lower,upper)
            hold_residual,hold_velocity,hold_feature = predict(hold)
            predictions += 1
            options.append((hold,hold_residual,hold_velocity,hold_feature))
            # When forward motion is not confirmed, select the evaluated
            # braking option with least ORIGINAL carrier-threshold violation.
            # Do not silently dispatch the known-unsafe kinematic proposal.
            def support_score(option):
                deficit = np.maximum(feature_floor[:len(required)]-option[3][:len(required)],0.)
                return (float(deficit@deficit),float(np.linalg.norm(option[1])))
            final,corrected,corrected_velocity,corrected_feature = min(options,key=support_score)
            support_fallback = True
        return final,dict(horizon_s=DT,physics_steps_per_prediction=steps,
            predicted_candidates=predictions,linearizations=iteration+1,accepted=bool(accepted),
            support_fallback=support_fallback,forward_probe_rejected=bool(forward and not feasible),
            baseline_task_residual=float(np.linalg.norm(baseline)),
            corrected_task_residual=float(np.linalg.norm(corrected)),
            maximum_ctrl_correction_deg=float(np.degrees(abs(final-target)).max()),
            baseline_predicted_root_velocity=baseline_velocity[:6].tolist(),
            corrected_predicted_root_velocity=corrected_velocity[:6].tolist(),
            required_actual_carriers=required,feature_floor=feature_floor.tolist(),
            axial_goal_kind='soft common-shift motion; continuous actual net progress decides success',
            nominal_axis_goal_mm_s=(nominal_axis_goal*1000).tolist(),
            baseline_support_and_axial_features=baseline_feature.tolist(),
            corrected_support_and_axial_features=corrected_feature.tolist(),
            corrected_feasible=feasible,constrained_status=str(constrained.message),
            basis='private original-model joint ctrl rollouts from full INTEGRATION state; no root assignment, actual progress credit or parameter change')
