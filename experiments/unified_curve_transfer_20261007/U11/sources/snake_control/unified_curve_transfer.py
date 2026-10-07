"""Three spatial regions, one material shift, and joint-only physical actuation.

The route is registered to a saved, supported posture. Captured bends join the
two known pipe axes; helical extensions provide space for the complete body.
This first implementation explicitly uses simulated world poses/contact points
in its shape tracking and traction solve. Encoder errors and actuator loads
are separate inputs, not a claim of hardware-ready global observation.
Units: metres, radians, seconds.
"""
import json
import math

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import lsq_linear
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
        self.clearance = None
        self.clearance_enabled = False
        self.clearance_margin = .036
        self.grip_clearance_profile = None
        self.reference_clearance_deficit_mm = None
        self.bridge_offset = np.zeros(3)
        self.shape_grid = np.arange(self.domain[0], self.domain[1]+LINK/2, LINK/2)
        self.shape_offsets = np.zeros((len(self.shape_grid), 3))
        self.clearance_offsets = np.zeros(len(self.shape_grid))
        self.reference_pipe_clearance_mm = None
        self.arc_grid_u = np.unique(np.r_[locations, np.linspace(self.domain[0], self.domain[1],
            math.ceil((self.domain[1]-self.domain[0])/.0025)+1)])
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
        if self.clearance_enabled:
            margin = self.clearance_margins(s)+np.interp(s, self.shape_grid, self.clearance_offsets)
            point = self.clearance.project(point, margin)
        return point

    def capture_grip_clearance(self, physical_nodes):
        """Retain observed usable mesh wrapping, without a uniform tube claim.

        This local centreline allowance comes from the restored physical links,
        whose real collision shells are meshes. It is not an exact mesh swept
        volume. The bridge keeps its original conservative envelope; dynamics
        and original collision contacts still decide whether squeezing works.
        """
        gaps = self.clearance.link_minimum(physical_nodes, samples=65)
        self.grip_clearance_profile = np.clip(gaps-.0015, .004, BODY_RADIUS+.00075)
        self.shape_revision += 1

    def clearance_margins(self, parameters, links=False):
        parameters = np.asarray(parameters)
        base = BODY_RADIUS+.00075 if links else self.clearance_margin
        margin = np.full(parameters.shape, base)
        if self.grip_clearance_profile is not None:
            observed = np.interp(parameters, self.center_s0[::-1], self.grip_clearance_profile[::-1])
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
               self.clearance_enabled, self.clearance_margin, self.shape_revision)
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
        node_arc = [self.initial_head_arc+shift]
        if node_arc[0] >= arc[-1]-.3:
            raise ValueError('Head has exhausted the target helix extension')
        nodes = [point(node_arc[0])]
        for module in range(21):
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
                if _clearance_attempt >= 5:
                    raise ValueError(f'Cannot keep reference links outside original pipe envelope: {margins.min()*1000:.3f} mm')
                bumps = np.exp(-.5*((self.shape_grid[:, None]-locations[None, invalid])/(.8*LINK))**2)
                increments = deficits[invalid]+.00025
                self.clearance_offsets += np.max(bumps*increments[None, :], axis=1)
                self.clearance_offsets = np.minimum(self.clearance_offsets, .04)
                self.shape_revision += 1
                return self.reference(shift, phase, _clearance_attempt+1)
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
        self.shape_offsets += rate*dt*averaged
        norm = np.linalg.norm(self.shape_offsets, axis=1)
        self.shape_offsets *= np.minimum(1., .012/np.maximum(norm, 1e-9))[:, None]
        self.shape_revision += 1

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
                    observed_grip_clearance_profile_mm=None if self.grip_clearance_profile is None else (self.grip_clearance_profile*1000).tolist(),
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
                  'physical_pipe_speed', 'physical_axial_sines', 'physical_last_centers', 'physical_continuous_groups'}
        for key, value in state['controller'].items():
            setattr(self, key, np.asarray(value) if key in arrays else value)
        curve_arrays = {'node_s0', 'center_s0', 'phase', 'phase_correction', 'radius_delta', 'pipe_smoothing',
            'bridge_offset', 'shape_grid', 'shape_offsets', 'clearance_offsets', 'arc_grid_u', 'initial_node_arc',
            'reference_node_arc', 'reference_center_arc', 'reference_node_parameters', 'grip_clearance_profile'}
        for key, value in state['curve'].items():
            if key not in {'base_x', 'base_c'}:
                setattr(self.curve, key, np.asarray(value) if key in curve_arrays and value is not None else value)
        self.curve.base = PPoly(np.asarray(state['curve']['base_c']), np.asarray(state['curve']['base_x']))
        if self.curve.clearance_enabled:
            self.curve.clearance = self.pipe_geometry
        self.curve.arc_cache = None
        self.first_dispatch = True
        reference, _, _ = self.curve.reference(self.shift)
        if 'reference_points_mm' in state:
            self.restored_reference_error_mm = float(np.max(np.abs(reference*1000-state['reference_points_mm'])))
            if self.restored_reference_error_mm > 1e-6:
                raise ValueError('Restored curve differs from checkpoint reference')
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
        if self.grip_clearance_enabled:
            self.curve.capture_grip_clearance(p)
        arc, _ = self.curve.arc_map()
        self.curve.initial_node_arc = np.interp(self.curve.node_s0, self.curve.arc_grid_u, arc)
        self.curve.initial_head_arc = float(self.curve.initial_node_arc[0])-self.shift
        _, _, locations = self.curve.reference(self.shift)
        self.projected_locations = locations.copy()
        self.last_route_registration_time = t
        self.last_shape_time = t
        self.observer_rebased = True
        self.events.append(dict(t_s=t, mode='repair_external_supported_route', source_node=source_node,
            target_node=target_node, shared_shift_mm=self.shift*1000,
            physical_progress_unchanged_mm=self.observed_shift*1000,
            observed_grip_clearance_enabled=self.grip_clearance_enabled,
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
        """Measured bends and load change the route before target generation."""
        residual = (p[:-1]+p[1:])/2-reference
        regions = self.curve.regions(s)
        stressed = (self.filtered_load > 1.9) & (tracking > np.radians(3.))
        module_stress = np.r_[stressed[0], stressed] | np.r_[stressed, stressed[-1]]
        centers = (p[:-1]+p[1:])/2
        for side, code in [(0, 0), (1, 2)]:
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
        if supported and captured_target_limit.any():
            previous = self.curve.pipe_smoothing[1]
            self.curve.pipe_smoothing[1] = min(1., previous+.015*dt)
            self.curve.shape_revision += 1
            if previous == 0.:
                self.events.append(dict(t_s=self.current_control_time, mode='smooth_captured_receiving_bends',
                    joints=(np.flatnonzero(captured_target_limit)+1).tolist(),
                    action='same fitted target helix gradually replaces captured pipe-span curvature peaks; local limit feedback relaxes curvature'))
        self.curve_grip_strength = strength.copy()
        ordinary = measured_bone.sample(step_mm=3.)
        tightened = measured_bone.sample(field=measured_bone.field*(1+gain[:, None]), step_mm=3.)
        material_nodes = np.arange(22)*LINK*1000
        delta = np.column_stack([np.interp(material_nodes, tightened['s_mm'], np.array(tightened['points_mm'])[:, j])-
                                 np.interp(material_nodes, ordinary['s_mm'], np.array(ordinary['points_mm'])[:, j]) for j in range(3)])/1000
        self.curve.deposit(self.curve.reference_node_parameters, delta, .18, dt)
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

    def command(self, data, t, forces, torque, support_forces=None):
        self.current_control_time = t
        self.observer_rebased = False
        self.physical_observer_just_initialized = False
        p, centers = self.observe(data)
        q = data.qpos[self.qa]
        reference_p, reference_c, locations = self.curve.reference(self.shift)
        if t-self.last_route_registration_time > 20.:
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
        maximum_error = float(tracking.max())
        front_progress = float(np.median(centers[:7, 1]-self.initial_centers[:7, 1]))
        rear_progress = float(np.median(centers[14:, 0]-self.initial_centers[14:, 0]))
        actual_material = self.observe_progress(centers, forces, t)
        self.last_body_progress = actual_material
        lag = self.shift-self.physical_progress_origin_shift-actual_material
        self.progress_history.append((t, actual_material))
        self.progress_history = [(a, b) for a, b in self.progress_history if a >= t-10.]
        actual_speed = ((actual_material-self.progress_history[0][1])/
                        max(t-self.progress_history[0][0], DT))
        error_scale = np.clip((np.radians(18)-maximum_error)/np.radians(14), 0., 1.)
        lag_scale = np.clip((.075-lag)/.05, 0., 1.)
        source_count, target_count = [int(np.count_nonzero(f > .1)) for f in forces]
        support_forces = forces if support_forces is None else support_forces
        source_grip_mask, target_grip_mask = physical_wrap_masks(centers)
        posterior_count = int(np.count_nonzero((support_forces[0] > .1) & source_grip_mask))
        receiver_count = int(np.count_nonzero((support_forces[1] > .1) & target_grip_mask))
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
        self.speed += np.clip(requested-self.speed, -.002*DT, .001*DT)
        mode = 'advance' if self.speed > .00015 else 'reshape_with_shift_braked'
        if mode != self.last_mode:
            self.events.append(dict(t_s=t, mode=mode, shift_mm=self.shift*1000,
                maximum_tracking_error_deg=math.degrees(maximum_error), lag_mm=lag*1000))
            self.last_mode = mode
        if t-self.last_shape_time >= .5:
            self.feedback_shape(p, q, reference_c, locations, tracking, forces, .5)
            self.last_shape_time = t
        ds = DT*self.speed
        self.curve.increment(ds)
        self.shift += ds
        wanted_p, wanted_c, locations = self.curve.reference(self.shift)
        feedforward = (wanted_c-reference_c)/DT
        # SpatialBishopBackbone supplies one coherent finite-link bending field
        # for diagnostics. Joint execution additionally projects the entire curve
        # through the original model's Jacobians and current support geometry.
        bone = SpatialBishopBackbone(wanted_p*1000, np.full(21, LINK*1000))
        material_field, _ = bone.material_field(q)
        bone_angles = bone.joint_angles(field=material_field)
        source_grip_mask, target_grip_mask = self.supported_curve_regions(locations, centers)

        n = self.m.nv
        A, b = [], []
        jp, jr = np.zeros((3, n)), np.zeros((3, n))
        def add(j, value, weight):
            A.append(weight*np.atleast_2d(j).copy())
            b.append(weight*np.atleast_1d(value).copy())
        weights = self.curve.weights(locations)
        current_regions = self.curve.regions(locations)
        source_axial = float(np.median(feedforward[current_regions == 0, 0])) if np.any(current_regions == 0) else 0.
        target_axial = float(np.median(feedforward[current_regions == 2, 1])) if np.any(current_regions == 2) else 0.
        source_roll = float(np.clip(-np.sign(self.curve.theta_rates[0])*source_axial/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[0]**2)), -.12, .12))
        target_roll = float(np.clip(np.sign(self.curve.theta_rates[1])*target_axial/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[1]**2)), -.12, .12))
        nominal_roll = np.array([source_roll, target_roll])
        recovery_roll = np.zeros(2)
        correction = .35*(wanted_c-centers)
        correction *= np.minimum(1., .008/np.maximum(np.linalg.norm(correction, axis=1), 1e-9))[:, None]
        for side, code, supported_mask in [(0, 0, source_grip_mask), (1, 2, target_grip_mask)]:
            mask = (current_regions == code) & supported_mask
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
        source_roll, target_roll = np.clip(nominal_roll+recovery_roll, -.12, .12)
        tangent_feedforward = []
        solved_spin_rows = []
        avoided_corrections = []
        for module, center in enumerate(centers):
            mujoco.mj_jac(self.m, self.d, jp, jr, center, self.head+module)
            u = self.d.xmat[self.head+module].reshape(3, 3)[:, 2]
            v = feedforward[module]+.35*(wanted_c[module]-center)
            region = self.curve.regions(locations[module])
            supported_side = (0 if region == 0 and source_grip_mask[module] else
                              1 if region == 2 and target_grip_mask[module] else None)
            retained_margin = (float(self.curve.clearance_margins(locations[module], links=True))
                               if self.grip_clearance_enabled and supported_side is not None else None)
            v, avoided = self.pipe_geometry.guard_velocity(center, wanted_c[module], v, BODY_RADIUS,
                supported_side=supported_side, retained_margin=retained_margin)
            if avoided:
                avoided_corrections.append(module)
            v *= min(1., .008/max(np.linalg.norm(v), 1e-9))
            add(jp, v, 2. if region != 1 else 2.5)
            wanted_u = wanted_p[module]-wanted_p[module+1]
            wanted_u /= max(np.linalg.norm(wanted_u), 1e-9)
            previous_u = reference_p[module]-reference_p[module+1]
            previous_u /= max(np.linalg.norm(previous_u), 1e-9)
            rotation_axis = np.cross(previous_u, wanted_u)
            sine = np.linalg.norm(rotation_axis)
            angle = math.atan2(sine, float(previous_u@wanted_u))
            omega_ff = rotation_axis*(angle/(max(sine, 1e-12)*DT))
            tangent_feedforward.append(omega_ff)
            # A tangent fixes only the two perpendicular angular components.
            # Axial material spin is a separate row in the same finite-joint
            # solve; the tangent row must not also ask that spin to be zero.
            perpendicular = np.eye(3)-np.outer(u, u)
            add(perpendicular@jr, perpendicular@(omega_ff+.35*np.cross(u, wanted_u)), .055)
            spin = weights[module, 0]*source_roll+weights[module, 1]*target_roll
            add(u@jr, spin, .12)
            solved_spin_rows.append((u@jr).copy())

        # Supporting points constrain the predicted free-root velocity. Only
        # solved joint increments are sent; the physical root is never assigned.
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
        for (module, side), (_, point, contact_normal) in selected.items():
            mujoco.mj_jac(self.m, self.d, jp, jr, point, self.head+module)
            axis = np.array([1., 0., 0.]) if side == 0 else np.array([0., 1., 0.])
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            normal = point-origin
            normal -= normal@axis*axis
            normal /= max(np.linalg.norm(normal), 1e-9)
            # U04's unconstrained pose correction undid an additive clamp:
            # grip increment grew while actual contact pressure collapsed.
            # Regulate pressure in this SAME solve, on current support regions.
            current_mask = source_grip_mask if side == 0 else target_grip_mask
            is_support = current_mask[module] and cylindrical_contact(side, point, contact_normal)
            count = max(1, int(np.count_nonzero(current_mask & (support_forces[side] > .1))))
            goal_force = self.support_force_goals[side]/count if is_support else 0.
            normal_velocity = np.clip(.00002*(forces[side, module]-goal_force), -.0006, .0006)
            # An ordinary bridge contact is free to peel; it is not a grasp anchor.
            add(normal@jp, normal_velocity, 8. if is_support else 2.)
            add((np.eye(3)-np.outer(normal, normal))@jp, np.zeros(3), 3.)
        joint_jac = np.zeros((20, n))
        joint_jac[np.arange(20), self.va] = 1.
        inward = -.2*np.sign(q)*np.maximum(np.abs(q)-np.radians(80.), 0.)
        add(joint_jac, inward, .025)
        low, high = np.r_[np.full(3, -.025), np.full(3, -.15), np.full(20, -np.radians(10))], np.r_[np.full(3, .025), np.full(3, .15), np.full(20, np.radians(10))]
        speed = np.radians(10.)
        low[self.va] = np.clip((np.radians(-88)-self.last)/DT, -speed, speed-1e-7)
        high[self.va] = np.clip((np.radians(88)-self.last)/DT, -speed+1e-7, speed)
        matrix, vector = np.vstack(A), np.concatenate(b)
        solution = lsq_linear(matrix, vector, bounds=(low, high), tol=1e-6, max_iter=60)
        # The single updated route and its support objectives generate execution;
        # there is no separate additive joint-grip target for tracking to undo.
        raw = self.last+DT*solution.x[self.va]
        # Symmetric anti-windup around the CURRENT encoders. Dynamic gripping
        # does not shift this window, so every joint can receive either sign of
        # drive. Rate limiting unwinds the first saved clamp continuously.
        raw = np.clip(raw, q-np.radians(3.), q+np.radians(3.))
        target = np.clip(raw, np.maximum(-np.radians(88), self.last-speed*DT),
                        np.minimum(np.radians(88), self.last+speed*DT))
        if self.first_dispatch:
            target = self.last.copy()
            self.first_dispatch = False
        self.last = target.copy()
        info = dict(stage=mode, shift_mm=float(self.shift*1000), shift_speed_mm_s=float(self.speed*1000),
            actual_material_estimate_mm=actual_material*1000, actual_material_speed_mm_s=actual_speed*1000,
            reference_lead_mm=lag*1000, fixed_front_y_mm=front_progress*1000,
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
            nominal_material_roll_rad_s=nominal_roll.tolist(), recovery_material_roll_rad_s=recovery_roll.tolist(),
            reference_tangent_feedforward_rad_s=np.asarray(tangent_feedforward).tolist(),
            solved_material_spin_rad_s=(np.asarray(solved_spin_rows)@solution.x).tolist(),
            tangent_task='perpendicular angular velocity plus reference tangent change; axial spin has a separate row',
            curve_parameters=self.curve.parameters(), shape_updates=self.n_adaptations,
            bishop_joint_fit_max_error_deg=float(np.degrees(np.abs(bone_angles-q)).max()),
            solve_residual=float(np.linalg.norm(matrix@solution.x-vector)), solve_status=int(solution.status),
            feedback_fields=dict(hardware_candidates=['joint_angles', 'actuator_load', 'last_sent_targets'],
                simulation_truth_used=['world_body_pose', 'pipe_contact_points', 'pipe_contact_force_for_support_gate']),
            limitation='physical cohort axial progress requires world-pose truth; pitch-normalized progress is approximate; contacts/positions score completion')
        return target, info
