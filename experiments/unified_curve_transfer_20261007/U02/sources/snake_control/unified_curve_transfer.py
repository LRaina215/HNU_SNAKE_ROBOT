"""Three spatial regions, one material shift, and joint-only physical actuation.

The route is registered to a saved, supported posture. Captured bends join the
two known pipe axes; helical extensions provide space for the complete body.
This first implementation explicitly uses simulated world poses/contact points
in its shape tracking and traction solve. Encoder errors and actuator loads
are separate inputs, not a claim of hardware-ready global observation.
Units: metres, radians, seconds.
"""
import math

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from snake_control.spatial_bishop_backbone import SpatialBishopBackbone


DT = .05
LINK = .0645
BODY_RADIUS = .029
CORNER = np.array([1.1, 0., 1.])


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


class RegisteredThreeRegionCurve:
    """A single increasing tail-to-head route; all material locations share shift.

    The initial coordinate is measured link length, as in FiveSegmentCurve's
    material_positions. A spline between link nodes is an approximation to
    arclength, so the finite-link solver must enforce the real mechanism.
    Region bounds, node samples and feedback deposition use this same coordinate.
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
        self.radius_delta = np.zeros(2)
        self.bridge_offset = np.zeros(3)
        self.shape_grid = np.arange(self.domain[0], self.domain[1]+LINK/2, LINK/2)
        self.shape_offsets = np.zeros((len(self.shape_grid), 3))

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
        return np.asarray(shift)[..., None]*(-np.sign(self.theta_rates)/(
            np.asarray(self.radii)*np.sqrt(1-np.asarray(self.axial_sines)**2)))

    def sample(self, s, phase=None):
        s = np.asarray(s)
        base = self.base(s)
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
        return point

    def reference(self, shift):
        nodes_s = self.node_s0+shift
        p = self.sample(nodes_s)
        return p, (p[:-1]+p[1:])/2, self.center_s0+shift

    def deposit(self, locations, changes, rate, dt):
        weights = np.exp(-.5*((self.shape_grid[:, None]-locations[None, :])/(.8*LINK))**2)
        averaged = weights@changes/np.maximum(weights.sum(1)[:, None], 1.)
        self.shape_offsets += rate*dt*averaged
        norm = np.linalg.norm(self.shape_offsets, axis=1)
        self.shape_offsets *= np.minimum(1., .012/np.maximum(norm, 1e-9))[:, None]

    def parameters(self):
        return dict(source_phase_rad=float(self.phase[0]), target_phase_rad=float(self.phase[1]),
                    source_radius_delta_mm=float(self.radius_delta[0]*1000),
                    target_radius_delta_mm=float(self.radius_delta[1]*1000),
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
        self.preload = data.ctrl-data.qpos[self.qa]
        self.curve = RegisteredThreeRegionCurve(p)
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

    def observe_progress(self, centers, forces):
        """Fit the common shift-and-roll trajectory, including after turning.

        This is a world-pose-based simulation observer. It is an estimate of
        correspondence to the deformed route. Candidate shift AND its pipe phase
        are evaluated together: projecting onto the already rotated current
        helix alone can return the commanded shift even for a stationary body.
        Physical contact/position completion remains separate.
        Taking the slower occupied pipe region prevents rear movement from
        masking a retreating front. No fixed module group remains a source-X
        progress sensor once that group has crossed the junction.
        """
        candidates = self.shift+np.linspace(-.16, .16, 65)
        grid = self.curve.center_s0[:, None]+candidates
        phases = np.tile(self.curve.phase_at(candidates), (21, 1))
        points = (self.curve.sample((grid+LINK/2).ravel(), phases)+
                  self.curve.sample((grid-LINK/2).ravel(), phases)).reshape(21, 65, 3)/2
        distance = np.linalg.norm(points-centers[:, None, :], axis=2)
        indices = np.argmin(distance, axis=1)
        self.projected_locations = grid[np.arange(21), indices]
        shifts = self.projected_locations-self.curve.center_s0
        regions = self.curve.regions(self.projected_locations)
        usable = (distance[np.arange(21), indices] < .065) | (forces.sum(0) > .1)
        group_progress = []
        for region in [0, 2]:
            mask = usable & (regions == region)
            if mask.sum() >= 2:
                group_progress.append(float(np.median(shifts[mask])))
        return min(group_progress) if group_progress else float(np.median(shifts[usable])) if usable.any() else self.last_body_progress

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
        for side, code in [(0, 0), (1, 2)]:
            ids = np.flatnonzero(regions == code)
            if len(ids) < 3:
                continue
            axis = np.array([1., 0., 0.]) if side == 0 else np.array([0., 1., 0.])
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            centers = (p[:-1]+p[1:])/2
            radial = centers[ids]-origin
            radial -= (radial@axis)[:, None]*axis
            normal = radial/np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-8)
            mismatch = float(np.median(np.sum(residual[ids]*normal, axis=1)))
            # Opening a loaded source wrap is bounded; no progressive unwrapping.
            # Target capture retains a small inward bias if position tracks.
            bias = .001 if module_stress[ids].any() else (-.0006 if side == 1 else -.0003)
            wanted = np.clip(self.curve.radius_delta[side]+.15*mismatch+bias, -.004, .004)
            if side == 1:
                # U01 opened the receiving wrap under tracking stress, then lost
                # M3 and M0. Receiving support takes priority over unloading it.
                count = np.count_nonzero(forces[1] > .1)
                upper = 0. if count >= 3 else -.0015
                wanted = min(wanted, upper)
            self.curve.radius_delta[side] += np.clip(wanted-self.curve.radius_delta[side], -.00025*dt, .00025*dt)
        transition = regions == 1
        if transition.any():
            correction = residual.copy()
            correction[~transition] = 0.
            # Remove path-tangent mismatch: shape yielding must not silently
            # replace a stalled shift with a different feed for each module.
            tangent = self.curve.base(s, 1)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
            correction -= np.sum(correction*tangent, axis=1)[:, None]*tangent
            lengths = np.linalg.norm(correction, axis=1)
            correction *= np.minimum(1., .008/np.maximum(lengths, 1e-9))[:, None]
            self.curve.deposit(s, correction, .18, dt)
        self.n_adaptations += 1

    def command(self, data, t, forces, torque):
        p, centers = self.observe(data)
        q = data.qpos[self.qa]
        reference_p, reference_c, locations = self.curve.reference(self.shift)
        self.filtered_load += .1*(np.abs(torque)-self.filtered_load)
        tracking = np.abs(self.last-q-self.preload)
        maximum_error = float(tracking.max())
        front_progress = float(np.median(centers[:7, 1]-self.initial_centers[:7, 1]))
        rear_progress = float(np.median(centers[14:, 0]-self.initial_centers[14:, 0]))
        actual_material = self.observe_progress(centers, forces)
        self.last_body_progress = actual_material
        lag = self.shift-actual_material
        self.progress_history.append((t, actual_material))
        self.progress_history = [(a, b) for a, b in self.progress_history if a >= t-10.]
        actual_speed = ((actual_material-self.progress_history[0][1])/
                        max(t-self.progress_history[0][0], DT))
        error_scale = np.clip((np.radians(18)-maximum_error)/np.radians(14), 0., 1.)
        lag_scale = np.clip((.075-lag)/.05, 0., 1.)
        source_count, target_count = [int(np.count_nonzero(f > .1)) for f in forces]
        support_scale = 1. if target_count >= 2 and (source_count >= 2 or (locations[-1] > self.curve.target_bound)) else 0.
        requested = .004*float(error_scale)*float(lag_scale)*support_scale*smooth(t/2.)
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

        n = self.m.nv
        A, b = [], []
        jp, jr = np.zeros((3, n)), np.zeros((3, n))
        def add(j, value, weight):
            A.append(weight*np.atleast_2d(j).copy())
            b.append(weight*np.atleast_1d(value).copy())
        weights = self.curve.weights(locations)
        source_roll = -np.sign(self.curve.theta_rates[0])*self.speed*self.curve.axial_sines[0]/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[0]**2))
        target_roll = np.sign(self.curve.theta_rates[1])*self.speed*self.curve.axial_sines[1]/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[1]**2))
        for module, center in enumerate(centers):
            mujoco.mj_jac(self.m, self.d, jp, jr, center, self.head+module)
            u = self.d.xmat[self.head+module].reshape(3, 3)[:, 2]
            v = feedforward[module]+.35*(wanted_c[module]-center)
            v *= min(1., .008/max(np.linalg.norm(v), 1e-9))
            region = self.curve.regions(locations[module])
            add(jp, v, 2. if region != 1 else 2.5)
            wanted_u = wanted_p[module]-wanted_p[module+1]
            wanted_u /= max(np.linalg.norm(wanted_u), 1e-9)
            add(jr, .35*np.cross(u, wanted_u), .055)
            spin = weights[module, 0]*source_roll+weights[module, 1]*target_roll
            add(u@jr, spin, .12)

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
                    if module not in selected or contact.dist < selected[module][0]:
                        selected[module] = (contact.dist, contact.pos.copy(), side)
        for module, (_, point, side) in selected.items():
            mujoco.mj_jac(self.m, self.d, jp, jr, point, self.head+module)
            axis = np.array([1., 0., 0.]) if side == 0 else np.array([0., 1., 0.])
            origin = np.array([0., 0., 1.]) if side == 0 else CORNER
            normal = point-origin
            normal -= normal@axis*axis
            normal /= max(np.linalg.norm(normal), 1e-9)
            add(normal@jp, 0., 8.)
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
        raw = self.last+DT*solution.x[self.va]
        # Position targets cannot accumulate indefinitely through a saturated
        # actuator while shift is braked. Retain initial clamp preload; bound
        # only the EXTRA target-minus-encoder error from this new motion.
        raw = np.clip(raw, q+self.preload-np.radians(3.), q+self.preload+np.radians(3.))
        target = np.clip(raw, np.maximum(-np.radians(88), self.last-speed*DT),
                        np.minimum(np.radians(88), self.last+speed*DT))
        if t == 0.:
            target = self.last.copy()
        self.last = target.copy()
        info = dict(stage=mode, shift_mm=float(self.shift*1000), shift_speed_mm_s=float(self.speed*1000),
            actual_material_estimate_mm=actual_material*1000, actual_material_speed_mm_s=actual_speed*1000,
            reference_lead_mm=lag*1000, fixed_front_y_mm=front_progress*1000,
            fixed_rear_x_mm=rear_progress*1000, tracking_error_deg=math.degrees(maximum_error),
            source_modules=np.flatnonzero(forces[0] > .1).tolist(), target_modules=np.flatnonzero(forces[1] > .1).tolist(),
            material_locations_mm=(locations*1000).tolist(), material_region_codes=self.curve.regions(locations).tolist(),
            observed_projected_locations_mm=(self.projected_locations*1000).tolist(),
            observed_projected_region_codes=self.curve.regions(self.projected_locations).tolist(),
            reference_points_mm=(wanted_p*1000).tolist(), reference_centers_mm=(wanted_c*1000).tolist(),
            body_curve_rms_error_mm=float(np.sqrt(np.mean((wanted_c-centers)**2))*1000),
            source_material_roll_rad_s=source_roll, target_material_roll_rad_s=target_roll,
            curve_parameters=self.curve.parameters(), shape_updates=self.n_adaptations,
            bishop_joint_fit_max_error_deg=float(np.degrees(np.abs(bone_angles-q)).max()),
            solve_residual=float(np.linalg.norm(matrix@solution.x-vector)), solve_status=int(solution.status),
            feedback_fields=dict(hardware_candidates=['joint_angles', 'actuator_load', 'last_sent_targets'],
                simulation_truth_used=['world_body_pose', 'pipe_contact_points', 'pipe_contact_force_for_support_gate']),
            limitation='curve-projected material progress requires world-pose truth; actual target contacts/positions score completion')
        return target, info
