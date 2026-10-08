"""Continuation of CLOSE1B with an elastic elbow reservoir and shared feed.

Pipe wraps keep their independently observed winding. The spatial plan is
registered once at startup; subsequent observations change loading and local
deformation, never replace its destination. Only joint targets are dispatched.
"""
import copy
import math
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear, minimize

from snake_control.global_transfer_route import FrozenRouteSpec, route_body, sample_route
from snake_control.unified_curve_transfer import (
    UnifiedCurveTransfer, RegisteredThreeRegionCurve, LINK, smooth,
    physical_wrap_masks,
    cylindrical_contact, DT,
)


class CoordinatedRouteCurve(RegisteredThreeRegionCurve):
    def sample(self, s, phase=None):
        # A pointwise mesh-envelope projection has a discontinuous active
        # branch near the elbow. Generate a continuous guide instead; retain
        # full-rod original-solid checks and physical contact/model constraints.
        clearance = self.clearance_enabled
        self.clearance_enabled = False
        try:
            return super().sample(s,phase)
        finally:
            self.clearance_enabled = clearance

    def reference(self, shift, phase=None, _clearance_attempt=0):
        """Read current guide without silently depositing clearance repairs."""
        arc, xyz = self.arc_map(phase)
        head = self.initial_head_arc+shift
        if self.deformation_anchor_parameter is not None:
            head = float(np.interp(self.deformation_anchor_parameter, self.arc_grid_u, arc)) + shift-self.deformation_anchor_shift
        if self.ordered_node_arc_offsets is None:
            self.ordered_node_arc_offsets = self.initial_node_arc[0]-self.initial_node_arc
        node_arc = head-self.ordered_node_arc_offsets
        if node_arc[-1] <= arc[0] or node_arc[0] >= arc[-1]:
            raise ValueError('Shared material guide exhausted route')
        nodes = np.column_stack([np.interp(node_arc, arc, xyz[:, axis]) for axis in range(3)])
        center_arc = .5*(node_arc[:-1]+node_arc[1:])
        locations = np.interp(center_arc, arc, self.arc_grid_u)
        if phase is None:
            self.reference_node_arc = node_arc
            self.reference_center_arc = center_arc
            self.reference_node_parameters = np.interp(node_arc, arc, self.arc_grid_u)
            self.current_target_arc_bound = float(np.interp(self.target_bound, self.arc_grid_u, arc))
            if self.clearance_enabled:
                margin = float(self.clearance.link_minimum(nodes, samples=65).min())
                if margin <= 0.:
                    raise ValueError('Execution guide intersects original pipe solid')
                self.reference_pipe_clearance_mm = margin*1000
                self.reference_clearance_deficit_mm = 0.
        return nodes, .5*(nodes[:-1]+nodes[1:]), locations

    def arch_extra_arc(self, height, phase=None):
        spec = self.planned_route_spec
        grid = np.linspace(spec['source_bound'], spec['target_bound'], 201)
        points = sample_route(spec, grid, phase)
        f = np.linspace(0., 1., len(grid))
        bowed = points.copy()
        window = np.sin(np.pi*f)**2
        bowed[:, 2] += window*height
        bowed[:, :2] -= window[:, None]*.004*min(1., height/.020)
        return float(np.linalg.norm(np.diff(bowed, axis=0), axis=1).sum()-
                     np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def advance_planned_guidance(self, shift, dt, physical_nodes, forces, tracking, load):
        before, _, _ = self.reference(shift)
        parameters = self.reference_node_parameters.copy()
        spec = self.planned_route_spec
        phase = self.phase_at(shift)
        reservoir = self.reservoir_height
        extra_arc = self.arch_extra_arc(reservoir, phase)
        # The extra source material demanded by the bowed curve receives the
        # same rolling phase compensation as shared material transport.
        phase[0] += -np.sign(self.theta_rates[0])*extra_arc/(self.radii[0]*math.sqrt(1-self.axial_sines[0]**2))
        goal, goal_parameters = route_body(spec, shift, phase, bridge_height=reservoir)
        delta = goal-before
        target = goal_parameters >= spec['target_bound']
        if target.any():
            ids = np.flatnonzero(target)
            old = before[ids][:, [0, 2]]-[1.1, 1.]
            new = goal[ids][:, [0, 2]]-[1.1, 1.]
            old_phi = np.unwrap(np.arctan2(old[:, 1], old[:, 0]))
            new_phi = np.unwrap(np.arctan2(new[:, 1], new[:, 0]))
            branch = round(float(np.mean(old_phi-new_phi))/(2*np.pi))
            new_phi += branch*2*np.pi
            radial = old/np.maximum(np.linalg.norm(old, axis=1)[:, None], 1e-10)
            azimuth = np.column_stack([-radial[:, 1], radial[:, 0]])
            radial_delta = np.linalg.norm(new, axis=1)-np.linalg.norm(old, axis=1)
            delta[np.ix_(ids, [0, 2])] = (
                radial_delta[:, None]*radial +
                (np.linalg.norm(old, axis=1)*(new_phi-old_phi))[:, None]*azimuth)
        # One shape increment; avoid replacing the inherited hook with an
        # unfold/unwind stage or imposing the source's handedness on the target.
        constraints = []
        for module, direction in enumerate(np.diff(before, axis=0)):
            row = np.zeros((22, 3))
            direction = direction/max(np.linalg.norm(direction), 1e-12)
            row[module], row[module+1] = -direction, direction
            constraints.append(row.ravel())
        matrix = np.asarray(constraints)
        residual = matrix@delta.ravel()
        delta -= (matrix.T@np.linalg.lstsq(matrix@matrix.T, residual, rcond=1e-10)[0]).reshape(22, 3)
        rate = .004 * float(np.clip((np.radians(18)-np.max(tracking))/np.radians(15), .1, 1.))
        delta *= min(1., rate*dt/max(np.linalg.norm(delta, axis=1).max(), 1e-12))
        clamped = np.clip(self.arc_grid_u, parameters[-1], parameters[0])
        field = np.column_stack([np.interp(clamped, parameters[::-1], delta[::-1, axis])
                                 for axis in range(3)])
        field *= (1-smooth(np.abs(self.arc_grid_u-clamped)/(2*LINK)))[:, None]
        saved = copy.deepcopy(self.__dict__)
        reason = None
        for fraction in (1., .5, .25, .125, .0625, .03125):
            self.__dict__.clear()
            self.__dict__.update(copy.deepcopy(saved))
            self.route_guidance_offsets += fraction*field
            self.deformation_anchor_parameter = float(parameters[0])
            self.deformation_anchor_shift = shift
            self.shape_revision += 1
            self.arc_cache = None
            try:
                self.relax_ordered_spacing(shift, fraction*dt)
                after, _, _ = self.reference(shift)
                step = float(np.linalg.norm(after-before, axis=1).max())
                if step > .008*dt+1e-8:
                    reason = 'full curve deformation exceeds original 8 mm/s step'
                    continue
                self.route_guidance_last_step = dict(
                    accepted_fraction=fraction, maximum_node_step_mm=step*1000,
                    global_goal_rms_mm=float(np.sqrt(np.mean((goal-after)**2))*1000),
                    actual_global_goal_rms_mm=float(np.sqrt(np.mean((goal-physical_nodes)**2))*1000),
                    global_phase_rad=phase.tolist(), goal_node_parameters_mm=(goal_parameters*1000).tolist(),
                    target_shape_stage='continue_existing_hook', reservoir_height_mm=reservoir*1000,
                    geometric_arch_extra_arc_mm=extra_arc*1000,
                    basis='one startup route; one shared shift; distributed outside arch and pressure feedback')
                return self.route_guidance_last_step
            except ValueError as rejected:
                reason = str(rejected)
        self.__dict__.clear()
        self.__dict__.update(saved)
        self.route_guidance_last_step = dict(accepted_fraction=0., maximum_node_step_mm=0.,
                                             rejected_reason=reason, reservoir_height_mm=reservoir*1000)
        return self.route_guidance_last_step


class CoordinatedCurveTransfer(UnifiedCurveTransfer):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.curve.__class__ = CoordinatedRouteCurve
        curve = self.curve
        # The head's last two rods already have substantial forward axial
        # inclination. Give the receiving extension an open pitch so those
        # rods can continue around the pipe without entering a compact turn.
        points, _ = self.observe(data)
        curve.axial_sines[1] = .48
        curve.theta_rates[1] = -math.sqrt(1-.48**2)/curve.radii[1]
        u = curve.base.x
        values = curve.base(u)
        mask = u > curve.node_s0[0]
        distance = u[mask]-curve.node_s0[0]
        radial = points[0, [0, 2]]-[1.1, 1.]
        angle = math.atan2(radial[1], radial[0])+curve.theta_rates[1]*distance
        radius = np.linalg.norm(radial)+(curve.radii[1]-np.linalg.norm(radial))*smooth(distance/(3*LINK))
        values[mask, 1] = points[0, 1]+.48*distance
        values[np.ix_(mask, [0, 2])] = [1.1, 1.]+radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)])
        curve.base = CubicSpline(u, values, axis=0)
        curve.arc_cache = None
        curve.shape_revision += 1
        arc, _ = curve.arc_map()
        curve.initial_node_arc = np.interp(curve.node_s0, curve.arc_grid_u, arc)
        curve.initial_head_arc = float(curve.initial_node_arc[0])
        curve.reference_node_arc = curve.initial_node_arc.copy()
        self.physical_axial_sines = np.asarray(curve.axial_sines)
        plan_values = values.copy()
        target_mask = u >= curve.target_bound
        target_distance = u[target_mask]-curve.target_bound
        target_radial = points[8,[0,2]]-[1.1,1.]
        target_angle = math.atan2(target_radial[1],target_radial[0])+curve.theta_rates[1]*target_distance
        plan_values[target_mask,1] = points[8,1]+curve.axial_sines[1]*target_distance
        plan_values[np.ix_(target_mask,[0,2])] = [1.1,1.]+curve.radii[1]*np.column_stack([np.cos(target_angle),np.sin(target_angle)])
        plan = CubicSpline(u,plan_values,axis=0)
        spec = dict(version=2, origin='CLOSE1B startup curve, inherited front winding, outside arch reservoir',
                    source_bound=float(curve.source_bound), target_bound=float(curve.target_bound),
                    initial_head_parameter=float(curve.node_s0[0]), initial_head_arc=float(curve.initial_head_arc),
                    spline_x=plan.x.tolist(), spline_c=plan.c.tolist(),
                    radius=curve.radii[0], axial_sine=curve.axial_sines[0],
                    source_polar_rate=curve.theta_rates[0], target_polar_rate=curve.theta_rates[1],
                    source_right_hand_dphi_daxis=curve.theta_rates[0]/curve.axial_sines[0],
                    target_right_hand_dphi_daxis=-curve.theta_rates[1]/curve.axial_sines[1],
                    pipe_radii=list(curve.radii), pipe_axial_sines=list(curve.axial_sines),
                    bridge_controls=curve.base([curve.source_bound, curve.target_bound]).tolist(), frozen=True)
        # Register the new whole-body target by retaining the source tail
        # coordinate, rather than allowing its shorter open target span to
        # request a backward source movement at startup.
        low, high = 1.25, 1.75
        for _ in range(24):
            mid = .5*(low+high)
            spec['initial_head_parameter'] = mid
            _, locations = route_body(spec,0.)
            if locations[-1] < 0.:
                low = mid
            else:
                high = mid
        spec['initial_head_parameter'] = .5*(low+high)
        curve.planned_route_spec = FrozenRouteSpec(spec)
        curve.reservoir_height = 0.
        self.coordination_time = 0.
        self.nominal_support_goals = None
        self.coordination_feedback = {}
        self.coordination_stage = 'form_receiving_pitch'
        self.coordination_origin = np.zeros(2)
        self.coordination_shift_origin = 0.
        self.coordination_started = 0.
        self.actual_feed_credit = 0.
        self.maximum_arch_height = .045
        self.stroke_height = 0.
        self.stroke_extra_arc = 0.
        self.prediction_cache = None
        self.localized_yield_enabled = False
        self.alternating_yield_enabled = False

    def command(self, data, t, forces, torque, support_forces=None):
        self.coordination_time = t
        if self.nominal_support_goals is None:
            self.nominal_support_goals = np.maximum(np.asarray(forces).sum(1), 1.)
        material = self.physical_pipe_progress/self.physical_axial_sines
        delivered = material-self.coordination_origin
        supply = max(0., float(delivered[0]))
        drawn = max(0., float(delivered[1]))
        credit = max(0., supply-drawn)
        if self.coordination_stage == 'form_receiving_pitch':
            scales = np.array([.85,1.])
            self.curve.reservoir_height = 0.
            actual,_ = self.observe(data)
            goal,_ = route_body(self.curve.planned_route_spec,self.shift,self.curve.phase_at(self.shift))
            receiving_error = float(np.linalg.norm(goal[:9]-actual[:9],axis=1).max())
            if receiving_error < LINK/2 and np.count_nonzero(forces[1]>.1)>=2:
                self.coordination_stage = 'rear_feed'
                self.coordination_origin = material.copy()
                self.coordination_started = t
                self.events.append(dict(t_s=t,mode='actual_receiving_shape_ready_for_new_feed',
                                        maximum_front_goal_error_mm=receiving_error*1000,
                                        material_origin_mm=(material*1000).tolist(),
                                        meaning='stage-local material meter only; shape motion not charged as front draw'))
        if self.coordination_stage == 'rear_feed':
            scales = np.array([.60, 1.10])
            # Shape first supplies a finite rear-feed stroke. Its commanded
            # height is not credited as length; only measured source advance is.
            self.curve.reservoir_height = min(self.maximum_arch_height,
                                             self.curve.reservoir_height+.0008*.05)
            if supply >= .008 and credit>=.003 and t-self.coordination_started >= 4.:
                self.actual_feed_credit = supply
                self.stroke_height = self.curve.reservoir_height
                self.stroke_extra_arc = self.curve.arch_extra_arc(self.stroke_height,self.curve.phase_at(self.shift))
                self.coordination_stage = 'front_draw'
                self.coordination_shift_origin = self.shift
                self.events.append(dict(t_s=t, mode='actual_rear_feed_available_for_front_draw',
                                        measured_credit_mm=credit*1000))
        if self.coordination_stage == 'front_draw':
            scales = np.array([1.10, .75])
            self.actual_feed_credit = max(self.actual_feed_credit, supply)
            # One commanded handoff: ds + d(bridge arc) = 0 holds the rear
            # material coordinate while the front draws known supplied length.
            # This is a shape request, not proof that actual storage was released.
            wanted_arc = max(0., self.stroke_extra_arc-(self.shift-self.coordination_shift_origin))
            phase = self.curve.phase_at(self.shift)
            low, high = 0., self.stroke_height
            for _ in range(14):
                mid = .5*(low+high)
                if self.curve.arch_extra_arc(mid,phase) < wanted_arc:
                    low = mid
                else:
                    high = mid
            self.curve.reservoir_height = .5*(low+high)
            if drawn >= .8*self.actual_feed_credit and credit <= .003:
                self.coordination_stage = 'rear_feed'
                self.coordination_origin = material.copy()
                self.coordination_started = t
                self.events.append(dict(t_s=t, mode='actual_front_draw_completes_feed_cycle', drawn_mm=drawn*1000))
        counts = np.count_nonzero(np.asarray(forces)>.1, axis=1)
        scales[counts <= 2] = np.maximum(scales[counts <= 2], 1.)
        self.support_force_goals = self.nominal_support_goals*scales
        target, info = super().command(data, t, forces, torque, support_forces)
        info['coordination'] = dict(phase=self.coordination_stage,
                                   reservoir_height_mm=self.curve.reservoir_height*1000,
                                   measured_stored_material_mm=credit*1000,
                                   measured_positive_source_supply_mm=supply*1000,
                                   measured_positive_front_draw_mm=drawn*1000,
                                   pressure_scales=scales.tolist(), support_force_goals_N=self.support_force_goals.tolist(),
                                   feed_policy=self.coordination_feedback)
        info['stage'] = info['coordination']['phase']+'_'+info['stage']
        return target, info

    def coordinated_feed_limit(self, lag, error_scale, support_scale, physical_speeds):
        supplied = max(0., float((self.physical_pipe_progress/self.physical_axial_sines-self.coordination_origin)[0]))
        remaining = max(0., supplied-(self.shift-self.coordination_shift_origin))
        requested = min(.002*error_scale*support_scale, remaining/.05) if self.coordination_stage=='front_draw' else 0.
        self.coordination_feedback = dict(lead_mm=lag*1000, measured_supply_mm=supplied*1000,
                                         unconsumed_measured_supply_mm=remaining*1000,
                                         requested_mm_s=float(requested*1000),
                                         observed_side_speeds_mm_s=(np.asarray(physical_speeds)*1000).tolist())
        return float(requested)

    def repair_registered_route(self, p, centers, t, forces):
        # Keep the selected spatial destination instead of re-registering a
        # failed actual posture as the new plan.
        return False

    def refine_physical_execution(self, data, target, matrix, vector, local_bend, forces):
        """Joint-servo model response, refreshed periodically and checked each tick.

        Derivatives may be cached; the selected command is always forecast from
        the current full physical state. Forecasts never become recorded motion.
        """
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.m, state_type))
        mujoco.mj_getState(self.m, data, state, state_type)
        shadow = mujoco.MjData(self.m)
        steps = round(DT/self.m.opt.timestep)
        _, centers = self.observe(data)
        masks = np.asarray(physical_wrap_masks(centers)) & (forces>.1)
        required = []
        for side in range(2):
            if side == 0 and self.target_carries_transfer(centers, forces):
                continue
            count = getattr(self,'target_carrier_requirement',2) if side==1 else 2
            selected = sorted(np.flatnonzero(masks[side]), key=lambda i:forces[side,i], reverse=True)[:count]
            required.extend((side, int(module)) for module in selected)
        pipe_ids = {self.m.geom(name).id:side for side,names in enumerate([
            ['pipe_straight_01','pipe_sharp_in_02'], ['pipe_sharp_out_03','pipe_straight_04']]) for name in names}

        def predict(candidate):
            mujoco.mj_setState(self.m, shadow, state, state_type)
            shadow.ctrl[:] = candidate
            for _ in range(steps):
                mujoco.mj_step(self.m, shadow)
            velocity = np.empty(self.m.nv)
            mujoco.mj_differentiatePos(self.m, velocity, DT, data.qpos, shadow.qpos)
            mujoco.mj_forward(self.m, shadow)
            pf = np.zeros((2,21))
            for index, contact in enumerate(shadow.contact):
                for pipe, robot in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
                    side = pipe_ids.get(pipe)
                    module = int(self.m.geom_bodyid[robot])-self.head
                    if side is not None and 0<=module<21 and cylindrical_contact(side,contact.pos,contact.frame[:3]):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.m,shadow,index,force)
                        pf[side,module] += max(0.,force[0])
            return velocity, pf

        q = data.qpos[self.qa]
        lower = np.maximum(-np.radians(88), self.last-np.radians(10)*DT)
        upper = np.minimum(np.radians(88), self.last+np.radians(10)*DT)
        # The saved preload must enter anti-windup without a rate-limit jump.
        lower = np.maximum(lower, np.minimum(q-np.radians(3), upper))
        upper = np.minimum(upper, np.maximum(q+np.radians(3), lower))
        base_v, base_f = predict(target)
        predictions = 1
        cache = self.prediction_cache
        refreshed = cache is None or self.current_control_time-cache['t']>=.5
        if refreshed:
            vjac, fjac = np.zeros((self.m.nv,20)), np.zeros((2,21,20))
            for joint in range(20):
                plus = min(.002, max(0.,upper[joint]-target[joint]))
                minus = min(.002, max(0.,target[joint]-lower[joint]))
                step = plus if plus>=minus else -minus
                if abs(step)<1e-9:
                    continue
                changed = target.copy()
                changed[joint] += step
                v, f = predict(changed)
                predictions += 1
                vjac[:,joint] = (v-base_v)/step
                fjac[:,:,joint] = (f-base_f)/step
            cache = dict(t=self.current_control_time, velocity_jac=vjac, force_jac=fjac)
            self.prediction_cache = cache
        vjac, fjac = np.asarray(cache['velocity_jac']), np.asarray(cache['force_jac'])
        residual = matrix@base_v-vector
        jac = matrix@vjac
        low, high = lower-target, upper-target
        active = np.flatnonzero(high-low>1e-9)
        floor = np.array([.100001]*len(required))
        features = np.array([base_f[s,j] for s,j in required])
        force_jac = np.array([fjac[s,j] for s,j in required]).reshape(len(required),20)
        if len(active):
            fit = lsq_linear(np.vstack([jac[:,active],.25*np.eye(len(active))]),
                             np.r_[-residual,np.zeros(len(active))],
                             bounds=(low[active],high[active]),max_iter=60,tol=1e-7)
            def objective(change):
                r = residual+jac[:,active]@change
                return .5*(r@r+.0625*(change@change))
            constrained = minimize(objective,fit.x,method='SLSQP',
                jac=lambda x:jac[:,active].T@(residual+jac[:,active]@x)+.0625*x,
                bounds=list(zip(low[active],high[active])),
                constraints=[dict(type='ineq',fun=lambda x:features+force_jac[:,active]@x-floor,
                                  jac=lambda x:force_jac[:,active])],options=dict(maxiter=60,ftol=1e-10))
            candidate = target.copy()
            candidate[active] += constrained.x
        else:
            candidate = target.copy()
        candidate = np.clip(candidate, lower, upper)
        cv, cf = predict(candidate)
        predictions += 1
        options = [(target,base_v,base_f),(candidate,cv,cf)]
        def score(option):
            f = np.array([option[2][s,j] for s,j in required])
            deficit = np.maximum(floor-f,0.)
            return (float(deficit@deficit),float(np.linalg.norm(matrix@option[1]-vector)))
        if score(options[1])[0] > 1e-12:
            hold = np.clip(self.last,lower,upper)
            hv,hf = predict(hold)
            predictions += 1
            options.append((hold,hv,hf))
        chosen = min(options,key=score)
        feasible = score(chosen)[0] < 1e-12
        return chosen[0].copy(), dict(horizon_s=DT,predicted_candidates=predictions,
            response_derivatives_refreshed=refreshed,response_derivative_age_s=self.current_control_time-cache['t'],
            required_actual_carriers=required,corrected_feasible=feasible,
            baseline_task_residual=float(np.linalg.norm(residual)),corrected_task_residual=score(chosen)[1],
            forward_probe_rejected=bool(not feasible),
            basis='current full-state original-model joint ctrl forecasts; cached derivatives only; no root assignment')

    def load_state(self, path):
        elapsed = super().load_state(path)
        self.prediction_cache = None
        self.nominal_support_goals = np.asarray(self.nominal_support_goals)
        self.coordination_origin = np.asarray(self.coordination_origin)
        return elapsed
