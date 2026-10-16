"""CMU rolling windows with a moving six-hinge L-junction connector.

Head/tail targets retain absolute sinusoidal phase. Only connector hinges are
projected against an observed carrying frame; source targets are never fitted
or reset to that frame. Physical free-root state is never assigned here.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from snake_control.cmu_window_gait import wave, cylinder, placed_pose
from snake_control.suspended_supply_transfer import LINK, DT, LIMIT, nodes, skew

A = 1.2351672327411283
WS = .19063617553842116
SOURCE_PHASE = -.9
FREQUENCY = .02
MODEL_NAME = 'Snake_Robot_o20_full_source_inner.xml'


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.-15.*x+6.*x*x)


def blend_rotation(a, b, weight):
    return Slerp([0., 1.], Rotation.from_matrix([a, b]))([float(np.clip(weight, 0, 1))]).as_matrix()[0]


def initial_pose(model):
    return placed_pose(model, A, WS, SOURCE_PHASE, 12, .83)[0]


class TargetPrimitive:
    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id

    def pose(self, phase, boundary, radius=.088):
        self.d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        self.d.qpos[self.qa] = wave(A, WS, phase)
        mujoco.mj_fwdPosition(self.m, self.d)
        p = nodes(self.d)
        helix = cylinder(p)
        align = Rotation.align_vectors([[0., 1., 0.]], [helix['axis']])[0].as_matrix()
        local = (p-helix['origin'])@align.T
        index = np.clip(boundary+1, 0, 21)
        i = min(int(index), 20)
        fraction = index-i
        radial = (1-fraction)*local[i]+fraction*local[i+1]
        # World XZ angle decreases along +Y for this positive-handed helix.
        angle = np.arctan2(radial[2], radial[0])
        rot = Rotation.from_rotvec([0., angle-(-5*np.pi/6), 0.]).as_matrix()@align
        pp = (p-helix['origin'])@rot.T
        selected = (1-fraction)*pp[i]+fraction*pp[i+1]
        # Radius offset is a connector approach, not a scaling of real rods.
        shift = np.array([1.1, .20-selected[1], 1.])
        shift += (radius-.088)*np.array([np.cos(-5*np.pi/6), 0, np.sin(-5*np.pi/6)])
        trans = shift-rot@helix['origin']
        self.d.qpos[:3] = trans
        self.d.qpos[3:7] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_fwdPosition(self.m, self.d)
        body = self.head+min(int(boundary), 20)
        return self.d.xpos[body].copy(), self.d.xmat[body].reshape(3, 3).copy(), nodes(self.d).copy()


class ConnectorFit:
    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.va = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def register(self, actual, q, anchor):
        self.d.qpos[:] = actual.qpos
        self.d.qpos[self.qa] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        body = self.head+anchor
        rr = actual.xmat[body].reshape(3, 3)@self.d.xmat[body].reshape(3, 3).T
        translation = actual.xpos[body]-rr@self.d.xpos[body]
        root_rotation = rr@self.d.xmat[self.head].reshape(3, 3)
        self.d.qpos[:3] = rr@self.d.qpos[:3]+translation
        self.d.qpos[3:7] = Rotation.from_matrix(root_rotation).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_fwdPosition(self.m, self.d)

    def solve(self, actual, direct, previous, active, anchor, endpoint, goal_p, goal_r, unlimited=False, orientation_weight=1.):
        active = np.asarray(active, int)
        if not len(active):
            self.register(actual, direct, anchor)
            return direct.copy(), dict(position_error_mm=0., orientation_error_deg=0., predicted_pipe_depth_mm=0.,
                                       points_mm=nodes(self.d)*1000, active_connector_hinges=[])
        origin = actual.xpos[self.head+anchor].copy()
        step = np.radians(20)*DT
        low = np.full(len(active), -LIMIT) if unlimited else np.maximum(-LIMIT, previous[active]-step)
        high = np.full(len(active), LIMIT) if unlimited else np.minimum(LIMIT, previous[active]+step)
        cache = {}

        def evaluate(x):
            if 'x' in cache and np.array_equal(cache['x'], x):
                return cache
            q = direct.copy()
            q[active] = x
            self.register(actual, q, anchor)
            d = self.d
            mujoco.mj_jac(self.m, d, self.jp, self.jr, origin, self.head+anchor)
            ap, ar = self.jp[:, self.va[active]].copy(), self.jr[:, self.va[active]].copy()

            def jac(point, body):
                mujoco.mj_jac(self.m, d, self.jp, self.jr, point, body)
                return self.jp[:, self.va[active]]-ap+skew(point-origin)@ar, self.jr[:, self.va[active]]-ar
            body = self.head+endpoint
            position = d.xpos[body].copy()
            orientation = d.xmat[body].reshape(3, 3).copy()
            jj, angular = jac(position, body)
            error = Rotation.from_matrix(orientation@goal_r.T).as_rotvec()
            theta = np.linalg.norm(error)
            cross = skew(error)
            factor = (1.-.5*theta/np.tan(.5*theta))/theta**2 if theta > 1e-5 else 1/12
            angular = (np.eye(3)-.5*cross+factor*(cross@cross))@angular
            penalty, pj, depth, self_depth = np.zeros(42), np.zeros((42, len(active))), 0., 0.
            for con in d.contact:
                b1, b2 = map(int, self.m.geom_bodyid[[con.geom1, con.geom2]])
                one, two = self.head <= b1 < self.head+21, self.head <= b2 < self.head+21
                if one != two:
                    pipe = con.geom2 if one else con.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):
                        continue
                    b = b1 if one else b2
                    index, value = b-self.head, max(0., -con.dist-.001)
                    depth = max(depth, -float(con.dist))
                    if value > penalty[index]:
                        penalty[index] = value
                        pj[index] = (1 if one else -1)*con.frame[:3]@jac(con.pos, b)[0]
                elif one and two and abs(b1-b2) > 1:
                    index, value = 21+min(b1, b2)-self.head, max(0., -con.dist-.0003)
                    self_depth = max(self_depth, -float(con.dist))
                    if value > penalty[index]:
                        penalty[index] = value
                        pj[index] = con.frame[:3]@(jac(con.pos, b1)[0]-jac(con.pos, b2)[0])
            cache.clear()
            cache.update(x=x.copy(), q=q, position=position, error=error, depth=depth, self_depth=self_depth,
                         residual=np.r_[(position-goal_p)/.025, orientation_weight*error/.35, penalty/.001, .03*(x-direct[active])],
                         jac=np.vstack([jj/.025, orientation_weight*angular/.35, pj/.001, .03*np.eye(len(active))]),
                         points=nodes(d).copy())
            return cache
        seed = np.clip(previous[active], low+1e-8, high-1e-8)
        result = least_squares(lambda x: evaluate(x)['residual'], seed, jac=lambda x: evaluate(x)['jac'],
                               bounds=(low, high), max_nfev=70 if unlimited else 14,
                               ftol=1e-6, xtol=1e-6, gtol=1e-5)
        state = evaluate(result.x)
        return state['q'].copy(), dict(position_error_mm=float(np.linalg.norm(state['position']-goal_p)*1000),
             orientation_error_deg=float(np.degrees(np.linalg.norm(state['error']))),
             predicted_pipe_depth_mm=state['depth']*1000, predicted_self_depth_mm=state['self_depth']*1000,
             points_mm=state['points']*1000, active_connector_hinges=active+1,
             solver_evaluations=int(result.nfev), cumulative_source_wave_direct=True,
             physical_root_assigned=False)


def front_status(data, forces):
    p = nodes(data)
    c = (p[:-1]+p[1:])/2
    radius = np.linalg.norm(c[:, [0, 2]]-[1.1, 1.], axis=1)
    near = (radius > .073) & (radius < .112) & (c[:, 1] > .15)
    phi = np.unwrap(np.arctan2(c[:, 2]-1., c[:, 0]-1.1))
    prefix = 0
    while prefix < 21 and near[prefix]:
        if prefix and not (.20 < phi[prefix]-phi[prefix-1] < 1.20 and -.05 < c[prefix, 1]-c[prefix-1, 1] < .005):
            break
        prefix += 1
    wrap = float(np.ptp(phi[:prefix])) if prefix > 1 else 0.
    grip = float(forces[1, :max(prefix, 7)].sum())
    return prefix, wrap, grip, near


class CMURollingTransfer:
    def __init__(self, model, data, variant=1):
        self.m = model
        self.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id
        self.target = TargetPrimitive(model)
        self.connector = ConnectorFit(model)
        self.last = data.ctrl.copy()
        self.phase = SOURCE_PHASE
        self.clock = 0.
        self.boundary = 6.
        self.capture_age = 0.
        self.acquired = False
        self.failure = None
        self.best_prefix = 0
        self.last_gain_time = 0.
        self.events = []
        self.variant = variant
        self.unfold_seconds, self.coil_seconds = 30., 25.
        self.phase_delta = -np.pi/2

    def command(self, data, t, forces, source_any):
        tracking = float(np.degrees(abs(self.last-data.qpos[self.qa])).max())
        clock_rate = float(np.clip((35.-tracking)/25., .20, 1.))
        self.clock += DT*clock_rate
        self.phase = SOURCE_PHASE-2*np.pi*FREQUENCY*self.clock
        prefix, wrap, grip, near = front_status(data, forces)
        capture = prefix >= 5 and wrap > 2.3 and grip > 1.
        self.capture_age = self.capture_age+DT if capture else 0.
        if self.capture_age >= .5 and not self.acquired:
            self.acquired = True
            self.last_gain_time = t
            self.events.append(dict(t_s=t, event='active_front_capture', prefix=prefix, wrap_rad=wrap, bearing_n=grip))
        if prefix > self.best_prefix:
            self.best_prefix = prefix
            self.last_gain_time = t
            self.events.append(dict(t_s=t, event='new_target_prefix', prefix=prefix))
        if self.acquired and capture:
            # Reviewed paper-prose interpretation; not joint-wave phase speed.
            rate = np.pi*.058*FREQUENCY/LINK
            self.boundary = min(20., prefix+1., self.boundary+rate*DT*clock_rate)
        u = float(smooth(self.clock/self.unfold_seconds))
        coil = float(smooth((self.clock-self.unfold_seconds)/self.coil_seconds))
        loosen = 1.-.6*float(smooth(self.clock/(self.unfold_seconds*.5)))+.6*coil
        head_phase = self.phase-self.phase_delta*u
        k = np.arange(20)
        cutoff = min(int(self.boundary), 20)
        active = np.arange(cutoff, min(cutoff+6, 20))
        direct = wave(A, WS, self.phase)
        direct[:cutoff] = wave(A*loosen, WS, head_phase)[:cutoff]
        if self.variant == 2 and not self.acquired:
            # R02 macro change: loosen the complete released prefix, including
            # the connector, before asking it to leave its source-coil basin.
            direct[active] = wave(A*loosen, WS, self.phase)[active]
        # Initial source targets remain exactly the accepted sinusoidal pose.
        direct = np.clip(direct, np.maximum(-LIMIT, self.last-np.radians(20)*DT),
                         np.minimum(LIMIT, self.last+np.radians(20)*DT))
        source_bearing = np.flatnonzero(source_any & (np.arange(21) >= min(cutoff+6, 20)))
        target_bearing = np.flatnonzero(forces[1] > .1)
        anchor = int(source_bearing[-1]) if len(source_bearing) else int(target_bearing[0]) if len(target_bearing) else 18
        endpoint = min(cutoff, 20)
        radius = .088+.09*(1.-loosen)
        goal_p, goal_r, ideal = self.target.pose(head_phase, self.boundary, radius)
        if not self.acquired and u < 1:
            self.connector.register(data, wave(A, WS, self.phase), anchor)
            start_p = self.connector.d.xpos[self.head+6].copy()
            start_r = self.connector.d.xmat[self.head+6].reshape(3, 3).copy()
            # Unwrap around the source bottom rather than cutting through its
            # solid cross-section from the negative-Y initial connector end.
            radial0 = start_p[[1, 2]]-[0., 1.]
            phi0 = np.arctan2(radial0[1], radial0[0])
            phi1 = -np.pi/4
            while phi1 < phi0:
                phi1 += 2*np.pi
            phi = (1-u)*phi0+u*phi1
            radial = (1-u)*np.linalg.norm(radial0)+u*.17
            around = np.array([(1-u)*start_p[0]+u*.99, radial*np.cos(phi), 1.+radial*np.sin(phi)])
            late = float(smooth((u-.65)/.35))
            goal_p = (1-late)*around+late*goal_p
            goal_r = blend_rotation(start_r, goal_r, u)
        if self.variant == 2 and self.clock < self.unfold_seconds*.5:
            q = direct.copy()
            self.connector.register(data, q, anchor)
            body = self.head+endpoint
            report = dict(position_error_mm=float(np.linalg.norm(self.connector.d.xpos[body]-goal_p)*1000),
                 orientation_error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(
                     self.connector.d.xmat[body].reshape(3,3)@goal_r.T).as_rotvec()))),
                 points_mm=nodes(self.connector.d)*1000, active_connector_hinges=active+1,
                 predicted_pipe_depth_mm=0., source_wave_direct_release=True, physical_root_assigned=False)
        else:
            q, report = self.connector.solve(data, direct, self.last, active, anchor, endpoint, goal_p, goal_r,
                                             orientation_weight=coil if not self.acquired else 1.)
        q = np.clip(q, np.maximum(-LIMIT, self.last-np.radians(20)*DT), np.minimum(LIMIT, self.last+np.radians(20)*DT))
        before = self.last.copy()
        self.last = q.copy()
        if not self.acquired and self.clock >= self.unfold_seconds+self.coil_seconds+1/FREQUENCY:
            self.failure = 'full_unfold_restore_and_one_roll_cycle_without_front_capture'
        if self.acquired and t-self.last_gain_time > 60 and self.boundary < 20:
            self.failure = 'connector_advance_without_new_target_prefix_60s'
        stage = 'unfold_lower_approach' if self.clock < self.unfold_seconds else 'restore_target_wrap' if self.clock < self.unfold_seconds+self.coil_seconds else 'continued_front_roll' if not self.acquired else 'rolling_tailward_handoff'
        return q, dict(stage=stage, controller_clock_s=self.clock, clock_rate=clock_rate,
            shift_mm=self.boundary*LINK*1000, window_transition_modules=self.boundary,
            roll_phase_rad=self.phase, head_phase_rad=head_phase, joint_phase_delta_rad=self.phase_delta*u,
            loosening_factor=loosen, head_window_mask=k<cutoff, connector_mask=np.isin(k, active),
            actual_target_prefix=prefix, actual_front_wrap_rad=wrap, actual_front_bearing_n=grip,
            actual_received_modules=np.flatnonzero(near), actual_new_retained_modules=np.flatnonzero(near),
            actual_source_exited_modules=np.flatnonzero(~source_any), front_capture=self.acquired,
            goal_connector_point_mm=goal_p*1000, reference_points_mm=report['points_mm'], ideal_target_points_mm=ideal*1000,
            reference_anchor_module=anchor, projection=report, tracking_deg=tracking,
            maximum_dispatch_step_deg=float(np.degrees(abs(q-before)).max()),
            kp_nm_per_rad=self.m.actuator_gainprm[:, 0].copy(), physical_root_assigned=False,
            source_wave_uses_absolute_phase=True, actual_bearing_registration_applies_to_connector_only=True,
            failure=self.failure)
