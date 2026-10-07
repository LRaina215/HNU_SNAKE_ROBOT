"""One-module-at-a-time transfer with support and shape in one velocity solve.

Observations are simulation world poses and pipe contacts. Only joint position
targets leave this controller; the free-root variables are predictions. The
LocalSection term is a fixed-parameter shape increment, not force adaptation.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from snake_control.pole_compliance import ComplianceParameters
from snake_control.three_segment_compliance import LocalSection


DT = .05
CORNER = np.array([1.1, 0., 1.])


class ProgressiveSupportHandoff:
    def __init__(self, model, data, forces, revision=1):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.va = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id
        self.last = data.ctrl.copy()
        self.revision = revision
        self.events = []
        self.contact_age = np.zeros((2, 21))
        self.last_seen = np.full((2, 21), -100.)
        self.last_seen[forces > .1] = 0.
        p, c = self.observe(data)
        self.initial_centers = c.copy()
        self.initial_radii = np.linalg.norm(c[:, [1, 2]] - [0, 1], axis=1)
        target = np.flatnonzero((forces[1] > .1) & (c[:, 1] > .085))
        if not len(target):
            raise ValueError('Initial front must already contact target pipe')
        self.front = int(target.max())
        self.initial_front = self.front
        self.candidate = min(20, self.front + 1)
        self.acquired = set(range(self.front + 1))
        self.last_admission_time = 0.
        self.stage = 'receive_one'
        self.stage_start = 0.
        self.goal = None
        self.goal_distance = None
        self.previous_source = set()
        self.config = dict(joint_count=20, head_length_mm=64.5,
            joints=[dict(joint_index=i+1, axis='Yaw' if i % 2 == 0 else 'Pitch',
                         link_length_mm=64.5) for i in range(20)])

    def observe(self, data):
        self.d.qpos[:] = data.qpos
        mujoco.mj_fwdPosition(self.m, self.d)
        p = np.vstack([self.d.site('s_head').xpos,
                      *[self.d.joint('joint'+str(i)).xanchor for i in range(1, 21)],
                      self.d.site('s_tail').xpos])
        return p, (p[:-1]+p[1:])/2

    @staticmethod
    def pipe_geometry(center, side):
        axis = np.array([1., 0, 0]) if side == 0 else np.array([0., 1, 0])
        origin = np.array([0., 0., 1.]) if side == 0 else CORNER
        radial = center-origin
        radial -= axis*(radial@axis)
        radius = max(float(np.linalg.norm(radial)), 1e-8)
        return axis, radial/radius, radius

    def command(self, data, t, forces):
        p, centers = self.observe(data)
        q = data.qpos[self.qa]
        touching = forces > .1
        self.last_seen[touching] = t
        self.contact_age = np.where(touching, self.contact_age+DT, 0.)
        target_radii = np.linalg.norm(centers[:, [0, 2]] - [1.1, 1.], axis=1)
        # Only actual sustained contact admits the next material module.
        k = self.candidate
        admitted = (k > self.front and self.contact_age[1, k] >= .75
                    and centers[k, 1] > .09 and target_radii[k] < .115
                    and np.count_nonzero(touching[1, :k]) >= 2)
        if admitted:
            self.front = k
            self.acquired.add(k)
            self.last_admission_time = t
            self.events.append(dict(t_s=t, event='target_admission', module=k,
                                    target_force_N=float(forces[1, k]),
                                    center_mm=(centers[k]*1000).tolist()))
            self.candidate = min(20, k+1)
            self.stage, self.stage_start = 'receive_one', t
            self.goal = None
        k = self.candidate
        # Recomputed every observation: source support can disappear AND return.
        # Never enlarge a free region to satisfy a chosen connector length.
        source_geometry = ((centers[:, 0] < 1.065)
            & (np.linalg.norm(centers[:, [1, 2]]-[0., 1.], axis=1) < .13))
        source_recent = t-self.last_seen[0] < .6
        protected = set(np.flatnonzero(source_geometry & source_recent
                                      & (np.arange(21) > k)).tolist())
        for module in sorted(protected-self.previous_source):
            self.events.append(dict(t_s=t, event='source_support_observed', module=module))
        for module in sorted(self.previous_source-protected):
            self.events.append(dict(t_s=t, event='source_support_temporarily_missing', module=module))
        self.previous_source = protected.copy()
        rear = min(protected) if protected else None
        if self.goal is None:
            uv = centers[:self.front+1, [0, 2]]-[1.1, 1.]
            theta = np.unwrap(np.arctan2(uv[:, 1], uv[:, 0]))
            sign = np.sign(np.median(np.diff(theta))) or 1.
            angle = theta[-1] + sign*2*np.arcsin(.0645*np.sqrt(1-.32**2)/(2*.093))
            self.goal = np.array([1.1+.093*np.cos(angle),
                                  max(.105, centers[self.front, 1]-.32*.0645),
                                  1.+.093*np.sin(angle)])
            self.goal_distance = float(np.linalg.norm(self.goal-centers[k]))
        # Alternate within the same admission attempt; protected rear is retained.
        elapsed = t-self.stage_start
        if self.stage == 'receive_one' and elapsed >= 6.:
            self.stage, self.stage_start = 'rear_feed_one', t
        elif self.stage == 'rear_feed_one' and elapsed >= 10.:
            self.stage, self.stage_start = 'receive_one', t

        n = self.m.nv
        A, b = [], []
        jp, jr = np.zeros((3, n)), np.zeros((3, n))
        def add(jacobian, velocity, weight):
            A.append(weight*np.atleast_2d(jacobian))
            b.append(weight*np.atleast_1d(velocity))
        source_modules = np.flatnonzero(source_geometry & (np.arange(21) > k))
        source_start = int(source_modules.min()) if len(source_modules) else None
        for module, center in enumerate(centers):
            mujoco.mj_jac(self.m, self.d, jp, jr, center, self.head+module)
            u = self.d.xmat[self.head+module].reshape(3, 3)[:, 2]
            if module <= self.front:
                axis, normal, radius = self.pipe_geometry(center, 1)
                roll = -.045 if self.stage == 'receive_one' else -.012
                velocity = .029*roll*np.cross(u, normal)
                velocity += normal*np.clip((.092-radius)*.18, -.002, .002)
                # Shape corrections preserve a cloud, not fixed world anchors.
                wanted_y = centers[self.front, 1]+.32*.0645*(self.front-module)
                velocity += axis*np.clip((wanted_y-center[1])*.08, -.001, .002)
                add(jp, velocity, 2.)
                add(u@jr, roll, .06)
            elif module == k:
                direction = self.goal-center
                velocity = direction*min(.18, .003/max(np.linalg.norm(direction), 1e-8))
                add(jp, velocity, 3.)
                # Align candidate's tangent with the newly extended helix.
                wanted_u = centers[self.front]-self.goal
                wanted_u /= max(np.linalg.norm(wanted_u), 1e-8)
                add(jr, .12*np.cross(u, wanted_u), .05)
            else:
                # Remaining rear retains its source wrap, including modules
                # that momentarily lose a contact. No joint-zeroing override.
                axis, normal, radius = self.pipe_geometry(center, 0)
                roll = .065 if self.stage == 'rear_feed_one' else .008
                velocity = .029*roll*np.cross(u, normal)
                wanted_radius = np.clip(self.initial_radii[module], .086, .105)
                velocity += normal*np.clip((wanted_radius-radius)*.16, -.002, .002)
                add(jp, velocity, 1.5)
                add(u@jr, roll, .06)

        # Fixed LocalSection shaping is an objective in this same solve.
        shape_velocity = np.zeros(20)
        shape_status = 'inactive'
        if self.stage == 'rear_feed_one' and source_start is not None and source_start < 17:
            geometry = dict(config=self.config, points_mm=p*1000,
                            curve=dict(gamma_min_rad=np.radians(1.)))
            length = np.linalg.norm(np.diff(p[source_start:], axis=0), axis=1).sum()
            pitch = float(np.arcsin(np.clip(abs(p[-1, 0]-p[source_start, 0])/length, .1, .6)))
            try:
                zero = LocalSection(geometry, ComplianceParameters(0., pitch),
                                    source_start, 20, np.array([1., 0., 0.]), np.array([0., 0., 1000.]))
                shaped = LocalSection(geometry, ComplianceParameters(.01, pitch),
                                      source_start, 20, np.array([1., 0., 0.]), np.array([0., 0., 1000.]))
                phase = zero.align(q[source_start:20], 0.)
                shape_velocity[source_start:20] = np.clip(
                    shaped.angles(phase)-zero.angles(phase), -np.radians(2), np.radians(2))
                shape_status = 'fixed_parameter_LocalSection_shape_objective'
            except (ValueError, FloatingPointError) as error:
                shape_status = 'unavailable: '+str(error)
        joint_jac = np.zeros((20, n))
        joint_jac[np.arange(20), self.va] = 1.
        inward = -.25*np.sign(q)*np.maximum(abs(q)-np.radians(78), 0.)
        add(joint_jac, inward+shape_velocity, .045)

        # Actual contact Jacobians tie the predicted root to the supporting
        # bodies. Release only the current candidate; never a whole span.
        selected = {}
        for contact in self.d.contact:
            for robot_geom, pipe_geom in [(contact.geom1, contact.geom2), (contact.geom2, contact.geom1)]:
                body = int(self.m.geom_bodyid[robot_geom])
                name = self.m.geom(pipe_geom).name or ''
                if not self.head <= body < self.head+21 or not name.startswith('pipe_'):
                    continue
                module = body-self.head
                if module == k or not (module <= self.front or module in protected):
                    continue
                if module not in selected or contact.dist < selected[module][0]:
                    selected[module] = (contact.dist, contact.pos.copy())
        for module, (_, point) in selected.items():
            mujoco.mj_jac(self.m, self.d, jp, jr, point, self.head+module)
            add(jp, np.zeros(3), 10.)
        # Loss of all posterior contacts freezes advance objectives through the
        # caller's support gate; no change of material ownership is inferred.
        speed = np.radians(10.)
        low, high = np.full(n, -1.), np.full(n, 1.)
        low[self.va] = np.clip((np.radians(-88)-self.last)/DT, -speed, speed-1e-7)
        high[self.va] = np.clip((np.radians(88)-self.last)/DT, -speed+1e-7, speed)
        solution = lsq_linear(np.vstack(A), np.concatenate(b), bounds=(low, high),
                              tol=1e-6, max_iter=60)
        raw = self.last+DT*solution.x[self.va]
        # Output projection only; no extra shape or zero-angle command afterward.
        target = np.clip(raw, np.maximum(-np.radians(88), self.last-speed*DT),
                        np.minimum(np.radians(88), self.last+speed*DT))
        if t == 0.:
            target = self.last.copy()
        self.last = target.copy()
        info = dict(stage=self.stage, front_last_module=self.front,
                    candidate_module=k, protected_source_modules=sorted(protected),
                    rear_first_module=rear, current_source_modules=np.flatnonzero(touching[0]).tolist(),
                    current_target_modules=np.flatnonzero(touching[1]).tolist(),
                    candidate_goal_mm=(self.goal*1000).tolist(),
                    candidate_distance_mm=float(np.linalg.norm(self.goal-centers[k])*1000),
                    candidate_contact_seconds=float(self.contact_age[1, k]),
                    newly_admitted_modules=sorted(self.acquired-set(range(self.initial_front+1))),
                    fixed_shape_status=shape_status, fixed_shape_velocity_rad_s=shape_velocity.tolist(),
                    solve_residual=float(np.linalg.norm(np.vstack(A)@solution.x-np.concatenate(b))),
                    solve_status=int(solution.status))
        return target, info
