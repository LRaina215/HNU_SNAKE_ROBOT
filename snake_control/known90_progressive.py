"""Progressive source-to-target feed with exact physical rod chords.

One common window moves through a fixed source / inner upward bridge / target
route. Local deformation is resampled; no fixed-node polar unwinding is used.
"""
import numpy as np

from snake_control.known90_global_local import ActualBearingMapper, LocatedCurve
from snake_control.inner_ccw_compliance import InnerCCWRoute
from snake_control.suspended_supply_transfer import DT, LINK, nodes


class ProgressiveFeed:
    def __init__(self, model, data, feedback=True, speed=.005):
        self.m = model
        self.mapper = ActualBearingMapper(model)
        self.qa, self.va = self.mapper.qa, self.mapper.va
        self.route = InnerCCWRoute(initial_points=nodes(data), source_phi=-np.pi/4)
        self.feedback, self.speed = feedback, speed
        self.shift = self.phase = self.angle = self.axial = self.target_delta = 0.
        self.coeff = np.zeros(2)
        self.last = data.ctrl.copy()
        self.kp = np.full(20, 127.)
        self.capture_age = 0.
        self.acquired = False
        self.events = []
        self.gated = False
        self.last_stations = self.route.window(0.)[1]
        self.last_scale = 1.

    def curve(self):
        return LocatedCurve(self.route, self.angle, self.axial, self.coeff, self.target_delta)

    def progress_path(self):
        s = np.linspace(0, self.route.total, 4001)
        p = self.route.sample(s)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        return s, p, arc

    def locate(self, c, source):
        sm = (self.last_stations[:-1] + self.last_stations[1:]) / 2
        carrying = source & (sm < -.10)
        base = self.route.sample(self.last_stations)
        bc = (base[:-1] + base[1:]) / 2
        if carrying.sum() >= 4:
            a = np.angle(np.exp(1j * (np.arctan2(c[carrying, 2]-1, c[carrying, 1])
                                     - np.arctan2(bc[carrying, 2]-1, bc[carrying, 1]))))
            wanted = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
            self.angle += np.clip(np.angle(np.exp(1j*(wanted-self.angle))), -.015, .015)
            # Keep the physical source departure on its existing straight pipe.
            # Body axial mismatch also alters feed cooperation below; it cannot
            # move the bridge cut past the elbow and compress the free prefix.
            wanted_x = np.median(c[carrying, 0] - bc[carrying, 0])
            wanted_x = np.clip(wanted_x, .91-self.route.q[0], .995-self.route.q[0])
            self.axial += np.clip(wanted_x-self.axial, -.0005, .0005)
        return carrying, bc

    def command(self, data, t):
        p = nodes(data)
        c = (p[:-1] + p[1:]) / 2
        q, load = data.qpos[self.qa], data.qfrc_actuator[self.va]
        sr = np.linalg.norm(c[:, [1, 2]]-[0, 1], axis=1)
        rt = np.linalg.norm(c[:, [0, 2]]-[1.1, 1], axis=1)
        source = (sr > .074) & (sr < .120) & (c[:, 0] < 1.035)
        target = (rt > .069) & (rt < .120) & (c[:, 1] > .15)
        prefix = 0
        while prefix < 21 and target[prefix]:
            prefix += 1
        wrap = np.ptp(np.unwrap(np.arctan2(c[:prefix, 2]-1, c[:prefix, 0]-1.1))) if prefix >= 2 else 0.
        tracking = float(np.degrees(abs(self.last-q)).max())
        condition = prefix >= 5 and wrap > 2.3 and tracking < 25
        self.capture_age = self.capture_age + DT if condition else 0.
        if not self.acquired and self.capture_age >= 1:
            self.acquired = True
            self.events.append(dict(t_s=t, event="active_geometric_wrap", prefix=prefix, wrap_rad=float(wrap)))
        carrying, bc = self.locate(c, source)
        execution, es = self.curve().window(self.shift)
        head_error = float(np.linalg.norm(p[0]-execution[0]))
        self.gated = tracking > 32 or head_error > .10 or (not self.acquired and self.shift > self.route.bridge_length+.40)
        scale = 0. if self.gated else float(np.clip((30-tracking)/22, .15, 1.))
        if head_error > .04:
            scale *= .4
        inc = self.speed * DT * scale
        self.shift = min(self.route.total, self.shift + inc)
        axial_error = float(np.median(bc[carrying, 0] + self.axial-c[carrying, 0])) if carrying.any() else 0.
        factor = float(np.clip(self.route.alpha/self.route.beta**2 + 1.5*axial_error, .06, .65))
        phase_inc = (inc if inc else self.speed*DT*.08) * self.route.beta/.029 * factor
        self.phase += phase_inc
        nominal, ns = self.route.window(self.shift)
        execution, es = self.curve().window(self.shift)
        sm = (es[:-1]+es[1:])/2
        planned = (sm > -2*LINK) & (sm < self.route.bridge_length) & (~target)
        if self.feedback and planned.any():
            err = (execution[:-1]+execution[1:])/2 - c
            wanted = .25*np.mean(err[planned][:, [1, 2]], axis=0)
            wanted = np.clip(wanted, -.012, .012)
            self.coeff += np.clip(wanted-self.coeff, -.00015, .00015)
        if self.feedback and self.acquired and prefix >= 5:
            wanted = np.clip(.25*(.087-np.median(rt[:prefix])), -.003, .003)
            self.target_delta += np.clip(wanted-self.target_delta, -.0001, .0001)
        # Re-walk the deformed spatial curve. All 21 chords are LINK even when
        # its departure, shape, and receiving radius change.
        execution, es = self.curve().window(self.shift)
        self.last_stations = es.copy()
        src_ids = np.flatnonzero(source & ((es[:-1]+es[1:])/2 < -.03))
        anchor = 3 if self.acquired and len(src_ids) < 5 else int(src_ids[-3]) if len(src_ids) >= 3 else 18
        weights = np.ones(22)
        active = np.count_nonzero((es[:-1]+es[1:])/2 >= -.10)
        weights[:min(22, max(3, active+2))] = 2.
        desired, report = self.mapper.solve(execution, data, self.last, anchor, phase_inc, weights)
        kp = np.full(20, 127.)
        if tracking > 18 and abs(load).max() > 1.8:
            kp[planned[:20]] = 95.25
        self.kp += np.clip(kp-self.kp, -127*DT/1.5, 127*DT/1.5)
        self.m.actuator_gainprm[:, 0] = self.kp
        self.m.actuator_biasprm[:, 1] = -self.kp
        step = float(np.degrees(abs(desired-self.last)).max())
        self.last = desired.copy()
        self.last_scale = scale
        stage = "continuous_target_intake" if self.acquired else "progressive_inner_upward_entry"
        return desired, dict(stage=stage, shared_shift_mm=self.shift*1000, roll_phase_rad=self.phase,
            tracking_deg=tracking, head_global_error_mm=float(np.linalg.norm(p[0]-nominal[0]))*1000,
            head_execution_error_mm=head_error*1000, estimated_prefix=prefix, estimated_wrap_rad=float(wrap),
            estimated_capture=self.acquired, advancement_gate=self.gated, speed_scale=scale,
            global_points_mm=nominal*1000, execution_points_mm=execution*1000, estimated_points_mm=p*1000,
            material_stations_mm=es*1000, bridge_coeff_mm=np.array([[0,*self.coeff],[0,0,0]])*1000,
            radius_delta_mm=np.array([0,self.target_delta])*1000, source_pose_azimuth_rad=self.angle,
            source_pose_axial_mm=self.axial*1000, kp_nm_per_rad=self.kp.copy(), roll_factor=factor,
            actual_source_axial_matching_error_mm=axial_error*1000, maximum_dispatch_step_deg=step,
            controller_source_geometry_mask=source, controller_target_geometry_mask=target,
            maximum_reference_rod_length_error_mm=float(np.max(abs(np.linalg.norm(np.diff(execution,axis=0),axis=1)-LINK)))*1000,
            physical_root_assigned=False, exact_contact_used_for_local_feedback=False, **report)
