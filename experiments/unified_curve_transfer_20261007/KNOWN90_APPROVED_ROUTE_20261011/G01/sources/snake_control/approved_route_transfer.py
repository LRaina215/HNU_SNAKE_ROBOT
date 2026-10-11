"""Approved complete route -> exact rods -> actual 20 hinges -> joint control.

MuJoCo actual pose and drive loads are allowed observations for this experiment.
Only private FK data are registered; the physical free root is never assigned
after initialization. Source sizing is fixed, not feedback-driven relaxation.
"""
from pathlib import Path
import importlib.util
import numpy as np
import mujoco
from scipy.optimize import brentq
from snake_control.suspended_supply_transfer import DT, LINK, nodes, WholeCurveMapper
from snake_control.known90_global_local import ActualBearingMapper

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_ROUTE_CONFIRM_20261011'
spec = importlib.util.spec_from_file_location('approved_route_geometry', APPROVED/'draw_global_route.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class ExecutionCurve:
    """Fixed support sizing plus bounded C1 local shape, with exact rod chords."""
    def __init__(self, global_route, source_delta=-.0015, coeff=None, target_delta=0., outlet=None):
        self.global_route = global_route
        self.source_delta = source_delta
        self.coeff = np.zeros(3) if coeff is None else np.asarray(coeff)
        self.target_delta = target_delta
        self.outlet = np.zeros(3) if outlet is None else np.asarray(outlet)

    def sample(self, s):
        s = np.atleast_1d(s)
        route = self.global_route
        p = route.sample(s)
        bl = route.bridge_length
        # Source cross-section chosen once for true-mechanism initialization.
        # Tiny inlet displacement joins back to the approved fixed target E.
        phi = geometry.SOURCE_PHI-geometry.BETA*s/geometry.R
        radial = np.column_stack([np.zeros(len(s)), np.cos(phi), np.sin(phi)])
        dq = self.source_delta*np.array([0, np.cos(geometry.SOURCE_PHI), np.sin(geometry.SOURCE_PHI)])
        dt = self.source_delta*geometry.BETA/geometry.R*np.array([0, np.sin(geometry.SOURCE_PHI), -np.cos(geometry.SOURCE_PHI)])
        source = s < 0
        p[source] += self.source_delta*radial[source]
        w = np.clip(s/bl, 0, 1)
        mid = (s >= 0) & (s < bl)
        h00 = 2*w**3-3*w**2+1
        h10 = w**3-2*w**2+w
        p[mid] += h00[mid, None]*dq + bl*h10[mid, None]*dt
        h = 16*w*w*(1-w)*(1-w)
        p += h[:, None]*self.coeff
        # Outlet cooperation affects the leaving arc, not source bearing size.
        p += h[:, None]*self.outlet
        # Receiving radius may adjust only after a real geometric wrap.
        blend = np.clip((s-bl)/(2*LINK), 0, 1)
        blend = blend*blend*(3-2*blend)
        radial_target = p[:, [0, 2]]-[1.1, 1.]
        radial_target /= np.maximum(np.linalg.norm(radial_target, axis=1)[:, None], 1e-9)
        p[:, [0, 2]] += blend[:, None]*self.target_delta*radial_target
        return p

    def window(self, shift):
        stations, points = [float(shift)], [self.sample([shift])[0]]
        for _ in range(21):
            s, p = stations[-1], points[-1]
            grid = s-np.linspace(.0001, .20, 38)
            residual = np.linalg.norm(self.sample(grid)-p, axis=1)-LINK
            j = np.flatnonzero((residual[1:] >= 0) & (residual[:-1] < 0))
            if not len(j):
                raise ValueError('No continuous preceding physical rod on local curve')
            k = int(j[0])
            previous = brentq(lambda v: np.linalg.norm(self.sample([v])[0]-p)-LINK,
                              grid[k+1], grid[k], xtol=1e-11)
            stations.append(previous)
            points.append(self.sample([previous])[0])
        return np.array(points), np.array(stations)


def initial_fit(model, route, source_delta=-.0015):
    execution = ExecutionCurve(route, source_delta)
    guide, stations = execution.window(0)
    mapper = WholeCurveMapper(model)
    pose, report = mapper.initialize(guide)
    p = nodes(mapper.d)
    centers = (p[:-1]+p[1:])/2
    radii = np.linalg.norm(centers[:, [1, 2]]-[0, 1], axis=1)
    bearing = set()
    target = set()
    for contact in mapper.d.contact:
        names = [model.geom(int(g)).name or '' for g in (contact.geom1, contact.geom2)]
        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
        robot = [int(b)-mapper.head for b in bodies if mapper.head <= b < mapper.head+21]
        if any(n in ('pipe_straight_01', 'pipe_sharp_in_02') for n in names):
            bearing.update(robot)
        if any(n in ('pipe_straight_04', 'pipe_sharp_out_03') for n in names):
            target.update(robot)
    report.update(source_center_radii_mm=radii*1000, initial_source_contact_modules=sorted(bearing),
                  initial_target_contact_modules=sorted(target), reference_stations_mm=stations*1000,
                  fixed_source_radius_delta_mm=source_delta*1000,
                  actual_rod_length_error_mm=float(np.max(abs(np.linalg.norm(np.diff(p, axis=0), axis=1)-LINK)))*1000,
                  physical_steps=0, global_route_overwritten=False)
    report['ready'] = bool(report['guide_rms_mm'] < 35 and report['pipe_depth_mm'] <= .9
                           and report['self_depth_mm'] <= .5 and report['maximum_joint_deg'] < 88.01
                           and np.all((radii > .075) & (radii < .11))
                           and len(bearing.intersection(range(12, 21))) >= 4 and not target)
    return pose, guide, report


class ApprovedRouteTransfer:
    def __init__(self, model, data, route, feedback=True, speed=.005, variant=1, source_delta=-.0015):
        self.m, self.route = model, route
        self.mapper = ActualBearingMapper(model)
        self.qa, self.va = self.mapper.qa, self.mapper.va
        self.last = data.ctrl.copy()
        self.feedback, self.speed, self.variant = feedback, speed, variant
        self.source_delta = source_delta
        self.shift = self.phase = 0.
        self.coeff, self.outlet = np.zeros(3), np.zeros(3)
        self.target_delta = 0.
        self.kp = np.full(20, 127.)
        self.capture_age = 0.
        self.acquired = False
        self.events = []
        self.stations = ExecutionCurve(route, source_delta).window(0)[1]

    def curve(self):
        return ExecutionCurve(self.route, self.source_delta, self.coeff, self.target_delta, self.outlet)

    def command(self, data, t):
        p = nodes(data)
        c = (p[:-1]+p[1:])/2
        q, load = data.qpos[self.qa], data.qfrc_actuator[self.va]
        sr = np.linalg.norm(c[:, [1, 2]]-[0, 1], axis=1)
        rt = np.linalg.norm(c[:, [0, 2]]-[1.1, 1], axis=1)
        source = (sr > .074) & (sr < .118) & (c[:, 0] < 1.035)
        target = (rt > .073) & (rt < .118) & (c[:, 1] > .15)
        prefix = 0
        while prefix < 21 and target[prefix]:
            prefix += 1
        wrap = float(np.ptp(np.unwrap(np.arctan2(c[:prefix, 2]-1, c[:prefix, 0]-1.1)))) if prefix >= 2 else 0.
        tracking = float(np.degrees(abs(self.last-q)).max())
        capture = prefix >= 5 and wrap > 2.3 and tracking < 30
        self.capture_age = self.capture_age+DT if capture else 0.
        if self.capture_age >= 1. and not self.acquired:
            self.acquired = True
            self.events.append(dict(t_s=t, event='controller_actual_geometry_wrap', prefix=prefix, wrap_rad=wrap))
        execution, es = self.curve().window(self.shift)
        head_error = float(np.linalg.norm(p[0]-execution[0]))
        # Body correspondence is the last exact-rod window, not nearest arbitrary
        # curves. Local admittance accepts bounded deflection under real load.
        sm = (es[:-1]+es[1:])/2
        bridge = (sm > -.02) & (sm < self.route.bridge_length) & (~source) & (~target)
        if self.feedback and bridge.any():
            w = np.clip(sm/self.route.bridge_length, 0, 1)
            h = 16*w*w*(1-w)*(1-w)
            useful = bridge & (h > .12)
            if useful.any():
                err = c-(execution[:-1]+execution[1:])/2
                resistance = float(abs(load[np.minimum(np.flatnonzero(useful), 19)]).max())
                gain = .25 if resistance < 1.8 and tracking < 18 else .40
                wanted = gain*np.sum(h[useful, None]*err[useful], axis=0)/max(float(np.sum(h[useful]**2)), .1)
                wanted = np.clip(wanted, [-.010, -.012, -.018], [.010, .012, .012])
                self.coeff += np.clip(wanted-self.coeff, -.0002, .0002)
        else:
            self.coeff += np.clip(-self.coeff, -.0002, .0002)
        if self.feedback and self.acquired and prefix >= 5:
            wanted = float(np.clip(.25*(.087-np.median(rt[:prefix])), -.0025, .0025))
            self.target_delta += np.clip(wanted-self.target_delta, -.0001, .0001)
        # Both feed and roll slow together. No independent rolling on a stopped
        # shared shift, and no early headless receiving-radius contraction.
        scale = float(np.clip((32-tracking)/24, .10, 1.))
        scale *= float(np.clip((.14-head_error)/.10, .08, 1.))
        if tracking > 45:
            scale = 0.
        inc = min(self.speed*DT*scale, self.route.final_shift-self.shift)
        self.shift += inc
        roll_factor = geometry.ALPHA/geometry.BETA**2
        phase_inc = inc*geometry.BETA/.029*roll_factor
        self.phase += phase_inc
        nominal, _ = self.route.window(self.shift)
        execution, es = self.curve().window(self.shift)
        self.stations = es.copy()
        carrying = np.flatnonzero(source & ((es[:-1]+es[1:])/2 < -.06))
        anchor = int(carrying[-3]) if len(carrying) >= 3 else min(3, max(prefix-1, 0)) if self.acquired else 18
        weights = np.ones(22)
        active = int(np.count_nonzero((es[:-1]+es[1:])/2 > -.03))
        weights[:min(22, max(3, active+2))] = 2.
        desired, report = self.mapper.solve(execution, data, self.last, anchor, phase_inc, weights)
        desired_kp = np.full(20, 127.)
        if tracking > 18 and abs(load).max() > 1.8:
            joint_bridge = (es[1:-1] > 0) & (es[1:-1] < self.route.bridge_length)
            desired_kp[joint_bridge] = 95.25
        self.kp += np.clip(desired_kp-self.kp, -127*DT/1.5, 127*DT/1.5)
        self.m.actuator_gainprm[:, 0] = self.kp
        self.m.actuator_biasprm[:, 1] = -self.kp
        step = float(np.degrees(abs(desired-self.last)).max())
        self.last = desired.copy()
        return desired, dict(stage='continuous_target_intake' if self.acquired else 'approved_inner_upward_entry',
            shared_shift_mm=self.shift*1000, roll_phase_rad=self.phase, tracking_deg=tracking,
            head_global_error_mm=float(np.linalg.norm(p[0]-nominal[0]))*1000,
            head_execution_error_mm=head_error*1000, estimated_prefix=prefix, estimated_wrap_rad=wrap,
            estimated_capture=self.acquired, advancement_gate=bool(scale == 0), speed_scale=scale,
            global_points_mm=nominal*1000, execution_points_mm=execution*1000, estimated_points_mm=p*1000,
            material_stations_mm=es*1000, bridge_coeff_mm=np.array([self.coeff, self.outlet])*1000,
            radius_delta_mm=np.array([self.source_delta, self.target_delta])*1000,
            kp_nm_per_rad=self.kp.copy(), maximum_dispatch_step_deg=step,
            maximum_reference_rod_length_error_mm=float(np.max(abs(np.linalg.norm(np.diff(execution, axis=0), axis=1)-LINK)))*1000,
            local_feedback_enabled=self.feedback, physical_root_assigned=False, independent_roll_when_shift_stopped=False,
            source_radius_feedback=False, **report)
