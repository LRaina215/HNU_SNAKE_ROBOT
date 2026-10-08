"""Direct front receiving and rear rolling through a smooth outside bridge."""
import math
import numpy as np
from scipy.interpolate import CubicSpline

from snake_control.coordinated_curve_transfer import CoordinatedCurveTransfer
from snake_control.unified_curve_transfer import UnifiedCurveTransfer, DT, smooth, physical_wrap_masks
from snake_control.global_transfer_route import FrozenRouteSpec, route_body


class DirectCurveTransfer(CoordinatedCurveTransfer):
    def __init__(self, model, data):
        super().__init__(model,data)
        curve = self.curve
        spec = dict(curve.planned_route_spec)
        base = curve.base
        us,ut = curve.source_bound,curve.target_bound
        source,target = base(us),base(ut)
        ts,tt = base(us,1),base(ut,1)
        ts /= np.linalg.norm(ts)
        tt /= np.linalg.norm(tt)
        controls = np.vstack([source,source+.14*ts,target-.14*tt,target])
        bu = np.linspace(us,ut,65)
        f = ((bu-us)/(ut-us))[:,None]
        bridge = ((1-f)**3*controls[0]+3*(1-f)**2*f*controls[1]+
                  3*(1-f)*f*f*controls[2]+f**3*controls[3])
        source_u,target_u = base.x[base.x<us],base.x[base.x>ut]
        u = np.r_[source_u,bu,target_u]
        xyz = np.vstack([base(source_u),bridge,base(target_u)])
        plan = CubicSpline(u,xyz,axis=0)
        spec.update(version=3,origin='direct receiving from existing hook; smooth outer bridge; simultaneous pipe rolling',
                    spline_x=plan.x.tolist(),spline_c=plan.c.tolist(),bridge_controls=controls.tolist())
        # The captured receiving hook stays intact at entry; its forward
        # extension already has the open pitch. No preliminary global stretch.
        low,high = 1.1,1.6
        for _ in range(24):
            mid = .5*(low+high)
            spec['initial_head_parameter'] = mid
            _,locations = route_body(spec,0.)
            if locations[-1]<0.:low=mid
            else:high=mid
        spec['initial_head_parameter'] = .5*(low+high)
        curve.planned_route_spec = FrozenRouteSpec(spec)
        curve.reservoir_height = 0.
        self.coordination_stage = 'direct_receive_and_feed'
        self.incoming_age = np.zeros(21)
        self.direct_feedback = {}

    def command(self,data,t,forces,torque,support_forces=None):
        self.coordination_time = t
        support = np.asarray(forces if support_forces is None else support_forces)
        self.current_support_forces = support.copy()
        if self.nominal_support_goals is None:
            self.nominal_support_goals = np.maximum(support.sum(1),1.)
        self.support_force_goals = self.nominal_support_goals.copy()
        _,centers = self.observe(data)
        target_mask = physical_wrap_masks(centers)[1]
        self.incoming_age = np.where((support[1]>.1)&target_mask,self.incoming_age+DT,0.)
        for module in np.flatnonzero(self.incoming_age>=1.):
            if module>=8 and int(module) not in self.transferred_material_modules:
                self.transferred_material_modules.append(int(module))
                self.events.append(dict(t_s=t,mode='actual_direct_target_handover',module=int(module),
                                        meaning='actual cylinder contact and target geometry maintained for 1 s'))
        target,info = UnifiedCurveTransfer.command(self,data,t,forces,torque,support_forces)
        info['coordination'] = dict(phase='direct_receive_and_feed',reservoir_height_mm=0.,
                                   pressure_scales=[1.,1.],support_force_goals_N=self.support_force_goals.tolist(),
                                   feed_policy=self.direct_feedback)
        info['stage'] = 'direct_receive_and_feed_'+info['stage']
        return target,info

    def coordinated_feed_limit(self,lag,error_scale,support_scale,physical_speeds):
        speeds = np.asarray(physical_speeds)
        _,centers = self.observe(self.d)
        sides = speeds[1:] if self.target_carries_transfer(centers,self.current_support_forces) else speeds
        slower = float(np.min(sides))
        lead_scale = float(np.clip((.004-lag)/.004,0.,1.))
        bootstrap = .0004*lead_scale
        # A finite 4 mm lead starts a joint-driven rolling probe; it is never
        # recorded as actual movement or replenished by commanded bridge shape.
        measured_limit = 0. if slower<-.0005 else max(0.,slower)+bootstrap
        requested = min(.0015*error_scale*support_scale*smooth(self.coordination_time/2.),measured_limit)
        self.direct_feedback = dict(lead_mm=lag*1000,finite_probe_room_mm=max(0.,.004-lag)*1000,
                                    slower_measured_material_speed_mm_s=slower*1000,
                                    requested_mm_s=float(requested*1000),
                                    observed_side_speeds_mm_s=(speeds*1000).tolist())
        return float(requested)

    def load_state(self,path):
        elapsed = super().load_state(path)
        self.incoming_age = np.asarray(self.incoming_age)
        return elapsed
