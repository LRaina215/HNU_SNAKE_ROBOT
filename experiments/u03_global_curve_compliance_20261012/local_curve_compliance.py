"""One receiving normal coordinate; A16 remains the nominal global route.

Simulation center radius and classified normal contact force are privileged
observations. No actual pose is used to replace/register the global route.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from receiver_phase_reference import ReceiverPhaseReference
from physics import nodes, mujoco


def cached_observation(m, d):
    """Read latest step cache, with <=1ms latency; never change warmstart."""
    points=nodes(d)
    centers=.5*(points[:-1]+points[1:])
    normal=np.zeros(21)
    head=m.body('head').id
    for k,contact in enumerate(d.contact):
        geoms=[contact.geom1,contact.geom2]
        for j,g in enumerate(geoms):
            name=m.geom(g).name or ''
            robot=int(m.geom_bodyid[geoms[1-j]])-head
            if name.startswith('pipe_') and name not in ['pipe_straight_01','pipe_sharp_in_02'] and 0<=robot<21:
                force=np.zeros(6)
                mujoco.mj_contactForce(m,d,k,force)
                normal[robot]+=max(0.,force[0])
    return centers,normal


class LocalCurveCompliance(ReceiverPhaseReference):
    def __init__(self, points, angles):
        self.compliance_ready = False
        self.delta = 0.0
        self.filtered_error = 0.0
        self.filtered_load = 0.0
        self.observation = dict(compliance_delta_m=0., compliance_rate_m_s=0.,
            compliance_gap_m=0., compliance_mean_load_N=0.,
            compliance_active_material=np.zeros(21, dtype=bool),
            compliance_amplitude_saturated=False, compliance_rate_saturated=False)
        super().__init__(points, angles)
        self.compliance_ready = True
        self.spec.update(version='B1', high_level_feedback=True,
            local_compliance='one bounded radial displacement of receiving curve; original true-arc Bishop and alternating joint integrals',
            nominal_global_route='frozen A16; unchanged S/F/M/common roll gauge',
            compliance_parameters=self.parameters())

    @staticmethod
    def parameters():
        return dict(radius_goal_m=.0925, gap_deadband_m=.004,
            filter_time_s=.4, gap_gain_per_s=.06, load_relief_m_N_s=.00005,
            mean_load_relief_threshold_N=12., return_gain_per_s=.1,
            amplitude_bound_m=.002, rate_bound_m_s=.00025,
            curve_support_relative_target_arc_m=[-.125, .60],
            curve_taper_m=[.125, .20],
            observation_window_relative_target_arc_m=[-.075, .40],
            observation_actual_Y_m=[.15, 1.2])

    def observe(self, centers, target_normal, previous_info, dt):
        """Read once per25ms command; radius closes, excessive load opens."""
        loc=previous_info.get('global_reference_locations', previous_info['reference_locations'])
        station=.5*(loc[:-1]+loc[1:])-previous_info.get('global_target_arc_m', self.target_arc)
        mask=(station>=-.075)&(station<=.40)&(centers[:,1]>.15)&(centers[:,1]<1.2)
        rho=np.linalg.norm(centers[:,[0,2]]-[1.1,1.],axis=1)
        if mask.any():
            err=float(np.mean(rho[mask])-.0925)
            load=float(np.mean(target_normal[mask]))
        else:
            err=0.; load=0.
        alpha=1-np.exp(-dt/.4)
        self.filtered_error+=alpha*(err-self.filtered_error)
        self.filtered_load+=alpha*(load-self.filtered_load)
        gap=np.sign(self.filtered_error)*max(0.,abs(self.filtered_error)-.004)
        desired=-.06*gap+.00005*max(0.,self.filtered_load-12.)-.1*self.delta
        if not mask.any(): desired=-.1*self.delta
        rate=float(np.clip(desired,-.00025,.00025))
        before=self.delta
        self.delta=float(np.clip(before+dt*rate,-.002,.002))
        self.observation=dict(compliance_delta_m=self.delta,
            compliance_rate_m_s=(self.delta-before)/dt,
            compliance_gap_m=self.filtered_error,
            compliance_mean_load_N=self.filtered_load,
            compliance_active_material=mask,
            compliance_amplitude_saturated=abs(self.delta)>=.002-1e-12,
            compliance_rate_saturated=abs(desired)>.00025)

    def set_nominal_family(self,S):
        phase=super().set_nominal_family(S)
        if not self.compliance_ready:return phase
        self.nominal_spline=self.spline
        self.nominal_target_arc=self.target_arc
        self.nominal_body_points,self.nominal_body_locations=self.body(S,True)
        if abs(self.delta)<1e-15:return phase
        old=self.spline.x
        xyz=self.spline(old)
        station=old-self.target_arc
        weight=self.smooth((station+.125)/.125)*self.smooth((.60-station)/.20)
        radial=xyz[:,[0,2]]-[1.1,1.]
        radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
        xyz[:,[0,2]]+=self.delta*weight[:,None]*radial
        # Reparametrize on deformed geometric arc; source supports are unchanged.
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.start_arc=float(np.interp(self.start_arc,old,arc))
        self.prefix_end_arc=float(np.interp(self.prefix_end_arc,old,arc))
        self.target_arc=float(np.interp(self.target_arc,old,arc))
        self.connector_arc=self.target_arc-self.start_arc
        self.spline=CubicSpline(arc,xyz,axis=0)
        return phase

    def command(self,S):
        q,info=super().command(S)
        if self.compliance_ready:
            info.update(global_reference_points=self.nominal_body_points.copy(),
                global_reference_locations=self.nominal_body_locations.copy(),
                global_target_arc_m=self.nominal_target_arc,**self.observation)
        return q,info
