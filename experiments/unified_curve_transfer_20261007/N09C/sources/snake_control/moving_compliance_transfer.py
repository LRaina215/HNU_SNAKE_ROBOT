"""Actual-position moving compliance with independent local curve winding.

Reuses U101's low-following position-servo effort band, while retaining N05
local winding and measured-motion common following. No XML, physical root,
actuator gain or physical force limits are changed. Recovery requires actual
ordered target bearing, not nominal passage through a curve station.
"""
import numpy as np
from snake_control.adaptive_front_winding_transfer import AdaptiveFrontWindingTransfer
from snake_control.unified_curve_transfer import DT,LINK,physical_wrap_masks,smooth


class MovingComplianceTransfer(AdaptiveFrontWindingTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        self.entry_capture_enabled=False
        self.curve.target_entry_offsets[:]=0.
        self.ordered_handover_required=True
        self.soft_servo_weight=np.zeros(20)
        self.ever_soft=np.zeros(20,dtype=bool)
        self.body_curve_coordinates=None
        self.actual_target_bearing_age=np.zeros(20)
        self.compliance_feedback={}

    def speed_tracking_error(self,error,physical_points,guide,support,t):
        node_u=self.curve.reference_node_parameters
        nominal=node_u[1:-1]
        segment=guide[1:]-guide[:-1]
        delta=physical_points[1:-1,None,:]-guide[None,:-1,:]
        fraction=np.clip(np.einsum('ijk,jk->ij',delta,segment)/np.maximum(np.sum(segment**2,axis=1)[None,:],1e-12),0.,1.)
        projected=guide[None,:-1,:]+fraction[:,:,None]*segment[None,:,:]
        coordinates=node_u[:-1][None,:]+fraction*(node_u[1:]-node_u[:-1])[None,:]
        distance=np.linalg.norm(projected-physical_points[1:-1,None,:],axis=2)
        # Preserve material identity while measuring actual position: nearby
        # folds must not map a tail hinge to a distant front turn.
        distance[np.abs(coordinates-nominal[:,None])>2*LINK]=np.inf
        choice=np.argmin(distance,axis=1)
        measured=coordinates[np.arange(20),choice]
        measured=np.where(np.isfinite(distance[np.arange(20),choice]),measured,nominal)
        if self.body_curve_coordinates is None:self.body_curve_coordinates=measured.copy()
        else:self.body_curve_coordinates+=np.clip(.2*(measured-self.body_curve_coordinates),-.003,.003)
        spec=self.curve.planned_route_spec
        start=spec['source_bound']-3.5*LINK
        recovery_start=spec['target_bound']-LINK
        recovery_end=spec['target_bound']+.5*LINK
        spatial=smooth((self.body_curve_coordinates-start)/LINK)*(1-smooth((self.body_curve_coordinates-recovery_start)/(recovery_end-recovery_start)))
        self.ever_soft|=spatial>.05
        centers=.5*(physical_points[:-1]+physical_points[1:])
        target_region=physical_wrap_masks(centers)[1]
        confirmed=max(self.transferred_material_modules,default=7)
        bearing=target_region[:-1]&target_region[1:]&((support[1,:-1]+support[1,1:])>.1)&(np.arange(1,21)<=confirmed)
        self.actual_target_bearing_age=np.where(bearing,self.actual_target_bearing_age+DT,0.)
        recover=self.actual_target_bearing_age>=.5
        desired=spatial.copy()
        # Once a section enters compliance, nominal window passage cannot
        # restore it. It stays compliant until actual ordered target bearing.
        pending=self.ever_soft&~recover
        desired[pending]=np.maximum(desired[pending],self.soft_servo_weight[pending])
        desired[recover]=0.
        tau=np.where(desired>self.soft_servo_weight,.7,2.)
        self.soft_servo_weight+=np.minimum(1.,DT/tau)*(desired-self.soft_servo_weight)
        permitted=self.soft_servo_weight>.05
        effective=np.where(permitted,0.,error)
        self.compliance_feedback=dict(actual_projected_joint_curve_coordinates_mm=(measured*1000).tolist(),
            filtered_actual_joint_curve_coordinates_mm=(self.body_curve_coordinates*1000).tolist(),
            nominal_joint_curve_coordinates_mm=(nominal*1000).tolist(),
            actual_projection_distance_mm=(distance[np.arange(20),choice]*1000).tolist(),
            softness=self.soft_servo_weight.tolist(),spatial_requested_softness=spatial.tolist(),
            actual_ordered_target_bearing_age_s=self.actual_target_bearing_age.tolist(),
            actual_bearing_recovery_allowed=recover.tolist(),
            permitted_soft_posture_error_deg=np.where(permitted,np.degrees(error),0.).tolist(),
            rigid_speed_guard_error_deg=np.degrees(effective).tolist(),
            material_joint_members=(np.flatnonzero(self.soft_servo_weight>.5)+1).tolist(),
            lower_curve_station_mm=start*1000,target_recovery_curve_station_mm=recovery_end*1000,
            ordered_incoming_target_modules=self.transferred_material_modules.copy(),
            basis='actual hinges projected onto current curve within material neighbourhood; actual ordered target support permits gradual recovery; compliant posture error excluded from motion guards')
        return effective

    def servo_reference(self,geometric,q,bias):
        weight=self.soft_servo_weight
        following=1.-.97*weight
        gain=self.m.actuator_gainprm[:,0]
        effort=(2.355*(1-weight)+.35*weight)
        band=np.minimum(np.radians(3),np.where(weight>.01,effort/gain,np.radians(3)))
        raw=q+following*(geometric-q+bias)
        raw=np.clip(raw,q-band,q+band)
        self.compliance_feedback.update(position_following_scale=following.tolist(),
            software_effort_band_Nm=effort.tolist(),nominal_effective_position_gain_Nm_rad=(gain*following).tolist(),
            caveat='software ctrl-q effort band is not an instantaneous measured-torque cap; raw actual actuator forces are retained in trace')
        return raw

    def command(self,data,t,forces,torque,support_forces=None):
        target,info=super().command(data,t,forces,torque,support_forces)
        info['stage']='moving_compliance_front_winding_and_feed'
        info['moving_compliance']=self.compliance_feedback.copy()
        info['moving_compliance']['actual_measured_actuator_torque_Nm']=np.asarray(torque).tolist()
        info['adaptive_front_winding']['action']='independent bounded front winding with actual-position moving compliant exit/bridge; load feed; actual target bearing restores following'
        return target,info

    def load_state(self,path):
        elapsed=super().load_state(path)
        for key in ['soft_servo_weight','ever_soft','actual_target_bearing_age','body_curve_coordinates']:
            if getattr(self,key,None) is not None:setattr(self,key,np.asarray(getattr(self,key)))
        self.ever_soft=self.ever_soft.astype(bool)
        return elapsed
