"""Actual-position moving compliance with independent local curve winding.

Reuses U101's low-following position-servo effort band, while retaining N05
local winding and measured-motion common following. No XML, physical root,
actuator gain or physical force limits are changed. Recovery requires actual
ordered target bearing, not nominal passage through a curve station.
"""
import numpy as np
from snake_control.adaptive_front_winding_transfer import AdaptiveFrontWindingTransfer
from snake_control.unified_curve_transfer import DT,LINK,physical_wrap_masks,smooth
from snake_control.global_transfer_route import FrozenRouteSpec,sample_route,route_body


class MovingComplianceTransfer(AdaptiveFrontWindingTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        self.entry_capture_enabled=False
        self.curve.target_entry_offsets[:]=0.
        # N09's rod M10 crossed the source surface near [0.9565,0,1.0565]
        # despite exhausting its 50 mm avoidance field. Plan the full inlet
        # corridor around that obstacle, rather than increasing that cap.
        spec=dict(self.curve.planned_route_spec)
        spec.update(outer_receiving_waypoints=[[.970,-.045,1.220],[1.025,0.,1.220]],
            bridge_outward_offset=[0.,0.,0.],entry_residual=np.zeros((22,3)).tolist())
        grid=np.asarray(spec['spline_x']);xyz=sample_route(spec,grid)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        spec['initial_head_arc']=float(np.interp(spec['initial_head_parameter'],grid,arc))
        initial,_=self.observe(data);raw,parameters=route_body(spec,0.)
        spec['entry_residual']=(initial-raw).tolist()
        spec['entry_target_weights']=smooth((parameters-spec['source_bound'])/(spec['target_bound']-spec['source_bound'])).tolist()
        self.curve.planned_route_spec=FrozenRouteSpec(spec)
        target_arc=float(np.interp(spec['target_bound'],grid,arc))
        self.receiver_initial_required_window_shift=max(0.,target_arc-(spec['initial_head_arc']-9*LINK))
        self.ordered_handover_required=True
        self.soft_servo_weight=np.zeros(20)
        self.ever_soft=np.zeros(20,dtype=bool)
        self.body_curve_coordinates=None
        self.actual_target_bearing_age=np.zeros(20)
        self.compliance_feedback={}
        # The local departure cap ends a bounded probe, not the complete
        # opportunity for the next rod to pass the receiving route boundary.
        self.allow_common_follow_after_local_bound=True

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
        # Only the next ordered rod can leave pass-through compliance before
        # contact. Actual proximity permits active reception; full rigidity
        # still requires the real ordered bearing test above.
        receiver=confirmed+1
        reception=np.zeros(20)
        receiving_proximity=0.
        receiver_radius=None
        receiver_axis=None
        if receiver<21:
            receiver_radius=float(np.linalg.norm(centers[receiver,[0,2]]-[1.1,1.]))
            receiver_axis=float(centers[receiver,1])
            # One rod of radial reach, ramping past the end of the elbow.
            # The captured bearing radius comes from the initial real hook.
            radial=smooth((self.target_entry_radius+LINK-receiver_radius)/LINK)
            axial=smooth((receiver_axis-.05)/LINK)
            receiving_proximity=float(radial*axial)
            for joint in [receiver-1,receiver]:
                if 0<=joint<20:reception[joint]=receiving_proximity
            precontact_softness=1.-.75*reception
            desired=np.minimum(desired,precontact_softness)
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
            ordered_active_receiver_module=int(receiver) if receiver<21 else None,
            actual_receiver_radius_mm=receiver_radius*1000 if receiver_radius is not None else None,
            actual_receiver_axis_mm=receiver_axis*1000 if receiver_axis is not None else None,
            actual_receiver_proximity=receiving_proximity,
            active_reception_joint_weights=reception.tolist(),
            precontact_reception_minimum_softness=.25,
            receiver_initial_required_window_shift_mm=self.receiver_initial_required_window_shift*1000,
            next_receiver_whole_rod_in_target_reference=bool(receiver<21 and np.min(node_u[receiver:receiver+2])>=spec['target_bound']),
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
        info['stage']='moving_compliance_with_ordered_active_reception'
        info['moving_compliance']=self.compliance_feedback.copy()
        info['moving_compliance']['actual_measured_actuator_torque_Nm']=np.asarray(torque).tolist()
        info['adaptive_front_winding']['action']='front winding and load feed through continuous outer receiving corridor; actual next-rod proximity gradually activates reception; actual ordered bearing completes grip recovery'
        return target,info

    def load_state(self,path):
        elapsed=super().load_state(path)
        for key in ['soft_servo_weight','ever_soft','actual_target_bearing_age','body_curve_coordinates']:
            if getattr(self,key,None) is not None:setattr(self,key,np.asarray(getattr(self,key)))
        self.ever_soft=self.ever_soft.astype(bool)
        return elapsed
