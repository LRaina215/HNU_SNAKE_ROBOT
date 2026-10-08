"""Bounded local winding/feed parameters on one material-window curve.

The front action continues when the common window cannot advance. Source
feed follows measured load. Existing supported front motion permits the
common shift to absorb both local departures; new contacts confirm transfer.
Every position target comes from the
same full curve integral, with one initial material frame and no joint wave
added outside it.
"""
import copy
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from snake_control.explicit_winding_transfer import ExplicitWindingTransfer
from snake_control.unified_curve_transfer import DT,LINK,BODY_RADIUS,physical_wrap_masks,smooth


class AdaptiveFrontWindingTransfer(ExplicitWindingTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        self.front_local_material=0.
        self.rear_local_material=0.
        self.local_material_limit=LINK
        self.local_front_speed=.002
        self.actual_follow_lead_limit=.25*LINK
        self.front_bound_since=None
        self.last_verified_handover_time=0.
        self.local_action={}
        self.curve.local_pipe_motion=self.pipe_curve_motion(0.,0.)
        # Capture measured target-cylinder support, rather than fixed joints.
        self.initial_target_cohort=None
        self.target_entry_radius=None
        self.curve.target_entry_offsets=np.zeros_like(self.curve.shape_offsets)
        self.entry_capture_feedback={}
        self.entry_capture_enabled=True

    def pipe_curve_motion(self,front,rear):
        material=np.array([rear,front])
        axial=np.asarray(self.curve.axial_sines)
        beta=np.sqrt(1-axial**2)
        phase=-np.sign(self.curve.theta_rates)*axial**2*material/(np.asarray(self.curve.radii)*beta)
        return dict(phase_rad=phase.tolist(),axis_extension_m=(axial*material).tolist(),
                    local_material_extension_m=material.tolist())

    def set_local_curve(self,front,rear):
        self.curve.local_pipe_motion=self.pipe_curve_motion(front,rear)

    def draw_target_entry(self,front,dt):
        """Draw a continuous entry field toward captured bearing radius.

        The field is deposited at material curve stations, not joint numbers.
        It acts near the target bridge end. Each inward velocity is stopped
        below the measured bearing radius; overlapping fields and other curve
        deformation can still change the final sampled radius.
        """
        enabled=self.curve.clearance_enabled;self.curve.clearance_enabled=False
        try:_,centers,locations=self.curve.reference(self.shift)
        finally:self.curve.clearance_enabled=enabled
        radial=centers-np.array([1.1,0.,1.]);radial[:,1]=0.
        radius=np.linalg.norm(radial,axis=1);normal=radial/np.maximum(radius[:,None],1e-12)
        entry=self.curve.planned_route_spec['target_bound']
        weights=np.exp(-.5*((locations-entry)/(.9*LINK))**2)
        gain=smooth((front+self.shift)/LINK)
        excess=np.maximum(0.,radius-self.target_entry_radius-.0005)
        velocity=-normal*(np.minimum(.001,excess/max(dt,1e-12))*weights*gain*(centers[:,1]>.060))[:,None]
        spread=np.exp(-.5*((self.curve.shape_grid[:,None]-locations[None,:])/(.5*LINK))**2)
        update=spread@velocity/np.maximum(spread.sum(1)[:,None],1.)*dt
        self.curve.target_entry_offsets+=update
        lengths=np.linalg.norm(self.curve.target_entry_offsets,axis=1)
        self.curve.target_entry_offsets*=np.minimum(1.,.025/np.maximum(lengths,1e-12))[:,None]
        return dict(captured_bearing_radius_mm=self.target_entry_radius*1000,
            maximum_entry_curve_velocity_mm_s=float(np.linalg.norm(velocity,axis=1).max()*1000),
            maximum_entry_field_mm=float(np.linalg.norm(self.curve.target_entry_offsets,axis=1).max()*1000),
            entry_weight_at_curve_stations=weights.tolist(),nominal_curve_radius_mm=(radius*1000).tolist(),
            basis='continuous station field at target bridge end; bounded approach toward measured initial bearing radius; no per-joint angle additions')

    def update_source_release(self,t,p,centers,q,support,error,locations,dt):
        return None

    def command(self,data,t,forces,torque,support_forces=None):
        self.current_control_time=self.coordination_time=t
        p,centers=self.observe(data);q=data.qpos[self.qa]
        support=np.asarray(forces if support_forces is None else support_forces)
        self.current_support_forces=support.copy()
        if self.nominal_support_goals is None:self.nominal_support_goals=np.maximum(support.sum(1),1.)
        self.support_force_goals=self.nominal_support_goals.copy()
        if self.initial_target_cohort is None:
            self.initial_target_cohort=np.flatnonzero((support[1]>.1)&physical_wrap_masks(centers)[1])
            self.target_entry_radius=float(np.median(np.linalg.norm(centers[self.initial_target_cohort][:,[0,2]]-[1.1,1.],axis=1)))
        if self.filtered_pressure_forces is not None:self.filtered_pressure_forces+=.1*(support-self.filtered_pressure_forces)
        self.filtered_load+=.1*(np.abs(torque)-self.filtered_load)
        if self.first_dispatch:self.torque_feedforward=np.asarray(torque).copy()
        else:self.torque_feedforward+=.1*(np.asarray(torque)-self.torque_feedforward)
        mask=physical_wrap_masks(centers)[1]
        self.incoming_age=np.where((support[1]>.1)&mask,self.incoming_age+DT,0.)
        for module in np.flatnonzero(self.incoming_age>=1.):
            if module>=8 and int(module) not in self.transferred_material_modules:
                self.transferred_material_modules.append(int(module));self.last_verified_handover_time=t
                self.events.append(dict(t_s=t,mode='actual_local_curve_winding_handover',module=int(module)))
        observed=self.observe_progress(centers,forces,t);self.last_body_progress=observed
        tracking=np.abs(self.last-q);geometric_error=np.abs(self.joint_geometry-q)
        before,reference,locations=self.curve.reference(self.shift)
        spec=self.curve.planned_route_spec
        joint_blend=smooth((self.curve.reference_node_parameters[1:-1]-spec['source_bound'])/(spec['target_bound']-spec['source_bound']))
        front_joints=joint_blend>.8;rear_joints=joint_blend<.2
        front_error=float(geometric_error[front_joints].max()) if front_joints.any() else float(geometric_error.max())
        rear_error=float(geometric_error[rear_joints].max()) if rear_joints.any() else float(geometric_error.max())
        front_follow=float(np.clip((np.radians(20)-front_error)/np.radians(16),0.,1.))
        rear_follow=float(np.clip((np.radians(16)-rear_error)/np.radians(12),0.,1.))
        target_load=float(support[1].sum());source_load=float(support[0].sum())
        load_ratio=target_load/max(self.nominal_support_goals[1],1.)
        feed_need=float(np.clip(load_ratio,.25,1.))
        loaded_rear=(self.filtered_load>2.1)&(geometric_error>np.radians(6.))&rear_joints
        rear_load_scale=float(np.clip(1.-.7*np.count_nonzero(loaded_rear)/max(np.count_nonzero(rear_joints),1),.3,1.))
        front_requested=self.local_front_speed*smooth(t/3.)*front_follow
        df=min(front_requested*DT,max(0.,self.local_material_limit-self.front_local_material))
        rear_goal=min(self.local_material_limit,self.front_local_material+df)
        rear_requested=min(.002*feed_need*rear_follow*rear_load_scale,max(0.,rear_goal-self.rear_local_material)/DT)
        dr=rear_requested*DT
        saved=copy.deepcopy(self.curve.__dict__);accepted_shape=0.;shape_rejections=[]
        for fraction in (1.,.5,.25,.125,.0625,.03125):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(saved))
            try:
                self.update_source_release(t,p,centers,q,support,geometric_error,locations,DT*fraction)
                self.feedback_shape(p,q,reference,locations,geometric_error,forces,DT*fraction)
                shaped,_,_=self.curve.reference(self.shift)
                shape_step=float(np.linalg.norm(shaped-before,axis=1).max())
                if shape_step>.008*DT+1e-8:
                    shape_rejections.append(dict(fraction=fraction,reason='continuous curve shape node speed bound',maximum_node_step_mm=shape_step*1000));continue
                accepted_shape=fraction;break
            except ValueError as exc:shape_rejections.append(dict(fraction=fraction,reason=str(exc)))
        else:self.curve.__dict__.clear();self.curve.__dict__.update(saved)
        shaped_state=copy.deepcopy(self.curve.__dict__)
        local_rejections=[];front_accepted=rear_accepted=0.
        accepted_entry={}
        for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,.00048828125):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(shaped_state))
            self.set_local_curve(self.front_local_material+df*fraction,self.rear_local_material+dr*fraction)
            try:
                entry_feedback=self.draw_target_entry(self.front_local_material+df*fraction,DT*fraction) if self.entry_capture_enabled else {}
                guide,guide_centers,locations,_,_=self.route_outside_bridge(self.shift)
                node_step=float(np.linalg.norm(guide-before,axis=1).max())
                if node_step>.008*DT+1e-8:
                    local_rejections.append(dict(fraction=fraction,reason='continuous curve node speed bound',maximum_node_step_mm=node_step*1000));continue
                front_accepted=df*fraction;rear_accepted=dr*fraction;accepted_entry=entry_feedback;break
            except ValueError as exc:local_rejections.append(dict(fraction=fraction,reason=str(exc)))
        else:
            self.curve.__dict__.clear();self.curve.__dict__.update(shaped_state)
            guide,guide_centers,locations=self.curve.reference(self.shift)
        self.entry_capture_feedback=accepted_entry
        front=self.front_local_material+front_accepted;rear=self.rear_local_material+rear_accepted
        local_state=copy.deepcopy(self.curve.__dict__)
        cohort=self.initial_target_cohort
        live_cohort=cohort[(support[1,cohort]>.1)&mask[cohort]]
        actual_target_axis=max(0.,float(np.median(centers[live_cohort,1]-self.initial_centers[live_cohort,1]))) if len(live_cohort) else 0.
        actual_target_material=actual_target_axis/self.curve.axial_sines[1]
        verified_length=max(0.,max(self.transferred_material_modules,default=7)-7)*LINK
        # Existing inherited hook motion must be able to start transfer.
        # New-body contacts confirm handover; they are not a startup gate.
        actual_credit=actual_target_material+self.actual_follow_lead_limit if actual_target_axis>.002 and len(live_cohort) else 0.
        common_follow=float(np.clip((np.radians(20)-geometric_error.max())/np.radians(16),0.,1.))
        requested_shift=min(.0015*common_follow,max(0.,actual_credit-self.shift)/.5) if target_load>.1 else 0.
        ds=requested_shift*DT;shift_rejections=[]
        for _ in range(12):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(local_state))
            self.curve.increment(ds)
            self.set_local_curve(max(0.,front-ds),max(0.,rear-ds))
            try:
                candidate,candidate_centers,candidate_locations,_,_=self.route_outside_bridge(self.shift+ds)
                node_step=float(np.linalg.norm(candidate-before,axis=1).max())
                if node_step>.008*DT+1e-8:
                    shift_rejections.append(dict(proposed_shift_step_mm=ds*1000,reason='continuous curve node speed bound',maximum_node_step_mm=node_step*1000))
                else:
                    guide,guide_centers,locations=candidate,candidate_centers,candidate_locations;break
            except ValueError as exc:shift_rejections.append(dict(proposed_shift_step_mm=ds*1000,reason=str(exc)))
            ds*=.5
        else:
            ds=0.;self.curve.__dict__.clear();self.curve.__dict__.update(local_state)
            guide,guide_centers,locations=self.curve.reference(self.shift)
        new_front=max(0.,front-ds);new_rear=max(0.,rear-ds)
        material_change=np.array([ds+new_rear-self.rear_local_material,ds+new_front-self.front_local_material])
        alpha=np.asarray(self.curve.axial_sines);beta=np.sqrt(1-alpha**2)
        material_roll=np.array([-np.sign(self.curve.theta_rates[0]),np.sign(self.curve.theta_rates[1])])*alpha*material_change/(BODY_RADIUS*beta)
        axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        cross=np.cross(self.root_orientation[:,2],axis);sine=np.linalg.norm(cross)
        transport=Rotation.from_rotvec(cross*math.atan2(sine,float(self.root_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()
        orientation=Rotation.from_rotvec(axis*material_roll[1]).as_matrix()@transport@self.root_orientation
        phases=self.pipe_roll_phase+material_roll
        prior,field_locations,field_roll=self.integral_targets(guide,orientation,phases)
        geometric=np.clip(prior,-np.radians(88),np.radians(88))
        self.shift+=ds;self.speed=ds/DT;self.front_local_material=new_front;self.rear_local_material=new_rear
        self.root_material_phase+=material_roll[1];self.root_orientation=orientation;self.pipe_roll_phase=phases
        self.joint_geometry=geometric;self.integral_joint_prior=prior.copy()
        bias=np.clip(self.torque_feedforward,-2.355,2.355)/self.m.actuator_gainprm[:,0]
        raw=np.clip(geometric+bias,q-np.radians(3),q+np.radians(3))
        target=np.clip(raw,np.maximum(-np.radians(88),self.last-np.radians(10)*DT),np.minimum(np.radians(88),self.last+np.radians(10)*DT))
        if self.first_dispatch:target=self.last.copy();self.first_dispatch=False
        self.last=target.copy()
        self.fit_data.qpos[:]=data.qpos;self.fit_data.qpos[self.qa]=geometric
        mujoco.mj_fwdPosition(self.m,self.fit_data);d=self.fit_data
        body=np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
        self.fk_reference_points=body.copy();self.fk_root_pose=d.qpos[:7].copy()
        at_bound=new_front>=self.local_material_limit-1e-8
        if at_bound and self.front_bound_since is None:self.front_bound_since=t
        if not at_bound:self.front_bound_since=None
        failure=None
        if at_bound and not self.transferred_material_modules:failure='local_front_winding_bound_without_verified_handover'
        elif at_bound and t-max(self.front_bound_since,self.last_verified_handover_time)>=45.:failure='local_front_winding_bound_without_further_handover_45s'
        self.local_action=dict(action='bounded front curve winding; measured-load rear curve feed; existing supported front motion drives common shift and absorbs local departures',
            front_local_material_mm=new_front*1000,rear_local_material_mm=new_rear*1000,local_material_limit_mm=self.local_material_limit*1000,
            front_requested_speed_mm_s=front_requested*1000,front_accepted_speed_mm_s=front_accepted/DT*1000,
            rear_requested_speed_mm_s=rear_requested*1000,rear_accepted_speed_mm_s=rear_accepted/DT*1000,
            target_load_N=target_load,source_load_N=source_load,rear_feed_load_ratio=feed_need,rear_load_scale=rear_load_scale,
            front_integral_tracking_error_deg=float(np.degrees(front_error)),rear_integral_tracking_error_deg=float(np.degrees(rear_error)),
            curve_pipe_motion=copy.deepcopy(self.curve.local_pipe_motion),local_rejections=local_rejections,
            verified_handover_material_credit_mm=verified_length*1000,actual_target_cohort_material_estimate_mm=actual_target_material*1000,
            confirmed_new_body_material_span_mm=verified_length*1000,existing_supported_target_cohort=live_cohort.tolist(),
            existing_supported_target_axis_advance_mm=actual_target_axis*1000,
            bounded_common_window_credit_mm=actual_credit*1000,common_window_lead_limit_mm=self.actual_follow_lead_limit*1000,
            shape_update_fraction=accepted_shape,requested_common_shift_speed_mm_s=requested_shift*1000,accepted_common_shift_speed_mm_s=self.speed*1000,
            target_entry_capture_feedback=self.entry_capture_feedback,
            curve_shape_update_rejections=shape_rejections,
            common_shift_rejections=shift_rejections,pipe_material_roll_phases_rad=phases.tolist(),failure=failure,
            primary_front_execution='same full-curve integral; local parameter survives rejected common shift; no additional joint wave')
        self.curve.route_guidance_last_step=dict(actual_global_goal_rms_mm=float(np.sqrt(np.mean((guide-p)**2))*1000),basis='one curve, bounded local pipe motion and handover-following common window')
        info=dict(stage='bounded_local_front_winding_and_load_feed',shift_mm=self.shift*1000,shift_speed_mm_s=self.speed*1000,
            reference_points_mm=(body*1000).tolist(),curve_guide_points_mm=(guide*1000).tolist(),
            reference_lead_mm=(self.shift-observed)*1000,actual_material_estimate_mm=observed*1000,
            material_locations_mm=(locations*1000).tolist(),material_region_codes=self.curve.regions(locations).tolist(),
            reference_source_region_mask=self.source_span(locations).tolist(),reference_node_arclength_mm=(self.curve.reference_node_arc*1000).tolist(),
            maximum_reference_link_error_mm=float(np.abs(np.linalg.norm(np.diff(body,axis=0),axis=1)-LINK).max()*1000),
            reference_pipe_clearance_mm=float(self.pipe_geometry.link_minimum(body,65).min()*1000),
            body_curve_rms_error_mm=float(np.sqrt(np.mean((.5*(body[:-1]+body[1:])-centers)**2))*1000),
            tracking_error_deg=float(np.degrees(tracking.max())),fixed_front_y_mm=float(np.median(centers[:7,1]-self.initial_centers[:7,1])*1000),
            fixed_rear_x_mm=float(np.median(centers[14:,0]-self.initial_centers[14:,0])*1000),
            curve_parameters=self.curve.parameters(),global_route_guidance=self.curve.route_guidance_last_step,
            joint_geometry_reference_deg=np.degrees(geometric).tolist(),shift_integral_joint_prior_deg=np.degrees(prior).tolist(),
            adaptive_front_winding=self.local_action,curve_pressure_feedback=self.pressure_feedback,
            servo_torque_feedforward_Nm=self.torque_feedforward.tolist(),physics_execution_prediction=None,
            dynamic_grip_bias_deg=np.zeros(20).tolist(),transferred_material_modules=self.transferred_material_modules.copy(),
            coordination=dict(phase='bounded_local_front_winding',reservoir_height_mm=0.),
            limitation='actual motion and geometric cylinder handover alone establish progress; bounded local curve parameters are not progress')
        return target,info

    def load_state(self,path):
        elapsed=super().load_state(path)
        if self.initial_target_cohort is not None:self.initial_target_cohort=np.asarray(self.initial_target_cohort,dtype=int)
        self.curve.target_entry_offsets=np.asarray(getattr(self.curve,'target_entry_offsets',np.zeros_like(self.curve.shape_offsets)))
        return elapsed
