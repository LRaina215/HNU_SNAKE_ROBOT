"""A captured stationary spatial curve and a shared advancing body window.

No shift-induced spatial helix rotation, axial translation, local departure,
pure-rolling phase or decaying Cartesian entry residual. Initial material
phase is retained and parallel transported as the window tangent changes.
"""
import copy,math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from snake_control.explicit_winding_transfer import ExplicitWindingTransfer
from snake_control.articulated_shift_transfer import ArticulatedWindowCurve
from snake_control.global_transfer_route import FrozenRouteSpec
from snake_control.unified_curve_transfer import DT,LINK,smooth,physical_wrap_masks


class FixedSpatialWindowCurve(ArticulatedWindowCurve):
    def increment(self,shift_increment):
        # Shift changes only the window, never the spatial route phase.
        pass

    def reference(self,shift,phase=None,_clearance_attempt=0):
        grid=np.asarray(self.arc_grid_u)
        xyz=self.base(grid)+np.column_stack([np.interp(grid,self.shape_grid,self.shape_offsets[:,j]) for j in range(3)])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        head=float(np.interp(self.node_s0[0],grid,arc))+shift
        node_arc=head-np.asarray(self.fixed_initial_node_arc_offsets)
        if node_arc[-1]<arc[0] or node_arc[0]>arc[-1]:raise ValueError('Fixed spatial route exhausted')
        nodes=np.column_stack([np.interp(node_arc,arc,xyz[:,j]) for j in range(3)])
        parameters=np.interp(node_arc,arc,grid)
        self.reference_node_parameters=parameters
        self.reference_node_arc=node_arc
        self.reference_center_arc=.5*(node_arc[:-1]+node_arc[1:])
        self.current_target_arc_bound=float(np.interp(self.target_bound,grid,arc))
        if self.clearance_enabled:
            minimum=float(self.clearance.link_minimum(nodes,65).min())
            if minimum<=0.:raise ValueError(f'fixed spatial window intersects original solid: {minimum*1000:.6f}mm')
            self.reference_pipe_clearance_mm=minimum*1000
        return nodes,.5*(nodes[:-1]+nodes[1:]),.5*(parameters[:-1]+parameters[1:])


class FixedSpatialShiftTransfer(ExplicitWindingTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        c=self.curve
        # c.base still passes through all captured actual CLOSE nodes and
        # extends the inherited target hook; the later bridge plans did not.
        c.__class__=FixedSpatialWindowCurve
        c.shape_offsets[:]=0.;c.bridge_clearance_offsets[:]=0.
        c.phase[:]=0.;c.local_pipe_motion=None
        grid=np.asarray(c.arc_grid_u);xyz=c.base(grid)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        c.initial_node_arc=np.interp(c.node_s0,grid,arc)
        c.initial_head_arc=float(c.initial_node_arc[0])
        c.fixed_initial_node_arc_offsets=c.initial_head_arc-c.initial_node_arc
        spec=dict(c.planned_route_spec)
        spec.update(version=5,origin='captured CLOSE spatial curve; stationary source bridge and inherited hook extension',
            spline_x=c.base.x.tolist(),spline_c=c.base.c.tolist(),
            initial_head_parameter=float(c.node_s0[0]),initial_head_arc=c.initial_head_arc,
            material_window='fixed captured initial curve arclength intervals; one common shift',
            entry_residual=np.zeros((22,3)).tolist(),bridge_phase_transport=False,
            bridge_outward_offset=[0.,0.,0.])
        c.planned_route_spec=FrozenRouteSpec(spec)
        self.ordered_handover_required=True
        self.fixed_spatial_action={}
        self.action_opportunity_shift=float(np.interp(c.target_bound,grid,arc)-c.initial_node_arc[9])
        self.action_opportunity_time=max(90.,self.action_opportunity_shift/.0015+20.)
        initial,_=c.reference(0.)
        actual,_=self.observe(data)
        self.initial_fixed_route_error=float(np.linalg.norm(initial-actual,axis=1).max())

    def command(self,data,t,forces,torque,support_forces=None):
        self.current_control_time=self.coordination_time=t
        p,centers=self.observe(data);q=data.qpos[self.qa]
        support=np.asarray(forces if support_forces is None else support_forces)
        self.current_support_forces=support.copy()
        if self.nominal_support_goals is None:self.nominal_support_goals=np.maximum(support.sum(1),1.)
        self.support_force_goals=self.nominal_support_goals.copy()
        if self.filtered_pressure_forces is not None:self.filtered_pressure_forces+=.1*(support-self.filtered_pressure_forces)
        self.filtered_load+=.1*(np.abs(torque)-self.filtered_load)
        if self.first_dispatch:self.torque_feedforward=np.asarray(torque).copy()
        else:self.torque_feedforward+=.1*(np.asarray(torque)-self.torque_feedforward)
        mask=physical_wrap_masks(centers)[1]
        self.incoming_age=np.where((support[1]>.1)&mask,self.incoming_age+DT,0.)
        next_module=max(self.transferred_material_modules,default=7)+1
        while next_module<21 and self.incoming_age[next_module]>=1.:
            self.transferred_material_modules.append(next_module)
            self.events.append(dict(t_s=t,mode='fixed_spatial_window_actual_ordered_handover',module=next_module))
            next_module+=1
        observed=self.observe_progress(centers,forces,t);self.last_body_progress=observed
        tracking=np.abs(self.last-q);geometric_error=np.abs(self.joint_geometry-q)
        following=float(np.clip((np.radians(20)-geometric_error.max())/np.radians(16),0.,1.))
        stressed=(self.filtered_load>2.1)&(geometric_error>np.radians(6.))
        load_scale=float(np.clip(1.-.5*np.mean(stressed),.5,1.))
        requested=.0015*following*load_scale*smooth(t/2.)
        before,reference,locations=self.curve.reference(self.shift)
        saved=copy.deepcopy(self.curve.__dict__);shape_fraction=0.;shape_rejections=[]
        for fraction in (1.,.5,.25,.125,.0625,.03125):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(saved))
            try:
                self.feedback_shape(p,q,reference,locations,geometric_error,forces,DT*fraction)
                shaped,_,_=self.curve.reference(self.shift)
                step=float(np.linalg.norm(shaped-before,axis=1).max())
                if step>.008*DT+1e-8:
                    shape_rejections.append(dict(fraction=fraction,reason='curve shape node speed bound',maximum_node_step_mm=step*1000));continue
                shape_fraction=fraction;break
            except ValueError as exc:shape_rejections.append(dict(fraction=fraction,reason=str(exc)))
        else:self.curve.__dict__.clear();self.curve.__dict__.update(saved)
        ds=requested*DT;rejections=[]
        for _ in range(12):
            try:
                guide,guide_centers,locations=self.curve.reference(self.shift+ds)
                step=float(np.linalg.norm(guide-before,axis=1).max())
                if step<=.008*DT+1e-8:break
                rejections.append(dict(proposed_shift_mm=ds*1000,reason='curve window node speed bound',maximum_node_step_mm=step*1000))
            except ValueError as exc:rejections.append(dict(proposed_shift_mm=ds*1000,reason=str(exc)))
            ds*=.5
        else:ds=0.;guide,guide_centers,locations=self.curve.reference(self.shift)
        axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        cross=np.cross(self.root_orientation[:,2],axis);sine=np.linalg.norm(cross)
        transport=Rotation.from_rotvec(cross*math.atan2(sine,float(self.root_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()
        orientation=transport@self.root_orientation
        prior,field_locations,field_roll=self.integral_targets(guide,orientation,np.zeros(2))
        geometric=np.clip(prior,-np.radians(88),np.radians(88))
        self.shift+=ds;self.speed=ds/DT;self.root_orientation=orientation
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
        self.fixed_spatial_action=dict(action='stationary captured spatial curve; common window advances all integral angles; parallel transported initial material phase; tangential motion allowed',
            nominal_speed_mm_s=1.5,requested_speed_mm_s=requested*1000,accepted_speed_mm_s=self.speed*1000,
            full_integral_tracking_error_deg=float(np.degrees(geometric_error.max())),follow_scale=following,load_scale=load_scale,
            initial_curve_fit_error_mm=self.initial_fixed_route_error*1000,
            initial_M8_required_complete_receiver_window_mm=self.action_opportunity_shift*1000,
            next_receiver_whole_rod_in_target_reference=bool(next_module<21 and np.min(self.curve.reference_node_parameters[next_module:next_module+2])>=self.curve.target_bound),
            spatial_route_phase_rad=self.curve.phase.tolist(),material_axial_roll_phase_rad=self.pipe_roll_phase.tolist(),
            shape_update_fraction=shape_fraction,shape_rejections=shape_rejections,window_rejections=rejections,
            guide_global_rms_error_mm=float(np.sqrt(np.mean((guide-p)**2))*1000),
            material_correspondence='initial smooth-curve arc intervals captured once and shifted together; physical rods and integration cells remain 64.5 mm',
            observer_caveat='legacy rolling-motion estimate is diagnostic only and does not gate or establish progress')
        info=dict(stage='fixed_spatial_curve_common_window',shift_mm=self.shift*1000,shift_speed_mm_s=self.speed*1000,
            reference_points_mm=(body*1000).tolist(),curve_guide_points_mm=(guide*1000).tolist(),
            reference_lead_mm=(self.shift-observed)*1000,actual_material_estimate_mm=observed*1000,
            material_locations_mm=(locations*1000).tolist(),material_region_codes=self.curve.regions(locations).tolist(),
            reference_source_region_mask=self.source_span(locations).tolist(),reference_node_arclength_mm=(self.curve.reference_node_arc*1000).tolist(),
            maximum_reference_link_error_mm=float(np.abs(np.linalg.norm(np.diff(body,axis=0),axis=1)-LINK).max()*1000),
            reference_pipe_clearance_mm=float(self.pipe_geometry.link_minimum(body,65).min()*1000),
            body_curve_rms_error_mm=float(np.sqrt(np.mean((.5*(body[:-1]+body[1:])-centers)**2))*1000),
            tracking_error_deg=float(np.degrees(tracking.max())),fixed_front_y_mm=float(np.median(centers[:7,1]-self.initial_centers[:7,1])*1000),
            fixed_rear_x_mm=float(np.median(centers[14:,0]-self.initial_centers[14:,0])*1000),
            curve_parameters=self.curve.parameters(),global_route_guidance=dict(actual_global_goal_rms_mm=self.fixed_spatial_action['guide_global_rms_error_mm']),
            joint_geometry_reference_deg=np.degrees(geometric).tolist(),fixed_spatial_shift=self.fixed_spatial_action,
            curve_pressure_feedback=self.pressure_feedback,servo_torque_feedforward_Nm=self.torque_feedforward.tolist(),
            dynamic_grip_bias_deg=np.zeros(20).tolist(),transferred_material_modules=self.transferred_material_modules.copy(),
            coordination=dict(phase='fixed_spatial_window',reservoir_height_mm=0.))
        return target,info
