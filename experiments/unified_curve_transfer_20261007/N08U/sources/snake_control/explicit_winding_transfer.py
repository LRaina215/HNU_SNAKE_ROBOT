"""Explicit front winding and rear feed from one moving material window.

The common shifted curve integral directly generates all alternating-hinge
position targets. No support optimizer may replace the front rolling wave.
The two pipe rolling phases are driven by that same shift, with a continuous
transition through the bridge. Actual dynamics receive only twenty ctrl.
"""
import copy
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from snake_control.articulated_shift_transfer import ArticulatedShiftTransfer
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone
from snake_control.unified_curve_transfer import DT,LINK,BODY_RADIUS,physical_wrap_masks,smooth
from snake_control.global_transfer_route import FrozenRouteSpec,route_body,sample_route


class ExplicitWindingTransfer(ArticulatedShiftTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        # N01's pointwise bridge phase blend drove a rod through the source
        # cylinder and throttled the common shift. Transport endpoint control
        # frames instead, retaining tangency, with a fixed outer -Y allowance.
        actual,_=self.observe(data)
        spec=dict(self.curve.planned_route_spec)
        spec.update(bridge_phase_transport=True,bridge_outward_offset=[0.,-.035,0.],
                    entry_residual=np.zeros((22,3)).tolist())
        grid=np.asarray(spec['spline_x']);xyz=sample_route(spec,grid)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        spec['initial_head_arc']=float(np.interp(spec['initial_head_parameter'],grid,arc))
        raw,parameters=route_body(spec,0.)
        spec['entry_residual']=(actual-raw).tolist()
        spec['entry_target_weights']=smooth((parameters-spec['source_bound'])/(spec['target_bound']-spec['source_bound'])).tolist()
        self.curve.planned_route_spec=FrozenRouteSpec(spec)
        self.curve.bridge_clearance_offsets=np.zeros_like(self.curve.shape_offsets)
        self.pipe_roll_phase=np.zeros(2)
        self.action_opportunity_shift=2*LINK
        self.action_opportunity_time=90.
        self.winding_control={}

    def route_outside_bridge(self,shift):
        """Bounded continuous curve-field avoidance, before angle generation.

        Correct the bridge's spatial field, not individual sampled nodes or
        physical body poses. A full update still has the original 8mm/s node
        bound; rejected proposals are rolled back by the common shift loop.
        """
        spec=self.curve.planned_route_spec
        fractions=np.linspace(0.,1.,65)
        enabled=self.curve.clearance_enabled
        self.curve.clearance_enabled=False
        largest=0.
        try:
            for _ in range(5):
                nodes,centers,locations=self.curve.reference(shift)
                lines=nodes[:-1,None]*(1-fractions[None,:,None])+nodes[1:,None]*fractions[None,:,None]
                gaps=self.pipe_geometry.signed(lines)
                minima=gaps.min(1);nearest=gaps.argmin(1)
                active=np.flatnonzero((minima<.0001)&(locations>=spec['source_bound']-LINK)&(locations<=spec['target_bound']+LINK))
                if not len(active):break
                points=lines[active,nearest[active]];step=1e-6
                normal=np.column_stack([(self.pipe_geometry.signed(points+np.eye(3)[axis]*step)-
                    self.pipe_geometry.signed(points-np.eye(3)[axis]*step))/(2*step) for axis in range(3)])
                normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
                increments=(.000125-minima[active])[:,None]*normal
                weights=np.exp(-.5*((self.curve.shape_grid[:,None]-locations[active][None,:])/(.8*LINK))**2)
                update=weights@increments/np.maximum(weights.sum(1)[:,None],1.)
                self.curve.bridge_clearance_offsets+=update
                length=np.linalg.norm(self.curve.bridge_clearance_offsets,axis=1)
                self.curve.bridge_clearance_offsets*=np.minimum(1.,.05/np.maximum(length,1e-12))[:,None]
                largest=max(largest,float(np.linalg.norm(update,axis=1).max()))
            nodes,centers,locations=self.curve.reference(shift)
            minimum=float(self.pipe_geometry.link_minimum(nodes,65).min())
            if minimum<=0.:raise ValueError(f'bridge curve remains inside original solid: {minimum*1000:.6f}mm')
            return nodes,centers,locations,largest,minimum
        finally:
            self.curve.clearance_enabled=enabled

    def integral_targets(self,guide,orientation,phase):
        bone=SpatialBishopBackbone(guide*1000,np.full(21,LINK*1000))
        normals=orientation@self.initial_material_normals_local
        field=bone.field@(bone.start_frame[:,1:].T@normals)
        station=.5*(bone.bounds[:-1]+bone.bounds[1:])/1000
        locations=np.interp(station,np.arange(22)*LINK,self.curve.reference_node_parameters)
        spec=self.curve.planned_route_spec
        blend=smooth((locations-spec['source_bound'])/(spec['target_bound']-spec['source_bound']))
        local_roll=(1-blend)*phase[0]+blend*phase[1]
        relative=local_roll-phase[1]
        co,sn=np.cos(relative),np.sin(relative)
        rotation=np.array([[co,sn],[-sn,co]]).transpose(2,0,1)
        prior=bone.joint_angles(field=np.einsum('ni,nij->nj',field,rotation))
        return prior,locations,local_roll

    def command(self,data,t,forces,torque,support_forces=None):
        self.current_control_time=self.coordination_time=t
        p,centers=self.observe(data);q=data.qpos[self.qa]
        support=np.asarray(forces if support_forces is None else support_forces)
        self.current_support_forces=support.copy()
        if self.nominal_support_goals is None:
            self.nominal_support_goals=np.maximum(support.sum(1),1.)
        self.support_force_goals=self.nominal_support_goals.copy()
        if self.filtered_pressure_forces is not None:
            self.filtered_pressure_forces+=.1*(support-self.filtered_pressure_forces)
        self.filtered_load+=.1*(np.abs(torque)-self.filtered_load)
        if self.first_dispatch:self.torque_feedforward=np.asarray(torque).copy()
        else:self.torque_feedforward+=.1*(np.asarray(torque)-self.torque_feedforward)
        mask=physical_wrap_masks(centers)[1]
        self.incoming_age=np.where((support[1]>.1)&mask,self.incoming_age+DT,0.)
        for module in np.flatnonzero(self.incoming_age>=1.):
            if module>=8 and int(module) not in self.transferred_material_modules:
                self.transferred_material_modules.append(int(module))
                self.events.append(dict(t_s=t,mode='actual_explicit_winding_handover',module=int(module)))
        observed=self.observe_progress(centers,forces,t);self.last_body_progress=observed
        tracking=np.abs(self.last-q);geometric_error=np.abs(self.joint_geometry-q)
        following=float(np.clip((np.radians(20)-geometric_error.max())/np.radians(16),0.,1.))
        stressed=(self.filtered_load>2.1)&(geometric_error>np.radians(6.))
        load_scale=float(np.clip(1.-.5*np.mean(stressed),.5,1.))
        requested=.002*following*load_scale*smooth(t/3.)
        self.speed+=np.clip(requested-self.speed,-.003*DT,.001*DT)
        before,reference,locations=self.curve.reference(self.shift)
        saved=copy.deepcopy(self.curve.__dict__)
        accepted_shape=0.
        for fraction in (1.,.5,.25,.125,.0625,.03125):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(saved))
            try:
                self.feedback_shape(p,q,reference,locations,geometric_error,forces,DT*fraction)
                shaped,shaped_c,_=self.curve.reference(self.shift)
                if np.linalg.norm(shaped-before,axis=1).max()>.008*DT+1e-8:continue
                accepted_shape=fraction;break
            except ValueError:continue
        else:
            self.curve.__dict__.clear();self.curve.__dict__.update(saved)
            shaped,shaped_c,_=self.curve.reference(self.shift)
        phase_saved=copy.deepcopy(self.curve.__dict__);ds=self.speed*DT
        requested_ds=ds;rejections=[];clearance_update=0.
        for _ in range(12):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(phase_saved))
            self.curve.increment(ds)
            try:
                guide,guide_centers,locations,clearance_update,guide_gap=self.route_outside_bridge(self.shift+ds)
                node_step=float(np.linalg.norm(guide-before,axis=1).max())
                if node_step<=.008*DT+1e-8:break
                rejections.append(dict(proposed_shift_step_mm=ds*1000,reason='continuous curve node speed bound',maximum_node_step_mm=node_step*1000))
            except ValueError as exc:
                rejections.append(dict(proposed_shift_step_mm=ds*1000,reason=str(exc)))
            ds*=.5
        else:
            ds=0.;self.curve.__dict__.clear();self.curve.__dict__.update(phase_saved)
            guide,guide_centers,locations=self.curve.reference(self.shift)
        axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        cross=np.cross(self.root_orientation[:,2],axis);sine=np.linalg.norm(cross)
        transport=Rotation.from_rotvec(cross*math.atan2(sine,float(self.root_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()
        # Source azimuth is about +X, target azimuth about -Y in the curve
        # chart. Their local material rolling signs therefore differ.
        source_step=-np.sign(self.curve.theta_rates[0])*self.curve.axial_sines[0]*ds/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[0]**2))
        target_step=np.sign(self.curve.theta_rates[1])*self.curve.axial_sines[1]*ds/(BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[1]**2))
        orientation=Rotation.from_rotvec(axis*target_step).as_matrix()@transport@self.root_orientation
        phases=self.pipe_roll_phase+np.array([source_step,target_step])
        prior,field_locations,field_roll=self.integral_targets(guide,orientation,phases)
        geometric=np.clip(prior,-np.radians(88),np.radians(88))
        self.shift+=ds;self.speed=ds/DT;self.pipe_roll_phase=phases
        self.root_material_phase+=target_step;self.root_orientation=orientation
        self.joint_geometry=geometric;self.integral_joint_prior=prior.copy()
        gain=self.m.actuator_gainprm[:,0];bias=np.clip(self.torque_feedforward,-2.355,2.355)/gain
        raw=np.clip(geometric+bias,q-np.radians(3),q+np.radians(3))
        target=np.clip(raw,np.maximum(-np.radians(88),self.last-np.radians(10)*DT),np.minimum(np.radians(88),self.last+np.radians(10)*DT))
        if self.first_dispatch:target=self.last.copy();self.first_dispatch=False
        self.last=target.copy()
        # FK for honest reference display, registered to measured root only.
        # It does not alter the fixed material gauge or the actual root.
        self.fit_data.qpos[:]=data.qpos;self.fit_data.qpos[self.qa]=geometric
        mujoco.mj_fwdPosition(self.m,self.fit_data)
        d=self.fit_data
        body=np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
        self.fk_reference_points=body.copy();self.fk_root_pose=d.qpos[:7].copy()
        self.winding_control=dict(action='front continues inherited target hook; rear rolls/feed; bridge joins two pipe material phases on same shift',
            common_shift_mm=self.shift*1000,nominal_speed_mm_s=2.,requested_speed_mm_s=requested*1000,
            follow_scale=following,load_scale=load_scale,maximum_integral_tracking_error_deg=float(np.degrees(geometric_error.max())),
            pipe_material_roll_phases_rad=phases.tolist(),field_curve_parameters_mm=(field_locations*1000).tolist(),
            field_material_roll_phase_rad=field_roll.tolist(),shape_update_fraction=accepted_shape,
            requested_shift_step_mm=requested_ds*1000,accepted_shift_step_mm=ds*1000,
            advance_rejections=rejections,maximum_bridge_clearance_field_mm=float(np.linalg.norm(self.curve.bridge_clearance_offsets,axis=1).max()*1000),
            candidate_clearance_field_update_mm=clearance_update*1000,
            opportunity_shift_mm=self.action_opportunity_shift*1000,opportunity_window_s=self.action_opportunity_time,
            primary_front_execution='unmodified common-window integral position targets; no whole-body optimizer or contact action reselection')
        self.curve.route_guidance_last_step=dict(actual_global_goal_rms_mm=float(np.sqrt(np.mean((guide-p)**2))*1000),basis='one shared material window, direct full integral position wave')
        info=dict(stage='explicit_front_winding_and_rear_feed',shift_mm=self.shift*1000,shift_speed_mm_s=self.speed*1000,
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
            explicit_winding=self.winding_control,curve_pressure_feedback=self.pressure_feedback,
            servo_torque_feedforward_Nm=self.torque_feedforward.tolist(),physics_execution_prediction=None,
            dynamic_grip_bias_deg=np.zeros(20).tolist(),transferred_material_modules=self.transferred_material_modules.copy(),
            coordination=dict(phase='explicit_front_winding',reservoir_height_mm=0.),
            limitation='actual motion/contact feedback is simulation truth; only twenty position commands execute; virtual FK display is not physical progress')
        return target,info

    def load_state(self,path):
        elapsed=super().load_state(path);self.pipe_roll_phase=np.asarray(self.pipe_roll_phase)
        self.curve.bridge_clearance_offsets=np.asarray(getattr(self.curve,'bridge_clearance_offsets',np.zeros_like(self.curve.shape_offsets)))
        return elapsed
