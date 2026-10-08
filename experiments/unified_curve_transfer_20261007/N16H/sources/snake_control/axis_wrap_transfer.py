"""Stable-bearing located front wrap feedback, expressed in one curve.

The world goal follows the target pipe axis. Actual pose locates only a
bounded local connection and supplies physical FK feedback. Four continuous
curve/frame parameters are adjusted; no individual joint-angle patch.
"""
import copy,math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
from snake_control.fixed_spatial_shift_transfer import FixedSpatialShiftTransfer
from snake_control.unified_curve_transfer import DT,LINK,smooth,physical_wrap_masks


class AxisWrapTransfer(FixedSpatialShiftTransfer):
    def __init__(self,model,data):
        super().__init__(model,data,False)
        self.wrap_parameters=np.zeros(4)
        self.wrap_transport_orientation=self.root_orientation.copy()
        self.wrap_anchor_u=self.curve.target_bound
        self.curve.axis_wrap_offsets=np.zeros_like(self.curve.shape_offsets)
        self.bearing_age=np.zeros(21)
        self.wrap_feedback={}
        self.wrap_lost_since=None
        self.prediction_anchor_module=None
        self.prediction_registration={}
        _,centers=self.observe(data)
        phi=np.unwrap(np.arctan2(-(centers[:8,2]-1),centers[:8,0]-1.1))
        self.initial_front_span=float(phi[0]-phi[-1])
        self.parameter_limits=np.array([math.pi,.018,.15,.15])
        self.parameter_step_limits=np.array([np.radians(2)*DT,.001*DT,np.radians(1)*DT,np.radians(1)*DT])

    def curve_patch(self,parameters,anchor,head_u):
        grid=self.curve.shape_grid;native=self.curve.base(grid)
        radial=native[:,[0,2]]-[1.1,1.]
        radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-12)
        weights=smooth((grid-(self.curve.target_bound-LINK))/(2*LINK))*(1-smooth((grid-(head_u+LINK))/(2*LINK)))
        anchor_y=float(self.curve.base(anchor)[1])
        lever=np.clip(native[:,1]-anchor_y,-.30,.30)
        patch=np.zeros_like(native)
        patch[:,0]=weights*(parameters[1]*radial[:,0]+parameters[2]*lever)
        patch[:,2]=weights*(parameters[1]*radial[:,1]+parameters[3]*lever)
        patch*=np.minimum(1.,.040/np.maximum(np.linalg.norm(patch,axis=1),1e-12))[:,None]
        return patch

    def evaluate(self,parameters,shift,anchor,head_u,data,patch=None):
        self.curve.axis_wrap_offsets=self.curve_patch(parameters,anchor,head_u) if patch is None else patch.copy()
        guide,_,locations=self.curve.reference(shift)
        axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        cross=np.cross(self.wrap_transport_orientation[:,2],axis);sine=np.linalg.norm(cross)
        base=Rotation.from_rotvec(cross*math.atan2(sine,float(self.wrap_transport_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()@self.wrap_transport_orientation
        orientation=Rotation.from_rotvec(axis*parameters[0]).as_matrix()@base
        prior,field_locations,roll=self.integral_targets(guide,orientation,np.zeros(2))
        self.fit_data.qpos[:]=data.qpos
        self.fit_data.qpos[self.qa]=np.clip(prior,-np.radians(88),np.radians(88))
        mujoco.mj_fwdPosition(self.m,self.fit_data)
        points,_=self.observe(self.fit_data)
        centers=.5*(points[:-1]+points[1:])
        if self.prediction_anchor_module is not None:
            module=self.prediction_anchor_module
            actual_rotation=data.xmat[self.head+module].reshape(3,3)
            predicted_rotation=self.fit_data.xmat[self.head+module].reshape(3,3)
            registration=actual_rotation@predicted_rotation.T
            _,actual_centers=self.observe(data)
            translation=actual_centers[module]-registration@centers[module]
            centers=centers@registration.T+translation
            self.prediction_registration=dict(anchor_module=int(module),rotation=registration.tolist(),translation_mm=(translation*1000).tolist(),
                assumption='short-time physical FK prediction registered to measured bearing body pose; bearing is not actually pinned and physical root is never assigned')
        else:
            self.prediction_registration=dict(anchor_module=None,assumption='no stable bearing: actual-root FK fallback')
        return guide,orientation,prior,field_locations,roll,centers,base,locations

    def execution_reference(self,data,t,support,guide,orientation,shift):
        _,actual_centers=self.observe(data)
        source,target=physical_wrap_masks(actual_centers)
        bearing=(support[1]>.1)&target
        self.bearing_age=np.where(bearing,self.bearing_age+DT,0.)
        confirmed=max(self.transferred_material_modules,default=7)
        stable=bearing&(self.bearing_age>=.5)&(np.arange(21)<=confirmed)
        prior_anchor=False
        if not stable.any() and t<.5:
            stable=bearing&(np.arange(21)<=7);prior_anchor=True
        ids=np.flatnonzero(stable)
        if self.prediction_anchor_module not in ids:
            posterior_ids=ids[ids>=2]
            self.prediction_anchor_module=int(posterior_ids[-1]) if len(posterior_ids) else (int(ids[-1]) if len(ids) else None)
        saved_curve=copy.deepcopy(self.curve.__dict__)
        saved_patch=self.curve.axis_wrap_offsets.copy()
        saved_anchor=self.wrap_anchor_u
        self.curve.axis_wrap_offsets[:]=0.
        self.curve.shape_offsets[:]=0.
        self.curve.clearance_enabled=False
        goal_nodes,goal,locations=self.curve.reference(shift)
        self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(saved_curve))
        if len(ids):
            measured_anchor=float(np.average(locations[ids],weights=support[1,ids]))
            self.wrap_anchor_u+=float(np.clip(.2*(measured_anchor-self.wrap_anchor_u),-.002,.002))
        head_u=float(self.curve.reference_node_parameters[0])
        front=np.flatnonzero(locations>=self.curve.target_bound)
        if not len(front):front=np.arange(8)
        azimuth_offset=axis_offset=0.
        if len(ids):
            ap=np.arctan2(actual_centers[ids,2]-1,actual_centers[ids,0]-1.1)
            gp=np.arctan2(goal[ids,2]-1,goal[ids,0]-1.1)
            difference=np.angle(np.exp(1j*(ap-gp)))
            azimuth_offset=float(np.clip(np.angle(np.sum(support[1,ids]*np.exp(1j*difference))),-np.pi/4,np.pi/4))
            axis_offset=float(np.clip(np.median(actual_centers[ids,1]-goal[ids,1]),-.5*LINK,.5*LINK))
        radial=goal[front][:,[0,2]]-[1.1,1.]
        co,sn=np.cos(azimuth_offset),np.sin(azimuth_offset)
        goal[np.ix_(front,[0,2])]=[1.1,1.]+np.column_stack([co*radial[:,0]-sn*radial[:,1],sn*radial[:,0]+co*radial[:,1]])
        goal[front,1]+=axis_offset
        goal_radius=np.linalg.norm(goal[front][:,[0,2]]-[1.1,1.],axis=1)
        posterior=np.flatnonzero(source&(support[0]>.1))
        weights=np.where(stable[front],1.5,1.)[:,None]
        def residual(value):
            predicted=value[5]
            errors=[((predicted[front]-goal[front])*[1.,.5,1.]*weights).ravel()]
            r=np.linalg.norm(predicted[front][:,[0,2]]-[1.1,1.],axis=1)
            errors.append(.8*(r-goal_radius))
            if len(posterior):errors.append(.4*(predicted[posterior,1:3]-actual_centers[posterior,1:3]).ravel())
            return np.concatenate(errors)
        baseline_rejection=None
        try:baseline=self.evaluate(self.wrap_parameters,shift,self.wrap_anchor_u,head_u,data)
        except ValueError as exc:
            baseline_rejection=str(exc)
            self.wrap_anchor_u=saved_anchor
            baseline=self.evaluate(self.wrap_parameters,shift,self.wrap_anchor_u,head_u,data,patch=saved_patch)
        base_error=residual(baseline)
        columns=[];epsilon=[.001,.0001,.001,.001]
        for j,h in enumerate(epsilon):
            candidate=self.wrap_parameters.copy();candidate[j]+=h
            try:column=(residual(self.evaluate(candidate,shift,self.wrap_anchor_u,head_u,data))-base_error)/h
            except ValueError:column=np.zeros_like(base_error)
            columns.append(column*self.parameter_step_limits[j])
        jacobian=np.column_stack(columns)
        matrix=np.vstack([jacobian,.0001*np.eye(4)])
        rhs=np.r_[-.15*DT*base_error,-.00002*self.wrap_parameters/self.parameter_limits]
        lower=np.maximum(-1.,(-self.parameter_limits-self.wrap_parameters)/self.parameter_step_limits)
        upper=np.minimum(1.,(self.parameter_limits-self.wrap_parameters)/self.parameter_step_limits)
        normalized=lsq_linear(matrix,rhs,bounds=(lower,upper),tol=1e-8,max_iter=30).x
        proposal=np.clip(self.wrap_parameters+normalized*self.parameter_step_limits,-self.parameter_limits,self.parameter_limits)
        accepted=0.;rejects=[];chosen=baseline;accepted_patch=None
        if baseline_rejection:rejects.append(dict(fraction=0.,reason=baseline_rejection,stage='baseline'))
        for fraction in [1.,.5,.25,.125,.0625,.03125,0.]:
            candidate=self.wrap_parameters+fraction*(proposal-self.wrap_parameters)
            try:
                value=self.evaluate(candidate,shift,self.wrap_anchor_u,head_u,data)
                node_step=float(np.linalg.norm(value[0]-self.reference_before_execution,axis=1).max())
                if node_step>.008*DT+1e-8:
                    rejects.append(dict(fraction=fraction,reason='full curve node speed bound',node_step_mm=node_step*1000));continue
                if np.linalg.norm(residual(value))>np.linalg.norm(base_error)+1e-12:
                    rejects.append(dict(fraction=fraction,reason='physical FK predicted front goal residual increases'));continue
                chosen=value;accepted=fraction;self.wrap_parameters=candidate
                accepted_patch=self.curve.axis_wrap_offsets.copy();break
            except ValueError as exc:rejects.append(dict(fraction=fraction,reason=str(exc)))
        else:
            self.wrap_anchor_u=saved_anchor
            accepted_patch=saved_patch
        # Re-evaluate chosen state after finite-difference calls to retain its
        # guide, parameters and metadata, rather than the last perturbation.
        chosen=self.evaluate(self.wrap_parameters,shift,self.wrap_anchor_u,head_u,data,patch=accepted_patch)
        self.wrap_transport_orientation=chosen[6].copy()
        self.wrap_feedback=dict(action='pipe-axis front wrap goal located by stable actual bearing; local radial and tilt curve field plus bounded material phase from physical FK residual',
            stable_target_bearing_modules=ids.tolist(),bearing_age_s=self.bearing_age.tolist(),
            initial_CLOSE_anchor_prior=prior_anchor,actual_anchor_curve_station_mm=self.wrap_anchor_u*1000,
            local_connection_axis_offset_mm=axis_offset*1000,local_connection_azimuth_offset_deg=float(np.degrees(azimuth_offset)),
            goal_front_modules=front.tolist(),goal_front_centers_mm=(goal[front]*1000).tolist(),
            parameters=dict(material_phase_deg=float(np.degrees(self.wrap_parameters[0])),radial_curve_offset_mm=self.wrap_parameters[1]*1000,
                local_tilt_x_deg=float(np.degrees(self.wrap_parameters[2])),local_tilt_z_deg=float(np.degrees(self.wrap_parameters[3]))),
            proposed_normalized_parameter_step=normalized.tolist(),accepted_fraction=accepted,rejections=rejects,
            physical_FK_goal_residual_before_rms_mm=float(np.sqrt(np.mean(base_error**2))*1000),
            physical_FK_goal_residual_after_rms_mm=float(np.sqrt(np.mean(residual(chosen)**2))*1000),
            maximum_local_curve_field_mm=float(np.linalg.norm(self.curve.axis_wrap_offsets,axis=1).max()*1000),
            physical_FK_prediction_registration=self.prediction_registration.copy(),
            bounded_parameter_solver='four normalized curve/frame increments solved within remaining parameter bounds; reject predicted residual increase',
            feedback_truth='simulation actual world nodes, body/root orientation, measured cylinder load, target/source geometric membership; physical root is observed only; initial material normals fixed',
            long_term_goal='original fixed route and single inherited target helix about +Y; actual anchors bound only local connection, no unrestricted actual-pose overwrite')
        return chosen[:5]

    def command(self,data,t,forces,torque,support_forces=None):
        target,info=super().command(data,t,forces,torque,support_forces)
        _,centers=self.observe(data)
        phi=np.unwrap(np.arctan2(-(centers[:8,2]-1),centers[:8,0]-1.1))
        span=float(phi[0]-phi[-1]);radius=np.linalg.norm(centers[:8][:,[0,2]]-[1.1,1.],axis=1)
        lost=span<.65*self.initial_front_span or np.median(radius)>.125
        if lost and self.wrap_lost_since is None:self.wrap_lost_since=t
        if not lost:self.wrap_lost_since=None
        failure='front_wrap_recovery_failed_20s' if self.wrap_lost_since is not None and t-self.wrap_lost_since>=20. else None
        self.wrap_feedback.update(actual_front_span_rad=span,actual_front_median_radius_mm=float(np.median(radius)*1000),
            recovery_active=bool(lost),recovery_started_s=self.wrap_lost_since,recovery_budget_s=20.,failure=failure)
        info['stage']='axis_located_front_wrap_and_common_feed'
        info['axis_wrap']=self.wrap_feedback.copy()
        return target,info
