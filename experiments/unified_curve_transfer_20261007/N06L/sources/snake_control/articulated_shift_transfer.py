"""Shared curve window -> exact alternating-hinge body -> position targets.

The independent FK cache may use a virtual head pose to register the planned
curve. Neither that pose nor any planned node is assigned to actual dynamics.
"""
import copy
import math
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from snake_control.direct_curve_transfer import DirectCurveTransfer
from snake_control.unified_curve_transfer import DT, LINK, BODY_RADIUS, physical_wrap_masks, cylindrical_contact, smooth
from snake_control.coordinated_curve_transfer import CoordinatedRouteCurve
from snake_control.global_transfer_route import FrozenRouteSpec, route_body, sample_route
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone


class ArticulatedWindowCurve(CoordinatedRouteCurve):
    def reference(self,shift,phase=None,_clearance_attempt=0):
        phase = self.phase if phase is None else np.asarray(phase)
        spec = self.planned_route_spec
        points,parameters = route_body(spec,shift,phase,
            spatial_offsets=(self.shape_grid,self.shape_offsets+getattr(self,'bridge_clearance_offsets',0.)+
                             np.asarray(getattr(self,'target_entry_offsets',0.))),
            local_pipe_motion=getattr(self,'local_pipe_motion',None))
        locations = .5*(parameters[:-1]+parameters[1:])
        self.reference_node_parameters = parameters
        self.reference_node_arc = spec['initial_head_arc']+shift-np.arange(22)*LINK
        self.reference_center_arc = .5*(self.reference_node_arc[:-1]+self.reference_node_arc[1:])
        self.current_target_arc_bound = spec['target_bound']
        if self.clearance_enabled:
            minimum = float(self.clearance.link_minimum(points,65).min())
            if minimum<=0.:raise ValueError('Continuous material window crosses original pipe solid')
            self.reference_pipe_clearance_mm = minimum*1000
        return points,.5*(points[:-1]+points[1:]),locations

    def advance_planned_guidance(self,shift,dt,physical_nodes,forces,tracking,load):
        # The window itself is the target. No independent per-node pursuit or
        # finite-chord root search may change its material correspondence.
        points,_,_ = self.reference(shift)
        self.route_guidance_last_step = dict(accepted_fraction=1.,maximum_node_step_mm=0.,
            actual_global_goal_rms_mm=float(np.sqrt(np.mean((points-physical_nodes)**2))*1000),
            global_phase_rad=self.phase.tolist(),
            basis='s_i = s_head + shared_shift - i*physical_link; pressure deforms same curve; exact FK projects body')
        return self.route_guidance_last_step


class ArticulatedShiftTransfer(DirectCurveTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        actual,_ = self.observe(data)
        spec = dict(self.curve.planned_route_spec)
        grid = np.asarray(spec['spline_x'])
        xyz = sample_route(spec,grid)
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        head_arc = float(np.interp(0.,grid,arc))+21*LINK
        spec.update(version=4,material_window='uniform_arclength',
                    initial_head_parameter=float(np.interp(head_arc,arc,grid)),initial_head_arc=head_arc)
        raw,parameters = route_body(spec,0.)
        spec['entry_residual'] = (actual-raw).tolist()
        spec['entry_target_weights'] = smooth((parameters-spec['source_bound'])/(spec['target_bound']-spec['source_bound'])).tolist()
        self.curve.planned_route_spec = FrozenRouteSpec(spec)
        self.curve.__class__ = ArticulatedWindowCurve
        self.fit_data = mujoco.MjData(model)
        self.joint_geometry = data.qpos[self.qa].copy()
        self.root_orientation = self.d.xmat[self.head].reshape(3,3).copy()
        self.root_material_phase = 0.
        self.torque_feedforward = np.zeros(20)
        self.target_carrier_requirement = 3
        self.fk_reference_points, _ = self.observe(data)
        self.fk_root_pose = data.qpos[:7].copy()
        initial_bone = SpatialBishopBackbone(actual*1000,np.full(21,LINK*1000))
        _,material_rotation = initial_bone.material_field(data.qpos[self.qa])
        self.initial_material_normals_local = self.root_orientation.T@initial_bone.start_frame[:,1:]@material_rotation
        self.integral_joint_prior = self.joint_geometry.copy()
        self.pressure_profile_locations = None
        self.pressure_profile_forces = None
        self.filtered_pressure_forces = None
        self.pressure_feedback = {}

    def coordinated_feed_limit(self,lag,error_scale,support_scale,physical_speeds):
        # Shift generates the joint wave. Its startup does not require prior
        # positive body motion or a budget credited from historical motion.
        q = self.d.qpos[self.qa]
        joint_error = np.abs(self.joint_geometry-q)
        loaded_mismatch = (self.filtered_load>2.1)&(joint_error>np.radians(6.))
        load_scale = float(np.clip(1.-.75*np.mean(loaded_mismatch),.25,1.))
        requested = .0015*error_scale*support_scale*load_scale*smooth(self.coordination_time/2.)
        self.direct_feedback = dict(policy='persistent shared shift; actual joint tracking/load/support scales',
            nominal_speed_mm_s=1.5,requested_mm_s=float(requested*1000),
            tracking_scale=float(error_scale),support_scale=float(support_scale),load_scale=load_scale,
            geometric_tracking_max_deg=float(np.degrees(joint_error.max())),
            reference_minus_observed_material_mm=float(lag*1000),
            observed_side_speeds_mm_s=(np.asarray(physical_speeds)*1000).tolist(),
            actual_motion_is_success_evidence_only=True)
        return float(requested)

    def feedback_shape(self,p,q,reference,locations,tracking,forces,dt):
        """Transport the captured support-load profile on the same curve.

        Dividing total load equally among current contacts had also demanded
        that every zero-load wrapped rod squeeze inward. That changed the
        whole body even with shift stopped. Here the captured spatial load
        stations move through the shared material window; their total load is
        conserved and only measured local load error deforms the curve.
        """
        support = self.current_support_forces
        if self.pressure_profile_locations is None:
            order = np.argsort(locations)
            self.pressure_profile_locations = locations[order].copy()
            self.pressure_profile_forces = support[:,order].copy()
            self.filtered_pressure_forces = support.copy()
        centers = .5*(p[:-1]+p[1:])
        masks = np.asarray(physical_wrap_masks(centers))
        goals = np.array([np.interp(locations,self.pressure_profile_locations,row)
                          for row in self.pressure_profile_forces])
        goals *= masks
        for side in range(2):
            total = goals[side].sum()
            if total>.1:
                goals[side] *= self.support_force_goals[side]/total
        velocity = np.zeros_like(centers)
        local_velocity = np.zeros((2,21))
        error = self.filtered_pressure_forces-goals
        deadband = .05+.02*goals
        error = np.sign(error)*np.maximum(0.,np.abs(error)-deadband)
        for side in range(2):
            radial = centers-([0.,0.,1.] if side==0 else [1.1,0.,1.])
            radial[:,side] = 0.
            radial /= np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
            active = masks[side]&((goals[side]>.1)|(support[side]>.1))
            vn = np.clip(.00002*error[side],-.0006,.0006)*active
            velocity += vn[:,None]*radial
            local_velocity[side] = vn
        joint_mismatch = np.abs(self.joint_geometry-q)
        stressed = (self.filtered_load>1.9)&(joint_mismatch>np.radians(3.))
        module_stress = np.r_[stressed[0],stressed]|np.r_[stressed,stressed[-1]]
        tangent = p[:-1]-p[1:]
        tangent /= np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        residual = centers-reference
        normal_residual = residual-np.sum(residual*tangent,axis=1)[:,None]*tangent
        tracking_velocity = .1*normal_residual*module_stress[:,None]
        tracking_velocity *= np.minimum(1.,.001/np.maximum(np.linalg.norm(tracking_velocity,axis=1),1e-12))[:,None]
        velocity += tracking_velocity
        self.curve.deposit(locations,velocity,1.,dt)
        self.n_adaptations += 1
        self.pressure_feedback = dict(
            measured_cylinder_loads_N=support.tolist(),filtered_cylinder_loads_N=self.filtered_pressure_forces.tolist(),
            curve_station_load_goals_N=goals.tolist(),normal_curve_velocity_mm_s=(local_velocity*1000).tolist(),
            loaded_tracking_normal_velocity_mm_s=(tracking_velocity*1000).tolist(),
            basis='initial measured spatial support-load profile; common window transports material through fixed curve stations; conserved side load')

    def fit_articulated_body(self,data,guide,orientation):
        """Project the shifted curve with exact hinges and support registration.

        The material orientation passed here is integrated independently of
        the fitted virtual root. Real contact levers register that root and
        constrain rolling increments inside this one position-reference fit.
        """
        d = self.fit_data
        d.qpos[:] = data.qpos
        previous = self.joint_geometry.copy()
        root_position = self.fk_root_pose[:3].copy()
        root_rotation = Rotation.from_quat(self.fk_root_pose[3:7][[1,2,3,0]])
        nominal_position = guide[0]-orientation@self.m.site('s_head').pos
        nominal_rotation = (Rotation.from_matrix(orientation)*root_rotation.inv()).as_rotvec()
        d.qpos[:7] = self.fk_root_pose
        d.qpos[self.qa] = previous
        mujoco.mj_fwdPosition(self.m,d)
        real_points,real_centers = self.observe(data)
        masks = physical_wrap_masks(real_centers)
        registered = set()
        for side in range(2):
            candidates = sorted(np.flatnonzero(masks[side]&(self.current_support_forces[side]>.1)),
                key=lambda module:self.current_support_forces[side,module],reverse=True)
            selected = []
            for module in candidates:
                if all(abs(int(module)-other)>=2 for other in selected):
                    selected.append(int(module))
                if len(selected)==2:
                    break
            registered.update((side,module) for module in selected)
        contacts = []
        for contact in data.contact:
            for pipe,robot in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
                body = int(self.m.geom_bodyid[robot])
                if not self.head<=body<self.head+21:
                    continue
                name = self.m.geom(pipe).name or ''
                side = 0 if name in {'pipe_straight_01','pipe_sharp_in_02'} else 1 if name in {'pipe_sharp_out_03','pipe_straight_04'} else None
                if side is None or not cylindrical_contact(side,contact.pos,contact.frame[:3]):
                    continue
                module = body-self.head
                if (side,module) not in registered:
                    continue
                # A sensed material surface point, expressed in its real rod.
                # Preserve its instantaneous rolling position from the preceding
                # mathematical pose, never by assigning a physical body pose.
                local = data.xmat[body].reshape(3,3).T@(contact.pos-data.xpos[body])
                anchor = d.xpos[body]+d.xmat[body].reshape(3,3)@local
                normal = contact.frame[:3].copy()
                normal /= max(np.linalg.norm(normal),1e-12)
                contacts.append((body,local,anchor.copy(),side,module,normal))
        bone = SpatialBishopBackbone(guide*1000,np.full(21,LINK*1000))
        material_normals = orientation@self.initial_material_normals_local
        material_rotation = bone.start_frame[:,1:].T@material_normals
        prior = bone.joint_angles(field=bone.field@material_rotation)
        self.integral_joint_prior = prior.copy()
        lower = np.r_[np.full(3,-.008*DT),np.full(3,-.12*DT),
            np.maximum(-np.radians(88),previous-np.radians(10)*DT)]
        upper = np.r_[np.full(3,.008*DT),np.full(3,.12*DT),
            np.minimum(np.radians(88),previous+np.radians(10)*DT)]
        cache = {}

        def skew(v):
            return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

        def evaluate(x):
            if 'x' in cache and np.array_equal(cache['x'],x):
                return cache
            d.qpos[:3] = root_position+x[:3]
            quat = (Rotation.from_rotvec(x[3:6])*root_rotation).as_quat()
            d.qpos[3:7] = quat[[3,0,1,2]]
            d.qpos[self.qa] = x[6:]
            mujoco.mj_fwdPosition(self.m,d)
            angle = np.linalg.norm(x[3:6]);hat = skew(x[3:6])
            left = np.eye(3)+.5*hat+hat@hat/6 if angle<1e-6 else (
                np.eye(3)+(1-math.cos(angle))/angle**2*hat+(angle-math.sin(angle))/angle**3*(hat@hat))
            jp,jr = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
            def point_jac(point,body):
                mujoco.mj_jac(self.m,d,jp,jr,point,body)
                return np.column_stack([np.eye(3),-skew(point-d.qpos[:3])@left,jp[:,self.va]])
            points = np.vstack([d.site('s_head').xpos,
                *[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
            jac = np.asarray([point_jac(point,self.head+min(node,20)) for node,point in enumerate(points)])
            surface = np.asarray([d.xpos[body]+d.xmat[body].reshape(3,3)@local for body,local,*_ in contacts]).reshape(-1,3)
            surface_jac = np.asarray([point_jac(point,record[0]) for point,record in zip(surface,contacts)]).reshape(-1,3,26)
            fractions = np.linspace(0.,1.,9)
            lines = (points[:-1,None]*(1-fractions[None,:,None])+points[1:,None]*fractions[None,:,None]).reshape(-1,3)
            line_jac = (jac[:-1,None]*(1-fractions[None,:,None,None])+jac[1:,None]*fractions[None,:,None,None]).reshape(-1,3,26)
            gaps = self.pipe_geometry.signed(lines)
            active = gaps<.003
            gap_jac = np.zeros((len(lines),26))
            if active.any():
                step = 1e-6
                gradient = np.column_stack([(self.pipe_geometry.signed(lines[active]+np.eye(3)[axis]*step)-
                    self.pipe_geometry.signed(lines[active]-np.eye(3)[axis]*step))/(2*step) for axis in range(3)])
                gap_jac[active] = -8.*np.einsum('ni,nij->nj',gradient,line_jac[active])
            cache.update(x=x.copy(),points=points.copy(),jac=jac,surface=surface,surface_jac=surface_jac,
                         gap_error=8.*np.maximum(0.,.003-gaps),gap_jac=gap_jac)
            return cache

        regularizer = .075
        holders = [(side,int(i)) for side in range(2) for i in np.flatnonzero(
            masks[side]&(self.current_support_forces[side]>.1))]
        guide_centers = .5*(guide[:-1]+guide[1:])
        grip_radii = []
        for side,i in holders:
            axes = [1,2] if side==0 else [0,2]
            origin = [0.,1.] if side==0 else [1.1,1.]
            grip_radii.append(float(np.linalg.norm(guide_centers[i,axes]-origin)))
        def grip_terms(points,jac):
            centers = .5*(points[:-1]+points[1:])
            values,rows = [],[]
            for (side,i),radius in zip(holders,grip_radii):
                axes = [1,2] if side==0 else [0,2]
                origin = [0.,1.] if side==0 else [1.1,1.]
                radial = centers[i,axes]-origin
                length = max(float(np.linalg.norm(radial)),1e-12)
                values.append(3.*(length-radius))
                rows.append(3.*(radial/length)@(.5*(jac[i]+jac[i+1]))[axes])
            return np.asarray(values),np.asarray(rows).reshape(-1,26)
        anchors = np.asarray([record[2] for record in contacts]).reshape(-1,3)
        tangent_scale = float(getattr(self,'contact_tangent_constraint_scale',1.))
        contact_projection = np.asarray([np.eye(3) if tangent_scale==1. else
            np.outer(record[5],record[5])+tangent_scale*(np.eye(3)-np.outer(record[5],record[5]))
            for record in contacts]).reshape(-1,3,3)
        rolling_weight = 1./DT
        nominal_root = np.r_[nominal_position-root_position,nominal_rotation]
        root_weight = np.r_[np.full(3,.25),np.full(3,.010)]
        def residual(x):
            state = evaluate(x)
            grip,_ = grip_terms(state['points'],state['jac'])
            return np.r_[(state['points']-guide).ravel(),regularizer*(x[6:]-prior),grip,
                rolling_weight*np.einsum('nij,nj->ni',contact_projection,state['surface']-anchors).ravel(),root_weight*(x[:6]-nominal_root),state['gap_error']]
        def jacobian(x):
            state = evaluate(x)
            _,grip_jac = grip_terms(state['points'],state['jac'])
            return np.vstack([state['jac'].reshape(66,26),
                np.column_stack([np.zeros((20,6)),regularizer*np.eye(20)]),grip_jac,
                rolling_weight*np.einsum('nij,njk->nik',contact_projection,state['surface_jac']).reshape(-1,26),
                np.column_stack([np.diag(root_weight),np.zeros((6,20))]),state['gap_jac']])
        initial = np.clip(np.r_[nominal_root,previous],lower+1e-12,upper-1e-12)
        solved = least_squares(residual,initial,jac=jacobian,
                               bounds=(lower,upper),max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        state = evaluate(solved.x);points = state['points']
        self.fk_root_pose = d.qpos[:7].copy()
        return solved.x[6:].copy(),points.copy(),dict(
            fit_node_rms_mm=float(np.sqrt(np.mean((points-guide)**2))*1000),
            fit_maximum_node_error_mm=float(np.linalg.norm(points-guide,axis=1).max()*1000),
            maximum_fitted_joint_deg=float(np.degrees(np.abs(solved.x[6:])).max()),
            maximum_fk_link_error_mm=float(np.max(np.abs(np.linalg.norm(np.diff(points,axis=0),axis=1)-LINK))*1000),
            fixed_material_prior_max_error_deg=float(np.degrees(np.abs(solved.x[6:]-prior)).max()),
            registered_surface_point_count=len(contacts),
            registered_support_modules=sorted(registered),
            contact_tangent_constraint_scale=tangent_scale,
            maximum_virtual_root_translation_step_mm=float(np.linalg.norm(solved.x[:3])*1000),
            virtual_root_rotation_step_deg=float(np.degrees(np.linalg.norm(solved.x[3:6]))),
            maximum_registered_surface_velocity_mm_s=float(np.linalg.norm(state['surface']-anchors,axis=1).max()*1000/DT) if contacts else None,
            material_frame='initial material gauge with independently integrated phase; optimized virtual root never changes integral prior gauge',
            optimizer_evaluations=solved.nfev,optimizer_status=solved.message,
            reference='exact original robot FK with all 20 alternating hinge axes; virtual registration only')

    def command(self,data,t,forces,torque,support_forces=None):
        self.current_control_time = self.coordination_time = t
        p,centers = self.observe(data)
        q = data.qpos[self.qa]
        support = np.asarray(forces if support_forces is None else support_forces)
        self.current_support_forces = support.copy()
        if self.nominal_support_goals is None:
            self.nominal_support_goals = np.maximum(support.sum(1),1.)
        self.support_force_goals = self.nominal_support_goals.copy()
        if self.filtered_pressure_forces is not None:
            self.filtered_pressure_forces += .1*(support-self.filtered_pressure_forces)
        self.filtered_load += .1*(np.abs(torque)-self.filtered_load)
        if self.first_dispatch:
            self.torque_feedforward = np.asarray(torque).copy()
        else:
            self.torque_feedforward += .1*(np.asarray(torque)-self.torque_feedforward)
        mask = physical_wrap_masks(centers)[1]
        self.incoming_age = np.where((support[1]>.1)&mask,self.incoming_age+DT,0.)
        for module in np.flatnonzero(self.incoming_age>=1.):
            if module>=8 and int(module) not in self.transferred_material_modules:
                self.transferred_material_modules.append(int(module))
                self.events.append(dict(t_s=t,mode='actual_articulated_target_handover',module=int(module)))
        actual_material = self.observe_progress(centers,forces,t)
        self.last_body_progress = actual_material
        material = self.physical_pipe_progress/self.physical_axial_sines
        measured = float(material[1] if self.target_carries_transfer(centers,support) else min(material))
        lag = self.shift-measured
        tracking = np.abs(self.last-q)
        source_mask,target_mask = physical_wrap_masks(centers)
        posterior_count = np.count_nonzero((support[0]>.1)&source_mask)
        receiving_count = np.count_nonzero((support[1]>.1)&target_mask)
        support_scale = float(receiving_count>=2 and (posterior_count>=2 or self.target_carries_transfer(centers,support)))
        geometric_tracking = np.abs(self.joint_geometry-q)
        error_scale = float(np.clip((np.radians(18)-geometric_tracking.max())/np.radians(14),0.,1.))
        requested = self.coordinated_feed_limit(lag,error_scale,support_scale,
                                                self.physical_pipe_speed/self.physical_axial_sines)
        self.speed += np.clip(requested-self.speed,-.002*DT,.001*DT)
        before,reference,locations = self.curve.reference(self.shift)
        saved = copy.deepcopy(self.curve.__dict__)
        for fraction in (1.,.5,.25,.125,.0625,.03125):
            if fraction != 1.:
                self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(saved))
            try:
                self.feedback_shape(p,q,reference,locations,tracking,forces,DT*fraction)
                self.curve.advance_planned_guidance(self.shift,DT*fraction,p,support,tracking,self.filtered_load)
                shaped,shaped_c,_ = self.curve.reference(self.shift)
                if np.linalg.norm(shaped-before,axis=1).max()>.008*DT+1e-8:
                    continue
                break
            except ValueError:
                continue
        else:
            self.curve.__dict__.clear();self.curve.__dict__.update(saved)
            shaped,shaped_c,_ = self.curve.reference(self.shift)
        phase_saved = copy.deepcopy(self.curve.__dict__)
        ds = self.speed*DT
        for _ in range(12):
            self.curve.__dict__.clear();self.curve.__dict__.update(copy.deepcopy(phase_saved))
            self.curve.increment(ds)
            try:
                guide,guide_centers,locations = self.curve.reference(self.shift+ds)
                if np.linalg.norm(guide-before,axis=1).max()<=.008*DT+1e-8:
                    break
            except ValueError:
                pass
            ds *= .5
        else:
            ds = 0.
            self.curve.__dict__.clear();self.curve.__dict__.update(phase_saved)
            guide,guide_centers,locations = self.curve.reference(self.shift)
        wanted_axis = guide[0]-guide[1]
        wanted_axis /= np.linalg.norm(wanted_axis)
        previous_axis = self.root_orientation[:,2]
        cross = np.cross(previous_axis,wanted_axis)
        sine = np.linalg.norm(cross)
        transport = Rotation.from_rotvec(cross*math.atan2(sine,float(previous_axis@wanted_axis))/max(sine,1e-12)).as_matrix()
        nominal_y = float(guide_centers[0,1]-shaped_c[0,1])
        spin_step = float(np.clip(np.sign(self.curve.theta_rates[1])*nominal_y /
            (BODY_RADIUS*math.sqrt(1-self.curve.axial_sines[1]**2)),-.12*DT,.12*DT))
        orientation = Rotation.from_rotvec(wanted_axis*spin_step).as_matrix()@transport@self.root_orientation
        previous_fk_root = self.fk_root_pose.copy()
        previous_integral = self.integral_joint_prior.copy()
        geometric,body,fit = self.fit_articulated_body(data,guide,orientation)
        clearance = float(self.pipe_geometry.link_minimum(body,65).min())
        if clearance <= 0.:
            self.curve.__dict__.clear();self.curve.__dict__.update(saved)
            ds = 0.
            guide,guide_centers,locations = self.curve.reference(self.shift)
            geometric = self.joint_geometry.copy()
            body = self.fk_reference_points.copy()
            orientation = self.root_orientation.copy()
            self.fk_root_pose = previous_fk_root
            self.integral_joint_prior = previous_integral
            spin_step = 0.
            attempted = fit.copy()
            fit = dict(fit_node_rms_mm=float(np.sqrt(np.mean((body-guide)**2))*1000),
                fit_maximum_node_error_mm=float(np.linalg.norm(body-guide,axis=1).max()*1000),
                maximum_fk_link_error_mm=float(np.max(np.abs(np.linalg.norm(np.diff(body,axis=0),axis=1)-LINK))*1000),
                maximum_fitted_joint_deg=float(np.degrees(np.abs(geometric)).max()),
                candidate_rejected='exact FK body crosses original pipe solid',candidate_fit=attempted)
            clearance = float(self.pipe_geometry.link_minimum(body,65).min())
        self.shift += ds
        self.speed = ds/DT
        self.root_material_phase += spin_step
        self.root_orientation = orientation
        self.joint_geometry = geometric
        self.fk_reference_points = body
        gain = self.m.actuator_gainprm[:,0]
        bias = np.clip(self.torque_feedforward,-2.355,2.355)/gain
        raw = np.clip(geometric+bias,q-np.radians(3),q+np.radians(3))
        target = np.clip(raw,np.maximum(-np.radians(88),self.last-np.radians(10)*DT),
                         np.minimum(np.radians(88),self.last+np.radians(10)*DT))
        if self.first_dispatch:
            target = self.last.copy()
            self.first_dispatch = False
        self.last = target.copy()
        fk_centers = .5*(body[:-1]+body[1:])
        info = dict(stage='articulated_shared_shift',shift_mm=self.shift*1000,shift_speed_mm_s=self.speed*1000,
            reference_points_mm=(body*1000).tolist(),curve_guide_points_mm=(guide*1000).tolist(),
            reference_lead_mm=lag*1000,actual_material_estimate_mm=actual_material*1000,
            material_locations_mm=(locations*1000).tolist(),material_region_codes=self.curve.regions(locations).tolist(),
            reference_source_region_mask=self.source_span(locations).tolist(),
            reference_node_arclength_mm=(self.curve.reference_node_arc*1000).tolist(),
            maximum_reference_link_error_mm=fit['maximum_fk_link_error_mm'],
            reference_pipe_clearance_mm=clearance*1000,
            body_curve_rms_error_mm=float(np.sqrt(np.mean((fk_centers-centers)**2))*1000),
            tracking_error_deg=float(np.degrees(tracking.max())),
            fixed_front_y_mm=float(np.median(centers[:7,1]-self.initial_centers[:7,1])*1000),
            fixed_rear_x_mm=float(np.median(centers[14:,0]-self.initial_centers[14:,0])*1000),
            curve_parameters=self.curve.parameters(),global_route_guidance=self.curve.route_guidance_last_step,
            articulated_fit=fit,joint_geometry_reference_deg=np.degrees(geometric).tolist(),
            shift_integral_joint_prior_deg=np.degrees(self.integral_joint_prior).tolist(),
            fk_reference_root_pose=self.fk_root_pose.tolist(),root_material_phase_rad=self.root_material_phase,
            servo_torque_feedforward_Nm=self.torque_feedforward.tolist(),physics_execution_prediction=None,
            curve_pressure_feedback=self.pressure_feedback,
            position_execution='exact FK position reference plus measured servo-load compensation, bounded directly',
            dynamic_grip_bias_deg=np.zeros(20).tolist(),transferred_material_modules=self.transferred_material_modules.copy(),
            coordination=dict(phase='direct_exact_body_fit',reservoir_height_mm=0.,feed_policy=self.direct_feedback),
            limitation='virtual FK root is a reference registration only; actual root never assigned; world/contact feedback is simulation truth')
        return target,info

    def load_state(self,path):
        elapsed = super().load_state(path)
        for key in ['joint_geometry','root_orientation','torque_feedforward','fk_reference_points','fk_root_pose','initial_material_normals_local','integral_joint_prior','pressure_profile_locations','pressure_profile_forces','filtered_pressure_forces']:
            if getattr(self,key) is not None:
                setattr(self,key,np.asarray(getattr(self,key)))
        return elapsed
