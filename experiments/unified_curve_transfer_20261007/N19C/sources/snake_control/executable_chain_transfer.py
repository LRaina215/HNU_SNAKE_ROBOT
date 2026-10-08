"""Fit the real alternating hinge chain to the compact common curve window.

Twenty hinge variables only. A measured carrying rod locates the prediction,
with prescribed curve sliding/rolling; there are no free root fit variables.
Only ctrl is sent to the physical simulation. Integral angles seed the first
solve and are recorded, but never occur in its objective or constraints.
"""
import time
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from snake_control.compact_helix_transfer import CompactHelixTransfer
from snake_control.unified_curve_transfer import DT,LINK,physical_wrap_masks,cylindrical_contact,contact_curve_angular_motion


def skew(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])


class ExecutableChainTransfer(CompactHelixTransfer):
    def __init__(self,model,data):
        super().__init__(model,data,skin_aware=False)
        self.chain_feedback={}
        self.chain_fit_points=np.zeros((22,3))
        self.chain_fit_root=data.qpos[:7].copy()
        self.chain_integral_seed=np.zeros(20)
        self.chain_fit_initialized=False
        self.chain_anchor_previous=None
        self.chain_bearing_age=np.zeros((2,21))
        self.chain_rate_bounded=True

    def _points(self,d):
        return np.vstack([d.site('s_head').xpos,
            *[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])

    def _problem(self,data,support,guide,orientation,t):
        actual_points,actual_centers=self.observe(data)
        actual_root=data.qpos[:7].copy()
        actual_rotation=Rotation.from_quat(actual_root[3:7][[1,2,3,0]]).as_matrix()
        masks=np.asarray(physical_wrap_masks(actual_centers))
        bearings=(support>.1)&masks
        self.chain_bearing_age=np.where(bearings,self.chain_bearing_age+DT,0.)
        stable=bearings&(self.chain_bearing_age>=.5)
        if t<.5:stable=bearings.copy()
        ids=np.flatnonzero(stable[1])
        if len(ids):
            side=1;posterior=ids[ids>=2];candidates=posterior if len(posterior) else ids
        else:
            side=0;ids=np.flatnonzero(stable[0]);candidates=ids
        old=self.chain_anchor_previous
        if old is not None and old[0]==side and old[1] in candidates:module=int(old[1])
        elif len(candidates):module=int(candidates[-1])
        else:
            # Contact count is never itself a failure condition. This fallback
            # has an observed root, without optimization or physical assignment.
            side=None;module=0
        anchor_key=(side,module)
        delta=.5*(guide[:-1]+guide[1:])-.5*(self.reference_before_execution[:-1]+self.reference_before_execution[1:])
        anchor_step=delta[module].copy() if side is not None else np.zeros(3)
        length=float(np.linalg.norm(anchor_step))
        anchor_step*=min(1.,.008*DT/max(length,1e-12))
        anchor_rotation=self.d.xmat[self.head+module].reshape(3,3).copy()
        omega=np.zeros(3);rolling_scale=1.;lever=None
        best_contact=None
        contacts=[];seen=set()
        for contact in data.contact:
            for pipe,robot,sign in [(contact.geom1,contact.geom2,1.),(contact.geom2,contact.geom1,-1.)]:
                body=int(self.m.geom_bodyid[robot]);name=self.m.geom(pipe).name or ''
                pipe_side=0 if name in {'pipe_straight_01','pipe_sharp_in_02'} else 1 if name in {'pipe_sharp_out_03','pipe_straight_04'} else None
                if not self.head<=body<self.head+21 or pipe_side is None:continue
                i=body-self.head
                if not bearings[pipe_side,i] or not cylindrical_contact(pipe_side,contact.pos,contact.frame[:3]):continue
                if i==module and pipe_side==side and (best_contact is None or contact.dist<best_contact[0]):
                    best_contact=(float(contact.dist),contact.pos.copy())
                if (pipe_side,i) in seen:continue
                seen.add((pipe_side,i))
                normal=sign*contact.frame[:3].copy()
                local=self.d.xmat[body].reshape(3,3).T@(contact.pos-self.d.xpos[body])
                # Only the pressure/curve-normal component is prescribed.
                wanted=contact.pos+normal*float(normal@delta[i])
                contacts.append((body,local,wanted,normal))
        if side is not None and best_contact is not None:
            lever=best_contact[1]-actual_centers[module]
            origin=np.array([0.,0.,1.]) if side==0 else np.array([1.1,0.,1.])
            radial=actual_centers[module]-origin;radial[side]=0.;radial/=max(np.linalg.norm(radial),1e-12)
            pressure_velocity=radial*float(radial@anchor_step)/DT
            omega,rolling_scale=contact_curve_angular_motion(lever,anchor_rotation[:,2],anchor_step/DT,pressure_velocity,np.zeros(3),side)
        desired_center=actual_centers[module]+anchor_step
        desired_rotation=Rotation.from_rotvec(omega*DT).as_matrix()@anchor_rotation
        allowance=np.clip(self.pipe_skin_penetration(data)+.0005,.00075,.002)
        seed,locations,roll=self.integral_targets(guide,orientation,np.zeros(2))
        self.chain_integral_seed=seed.copy()
        cache={};d=self.fit_data
        jp,jr=np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
        node_weights=np.ones(22);node_weights[:9]=2.

        def evaluate(q):
            if 'q' in cache and np.array_equal(q,cache['q']):return cache
            d.qpos[:]=data.qpos;d.qpos[self.qa]=q
            mujoco.mj_fwdPosition(self.m,d)
            points=self._points(d);center=.5*(points[module]+points[module+1])
            register=desired_rotation@d.xmat[self.head+module].reshape(3,3).T
            translation=desired_center-register@center
            d.qpos[:3]=register@actual_root[:3]+translation
            d.qpos[3:7]=Rotation.from_matrix(register@actual_rotation).as_quat()[[3,0,1,2]]
            mujoco.mj_fwdPosition(self.m,d)
            points=self._points(d)
            mujoco.mj_jac(self.m,d,jp,jr,desired_center,self.head+module)
            center_jac=jp[:,self.va].copy();angular_jac=jr[:,self.va].copy()
            def point_jac(point,body):
                mujoco.mj_jac(self.m,d,jp,jr,point,body)
                return jp[:,self.va]-center_jac+skew(point-desired_center)@angular_jac
            jac=np.asarray([point_jac(point,self.head+min(i,20)) for i,point in enumerate(points)])
            normal_values=[];normal_jac=[]
            for body,local,wanted,normal in contacts:
                point=d.xpos[body]+d.xmat[body].reshape(3,3)@local
                normal_values.append(2.*float(normal@(point-wanted)))
                normal_jac.append(2.*normal@point_jac(point,body))
            values=np.r_[(node_weights[:,None]*(points-guide)).ravel(),normal_values]/LINK
            rows=np.vstack([(node_weights[:,None,None]*jac).reshape(66,20),np.asarray(normal_jac).reshape(-1,20)])/LINK
            gaps=np.r_[allowance+.010,np.full(21,.011)]
            gap_jac=np.zeros((42,20));pipe_depth=np.zeros(21);self_depth=np.zeros(21)
            for contact in d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[contact.geom1,contact.geom2]])
                one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                normal=contact.frame[:3];point=contact.pos
                if one!=two:
                    pipe=contact.geom2 if one else contact.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                    body=b1 if one else b2;i=body-self.head
                    gap=float(contact.dist)+allowance[i]
                    pipe_depth[i]=max(pipe_depth[i],-float(contact.dist))
                    if gap<gaps[i]:
                        gaps[i]=gap;gap_jac[i]=(-1. if one else 1.)*normal@point_jac(point,body)
                elif one and two and abs(b1-b2)>1:
                    i=min(b1,b2)-self.head;gap=float(contact.dist)+.001
                    self_depth[i]=max(self_depth[i],-float(contact.dist))
                    if gap<gaps[21+i]:
                        gaps[21+i]=gap;gap_jac[21+i]=normal@(point_jac(point,b2)-point_jac(point,b1))
            cache.clear();cache.update(q=q.copy(),points=points.copy(),jac=jac,root=d.qpos[:7].copy(),
                residual=values,residual_jac=rows,gaps=gaps/LINK,gap_jac=gap_jac/LINK,
                pipe_depth=pipe_depth,self_depth=self_depth,objective=.5*float(values@values))
            return cache

        metadata=dict(anchor_module=module,anchor_pipe_side=side,anchor_changed=old!=anchor_key,
            anchor_step_mm=(anchor_step*1000).tolist(),anchor_rolling_angular_velocity_rad_s=omega.tolist(),
            anchor_rolling_scale=rolling_scale,anchor_contact_lever_mm=(lever*1000).tolist() if lever is not None else None,
            stable_target_bearing_modules=np.flatnonzero(stable[1]).tolist(),stable_source_bearing_modules=np.flatnonzero(stable[0]).tolist(),
            actual_pipe_skin_penetration_mm=(self.pipe_skin_penetration(data)*1000).tolist(),
            pipe_skin_allowance_mm=(allowance*1000).tolist(),normal_only_contact_terms=len(contacts))
        return evaluate,seed,locations,roll,anchor_key,metadata

    def execution_reference(self,data,t,support,guide,orientation,shift):
        started=time.perf_counter()
        evaluate,seed,locations,roll,anchor_key,metadata=self._problem(data,support,guide,orientation,t)
        limit=np.radians(88.)
        previous=np.clip(self.joint_geometry,-limit,limit)
        step=np.radians(10.)*DT
        lower=np.maximum(-limit,previous-step);upper=np.minimum(limit,previous+step)
        warm=previous.copy()
        # An integral initialization is used once; subsequent solves continue
        # the physical-chain solution, rather than chasing an integral prior.
        if not self.chain_fit_initialized:warm=np.clip(seed,lower,upper)
        initial=evaluate(warm);initial_objective=initial['objective'];initial_feasible=initial['gaps'].min()>=-1e-7
        best_q=None;best_value=float('inf');evaluations=0
        def consider(q):
            nonlocal best_q,best_value,evaluations
            state=evaluate(q);evaluations+=1
            joint_feasible=bool(np.all(q>=lower-1e-10) and np.all(q<=upper+1e-10))
            if joint_feasible and state['gaps'].min()>=-1e-7 and state['objective']<best_value:
                best_q=q.copy();best_value=state['objective']
            return state
        consider(warm)
        actual=np.clip(data.qpos[self.qa],lower,upper)
        consider(actual)
        if not initial_feasible:warm=actual.copy()
        def objective(q):return consider(q)['objective']
        def gradient(q):
            s=evaluate(q);return s['residual_jac'].T@s['residual']
        result=minimize(objective,warm,jac=gradient,bounds=list(zip(lower,upper)),
            constraints=[dict(type='ineq',fun=lambda q:evaluate(q)['gaps'],jac=lambda q:evaluate(q)['gap_jac'])],
            method='SLSQP',options=dict(maxiter=24,ftol=1e-8))
        consider(result.x)
        feasible=best_q is not None
        chosen=best_q if feasible else previous
        final=evaluate(chosen)
        self.chain_fit_points=final['points'].copy();self.chain_fit_root=final['root'].copy()
        self.chain_fit_initialized=True;self.chain_anchor_previous=anchor_key
        self.chain_feedback=dict(action='20-variable exact alternating-hinge chain fits all common-window nodes; original collision shells constrained; integral angles initialize only',
            solver='SLSQP, analytic bearing-registered FK and collision-distance derivatives',
            free_virtual_root_variables=0,strong_integral_prior=False,physical_root_assigned=False,
            contact_tangent_and_axial_zero_constraints=False,front_node_weight=2.,other_node_weight=1.,
            solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),
            iterations=int(result.nit),evaluations=evaluations,elapsed_wall_s=time.perf_counter()-started,
            accepted_feasible_candidate=feasible,initial_objective=initial_objective,final_objective=final['objective'],
            joint_candidate_step_bound_deg=float(np.degrees(step)),
            maximum_joint_candidate_step_deg=float(np.degrees(np.abs(chosen-previous)).max()),
            geometry_continuity='joint reference itself advances at most 0.5deg per 50ms; original final ctrl limits and encoder anti-windup also remain',
            guide_fit_rms_mm=float(np.sqrt(np.mean((final['points']-guide)**2))*1000),
            maximum_guide_chord_error_mm=float(np.abs(np.linalg.norm(np.diff(guide,axis=0),axis=1)-LINK).max()*1000),
            maximum_exact_FK_rod_length_error_mm=float(np.abs(np.linalg.norm(np.diff(final['points'],axis=0),axis=1)-LINK).max()*1000),
            minimum_shell_constraint_margin_mm=float(final['gaps'].min()*LINK*1000),
            predicted_pipe_skin_penetration_mm=(final['pipe_depth']*1000).tolist(),
            predicted_nonadjacent_self_penetration_mm=(final['self_depth']*1000).tolist(),
            maximum_fit_minus_integral_seed_deg=float(np.degrees(np.abs(chosen-seed)).max()),
            integral_seed_deg=np.degrees(seed).tolist(),registered_fit_root_pose=self.chain_fit_root.tolist(),
            prediction_location='current bearing rod plus bounded curve displacement and contact-derived rolling; root algebraically registered, never freely optimized',
            feedback_truth='simulation world nodes, carrying-body frame, actual cylinder loads and original collision geometry; initial material gauge is retained',
            **metadata)
        self.wrap_feedback=dict(stable_target_bearing_modules=metadata['stable_target_bearing_modules'],
            physical_FK_prediction_registration=dict(anchor_module=metadata['anchor_module'],assumption=self.chain_feedback['prediction_location']),
            physical_FK_goal_residual_after_rms_mm=self.chain_feedback['guide_fit_rms_mm'],parameters=self.wrap_parameters.tolist())
        return guide,orientation,chosen,locations,roll

    def command(self,data,t,forces,torque,support_forces=None):
        saved_shift=self.shift;saved_target=self.last.copy();saved_geometry=self.joint_geometry.copy()
        saved_orientation=self.root_orientation.copy()
        target,info=super().command(data,t,forces,torque,support_forces)
        if not self.chain_feedback['accepted_feasible_candidate']:
            # The attempted guide and shell diagnostics remain explicit, but
            # neither its window increment nor its ctrl is physically committed.
            attempted=info['shift_mm'];self.shift=saved_shift;self.speed=0.
            self.last=saved_target;self.joint_geometry=saved_geometry;self.root_orientation=saved_orientation
            target=saved_target.copy();guide,_,_=self.curve.reference(self.shift)
            info.update(shift_mm=self.shift*1000,shift_speed_mm_s=0.,curve_guide_points_mm=(guide*1000).tolist(),
                joint_geometry_reference_deg=np.degrees(saved_geometry).tolist())
            info['fixed_spatial_shift'].update(accepted_speed_mm_s=0.,candidate_fit_rejected=True)
            self.chain_feedback.update(attempted_shift_mm=attempted,committed_shift_mm=self.shift*1000,
                no_feasible_candidate_policy='retain previous dispatch and common shift; do not count nominal failed advance as action')
        else:self.chain_feedback['candidate_fit_rejected']=False
        # Parent dispatch retains original anti-windup, rate and amplitude
        # limits. Correct its legacy integral naming for this distinct fit.
        self.integral_joint_prior=self.chain_integral_seed.copy()
        self.fk_reference_points=self.chain_fit_points.copy();self.fk_root_pose=self.chain_fit_root.copy()
        info['stage']='executable_chain_compact_common_window'
        info['reference_points_mm']=(self.chain_fit_points*1000).tolist()
        info['executable_chain']=self.chain_feedback.copy()
        info['fixed_spatial_shift']['action']='common window on one compact spatial curve; fitted real hinge-chain geometry generates 20 limited ctrl'
        info['fixed_spatial_shift']['chain_geometry_tracking_error_deg']=info['fixed_spatial_shift']['full_integral_tracking_error_deg']
        return target,info
