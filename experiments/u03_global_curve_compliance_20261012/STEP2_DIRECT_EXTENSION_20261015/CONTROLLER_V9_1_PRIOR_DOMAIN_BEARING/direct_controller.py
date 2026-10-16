"""G -> local geometric C -> explicit local modes -> original20 native hinges.

alpha/beta cannot advance source feed or material rolling; S starts at capture.
The only actual-state write belongs to the external physical servo runner.
"""
from types import SimpleNamespace
import numpy as np
from physics import mujoco,nodes
from geometry import Geometry,normalize
from rough_planner import RoughPlanner,LINK
from post_capture_curve import PostCaptureCurve
from support_policy import SourcePolicy
from contact_sensor import read_sensor,classified_observation,known_source
from cylinder_estimator import CylinderEstimator
from estimated_observer import EstimatedObserver
from direct_updater import DirectUpdater
from native_mapping import NativeMapping
from source_mode import reconstruct,SourceMode

class LocalReference(SimpleNamespace,SourceMode):
    def observe(self,*args):pass  # Receiving radial feedback is evaluated explicitly below.

def probe_command(t,first_contact,p):
    probe=np.zeros(2)
    if first_contact is None:return probe
    age=t-first_contact;period=p['direction_duration_s']
    if 0<=age<2*period:
        j=int(age//period);probe[j]=p['amplitude_per_m']*np.sin(2*np.pi*(age-j*period)/period)
    return probe

class DirectController(SourcePolicy):
    def __init__(self,m,d,registration,condition,write_global):
        self.m=m;self.reg=registration;self.p=registration['direct_extension'];self.last_t=None;self.S=0.;self.alpha=0.;self.beta=0.;self.theta=0.;self.phase='source_hold'
        self.source_ready=False;self.settle_age=0.;self.failure=None;self.first_release_t=None;self.first_capture_t=None;self.low_support_age=0.;self.alpha_finished_t=None
        self.support_tracks={};self.unknown_candidates={};self.source_chart_seen=np.zeros(21,bool);self.current_supply_materials=set();self.front_relief=np.zeros(2)
        geometry=Geometry.from_dict(registration['conditions'][condition]);self.planner=RoughPlanner(geometry,d.qpos.copy(),d.ctrl.copy(),registration['kinematic_solver'])
        self.ref=LocalReference(geometry=geometry.copy(),probe=np.zeros(2),front_relief=np.zeros(2),pitch=.18266863819422827,a=registration['source_constants']['a'],feed_per_cycle=registration['source_constants']['feed_per_cycle'],nominal_frequency_hz=registration['nominal_frequency_hz'],feed_early=.054,feed_late=.09,
            source_gamma=0.,source_direction=np.array([0.,1.]),source_gradient=0.,source_requested=0.)
        self.ref.feed_distance=NativeMapping.feed_distance.__get__(self.ref,LocalReference)
        self.estimator=CylinderEstimator(geometry,registration['source_known'],registration['estimator']);self.observer=EstimatedObserver(self.ref,registration['source_known']);self.updater=DirectUpdater(registration['global_update'],write_global)
        self.post=None;self.initial_preload=d.ctrl.copy()-self.planner.initial_q;self.raw=d.ctrl.copy();self.info=self.base_info(self.planner.initial_q,self.planner.given,d.qpos,0.)
        self.stable_source_P=None;self.stable_source_R=None;self.capture_geometric_q=None;self.delta=0.;self.filtered_gap=0.;self.filtered_load=0.;self.last_geo=self.planner.initial_q.copy()
        self.bearing_motion_start=None;self.bearing_motion_q=None;self.bearing_motion_command=None;self.bearing_motion_evidence=False
        self.observer.allow_prior_capture=True
        self.applied_probe_field=np.zeros_like(self.planner.coordinates.field(self.raw));self.applied_probe=np.zeros(2)
        self.probe_contact_clock=0.;self.previous_probe_request=np.zeros(2)
        self.transmitted_geometric=self.planner.initial_q.copy();self.last_command_transmission_factor=1.;self.geometric_progress_ready=True;self.sent_geometric_head_tracking_error_m=0.;self.actual_head_task_error_m=0.
        self.given_source_P=d.xpos[self.m.body('head').id:self.m.body('head').id+21].copy()
    def base_info(self,q,G,qpos,t):
        coordinates=self.planner.coordinates;field=coordinates.field(q);C=self.planner.fk(q);T=normalize(C[1]-C[0]);R=self.planner.d.xmat[self.planner.head].reshape(3,3)
        N=R[:,0]-R[:,0]@T*T;N=normalize(N);frame=np.column_stack([T,N,np.cross(T,N)]);dense,bishop=reconstruct(field,coordinates.bounds,frame,C[0],coordinates.material)
        sign=self.p['post_material_roll_sign']
        return dict(S=self.S,chi=sign*self.S/self.ref.feed_per_cycle,roll_phase=sign*2*np.pi*self.S/self.ref.feed_per_cycle,
            alpha=self.alpha,beta=self.beta,controller_phase=self.phase,source_commanded_feed_m=0. if self.post is None else self.ref.feed_distance(self.S),source_commanded_roll_rad=0. if self.post is None else sign*2*np.pi*self.S/self.ref.feed_per_cycle,
            curvature_field=field,geometric_curvature_field=field.copy(),curvature_bounds_mm=coordinates.bounds,material_arc_nodes_mm=coordinates.material,
            geometric_native_command=q.copy(),raw_native_command=q.copy(),geometric_reference_points=G.copy(),global_reference_points=G.copy(),global_reference_locations=np.arange(21,-1,-1)*LINK,global_target_arc_m=np.nan,spatial_guide_stations_m=np.full(22,np.nan),spatial_guide_stations_valid=False,
            reference_points=C,nominal_curve_points=C,native_exact_C_points=C,nominal_curve_dense=np.array([np.interp(np.linspace(0,21*LINK,257),np.arange(22)*LINK,C[:,j]) for j in range(3)]).T,
            bishop_curve_points=bishop,bishop_curve_dense=dense,reference_frame=frame,material_rotation=np.eye(2),local_curve_chord_error_mm=float(abs(np.linalg.norm(np.diff(C,axis=0),axis=1)-LINK).max()*1000),
            stance_projection_active=self.phase!='source_hold',stance_anchor_material=self.planner.anchor,stance_projection_head_error_mm=float(np.linalg.norm(C[0]-G[0])*1000),stance_projection_accepted=True)
    def command(self,d,t):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0 or dt>.026:raise ValueError('Non-continuous command clock')
        self.last_t=t;phase_progress_fraction=1.;phase_geometric_ready=self.geometric_progress_ready;sensor=read_sensor(self.m,d)
        raw_front_contact=any(ct['material']<8 and not known_source(ct,self.reg['source_known']) for ct in sensor['contacts'])
        probe_tracking=abs(sensor['joint_actual']-self.raw).max()<=np.radians(self.p['progress_tracking_gate_deg'])
        if self.estimator.first_front_contact is not None and raw_front_contact and probe_tracking and abs(self.applied_probe-self.previous_probe_request).max()<=self.reg['probe']['transmission_tracking_tolerance_per_m']:
            self.probe_contact_clock=min(2*self.reg['probe']['direction_duration_s'],self.probe_contact_clock+dt)
        probe=probe_command(self.probe_contact_clock,0. if self.estimator.first_front_contact is not None else None,self.reg['probe']) if raw_front_contact else np.zeros(2)
        self.previous_probe_request=probe.copy();self.ref.probe=probe
        executed_probe=self.applied_probe.copy() if abs(sensor['joint_actual']-self.raw).max()<=np.radians(self.p['progress_tracking_gate_deg']) else np.zeros(2)
        self.estimator.observe(sensor,t,executed_probe);observed=classified_observation(sensor,self.estimator.geometry,self.reg['source_known']);support=self.support(sensor,t,observed)
        weight=self.reg['robot_weight_N'];p=self.reg['support'];supply_ok=self.adequate_supply(support,weight);source=support['known_source_N']
        roles=self.observer.update(observed,self.S,dt,self.estimator.geometry,self.estimator.last.get('reliable',False))
        receiver=sum(ct['world_force_N'][2] for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']) if role['receiver_candidate_consistent'])
        bearing_candidate=bool(self.observer.front_confirmed and roles['front_current_valid'] and receiver>=p['receiver_load_min_fraction_after_capture']*weight)
        if bearing_candidate:
            if self.bearing_motion_start is None:self.bearing_motion_start=t;self.bearing_motion_q=sensor['joint_actual'][:14].copy();self.bearing_motion_command=self.raw[:14].copy()
            dq=sensor['joint_actual'][:14]-self.bearing_motion_q;dc=self.raw[:14]-self.bearing_motion_command;threshold=np.radians(self.reg['capture_motion_evidence']['minimum_actual_fore_motion_deg'])
            self.bearing_motion_evidence=bool(t-self.bearing_motion_start>=.5 and np.max(abs(dq))>=threshold and np.max(abs(dc))>=threshold and dq@dc>0.)
        else:self.bearing_motion_start=None;self.bearing_motion_evidence=False
        capture=bool(bearing_candidate and self.bearing_motion_evidence)
        if not self.source_ready:
            stable=support['all_source'] and supply_ok and p['initial_load_min_fraction']*weight<=source<=p['initial_load_max_fraction']*weight and np.max(abs(d.qvel))<p['initial_velocity_max']
            self.settle_age=self.settle_age+dt if stable else 0.
            if self.settle_age>=p['initial_stable_s']:
                self.source_ready=True;self.first_release_t=t;self.phase='front_release';self.stable_source_P=d.xpos[self.m.body('head').id:self.m.body('head').id+21].copy();self.stable_source_R=d.xmat[self.m.body('head').id:self.m.body('head').id+21].reshape(21,3,3).copy()
            if t>=p['initial_settle_timeout_s'] and not self.source_ready:self.failure='source_stabilization_failed'
        elif self.post is None:
            self.low_support_age=self.low_support_age+dt if not supply_ok else 0.
            if self.low_support_age>=p['rear_low_load_max_s']:self.failure='source_support_invalid_before_capture'
            if t-self.first_release_t>self.p['capture_timeout_after_release_s']:self.failure='front_capture_timeout'
            free=self.p['wrap_hinges'] if self.phase=='front_wrap' else self.p['released_hinges']
            tracking=bool(abs(sensor['joint_actual'][:free]-self.raw[:free]).max()<=np.radians(self.p['progress_tracking_gate_deg']))
            if self.phase=='front_release':
                if tracking:self.alpha=min(1.,self.alpha+phase_progress_fraction*dt/self.p['release_duration_s'])
                if raw_front_contact:
                    self.planner.start_wrap(self.alpha,self.theta);self.phase='front_wrap';self.alpha_finished_t=t
                if self.alpha>=1.:
                    if self.alpha_finished_t is None:self.alpha_finished_t=t
                    if not raw_front_contact and t-self.alpha_finished_t>self.p['contact_timeout_after_alpha_s']:self.failure='no_front_contact_after_extension'
            if self.phase=='front_wrap':
                maximum=1.  # Bounded probing and wrap may proceed before geometry converges.
                if tracking and self.beta<maximum:self.beta=min(maximum,self.beta+phase_progress_fraction*dt/self.p['wrap_duration_s'])
        start=self.p['wrap_hinges'] if self.phase in ['front_wrap','whole_body_feed_and_roll'] else self.p['released_hinges'];guard_start=start;given_delta=d.xpos[self.m.body('head').id+start:self.m.body('head').id+21]-self.given_source_P[start:]
        given_motion=float(np.linalg.norm(given_delta,axis=1).max());given_Xmotion=float(abs(given_delta[:,0]).max());oldphi=np.arctan2(self.given_source_P[start:,2]-1,self.given_source_P[start:,1]);newphi=np.arctan2(d.xpos[self.m.body('head').id+start:self.m.body('head').id+21,2]-1,d.xpos[self.m.body('head').id+start:self.m.body('head').id+21,1]);given_polar=float(abs(np.angle(np.exp(1j*(newphi-oldphi)))).max()*180/np.pi)
        if self.source_ready and self.post is None:
            actualP=d.xpos[self.m.body('head').id+start:self.m.body('head').id+21];drift=float(np.linalg.norm(actualP-self.stable_source_P[start:],axis=1).max());xdrift=float(abs(actualP[:,0]-self.stable_source_P[start:,0]).max())
            actualR=d.xmat[self.m.body('head').id+start:self.m.body('head').id+21].reshape(21-start,3,3);orientation=np.arccos(np.clip((np.einsum('nij,nij->n',self.stable_source_R[start:],actualR)-1)/2,-1,1));rotation=float(orientation.max()*180/np.pi)
        else:drift=0.;xdrift=0.;rotation=0.
        minimum=self.p['wrap_hinges'] if self.beta>0 else self.p['released_hinges']
        loaded=[ct for ct in sensor['contacts'] if ct['material']>=minimum and ct['material'] in self.current_supply_materials and ct['normal_N']>.5]
        if self.post is not None:loaded+=[ct for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']) if role['receiver_candidate_consistent'] and ct['normal_N']>.5]
        anchor=max(loaded,key=lambda ct:ct['normal_N'])['material'] if loaded else max(minimum,self.planner.anchor)
        self.planner.observe(d.qpos.copy(),anchor);self.planner.observe_contacts(sensor['contacts'])
        if self.source_ready and self.post is None:
            # C uses measured posterior joints at its bearing boundary. The
            # compensating local preload below keeps their total commands fixed.
            # Immutable given G and all released-front coordinates are unchanged.
            self.planner.q[minimum:]=sensor['joint_actual'][minimum:]
        if self.source_ready and self.post is None:
            self.planner.command_local_offset=self.raw-self.planner.q
            self.theta=self.updater.update(self.planner,self.estimator,self.alpha,self.beta,self.theta,t,position_cap=self.reg['kinematic_solver']['position_step_m'],joint_step_rad=np.radians(self.reg['servo_slew_deg_s'])*dt);self.ref.geometry=self.planner.geometry.copy()
            geometric,G,metrics=self.planner.solve(self.alpha,self.beta,self.theta,position_cap=self.reg['kinematic_solver']['position_step_m'],joint_step_rad=np.radians(self.reg['servo_slew_deg_s'])*dt);self.last_geo=geometric.copy()
        elif self.post is None:geometric=self.planner.initial_q.copy();G=self.planner.given.copy();metrics={}
        if capture and self.post is None and self.phase=='front_wrap':
            C=self.planner.fk(self.transmitted_geometric);headR=self.planner.d.xmat[self.planner.head].reshape(3,3);T=normalize(C[1]-C[0]);N=normalize(headR[:,0]-headR[:,0]@T*T);frame=np.column_stack([T,N,np.cross(T,N)])
            self.post=PostCaptureCurve(self.planner.given,self.ref.geometry,G,frame,self.reg);self.post.nominal_contact_phase=self.planner.nominal_contact_phase(self.beta);self.capture_theta=self.theta;self.capture_geometric_q=self.transmitted_geometric.copy();self.first_capture_t=t;self.phase='whole_body_feed_and_roll'
        if self.post is not None:
            self.S+=self.ref.feed_per_cycle*self.reg['nominal_frequency_hz']*dt
            self.theta=self.updater.update_post(self.post,self.estimator,self.S,self.theta,self.capture_theta,t);self.ref.geometry=self.post.geometry.copy();self.planner.geometry=self.ref.geometry.copy()
            native,post_info=self.post.command(self.S);blend=NativeMapping.smooth((t-self.first_capture_t)/15.)
            geometric=(1-blend)*self.capture_geometric_q+blend*native;G=post_info['global_reference_points'];metrics=dict(accepted=True,post_native_blend=blend)
        self.info=self.base_info(geometric,G,d.qpos,t)
        if self.post is not None:self.info.update(global_reference_locations=post_info['global_reference_locations'],global_target_arc_m=post_info['global_target_arc_m'],spatial_guide_stations_m=post_info['spatial_guide_stations_m'],spatial_guide_stations_valid=True)
        if self.post is None and self.beta>0:self.info['geometric_reference_points']=self.planner.last_geometric_targets.copy()
        if self.source_ready:self.local_front(d,sensor,dt)
        source_mode=self.ref.observe_all(self.m,d,observed,dict(roles,front_confirmed=capture),self.info,dt,True) if dt>0 else {}
        coordinates=self.planner.coordinates;field=coordinates.field(geometric);bounds=coordinates.bounds;n=coordinates.material;mid=.5*(bounds[:-1]+bounds[1:]);w=np.sin(np.pi*np.clip(mid/n[8],0,1))**2;w[bounds[1:]>n[8]]=0.
        probe_field=w[:,None]*probe[None,:]/1000;force_field=w[:,None]*self.front_relief[None,:]/1000
        # Measured qualified receiver loads adjust only local radial C, never G.
        qualified=roles['estimated_receiver_qualified'];ax,rad,rho=self.estimator.geometry.coordinates(sensor['centers'])
        gap=float(np.mean(rho[qualified]-(self.estimator.geometry.radius+.036))) if qualified.any() else 0.;load=float(np.mean(observed['forces'][1,qualified])) if qualified.any() else 0.
        gain=1-np.exp(-dt/.4);self.filtered_gap+=gain*(gap-self.filtered_gap);self.filtered_load+=gain*(load-self.filtered_load)
        dead=np.sign(self.filtered_gap)*max(0.,abs(self.filtered_gap)-.004);desired=-.06*dead+.00005*max(0.,self.filtered_load-12.)-.1*self.delta if qualified.any() else -.1*self.delta
        self.delta=float(np.clip(self.delta+dt*np.clip(desired,-.00025,.00025),-.002,.002))
        # Radial force relief is represented in the same curvature modes.
        radial_q=np.zeros(20)
        if self.delta and qualified.any():
            C=self.planner.fk(geometric);centers=.5*(C[:-1]+C[1:]);_,rr,rrho=self.estimator.geometry.coordinates(centers)
            nc=20 if self.post is not None else minimum;jpa=np.zeros((3,self.planner.m.nv));jra=np.zeros_like(jpa)
            mujoco.mj_jac(self.planner.m,self.planner.d,jpa,jra,self.planner.snapshot['anchorP'],self.planner.head+self.planner.anchor)
            jpa=jpa[:,self.planner.va[:nc]];jra=jra[:,self.planner.va[:nc]];gradients=[]
            for material in np.flatnonzero(qualified):
                J,_=self.planner.point_J(centers[material],self.planner.head+material,jpa,jra,nc);gradients.append(rr[material]/max(rrho[material],1e-12)@J)
            gradient=np.mean(gradients,axis=0);radial_q[:nc]=np.clip(self.delta*gradient/(gradient@gradient+1e-6),-np.radians(2),np.radians(2))
        preload=self.initial_preload.copy();blend=NativeMapping.smooth(self.alpha/.1)
        if self.source_ready and self.post is None:preload[minimum:]=self.planner.initial_ctrl[minimum:]-geometric[minimum:]
        gravity=self.planner.gravity_torque(geometric,minimum)/127.;preload[:minimum]=(1-blend)*preload[:minimum]+blend*gravity[:minimum]
        if self.post is not None:preload*=1-NativeMapping.smooth((t-self.first_capture_t)/15.)
        preload_field=coordinates.field(preload);radial_field=coordinates.field(radial_q)
        v=np.clip((mid-n[15])/(n[-1]-n[15]),0,1);tail_weight=np.sin(np.pi*v)**2;tail_weight[(bounds[:-1]<n[15])|(bounds[1:]>n[-1])]=0.
        source_field=self.ref.source_gamma*tail_weight[:,None]*self.ref.source_direction[None,:]/1000
        local=probe_field+force_field+preload_field+radial_field+source_field
        raw=coordinates.hinges(field+local);bounded=np.clip(raw,-np.radians(88),np.radians(88));bounded_without_probe=np.clip(raw-coordinates.hinges(probe_field),-np.radians(88),np.radians(88));local+=coordinates.field(bounded-raw);raw=coordinates.hinges(field+local)
        # Curvature rate is limited before the unchanged physical servo slew.
        factor=1.
        if dt>0:
            deltafield=coordinates.field(raw-self.raw);peak=np.linalg.norm(deltafield,axis=1).max()*1000;factor=min(1.,1.5*dt/max(peak,1e-12),np.radians(20)*dt/max(abs(raw-self.raw).max(),1e-12))
            oldfield=coordinates.field(self.raw);cap=.024;a=np.sum(deltafield**2,axis=1);b=2*np.sum(oldfield*deltafield,axis=1);c=np.sum(oldfield**2,axis=1)-cap**2
            roots=(-b+np.sqrt(np.maximum(0.,b*b-4*a*c)))/np.maximum(2*a,1e-30);roots[a<1e-20]=1.
            factor=min(factor,float(np.clip(roots,0,1).min()))
            for _ in range(16):
                step_metrics=self.planner.command_step_metrics(self.raw,self.raw+factor*(raw-self.raw))
                if step_metrics['maximum_geometry_step_m']<=self.reg['kinematic_solver']['position_step_m']+1e-10 and max(step_metrics['measured_plane_excess_compression_m'],step_metrics['known_source_command_excess_compression_m'])<=2e-6:break
                factor*=.5
            else:factor=0.
            raw=self.raw+factor*(raw-self.raw)
        else:step_metrics=dict(maximum_node_step_m=0.,maximum_geometry_step_m=0.,measured_plane_excess_compression_m=0.,known_source_command_excess_compression_m=0.)
        transmitted_probe=coordinates.field(bounded-bounded_without_probe);self.applied_probe_field+=factor*(transmitted_probe-self.applied_probe_field)
        for component in range(2):
            unit=np.zeros_like(probe_field);unit[:,component]=w/1000.;unit=coordinates.field(coordinates.hinges(unit))
            self.applied_probe[component]=float(np.sum(unit*self.applied_probe_field)/max(np.sum(unit*unit),1e-20))
        previous_transmitted_geometric=self.transmitted_geometric.copy();self.transmitted_geometric+=factor*(geometric-self.transmitted_geometric);self.last_command_transmission_factor=factor
        # The next private C solve starts from the geometric component actually sent.
        self.planner.q=self.transmitted_geometric.copy();sent_geometric_field=coordinates.field(self.transmitted_geometric)
        if self.source_ready and self.post is None:
            sent_C=self.planner.fk(self.transmitted_geometric);goal=self.planner.last_geometric_targets[0];self.sent_geometric_head_tracking_error_m=float(np.linalg.norm(sent_C[0]-goal));self.actual_head_task_error_m=float(np.linalg.norm(sensor['points'][0]-goal));self.geometric_progress_ready=bool(max(self.sent_geometric_head_tracking_error_m,self.actual_head_task_error_m)<=self.reg['kinematic_solver']['rough_receiver_uncertainty_m'])
        self.raw=raw;total=coordinates.field(raw);C=self.planner.fk(raw);frame=self.info['reference_frame'];dense,bishop=reconstruct(total,bounds,frame,C[0],n)
        self.info.update(curvature_field=total,reference_points=C,nominal_curve_points=C,native_exact_C_points=C,bishop_curve_points=bishop,bishop_curve_dense=dense,
            nominal_curve_dense=np.array([np.interp(np.linspace(0,21*LINK,257),np.arange(22)*LINK,C[:,j]) for j in range(3)]).T,
            raw_native_command=raw.copy(),local_modes_native_delta_rad=raw-self.transmitted_geometric,probe_curvature_field=probe_field,force_curvature_field=force_field,preload_curvature_field=preload_field,
            receiving_compliance_curvature_field=radial_field,source_force_curvature_field=source_field,command_limit_and_rate_curvature_field=total-sent_geometric_field-local,
            geometric_native_command=self.transmitted_geometric.copy(),geometric_curvature_field=sent_geometric_field,geometric_candidate_native_command=geometric.copy(),geometric_candidate_curvature_field=field.copy(),previous_transmitted_geometric_native_command=previous_transmitted_geometric,transmitted_local_curvature_field=total-sent_geometric_field,phase_progress_transmission_fraction=phase_progress_fraction,phase_geometric_progress_ready=phase_geometric_ready,geometric_progress_ready_next_command=self.geometric_progress_ready,sent_geometric_head_tracking_error_m=self.sent_geometric_head_tracking_error_m,actual_head_task_error_m=self.actual_head_task_error_m,probe_curvature_per_m=probe.copy(),applied_probe_curvature_per_m=self.applied_probe.copy(),probe_evidence_input_per_m=executed_probe.copy(),applied_probe_curvature_field=self.applied_probe_field.copy(),probe_contact_clock_s=self.probe_contact_clock,command_filter_factor=factor,
            command_maximum_C_node_step_m=step_metrics['maximum_node_step_m'],command_maximum_C_geometry_step_m=step_metrics['maximum_geometry_step_m'],command_measured_plane_excess_compression_m=step_metrics['measured_plane_excess_compression_m'],command_known_source_excess_compression_m=step_metrics['known_source_command_excess_compression_m'],compliance_delta_m=self.delta,
            **roles,**source_mode,source_ready=self.source_ready,source_settle_age_s=self.settle_age,controller_failure=self.failure or '',first_release_t=-1. if self.first_release_t is None else self.first_release_t,
            first_capture_t=-1. if self.first_capture_t is None else self.first_capture_t,current_capture_support=capture,current_supply_support=supply_ok,bearing_motion_evidence=self.bearing_motion_evidence,bearing_motion_age_s=0. if self.bearing_motion_start is None else t-self.bearing_motion_start,
            known_source_vertical_N=source,retained_supply_vertical_N=support['retained_supply_N'],retained_rear_vertical_N=support['fixed_rear8to20_N'],receiver_vertical_N=float(receiver),
            source_bearing_chart_start=support['bearing_chart_start'],source_bearing_chart_end=support['bearing_chart_end'],source_bearing_winding_deg=support['bearing_winding_deg'],
            source_guard_material_start=guard_start,actual_source_boundary_motion_m=drift,actual_source_boundary_X_motion_m=xdrift,actual_source_boundary_orientation_deg=rotation,
            actual_source_motion_from_given_m=given_motion,actual_source_Xmotion_from_given_m=given_Xmotion,actual_source_polar_rotation_from_given_deg=given_polar,
            all21_known_source_geometry=support['all_source'],raw_first_front_contact_t=-1. if self.estimator.first_front_contact is None else self.estimator.first_front_contact,
            geometry_reliable=bool(self.estimator.last.get('reliable',False)),geometry_confidence=float(self.estimator.last.get('confidence',0.)),
            estimated_entry_m=self.estimator.geometry.entry.copy(),estimated_axis=self.estimator.geometry.axis.copy(),estimated_radius_m=self.estimator.geometry.radius,
            applied_global_entry_m=self.ref.geometry.entry.copy(),applied_global_axis=self.ref.geometry.axis.copy(),applied_global_radius_m=self.ref.geometry.radius,
            global_update_count=sum(x['written'] for x in self.updater.history),planner_metrics=str(metrics),source_precontact_feed_zero=self.post is not None or self.S==0.,
            stance_projection_accepted=bool(metrics.get('accepted',self.phase=='source_hold')))
        return raw.copy(),self.info.copy()
