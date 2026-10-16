"""Native curve execution with source grip, local C and evidence-gated feed."""
import numpy as np
from native_mapping import NativeMapping
from native_step1_reference import NativeStep1Reference
from native_robot import NativeRobot
from native_updater import NativeUpdater
from geometry import Geometry
from contact_sensor import read_sensor,classified_observation,known_source
from cylinder_estimator import CylinderEstimator
from estimated_observer import EstimatedObserver
from support_policy import SourcePolicy
from source_mode import SourceMode,reconstruct
from source_balance import SourceBalance
from direct_controller import DirectController as LegacyInformation,probe_command

class DirectController(SourcePolicy):
    base_info=LegacyInformation.base_info
    def __init__(self,m,d,registration,condition,write_global):
        self.m=m;self.reg=registration;self.p=registration['direct_extension'];self.settings=registration['native_shape'];self.last_t=None
        self.S=0.;self.alpha=0.;self.beta=0.;self.phase='source_hold';self.post=None;self.source_ready=False;self.settle_age=0.;self.failure=None
        self.first_release_t=None;self.first_capture_t=None;self.low_support_age=0.;self.front_relief=np.zeros(2)
        self.support_tracks={};self.unknown_candidates={};self.source_chart_seen=np.zeros(21,bool);self.current_supply_materials=set()
        self.planner=NativeRobot(m,d);geometry=Geometry.from_dict(registration['conditions'][condition])
        self.ref=NativeStep1Reference(self.planner.given,self.planner.initial_q,geometry);self.ref.release_materials=12.;self.ref.shape_progress=6.
        self.ref.nominal_frequency_hz=registration['nominal_frequency_hz'];self.ref.probe=np.zeros(2);self.ref.front_relief=np.zeros(2)
        self.ref.source_gamma=0.;self.ref.source_direction=np.array([0.,1.]);self.ref.observe=lambda *args:None
        self.estimator=CylinderEstimator(geometry,registration['source_known'],registration['estimator']);self.observer=EstimatedObserver(self.ref,registration['source_known']);self.observer.allow_prior_capture=True
        self.updater=NativeUpdater(registration['global_update'],write_global)
        self.source_balance=SourceBalance(m,registration['source_known'],registration['source_balance'])
        self.raw=d.ctrl.copy();self.transmitted_geometric=self.planner.initial_q.copy();self.last_geo=self.transmitted_geometric.copy()
        self.applied_probe=np.zeros(2);self.probe_contact_clock=0.;self.previous_probe_request=np.zeros(2);self.applied_probe_field=np.zeros((256,2))
        self.bearing_motion_start=None;self.bearing_motion_q=None;self.bearing_motion_command=None;self.bearing_motion_evidence=False;self.partial_age=0.
        self.delta=0.;self.filtered_gap=0.;self.filtered_load=0.;self.given_source_P=read_sensor(m,d)['centers'].copy();self.given_source_R=d.xmat[self.planner.head:self.planner.head+21].reshape(21,3,3).copy();self.stable_source_P=None
        self.info=self.base_info(self.planner.initial_q,self.planner.given,d.qpos,0.)
        self.ref.delta=0.;self.ref.nominal_command(0.);self.initial_G=self.ref.spline(np.linspace(self.ref.start_arc-1.,self.ref.target_arc+1.5,700))
    def command(self,d,t):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0 or dt>.026:raise ValueError('Non-continuous command clock')
        self.last_t=t;sensor=read_sensor(self.m,d);weight=self.reg['robot_weight_N'];p=self.reg['support'];co=self.planner.coordinates
        free=min(20,int(np.ceil(self.ref.release_materials)));tracking=abs(sensor['joint_actual'][:free]-self.raw[:free]).max()<=np.radians(5.)
        raw_contact=any(ct['material']<8 and not known_source(ct,self.reg['source_known']) for ct in sensor['contacts'])
        head_probe_tracking=abs(sensor['joint_actual'][:2]-self.raw[:2]).max()<=np.radians(2.)
        executed_probe=self.applied_probe.copy() if head_probe_tracking else np.zeros(2)
        self.estimator.observe(sensor,t,executed_probe);observed=classified_observation(sensor,self.estimator.geometry,self.reg['source_known']);support=self.support(sensor,t,observed);supply=self.adequate_supply(support,weight)
        roles=self.observer.update(observed,self.S,dt,self.estimator.geometry,self.estimator.last.get('reliable',False))
        receiver=sum(ct['world_force_N'][2] for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']) if role['receiver_candidate_consistent'])
        bearing=bool(receiver>=.4*weight and any(ct['material']<8 and role['receiver_candidate_consistent'] for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes'])))
        if bearing:
            if self.bearing_motion_start is None:self.bearing_motion_start=t;self.bearing_motion_q=sensor['joint_actual'][:8].copy();self.bearing_motion_command=self.raw[:8].copy()
            dq=sensor['joint_actual'][:8]-self.bearing_motion_q;dc=self.raw[:8]-self.bearing_motion_command
            self.bearing_motion_evidence=bool(t-self.bearing_motion_start>=.5 and abs(dq).max()>=np.radians(.25) and abs(dc).max()>=np.radians(.25) and dq@dc>0)
        else:self.bearing_motion_start=None;self.bearing_motion_evidence=False
        capture=bool(bearing and self.bearing_motion_evidence and t-self.bearing_motion_start>=2.)
        if not self.source_ready:
            stable=support['all_source'] and supply and .9*weight<=support['known_source_N']<=1.1*weight and abs(d.qvel).max()<.03
            self.settle_age=self.settle_age+dt if stable else 0.
            if self.settle_age>=2.:self.source_ready=True;self.first_release_t=t;self.phase='front_release';self.stable_source_P=sensor['centers'].copy()
            if t>=10. and not self.source_ready:self.failure='source_stabilization_failed'
        elif self.post is None:
            self.low_support_age=self.low_support_age+dt if not supply else 0.
            if self.low_support_age>=.25:self.failure='source_support_invalid_before_capture'
            if t-self.first_release_t>self.p['capture_timeout_after_release_s']:self.failure='front_capture_timeout'
            self.alpha=min(1.,self.alpha+dt/self.p['release_duration_s'])
            self.phase='front_wrap' if raw_contact or self.alpha>=1. else 'front_release';self.beta=self.alpha
            self.partial_age=self.partial_age+dt if receiver>=.1*weight else 0.
            if self.partial_age>=.5 and tracking:
                goal=12.+2*np.clip((receiver/weight-.1)/.3,0,1);self.ref.release_materials=min(goal,self.ref.release_materials+.5*dt) if goal>self.ref.release_materials else self.ref.release_materials
        if capture and self.post is None and self.source_ready:
            self.first_capture_t=t;self.post=self.ref;self.phase='whole_body_feed_and_roll'
        if self.post is not None:self.S+=self.ref.feed_per_cycle*self.reg['nominal_frequency_hz']*dt
        self.ref.shape_progress=6.+min(3.,self.S);self.ref.delta=0.
        self.updater.update(self.ref,self.estimator,self.S,t)
        nominal,native_info=self.ref.nominal_command(self.S);G=native_info['reference_points'].copy()
        g_dense=self.ref.spline(np.linspace(self.ref.start_arc-1.,self.ref.target_arc+1.5,700))
        activation=NativeMapping.smooth(self.alpha);posterior=np.arange(20)>=self.ref.release_materials
        active=np.full(20,activation);active[posterior]=0.
        if self.post is not None:active[posterior]=NativeMapping.smooth((t-self.first_capture_t)/15.)
        geometric=self.planner.initial_q+active*(nominal-self.planner.initial_q)
        self.planner.observe(d,self.ref.release_materials);self.info=self.base_info(geometric,G,d.qpos,t)
        if self.source_ready:self.local_front(d,sensor,dt)
        source_mode=SourceMode.observe_all(self.ref,self.m,d,observed,dict(roles,front_confirmed=capture),self.info,dt,True) if dt>0 else {}
        if dt==0:source_mode=dict(source_gamma_per_m=0.,source_gamma_rate_per_m_s=0.,source_curve_direction=self.ref.source_direction.copy(),source_outward_gradient_m2=0.,source_gamma_requested_per_m=0.,source_mode_observed_material=0,source_mode_active=False)
        qualified=roles['estimated_receiver_qualified'];ax,rr,rho=self.ref.geometry.coordinates(sensor['centers']);gap=float(np.mean(rho[qualified]-(self.ref.geometry.radius+.036))) if qualified.any() else 0.;load=float(np.mean(observed['forces'][1,qualified])) if qualified.any() else 0.
        gain=1-np.exp(-dt/.4);self.filtered_gap+=gain*(gap-self.filtered_gap);self.filtered_load+=gain*(load-self.filtered_load)
        dead=np.sign(self.filtered_gap)*max(0.,abs(self.filtered_gap)-.004);desired=-.06*dead+.00005*max(0.,self.filtered_load-12.)-.1*self.delta if qualified.any() else -.1*self.delta
        self.delta=float(np.clip(self.delta+dt*np.clip(desired,-.00025,.00025),-.002,.002))
        self.ref.delta=self.delta
        adjusted,_=self.ref.nominal_command(self.S);radial_field=co.field(active*(adjusted-nominal))
        if self.estimator.first_front_contact is not None and raw_contact and head_probe_tracking and abs(self.applied_probe-self.previous_probe_request).max()<=self.reg['probe']['transmission_tracking_tolerance_per_m']:self.probe_contact_clock=min(2*self.reg['probe']['direction_duration_s'],self.probe_contact_clock+dt)
        probe=probe_command(self.probe_contact_clock,0. if self.estimator.first_front_contact is not None else None,self.reg['probe']) if raw_contact else np.zeros(2)
        self.previous_probe_request=probe.copy();self.ref.probe=probe
        n=co.material;bounds=co.bounds;mid=.5*(bounds[:-1]+bounds[1:]);w=np.sin(np.pi*np.clip(mid/n[8],0,1))**2;w[bounds[1:]>n[8]]=0.
        bearing_probe=np.zeros(2)
        if bearing and self.post is None:
            age=t-self.bearing_motion_start;bearing_probe[0]=.10*np.sin(2*np.pi*age/4.)
        probe_field=w[:,None]*(probe+bearing_probe)[None,:]/1000;force_field=w[:,None]*self.front_relief[None,:]/1000
        preload=(1-active)*(self.planner.initial_ctrl-self.planner.initial_q)
        gravity=self.planner.gravity_torque(geometric,free)/127.;preload[:free]+=active[:free]*gravity[:free]
        if self.post is not None:preload*=1-NativeMapping.smooth((t-self.first_capture_t)/15.)
        v=np.clip((mid-n[15])/(n[-1]-n[15]),0,1);tw=np.sin(np.pi*v)**2;tw[(bounds[:-1]<n[15])|(bounds[1:]>n[-1])]=0.
        source_field=self.ref.source_gamma*tw[:,None]*self.ref.source_direction[None,:]/1000
        balance,balance_info=self.source_balance.update(d,sensor,dt,self.source_ready,self.first_capture_t,t);balance_field=co.field(balance)
        field=co.field(geometric);local=probe_field+force_field+co.field(preload)+radial_field+source_field+balance_field;raw=co.hinges(field+local);raw=np.clip(raw,-np.radians(88),np.radians(88))
        factor=min(1.,np.radians(20)*dt/max(abs(raw-self.raw).max(),1e-12)) if dt>0 else 0.
        raw=self.raw+factor*(raw-self.raw);self.applied_probe_field+=factor*(probe_field-self.applied_probe_field);self.applied_probe+=factor*(probe-self.applied_probe)
        self.transmitted_geometric+=factor*(geometric-self.transmitted_geometric);self.raw=raw.copy();self.last_geo=geometric.copy()
        self.info=self.base_info(self.transmitted_geometric,G,d.qpos,t);C=self.planner.fk(raw);frame=self.info['reference_frame'];total=co.field(raw);dense,bishop=reconstruct(total,bounds,frame,C[0],n)
        start=free;drift=float(np.linalg.norm(sensor['centers'][start:]-self.given_source_P[start:],axis=1).max()) if start<21 else 0.;dx=float(abs(sensor['centers'][start:,0]-self.given_source_P[start:,0]).max()) if start<21 else 0.
        oldphi=np.arctan2(self.given_source_P[start:,2]-1,self.given_source_P[start:,1]);newphi=np.arctan2(sensor['centers'][start:,2]-1,sensor['centers'][start:,1]);polar=float(np.degrees(abs(np.angle(np.exp(1j*(newphi-oldphi))))).max())
        currentR=d.xmat[self.planner.head+start:self.planner.head+21].reshape(21-start,3,3);orientation=float(np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',self.given_source_R[start:],currentR)-1)/2,-1,1))).max())
        C_dense=np.column_stack([np.interp(np.linspace(0,n[-1],257),n,C[:,j]) for j in range(3)])
        self.info.update(curvature_field=total,reference_points=C,nominal_curve_points=C,native_exact_C_points=C,bishop_curve_points=bishop,bishop_curve_dense=dense,nominal_curve_dense=C_dense,
            raw_native_command=raw.copy(),geometric_native_command=self.transmitted_geometric.copy(),geometric_curvature_field=co.field(self.transmitted_geometric),geometric_candidate_native_command=geometric.copy(),
            local_modes_native_delta_rad=raw-self.transmitted_geometric,force_curvature_field=force_field,probe_curvature_field=probe_field,preload_curvature_field=co.field(preload),receiving_compliance_curvature_field=radial_field,source_force_curvature_field=source_field,
            transmitted_local_curvature_field=total-co.field(self.transmitted_geometric),command_filter_factor=factor,probe_curvature_per_m=probe,applied_probe_curvature_per_m=self.applied_probe.copy(),probe_evidence_input_per_m=executed_probe,probe_contact_clock_s=self.probe_contact_clock,
            compliance_delta_m=self.delta,source_ready=self.source_ready,source_settle_age_s=self.settle_age,controller_failure=self.failure or '',first_release_t=-1. if self.first_release_t is None else self.first_release_t,first_capture_t=-1. if self.first_capture_t is None else self.first_capture_t,
            current_capture_support=capture,current_supply_support=supply,bearing_motion_evidence=self.bearing_motion_evidence,bearing_motion_age_s=0. if self.bearing_motion_start is None else t-self.bearing_motion_start,
            known_source_vertical_N=support['known_source_N'],retained_supply_vertical_N=support['retained_supply_N'],retained_rear_vertical_N=support['fixed_rear8to20_N'],receiver_vertical_N=float(receiver),
            source_guard_material_start=start,actual_source_boundary_motion_m=drift,actual_source_boundary_X_motion_m=dx,actual_source_boundary_orientation_deg=orientation,actual_source_motion_from_given_m=drift,actual_source_Xmotion_from_given_m=dx,actual_source_polar_rotation_from_given_deg=polar,
            all21_known_source_geometry=support['all_source'],raw_first_front_contact_t=-1. if self.estimator.first_front_contact is None else self.estimator.first_front_contact,geometry_reliable=bool(self.estimator.last.get('reliable',False)),geometry_confidence=float(self.estimator.last.get('confidence',0.)),
            estimated_entry_m=self.estimator.geometry.entry.copy(),estimated_axis=self.estimator.geometry.axis.copy(),estimated_radius_m=self.estimator.geometry.radius,
            applied_global_entry_m=self.ref.geometry.entry.copy(),applied_global_axis=self.ref.geometry.axis.copy(),applied_global_radius_m=self.ref.geometry.radius,global_update_count=sum(x['written'] for x in self.updater.history),
            global_reference_dense=g_dense,global_reference_locations=native_info['reference_locations'],global_target_arc_m=self.ref.target_arc,spatial_guide_stations_m=native_info['reference_locations'],spatial_guide_stations_valid=True,
            native_geometric_integration_bounds_mm=native_info['curvature_bounds_mm'],native_geometric_material_nodes_mm=native_info['material_arc_nodes_mm'],source_release_materials=self.ref.release_materials,native_shape_progress=self.ref.shape_progress,
            source_balance_curvature_field=balance_field,**roles,**source_mode,**balance_info)
        return raw.copy(),self.info.copy()
