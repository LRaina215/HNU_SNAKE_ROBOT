"""Original Step1 coordination with Step2 observation/G and contact-based C."""
import numpy as np
from physics import mujoco
from geometry import Geometry
from from_source_curve import FromSourceCurve
from cylinder_estimator import CylinderEstimator
from estimated_observer import EstimatedObserver
from native_updater import NativeUpdater
from contact_sensor import read_sensor,classified_observation,known_source
from canonical_coordinates import CanonicalCoordinates

class RollingController:
    def __init__(self,m,d,registration,condition,write_global):
        self.m=m;self.reg=registration;self.last_t=None;self.S=0.;self.raw=d.ctrl.copy();self.co=CanonicalCoordinates()
        geometry=Geometry.from_dict(registration['conditions'][condition]);self.ref=FromSourceCurve(registration['source_constants'],geometry)
        self.ref.source_known=registration['source_known'];self.ref.nominal_frequency_hz=registration['nominal_frequency_hz']
        self.estimator=CylinderEstimator(geometry,registration['source_known'],registration['estimator']);self.observer=EstimatedObserver(self.ref,registration['source_known']);self.observer.allow_prior_capture=True
        self.updater=NativeUpdater(registration['global_update'],write_global);self.first_capture_t=None;self.first_release_t=None;self.bearing_start=None;self.bearing_q=None;self.bearing_command=None;self.bearing_motion=False
        self.probe_clock=0.;self.applied_probe=np.zeros(2);self.phase='source_settle';self.info=None
        self.given=read_sensor(m,d)['centers'].copy();self.failure=None
        _,i=self.ref.command(0.);self.previous=i.copy();lo=i['global_reference_locations'][-1];hi=self.ref.nominal_target_arc+1.35
        self.initial_G=self.ref.nominal_spline(np.linspace(lo,hi,650))
    def command(self,d,t):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0 or dt>.026:raise ValueError('Non-continuous original25ms command clock')
        self.last_t=t;self.S=self.ref.feed_per_cycle*self.reg['nominal_frequency_hz']*max(0.,t-self.reg['settle_s'])
        sensor=read_sensor(self.m,d);head_tracking=abs(sensor['joint_actual'][:2]-self.raw[:2]).max()<=np.radians(2.)
        sent=abs(sensor['joint_ctrl'][:2]-self.raw[:2]).max()<=np.radians(.1)
        executed_probe=self.applied_probe.copy() if head_tracking and sent else np.zeros(2)
        self.estimator.observe(sensor,t,executed_probe);obs=classified_observation(sensor,self.estimator.geometry,self.reg['source_known'])
        roles=self.observer.update(obs,self.S,dt,self.estimator.geometry,self.estimator.last.get('reliable',False))
        source=sum(ct['world_force_N'][2] for ct in sensor['contacts'] if known_source(ct,self.reg['source_known']))
        receiver=sum(ct['world_force_N'][2] for ct,role in zip(sensor['contacts'],obs['contact_estimated_classes']) if role['receiver_candidate_consistent'])
        bearing=receiver>=.4*self.reg['robot_weight_N']
        if bearing:
            if self.bearing_start is None:self.bearing_start=t;self.bearing_q=sensor['joint_actual'][:8].copy();self.bearing_command=self.raw[:8].copy()
            dq=sensor['joint_actual'][:8]-self.bearing_q;dc=self.raw[:8]-self.bearing_command
            self.bearing_motion=bool(t-self.bearing_start>=2. and abs(dq).max()>=np.radians(.25) and abs(dc).max()>=np.radians(.25) and dq@dc>0.)
        else:self.bearing_start=None;self.bearing_motion=False
        if self.bearing_motion and self.first_capture_t is None:self.first_capture_t=t
        if t>=self.reg['settle_s'] and self.first_release_t is None:self.first_release_t=t
        self.phase='source_settle' if t<self.reg['settle_s'] else 'release_contact_wrap' if self.first_capture_t is None else 'bearing_transfer_and_roll'
        # Estimated geometry classifies measurements; only applied G geometry
        # supplies C's radial frame/goal. Frozen G has the same observation/C.
        self.ref.receiving_qualified_mask=roles['estimated_receiver_qualified'];self.ref.sensor_geometry=self.ref.geometry.copy()
        if dt>0:source_info=self.ref.observe_all(self.m,d,obs,roles,self.previous,dt,t>=self.reg['settle_s'])
        else:source_info=dict(source_gamma_per_m=0.,source_gamma_rate_per_m_s=0.,source_curve_direction=self.ref.source_direction.copy(),source_outward_gradient_m2=0.,source_gamma_requested_per_m=0.,source_mode_observed_material=0,source_mode_active=False)
        self.updater.update(self.ref,self.estimator,self.S,t);self.ref.sensor_geometry=self.ref.geometry.copy()
        contact=any(ct['material']<8 and not known_source(ct,self.reg['source_known']) for ct in sensor['contacts'])
        if contact and head_tracking and sent and self.estimator.first_front_contact is not None:self.probe_clock=min(8.,self.probe_clock+dt)
        probe=np.zeros(2)
        if contact and self.probe_clock<8.:
            component=min(1,int(self.probe_clock//4.));probe[component]=self.reg['probe']['amplitude_per_m']*np.sin(2*np.pi*(self.probe_clock-4*component)/4.)
        self.ref.probe=probe;nominal,_=self.ref.nominal_command(self.S);raw,info=self.ref.command(self.S)
        if t<self.reg['settle_s']:raw=d.ctrl.copy()
        native_field=info['curvature_field'].copy();native_bounds=info['curvature_bounds_mm'].copy();native_nodes=info['material_arc_nodes_mm'].copy()
        probe_delta=info['probe_native_delta_rad'];receiving=info['geometric_native_command'];source_delta=raw-receiving-probe_delta if t>=self.reg['settle_s'] else np.zeros(20)
        self.applied_probe=probe.copy();self.raw=raw.copy();self.previous=info.copy();g=self.ref.geometry
        dense=self.ref.nominal_spline(np.linspace(min(info['global_reference_locations'][-1],self.ref.nominal_target_arc-.1),max(info['global_reference_locations'][0],self.ref.nominal_target_arc+1.1),650))
        delta=sensor['centers'][12:]-self.given[12:];motion=float(np.linalg.norm(delta,axis=1).max());axial=float(abs(delta[:,0]).max())
        info.update(S=self.S,alpha=float(self.ref.smooth(self.S/6.)),beta=float(self.ref.smooth((self.S-4.5)/1.5)),controller_phase=self.phase,
            curvature_field=self.co.field(raw),curvature_bounds_mm=self.co.bounds,material_arc_nodes_mm=self.co.material,
            native_geometric_curvature_field=native_field,native_geometric_curvature_bounds_mm=native_bounds,native_geometric_material_nodes_mm=native_nodes,
            global_nominal_native_command=nominal,geometric_native_command=nominal,geometric_curvature_field=self.co.field(nominal),raw_native_command=raw.copy(),
            receiving_compliance_curvature_field=self.co.field(receiving-nominal),force_curvature_field=self.co.field(source_delta),probe_curvature_field=self.co.field(probe_delta),
            local_modes_native_delta_rad=raw-nominal,command_filter_factor=1.,source_commanded_feed_m=self.ref.feed_distance(self.S),source_commanded_roll_rad=2*np.pi*self.S/self.ref.feed_per_cycle,
            shared_feed_rate_fraction=1.,nominal_time_S=self.S,lost_nominal_S=0.,normal_precontact_advance_allowed=True,probe_curvature_per_m=probe,applied_probe_curvature_per_m=self.applied_probe.copy(),probe_evidence_input_per_m=executed_probe,probe_contact_clock_s=self.probe_clock,
            first_release_t=-1. if self.first_release_t is None else self.first_release_t,first_capture_t=-1. if self.first_capture_t is None else self.first_capture_t,bearing_motion_evidence=self.bearing_motion,bearing_motion_age_s=0. if self.bearing_start is None else t-self.bearing_start,
            known_source_vertical_N=float(source),receiver_vertical_N=float(receiver),current_capture_support=self.bearing_motion,current_supply_support=source>=.2*self.reg['robot_weight_N'],
            actual_source_motion_from_given_m=motion,actual_source_Xmotion_from_given_m=axial,actual_source_boundary_motion_m=motion,
            geometry_reliable=bool(self.estimator.last.get('reliable',False)),geometry_confidence=float(self.estimator.last.get('confidence',0.)),
            estimated_entry_m=self.estimator.geometry.entry.copy(),estimated_axis=self.estimator.geometry.axis.copy(),estimated_radius_m=self.estimator.geometry.radius,
            applied_global_entry_m=g.entry.copy(),applied_global_axis=g.axis.copy(),applied_global_radius_m=g.radius,global_update_count=sum(x['written'] for x in self.updater.history),global_reference_dense=dense,
            native_exact_C_points=info['nominal_curve_points'],source_ready=t>=.5,controller_failure='',**roles,**source_info)
        self.info=info.copy();return raw.copy(),info
