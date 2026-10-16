"""One live curve controller from S=0. No physics integration in this module."""
import numpy as np
from physics import mujoco
from from_source_curve import FromSourceCurve
from geometry import Geometry
from estimated_observer import EstimatedObserver
from global_update import GlobalUpdater,probe_command
from contact_sensor import read_sensor,classified_observation,known_source

class FromSourceController:
    def __init__(self,m,d,registration,condition,write_global):
        from cylinder_estimator import CylinderEstimator
        self.reg=registration;self.m=m;self.S=0.;self.last_t=None;self.settle_age=0.;self.source_ready=False;self.low_support_age=0.;self.failure=None
        self.support_tracks={};self.first_release_t=None
        self.ref=FromSourceCurve(registration['source_constants'],Geometry.from_dict(registration['conditions'][condition]));self.ref.configure_initial(d.ctrl)
        self.ref.nominal_frequency_hz=registration['nominal_frequency_hz']
        self.estimator=CylinderEstimator(self.ref.geometry,registration['source_known'],registration['estimator'])
        self.observer=EstimatedObserver(self.ref,registration['source_known']);self.updater=GlobalUpdater(registration['global_update'],write_global)
        self.raw,self.info=self.ref.command(0.);self.front_relief=np.zeros(2)
    def support(self,sensor,t,observed):
        source=self.reg['source_known'];retained=0.;all_source=0.
        for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']):
            j=ct['material'];previous=self.support_tracks.get(j);known=known_source(ct,source)
            continuous=bool(previous is not None and t-previous['t']<=.10 and np.linalg.norm(ct['point']-previous['point'])<.015 and ct['normal']@previous['normal']>np.cos(np.radians(30)))
            eligible=known or (continuous and not role['receiver_candidate_consistent'])
            if eligible:
                self.support_tracks[j]=dict(t=t,point=ct['point'].copy(),normal=ct['normal'].copy())
                if j>=8:retained+=ct['world_force_N'][2]
            if known:all_source+=ct['world_force_N'][2]
        c=sensor['centers'];rho=np.linalg.norm(c[:,1:]-source['center_yz_m'],axis=1)
        geo=(abs(rho-(source['radius_m']+.036))<.031)&(c[:,0]>source['axial_bounds_m'][0])&(c[:,0]<source['axial_bounds_m'][1])
        return float(retained),float(all_source),bool(geo.all())
    def local_front(self,d,sensor,dt):
        info=self.info;bounds=info['curvature_bounds_mm'];nodes=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:]);v=np.clip(mid/nodes[8],0,1)
        w=np.sin(np.pi*v)**2;w[bounds[1:]>nodes[8]]=0.
        A=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]));span=A@w/1000
        mapping=np.zeros((20,2));odd=np.arange(20)%2==1;mapping[odd,0]=-span[odd];mapping[~odd,1]=-span[~odd]
        desired=np.zeros(2);head=self.m.body('head').id;columns=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]]
        supports=[ct for ct in sensor['contacts'] if ct['material']>=8 and ct['normal_N']>.5 and
            (known_source(ct,self.reg['source_known']) or ct['material'] in self.support_tracks)]
        if not supports:return
        anchor=max(supports,key=lambda ct:ct['normal_N'])['material']
        jpa=np.zeros((3,self.m.nv));jra=np.zeros((3,self.m.nv));mujoco.mj_jacBody(self.m,d,jpa,jra,head+anchor)
        J=np.vstack([jpa,jra]);root_mapping=-np.linalg.solve(J[:,:6],J[:,columns]@mapping)
        for ct in sensor['contacts']:
            if ct['material']>=8 or ct['normal_N']<=12:continue
            jp=np.zeros((3,self.m.nv));jr=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,d,jp,jr,ct['point'],head+ct['material'])
            gradient=ct['normal']@(jp[:,columns]@mapping+jp[:,:6]@root_mapping);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(ct['normal_N']-12)*gradient/n
        n=np.linalg.norm(desired);desired*=min(1.,.30/max(n,1e-12))
        step=desired-self.front_relief;n=np.linalg.norm(step);step*=min(1.,.06*dt/max(n,1e-12));self.front_relief+=step
        self.ref.front_relief=self.front_relief.copy()
    def command(self,d,t):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0 or dt>.026:raise ValueError('non-continuous controller command clock')
        self.last_t=t;sensor=read_sensor(self.m,d)
        probe=probe_command(t,self.estimator.first_front_contact,self.reg['probe']);self.estimator.observe(sensor,t,probe)
        self.ref.probe=probe;self.ref.sensor_geometry=self.estimator.geometry.copy()
        observed=classified_observation(sensor,self.estimator.geometry,self.reg['source_known'])
        rear,source,all_source=self.support(sensor,t,observed);weight=self.reg['robot_weight_N'];p=self.reg['support']
        if not self.source_ready:
            stable=all_source and p['initial_load_min_fraction']*weight<=source<=p['initial_load_max_fraction']*weight and np.max(abs(d.qvel))<p['initial_velocity_max']
            self.settle_age=self.settle_age+dt if stable else 0.
            if self.settle_age>=p['initial_stable_s']:self.source_ready=True;self.first_release_t=t
            if t>=p['initial_settle_timeout_s'] and not self.source_ready:self.failure='full_source_stabilization_failed'
        elif not self.observer.front_confirmed:
            self.low_support_age=self.low_support_age+dt if rear<p['rear_load_min_fraction_before_capture']*weight else 0.
            if self.low_support_age>=p['rear_low_load_max_s']:self.failure='rear_support_lost_before_front_capture'
        if self.source_ready and self.failure is None:self.S+=self.ref.feed_per_cycle*self.reg['nominal_frequency_hz']*dt
        roles=self.observer.update(observed,self.S,dt,self.estimator.geometry,self.estimator.last.get('reliable',False))
        self.ref.receiving_qualified_mask=roles['estimated_receiver_qualified'];self.local_front(d,sensor,dt)
        if dt>0:source_info=self.ref.observe_all(self.m,d,observed,roles,self.info,dt,True)
        else:source_info={}
        self.updater.update(self.ref,self.estimator,self.S,t);self.raw,self.info=self.ref.command(self.S)
        self.info.update(**roles,**source_info,source_ready=self.source_ready,source_settle_age_s=self.settle_age,retained_rear_vertical_N=rear,known_source_vertical_N=source,
            all21_known_source_geometry=all_source,controller_failure=self.failure or '',first_release_t=-1. if self.first_release_t is None else self.first_release_t,
            raw_first_front_contact_t=-1. if self.estimator.first_front_contact is None else self.estimator.first_front_contact,
            geometry_reliable=bool(self.estimator.last.get('reliable',False)),geometry_confidence=float(self.estimator.last.get('confidence',0.)),
            estimated_entry_m=self.estimator.geometry.entry.copy(),estimated_axis=self.estimator.geometry.axis.copy(),estimated_radius_m=self.estimator.geometry.radius,
            global_update_count=sum(x['written'] for x in self.updater.history),
            applied_global_entry_m=self.ref.geometry.entry.copy(),applied_global_axis=self.ref.geometry.axis.copy(),applied_global_radius_m=self.ref.geometry.radius)
        return self.raw.copy(),self.info.copy()
