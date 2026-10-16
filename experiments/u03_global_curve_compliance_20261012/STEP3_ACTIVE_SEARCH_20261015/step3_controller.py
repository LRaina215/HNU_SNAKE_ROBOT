"""Source crawl, bounded search, evidence, takeover, and transfer in one state."""
import numpy as np
from physics import mujoco
from source_controller import FromSourceController
from search_curve import ActiveSearchCurve
from step3_estimator import UnknownCylinderEstimator
from geometry import Geometry
from estimated_observer import EstimatedObserver
from global_update import GlobalUpdater, probe_command
from contact_sensor import read_sensor, classified_observation, known_source

class Step3Controller(FromSourceController):
    def __init__(self,m,d,reg,case):
        self.m=m;self.reg=reg;self.case=case
        self.S=0.;self.last_t=None;self.settle_age=0.;self.source_ready=False
        self.low_support_age=0.;self.failure=None;self.support_tracks={};self.first_release_t=None
        self.front_relief=np.zeros(2);self.events=[];self.mode='SOURCE_STABILIZE'
        self.search_started=None;self.search_slot=-1;self.release=0.;self.takeover_age=0.
        self.first_capture_t=None;self.current_geometry_installed=False;self.stopped_age=0.
        self.speed=0.;self.empty_directions=[];self.direction_evidence=False;self.new_contact_count=0
        self.reposition_completed=False;self.retreat=0.;self.previous_search_offset=np.zeros(2)
        source=reg['source_known'];constants=dict(reg['source_constants'])
        constants['x0']+=case['source_x_shift_m']
        # This inactive placeholder cannot influence the search family.
        placeholder=Geometry([source['axial_bounds_m'][1],.1,source['center_yz_m'][1]],reg['task_direction'],source['radius_m'])
        self.ref=ActiveSearchCurve(constants,placeholder,search_enabled=case['active_search'])
        self.ref.nominal_frequency_hz=reg['nominal_frequency_hz'];self.ref.configure_initial(d.ctrl)
        self.estimator=UnknownCylinderEstimator(source,reg['estimator'],reg['task_direction'])
        self.observer=EstimatedObserver(self.ref,source);self.updater=GlobalUpdater(reg['global_update'],True)
        self.raw,self.info=self.ref.command(0.)

    def event(self,t,kind,**fields):
        self.events.append(dict(t=float(t),S=float(self.S),kind=kind,**fields))

    def transition(self,t,mode,reason):
        if mode!=self.mode:self.event(t,'mode_change',before=self.mode,after=mode,reason=reason);self.mode=mode

    def _search(self,t,dt,rear,unknown):
        cfg=self.reg['search'];weight=self.reg['robot_weight_N']
        if self.search_started is None:return np.zeros(2)
        age=t-self.search_started;slot=int(age//cfg['direction_duration_s'])
        directions=np.asarray(cfg['directions_yz'],float)
        if slot!=self.search_slot:
            if self.search_slot>=0:
                empty=not self.direction_evidence
                if empty:self.empty_directions.append(self.search_slot)
                self.event(t,'search_direction_end',index=self.search_slot,empty=empty,new_non_source_contact=self.direction_evidence,
                    active=self.case['active_search'])
            self.search_slot=slot;self.direction_evidence=False
            if slot<len(directions):self.event(t,'search_direction_begin',index=slot,task_frame_direction=directions[slot].tolist(),
                applied_amplitude_m=cfg['amplitude_m'] if self.case['active_search'] else 0.)
        self.direction_evidence|=bool(unknown)
        if slot>=len(directions):
            if self.estimator.first_reliable is None and age>=cfg['evidence_timeout_s']:
                self.transition(t,'LIMITED_RETREAT','bounded search exhausted without identifiable receiving cylinder')
                self.retreat=min(cfg['maximum_retreat_S'],self.retreat+cfg['retreat_rate_S_s']*dt)
                if self.retreat>=cfg['maximum_retreat_S']-1e-9:self.failure='search_evidence_budget_exhausted'
            return np.zeros(2)
        phase=(age/cfg['direction_duration_s'])%1.
        pulse=np.sin(np.pi*phase)**2
        enabled=self.case['active_search'] and rear>=self.reg['support']['rear_load_min_fraction_before_capture']*weight
        return directions[slot]*cfg['amplitude_m']*pulse if enabled else np.zeros(2)

    def command(self,d,t):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0 or dt>.026:raise ValueError('noncontinuous command clock')
        self.last_t=t;sensor=read_sensor(self.m,d);src=self.reg['source_known'];cfg=self.reg['search']
        probe=probe_command(t,self.estimator.first_front_contact,self.reg['probe'])
        self.estimator.observe(sensor,t,probe);self.ref.probe=probe
        reliable=bool(self.estimator.last.get('reliable',False))
        geometry=self.estimator.geometry if self.estimator.geometry is not None else self.ref.geometry
        self.ref.sensor_geometry=geometry.copy()
        obs=classified_observation(sensor,geometry,src)
        if not reliable:
            obs['forces'][2]+=obs['forces'][1];obs['forces'][1]*=0
            for role in obs['contact_estimated_classes']:
                role['receiver_candidate_consistent']=False
                if role['estimated_class']==1:role['estimated_class']=2
        rear,source,all_source=self.support(sensor,t,obs);weight=self.reg['robot_weight_N'];p=self.reg['support']
        unknown=[x for x in sensor['contacts'] if x['material']<8 and not known_source(x,src)]
        if unknown:self.new_contact_count+=len(unknown)
        if not self.source_ready:
            stable=all_source and p['initial_load_min_fraction']*weight<=source<=p['initial_load_max_fraction']*weight and np.max(abs(d.qvel))<p['initial_velocity_max']
            self.settle_age=self.settle_age+dt if stable else 0.
            if self.settle_age>=p['initial_stable_s']:
                self.source_ready=True;self.transition(t,'SOURCE_CRAWL','full source load and velocity stable for registered duration')
                self.event(t,'full_source_stable',source_vertical_N=source)
            if t>=p['initial_settle_timeout_s'] and not self.source_ready:self.failure='full_source_stabilization_failed'
        roles=self.observer.update(obs,self.S,dt,geometry,reliable)
        front_vertical=sum(x['world_force_N'][2] for x,r in zip(sensor['contacts'],obs['contact_estimated_classes']) if x['material']<8 and r['receiver_candidate_consistent'])
        support_ok=rear>=p['rear_load_min_fraction_before_capture']*weight
        evidence_takeover=bool(reliable and roles['front_current_valid'] and front_vertical>=cfg['takeover_vertical_fraction']*weight and roles['actual_target_support_count']>=4)
        self.takeover_age=self.takeover_age+dt if evidence_takeover else 0.
        captured=self.takeover_age>=cfg['takeover_stable_s']
        if captured and self.first_capture_t is None:
            self.first_capture_t=t;self.event(t,'new_wrap_and_load_capture',front_vertical_N=float(front_vertical),target_support_count=roles['actual_target_support_count'])
        if self.source_ready and self.first_capture_t is None:
            self.low_support_age=self.low_support_age+dt if not support_ok else 0.
            if self.low_support_age>=p['rear_low_load_max_s']:self.failure='rear_support_lost_before_capture'
        if self.source_ready and self.failure is None:
            source_end_evidence=float(sensor['centers'][0,0])>=src['axial_bounds_m'][1]-cfg['source_endpoint_trigger_margin_m']
            if self.first_release_t is None and (unknown or source_end_evidence):
                self.first_release_t=t;self.event(t,'structure_boundary_evidence',known_finite_source_end=source_end_evidence,front_unknown_contacts=len(unknown))
                self.transition(t,'RELEASE_EXTEND','known source endpoint or new contact supports bounded exploration')
            if self.first_release_t is not None:
                limit=1. if captured else cfg['pre_capture_release_limit']
                if support_ok or captured:self.release=min(limit,self.release+cfg['release_rate_s']*dt)
                if self.release>=cfg['search_begin_release'] and self.search_started is None:
                    self.search_started=t;self.transition(t,'ACTIVE_SEARCH' if self.case['active_search'] else 'NATURAL_APPROACH','registered extension fraction reached')
            if unknown and self.search_started is not None:self.transition(t,'CONTACT_EVIDENCE','unlabeled non-source front contact; cylinder still requires independent evidence')
            if reliable and not self.current_geometry_installed:
                self.ref.geometry=geometry.copy();self.current_geometry_installed=True
                self.event(t,'first_contact_geometry_installed',geometry=geometry.as_dict(),confidence=self.estimator.last.get('confidence',0.))
            if reliable:self.transition(t,'CONTACT_SHAPING' if not captured else 'TRANSFER','reliable geometry; release remains takeover gated')
            lead=max(0.,roles['target_progress_lead_m']);desired=1.
            if not captured:desired=1. if self.S<cfg['pre_capture_max_S'] else 0.
            elif not evidence_takeover:desired=.1
            else:desired=float(np.clip(1-lead/.08,.15,1.))
            if not support_ok and self.first_capture_t is None:desired=0.
            self.speed+=np.clip(desired-self.speed,-dt/.25,dt/.75)
            if self.mode=='LIMITED_RETREAT':self.S=max(0.,self.S-cfg['retreat_rate_S_s']*dt)
            else:self.S+=self.ref.feed_per_cycle*self.reg['nominal_frequency_hz']*dt*self.speed
        offset=self._search(t,dt,rear,unknown)
        self.ref.set_search(offset,dt=dt,release_fraction=self.release,support_ok=support_ok)
        self.ref.set_capture_confidence(float(self.estimator.last.get('confidence',0.)) if reliable else 0.,dt)
        self.ref.receiving_qualified_mask=roles['estimated_receiver_qualified'] if reliable else np.zeros(21,bool)
        self.local_front(d,sensor,dt)
        local=self.ref.observe_all(self.m,d,obs,roles,self.info,dt,True) if dt>0 else {}
        if reliable:self.updater.update(self.ref,self.estimator,self.S,t)
        self.raw,self.info=self.ref.command(self.S)
        self.info.update(**roles,**local,mode=self.mode,source_ready=self.source_ready,source_settle_age_s=self.settle_age,
            retained_rear_vertical_N=rear,known_source_vertical_N=source,all21_known_source_geometry=all_source,
            controller_failure=self.failure or '',first_release_t=-1. if self.first_release_t is None else self.first_release_t,
            geometry_reliable=reliable,geometry_confidence=float(self.estimator.last.get('confidence',0.)),
            estimated_entry_m=geometry.entry.copy(),estimated_axis=geometry.axis.copy(),estimated_radius_m=geometry.radius,
            estimated_geometry_exists=self.estimator.geometry is not None,global_update_count=sum(x['written'] for x in self.updater.history),
            applied_global_entry_m=self.ref.geometry.entry.copy(),applied_global_axis=self.ref.geometry.axis.copy(),applied_global_radius_m=self.ref.geometry.radius,
            actual_front_vertical_N=front_vertical,actual_takeover_valid=captured,takeover_age_s=self.takeover_age,
            first_capture_t=-1. if self.first_capture_t is None else self.first_capture_t,search_slot=self.search_slot,
            search_requested_yz_m=offset,release_fraction=self.release,common_speed_scale=self.speed,
            raw_first_front_contact_t=-1. if self.estimator.first_front_contact is None else self.estimator.first_front_contact)
        return self.raw.copy(),self.info.copy()
