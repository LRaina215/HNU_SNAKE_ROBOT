"""Controller accepts servo registers only, never a simulator/model/case object.

All positions below are guide parameters in an unlocalized virtual chart.
Interaction, capture and support-risk flags are hypotheses, not scene facts.
"""
import numpy as np
from servo_curve import ServoSearchCurve
from servo_observer import ServoObservation, ServoObserver, score_probe, compare_probe_pair
from servo_geometry import fit_source, infer_front_novelty


def slew(current, desired, bound):
    step=np.asarray(desired)-np.asarray(current)
    length=float(np.linalg.norm(step))
    return np.asarray(current)+step*min(1.,bound/max(length,1e-12))


class ServoSearchController:
    def __init__(self, constants, initial_command, settings, active_search=True, hold_only=False):
        self.cfg=dict(settings['control']);self.frequency=settings['nominal_frequency_hz']
        self.active=bool(active_search);self.hold_only=bool(hold_only)
        self.ref=ServoSearchCurve(constants,initial_command)
        self.observer=ServoObserver(tracking_threshold_rad=np.deg2rad(.6),
            load_threshold=self.cfg['front_increment_load_threshold'],persistent_s=self.cfg['interaction_persistence_s'])
        self.S=0.;self.release=0.;self.offset=np.zeros(3);self.roll=0.;self.local=np.zeros(2);self.capture=0.
        self.direction=np.array([0.,1.,0.]);self.mode='CALIBRATE';self.mode_start=0.;self.last_t=None
        self.raw=np.asarray(initial_command).copy();self.info={};self.failure=None;self.events=[]
        self.attempt=0;self.probe_index=0;self.probe_samples=[];self.probe_start=None
        self.probe_reference=None;self.probe_direction_q=None;self.probe_scores=[];self.probe_baseline=None
        self.directions=np.vstack([np.eye(3)[0],-np.eye(3)[0],np.eye(3)[1],-np.eye(3)[1],np.eye(3)[2],-np.eye(3)[2]])
        self.selected_offset=np.zeros(3);self.current_interaction_age=0.;self.capture_evidence_age=0.
        self.overload_age=0.;self.continuity_rejections=0;self.first_interaction=None;self.first_capture_hypothesis=None
        self.reposition_target_S=0.;self.reposition_target_roll=0.;self.first_probe_t=None
        self.crawl_anomaly_age=0.;self.phase_records=[];self.last_sensor=None
        self.probe_phase='RAMP';self.probe_phase_start=0.;self.probe_hold_start=None;self.probe_hold_end=None
        self.probe_return_start=None;self.previous_sent=None;self.pair_scores=[];self.confirm_reference=None
        self.confirm_samples=[];self.confirm_increment_age=0.
        self.source_fit=None;self.next_source_fit=0.;self.novelty={};self.pair_amplitudes={}
        self.search_azimuth=0.

    def event(self,t,kind,**kw):
        self.events.append(dict(t=float(t),S=float(self.S),kind=kind,**kw))

    def transition(self,t,mode,reason):
        if self.mode!=mode:
            self.event(t,'mode_transition',before=self.mode,after=mode,reason=reason,attempt=self.attempt)
            self.mode=mode;self.mode_start=t

    def _curve(self):
        return self.ref.command(self.S,self.release,self.offset,self.roll,self.local,
                                capture_blend=self.capture,receiving_direction=self.direction,search_azimuth=self.search_azimuth)

    def _reset_sweep(self):
        self.probe_index=0;self.probe_start=None;self.probe_samples=[];self.probe_scores=[]
        self.probe_direction_q=None;self.pair_scores=[]
        self.observer.set_paired_response_support(False)

    def _request_reposition(self,t,reason):
        self.event(t,'reposition_requested',reason=reason,attempt=self.attempt,
                   inferred_actual_position=None)
        self.transition(t,'RETRACT',reason)

    def _probe(self,sample):
        t=sample.t;c=self.cfg
        if self.probe_start is None:
            if self.active and self.probe_index%2==0:
                # Use equal, locally opposing curve perturbations. Large guide
                # offsets can map into incomparable joint perturbations.
                direction=self.directions[self.probe_index];q0,_=self._curve();amplitude=c['search_offset_m']
                for _ in range(7):
                    qp,_=self.ref.command(self.S,self.release,direction*amplitude,self.roll,self.local,capture_blend=self.capture,receiving_direction=self.direction,search_azimuth=self.search_azimuth)
                    qm,_=self.ref.command(self.S,self.release,-direction*amplitude,self.roll,self.local,capture_blend=self.capture,receiving_direction=self.direction,search_azimuth=self.search_azimuth)
                    dp=qp-q0;dm=qm-q0;ap=np.linalg.norm(dp);am=np.linalg.norm(dm)
                    opposing=float(dp@dm/max(ap*am,1e-12))
                    if opposing<-.95 and abs(ap-am)<.2*max(ap,am):break
                    amplitude*=.7
                self.pair_amplitudes[self.probe_index//2]=amplitude;self._curve()
                self.event(t,'pair_excitation_registered',axis=self.probe_index//2,amplitude_m=amplitude,joint_opposition=opposing,
                           predicted_positive_norm_rad=float(ap),predicted_negative_norm_rad=float(am))
            self.probe_start=t;self.probe_reference=sample;self.probe_samples=[]
            self.probe_phase='RAMP';self.probe_phase_start=t;self.probe_hold_start=None;self.probe_hold_end=None;self.probe_return_start=None
            self.probe_baseline=dict(error_rad=(sample.sent-sample.q).copy(),load_fraction=sample.load_fraction.copy())
            self.event(t,'probe_begin',index=self.probe_index,direction=self.directions[self.probe_index].tolist(),
                       applied=self.active,ordinary_source_contact_not_excluded=True)
            if self.first_probe_t is None:self.first_probe_t=t
        self.probe_samples.append(sample)
        age=t-self.probe_start;r=c['probe_ramp_s'];h=c['probe_hold_s'];ret=c['probe_return_s'];z=c['probe_zero_s']
        phase_age=t-self.probe_phase_start
        if self.probe_phase=='RAMP':amplitude=self.ref.smooth(phase_age/r)
        elif self.probe_phase=='HOLD':amplitude=1.
        elif self.probe_phase=='RETURN':amplitude=1-self.ref.smooth(phase_age/ret)
        else:amplitude=0.
        requested=self.directions[self.probe_index]*self.pair_amplitudes.get(self.probe_index//2,c['search_offset_m'])*amplitude if self.active else np.zeros(3)
        old_offset=self.offset.copy()
        self.offset=slew(self.offset,requested,c['search_offset_rate_m_s']*self.dt)
        sent_still=self.previous_sent is not None and np.max(abs(sample.sent-self.previous_sent))<np.deg2rad(.1)
        if self.probe_phase=='RAMP' and phase_age>=r and np.linalg.norm(old_offset-requested)<.0005 and sent_still:
            self.probe_phase='HOLD';self.probe_phase_start=t;self.probe_hold_start=t
        elif self.probe_phase=='HOLD' and phase_age>=h:
            self.probe_hold_end=t;self.probe_phase='RETURN';self.probe_phase_start=t
        elif self.probe_phase=='RETURN' and phase_age>=ret and np.linalg.norm(old_offset)<.0005 and sent_still:
            self.probe_phase='ZERO';self.probe_phase_start=t;self.probe_return_start=t
        if self.probe_phase=='HOLD':
            dq=sample.sent-self.probe_reference.sent
            if np.linalg.norm(dq)>1e-5:self.probe_direction_q=dq.copy()
        if age>25 and self.probe_phase!='ZERO':
            self.event(t,'probe_timeout',index=self.probe_index,phase=self.probe_phase,applied_offset=self.offset.tolist())
            self._request_reposition(t,'probe could not reach or return under continuity bounds');return
        if self.probe_phase=='ZERO' and phase_age>=z:
            score=dict(response_available=False,return_recovered=False,interpretation='no commanded excitation')
            if self.active and self.probe_direction_q is not None:
                score=score_probe(self.probe_samples,self.probe_direction_q,self.probe_baseline,
                    self.probe_hold_start,self.probe_hold_end,self.probe_return_start,self.probe_reference)
            hold=[x for x in self.probe_samples if self.probe_hold_start<=x.t<=self.probe_hold_end]
            if hold:
                loads=np.array([x.load_fraction for x in hold]);errors=np.array([x.sent-x.q for x in hold])
                increment=float(np.sqrt(np.mean((loads[:,:8]-self.probe_baseline['load_fraction'][:8])**2)))
                error=float(np.sqrt(np.mean((errors[:,:8]-self.probe_baseline['error_rad'][:8])**2)))
            else:increment=0.;error=0.
            score.update(index=self.probe_index,task_chart_direction=self.directions[self.probe_index].tolist(),
                         front_increment_load=increment,front_increment_error_rad=error,
                         unexplained_interaction=bool(increment>c['front_increment_load_threshold'] and error>np.deg2rad(.3)))
            self.probe_scores.append(score)
            self.event(t,'probe_end',score=score,empty_response=not score['unexplained_interaction'])
            if self.probe_index%2 and self.active:
                pair=compare_probe_pair(self.probe_scores[-2],score)
                self.pair_scores.append(pair)
                self.event(t,'symmetric_pair',axis=self.probe_index//2,assessment=pair)
                self.observer.set_paired_response_support(bool(pair['paired_response_support']))
            self.probe_index+=1;self.probe_start=None;self.probe_direction_q=None
            if self.probe_index==6:
                if not self.active and self.current_interaction_age>=.5:
                    self.selected_offset=np.zeros(3);self.confirm_reference=sample;self.confirm_samples=[]
                    self.transition(t,'CONFIRM_INTERACTION','natural approach has sustained servo interaction; same subsequent shape trial')
                    return
                valid_axes=[j for j,p in enumerate(self.pair_scores) if p.get('pair_comparable',False) and p.get('paired_response_support',False)]
                candidates=[p for p in self.probe_scores if p['index']//2 in valid_axes and p['unexplained_interaction'] and p.get('return_recovered',False)]
                if candidates:
                    best=min(candidates,key=lambda p:p.get('return_cost',1e6)+p['front_increment_error_rad']*20)
                    self.selected_offset=.5*self.pair_amplitudes[best['index']//2]*np.asarray(best['task_chart_direction'])
                    # Low-resistance direction is an interaction hypothesis in the virtual guide chart.
                    axis_cost=[sum(p['front_increment_load'] for p in self.probe_scores[2*j:2*j+2]) if j in valid_axes else 1e6 for j in range(3)]
                    free_axis=np.eye(3)[int(np.argmin(axis_cost))]
                    self.direction=np.array([0.,1.,0.])+.25*free_axis;self.direction/=np.linalg.norm(self.direction)
                    self.event(t,'interaction_direction_hypothesis',direction=self.direction.tolist(),axis_cost=axis_cost,
                        interpretation='servo-response preference plus coarse task prior; not identified world pipe axis')
                    self.confirm_reference=sample;self.confirm_samples=[];self.confirm_increment_age=0.
                    self.transition(t,'CONFIRM_INTERACTION','reversible incremental-load response; confirmation required')
                else:self._request_reposition(t,'no repeatable, reversible interaction in bounded directional trials')

    def _local_admittance(self,obs,dt):
        if not self.info or self.mode in ['CALIBRATE','CRAWL','REPOSITION','RETRACT']:
            desired=np.zeros(2)
        else:
            bounds=self.info['curvature_bounds_mm'];nodes=self.info['material_arc_nodes_mm'];mid=(bounds[:-1]+bounds[1:])/2
            v=np.clip(mid/nodes[8],0,1);w=np.sin(np.pi*v)**2;w[bounds[1:]>nodes[8]]=0.
            span=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]))@w/1000
            A=np.zeros((20,2));odd=np.arange(20)%2==1;A[odd,0]=-span[odd];A[~odd,1]=-span[~odd]
            gradient=A.T@obs['baseline_load_deviation']
            desired=-self.cfg['local_admittance_gain']*gradient/max(np.linalg.norm(gradient),.02)
            desired=np.clip(desired,-self.cfg['local_curvature_limit_per_m'],self.cfg['local_curvature_limit_per_m'])
        self.local=slew(self.local,desired,self.cfg['local_curvature_rate_per_m_s']*dt)

    def command(self,sample):
        if type(sample) is not ServoObservation:raise TypeError('only ServoObservation is accepted')
        t=sample.t;self.dt=0. if self.last_t is None else t-self.last_t;dt=self.dt
        if self.last_t is not None and not 0<dt<=.051:raise ValueError('noncontinuous servo sampling')
        self.last_t=t;self.last_sensor=sample;c=self.cfg
        if self.mode=='CALIBRATE':self.observer.calibrate(sample)
        obs=self.observer.update(sample)
        if t>=self.next_source_fit:
            self.source_fit=fit_source(sample.q);self.next_source_fit=t+1.
        self.novelty=infer_front_novelty(sample.q,obs['baseline_load_deviation'],self.source_fit,
                                      load_threshold=c['front_increment_load_threshold'])
        old=(self.S,self.release,self.offset.copy(),self.roll,self.local.copy(),self.capture,self.direction.copy())
        overload=bool(np.count_nonzero(abs(sample.load_fraction)>.97)>=2)
        self.overload_age=self.overload_age+dt if overload else 0.
        if self.overload_age>c['overload_hold_s']:self.failure='servo_sustained_overload'
        rear_risk=bool(obs['rear_error_deviation_rms_rad']>np.deg2rad(c['rear_error_guard_deg']) and obs['rear_load_deviation_rms']>c['rear_load_jump_guard'])
        interaction=bool(obs['front_interaction_persistent'] and self.novelty['possible_front_interaction_beyond_source_envelope'])
        self.current_interaction_age=self.current_interaction_age+dt if interaction else 0.
        if interaction and self.first_interaction is None:
            self.first_interaction=t;self.event(t,'first_servo_interaction_hypothesis',verified_contact=False)
        if self.mode=='CALIBRATE':
            if t>=c['calibration_s'] and self.observer.baseline_ready and not self.hold_only:
                self.transition(t,'CRAWL','stationary servo baseline collected; physical support is unobserved')
            elif t>12 and not self.observer.baseline_ready:self.failure='servo_baseline_not_stationary'
        elif self.mode=='CRAWL':
            self.S+=self.ref.feed_per_cycle*self.frequency*dt
            changed=obs['front_load_persistent'] or obs['front_tracking_persistent']
            self.crawl_anomaly_age=self.crawl_anomaly_age+dt if changed else 0.
            if self.crawl_anomaly_age>.5 or t-self.mode_start>=c['crawl_probe_timeout_s']:
                why='persistent servo response change' if self.crawl_anomaly_age>.5 else 'bounded diagnostic probe after uninformative crawl'
                self.event(t,'probe_trigger',reason=why,source_endpoint_known=False,world_position_known=False)
                self.transition(t,'EXTEND',why)
        elif self.mode=='EXTEND':
            if rear_risk:self._request_reposition(t,'rear servo-pattern risk while extending')
            else:
                self.release=min(c['search_release'],self.release+c['release_rate_s']*dt)
                if old[1]>=c['search_release']-1e-9:
                    self._reset_sweep();self.transition(t,'SWEEP','bounded extension reached; position suitability still unverified')
        elif self.mode=='SWEEP':
            if rear_risk:self._request_reposition(t,'rear servo-pattern risk during search')
            else:self._probe(sample)
        elif self.mode=='CONFIRM_INTERACTION':
            self.offset=slew(self.offset,self.selected_offset,c['search_offset_rate_m_s']*dt)
            self.confirm_samples.append(sample)
            increment=np.sqrt(np.mean((sample.load_fraction[:8]-self.confirm_reference.load_fraction[:8])**2))
            response=increment>c['front_increment_load_threshold']*.5 if self.active else interaction
            self.confirm_increment_age=self.confirm_increment_age+dt if response else 0.
            if rear_risk:self._request_reposition(t,'rear risk during repeat confirmation')
            elif t-self.mode_start>=c['confirm_s']:
                if self.confirm_increment_age>=.5:
                    self.transition(t,'FORM_RECEIVING','incremental servo response reproduced from a fresh zero reference; bounded shape trial')
                    self.event(t,'nominal_shape_trial',radius_source='explicit nominal prior, not measured geometry')
                else:self._request_reposition(t,'interaction not retained when returning and continuing movement')
        elif self.mode=='FORM_RECEIVING':
            if rear_risk:self._request_reposition(t,'rear risk while testing receiving shape')
            else:
                if interaction:self.capture=min(1.,self.capture+c['capture_morph_rate_s']*dt)
                spread=np.count_nonzero(abs(sample.load_fraction[:8])>.10)>=4
                good=bool(spread and obs['front_error_deviation_rms_rad']<np.deg2rad(6) and interaction)
                self.capture_evidence_age=self.capture_evidence_age+dt if good else 0.
                if self.capture_evidence_age>=c['confirm_s'] and self.capture>=.25:
                    self.first_capture_hypothesis=t if self.first_capture_hypothesis is None else self.first_capture_hypothesis
                    self.event(t,'receiving_support_hypothesis',verified_wrapping=False,verified_target=False)
                    self.transition(t,'TRANSFER','distributed servo loading persisted during shape trial; actual takeover remains evaluation-only')
                elif t-self.mode_start>60:self._request_reposition(t,'receiving hypothesis not confirmed within bounded shape trial')
        elif self.mode=='TRANSFER':
            if rear_risk:self._request_reposition(t,'rear risk during transfer attempt')
            elif not interaction and t-self.mode_start>c['max_capture_without_recheck_s']:
                self._request_reposition(t,'receiving evidence lost during continuing feed')
            else:
                self.capture=min(1.,self.capture+c['capture_morph_rate_s']*dt)
                self.release=min(1.,self.release+c['release_rate_s']*dt*.5)
                self.S+=self.ref.feed_per_cycle*self.frequency*dt*c['transfer_speed_scale']
        elif self.mode=='RETRACT':
            self.offset=slew(self.offset,np.zeros(3),c['search_offset_rate_m_s']*dt)
            self.capture=max(0.,self.capture-c['capture_morph_rate_s']*dt)
            self.release=max(0.,self.release-c['release_rate_s']*dt)
            if old[1]<1e-9 and old[5]<1e-9 and np.linalg.norm(old[2])<1e-8:
                if self.attempt>=c['max_reposition_attempts']:self.failure='bounded_search_and_reposition_exhausted'
                else:
                    self.attempt+=1;self.reposition_target_S=self.S+c['reposition_delta_S']
                    self.reposition_target_roll=self.roll+c['reposition_roll_step_rad']
                    self.search_azimuth=c.get('search_sectors_rad',[0.,0.,0.,0.])[self.attempt]
                    self.event(t,'new_search_sector',azimuth_rad=float(self.search_azimuth),reason='previous bounded sector lacked repeatable receiver evidence',
                               front_retracted=True,source_geometry_not_observed=True)
                    self.observer.set_paired_response_support(False)
                    self.transition(t,'REPOSITION','retracted front; command source crawl and an additional internal phase change')
        elif self.mode=='REPOSITION':
            self.S=min(self.reposition_target_S,self.S+self.ref.feed_per_cycle*self.frequency*dt)
            self.roll=float(slew(self.roll,self.reposition_target_roll,c['roll_offset_rate_rad_s']*dt))
            if old[0]>=self.reposition_target_S-1e-8 and abs(old[3]-self.reposition_target_roll)<1e-8:
                self.event(t,'reposition_command_finished',actual_position_unobserved=True)
                self.transition(t,'EXTEND','repeat extension at changed source gait phase; verify by fresh probes')
        if self.mode not in ['SWEEP','CONFIRM_INTERACTION','FORM_RECEIVING','TRANSFER']:
            self.offset=slew(self.offset,np.zeros(3),c['search_offset_rate_m_s']*dt)
        # Paired directional trials hold local parameters fixed to avoid confounding their comparison.
        if self.mode not in ['SWEEP','CONFIRM_INTERACTION']:self._local_admittance(obs,dt)
        proposed=(self.S,self.release,self.offset.copy(),self.roll,self.local.copy(),self.capture,self.direction.copy())
        accepted=False;largest=0.
        for k in range(9):
            alpha=2.**(-k)
            self.S,self.release,self.offset,self.roll,self.local,self.capture,self.direction=[a+alpha*(b-a) for a,b in zip(old,proposed)]
            try:
                raw,info=self._curve();largest=float(np.max(abs(raw-self.raw))*180/np.pi)
                accepted=bool(np.all(np.isfinite(raw)) and np.max(abs(raw))<=np.deg2rad(88.) and (self.last_t==0 or largest<=c['command_change_limit_deg']))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError):accepted=False
            if accepted:break
        if not accepted:
            self.S,self.release,self.offset,self.roll,self.local,self.capture,self.direction=old
            raw,info=self._curve();self.continuity_rejections+=1
            if self.continuity_rejections>=c['max_continuity_rejections']:self.failure='curve_continuity_stalled'
        else:self.continuity_rejections=0
        self.raw=raw.copy();self.info=info
        self.previous_sent=sample.sent.copy()
        info.update(mode=self.mode,attempt=self.attempt,active_search=self.active,raw_step_deg=largest,
            curve_update_accepted=accepted,release_fraction=float(self.release),capture_blend=float(self.capture),
            search_offset_m=self.offset.copy(),roll_offset_rad=float(self.roll),local_relief_per_m=self.local.copy(),
            direction_hypothesis=self.direction.copy(),controller_failure=self.failure or '',
            front_interaction_hypothesis=interaction,rear_servo_pattern_risk=rear_risk,
            front_increment_load=obs['front_load_deviation_rms'],rear_increment_load=obs['rear_load_deviation_rms'],
            front_increment_error_rad=obs['front_error_deviation_rms_rad'],rear_increment_error_rad=obs['rear_error_deviation_rms_rad'],
            baseline_ready=self.observer.baseline_ready,S=float(self.S),probe_index=self.probe_index,
            source_shape_confidence=float(self.source_fit['confidence']),source_shape_reliable=bool(self.source_fit['reliable']),
            source_axis_headframe=np.asarray(self.source_fit['axis_headframe'] if self.source_fit['reliable'] else [np.nan]*3),
            front_beyond_source_envelope=bool(self.novelty['possible_front_interaction_beyond_source_envelope']),
            source_envelope_compatible=-1 if self.novelty['source_envelope_compatible'] is None else int(self.novelty['source_envelope_compatible']))
        return raw.copy(),dict(info)
