"""Permission-only G writeback contrast with a bounded command-continuity check."""
import numpy as np
from geometry import Geometry,normalize

class GlobalUpdater:
    def __init__(self,settings,enabled):self.settings=settings;self.enabled=enabled;self.next_update=0.;self.history=[];self.first_update=None
    def update(self,ref,estimator,S,t):
        if t<self.next_update:return
        interval=self.settings['interval_s'];self.next_update=t+interval
        if not estimator.last.get('reliable',False):return
        before=ref.geometry.copy();goal=estimator.geometry
        dc=goal.entry-before.entry;dc*=min(1.,self.settings['center_rate_m_s']*interval/max(np.linalg.norm(dc),1e-12));dc*=self.settings['gain']
        angle=np.arccos(np.clip(before.axis@goal.axis,-1,1));fraction=min(1.,np.radians(self.settings['axis_rate_deg_s'])*interval/max(angle,1e-12))*self.settings['gain']
        dr=np.clip(goal.radius-before.radius,-self.settings['radius_rate_m_s']*interval,self.settings['radius_rate_m_s']*interval)*self.settings['gain']
        q0,i0=ref.command(S);accepted=False;step=1.
        q1=None;delta=None;world=None;proposal=before.copy();failures=[]
        for attempt in range(7):
            q1=None;delta=None;world=None;proposal_factor=step
            proposal=Geometry(before.entry+dc*step,normalize((1-fraction*step)*before.axis+fraction*step*goal.axis),before.radius+dr*step)
            ref.geometry=proposal
            try:q1,i1=ref.command(S)
            except (ValueError,FloatingPointError) as exc:
                failures.append(dict(attempt=attempt,factor=step,reason=str(exc),exception_type=type(exc).__name__))
                step*=.5;continue
            delta=float(np.degrees(np.max(abs(q1-q0))));world=float(np.max(np.linalg.norm(i1['global_reference_points']-i0['global_reference_points'],axis=1)))
            accepted=bool(np.all(np.isfinite(q1)) and delta<=self.settings['same_S_command_bound_deg'] and world<=self.settings['material_node_step_m'])
            if accepted:break
            failures.append(dict(attempt=attempt,factor=step,reason='command or material-node continuity bound',same_S_change_deg=delta,world_change_m=world))
            step*=.5
        write=bool(self.enabled and accepted and (np.linalg.norm(dc)*step+fraction*step+abs(dr)*step)>1e-12)
        if not write:
            ref.geometry=before
            # Restore the spline/cache as well as its parameter object. q0 was
            # already a valid original command; no failed proposal is executed.
            restored,_=ref.command(S)
            if not np.allclose(restored,q0,rtol=0,atol=1e-12):raise ValueError('original reference changed during proposal rollback')
        elif self.first_update is None:self.first_update=t
        self.history.append(dict(t=float(t),S=float(S),evidence_t=estimator.last['latest_evidence_t'],fit_t=estimator.last['t'],fit_confidence=estimator.last['confidence'],
            write_permission=self.enabled,written=write,continuity_passed=accepted,backtracking_factor=proposal_factor,
            before=before.as_dict(),candidate=goal.as_dict(),proposed=proposal.as_dict(),after=ref.geometry.as_dict(),
            raw_before_rad=q0.tolist(),raw_proposed_rad=None if q1 is None else q1.tolist(),raw_after_rad=(q1 if write else q0).tolist(),
            same_S_raw_change_deg=delta,global_material_node_change_m=world,proposal_failures=failures,
            rejection_reason=None if accepted else 'all proposals rejected; restored original valid G and command',material_history_reset=False))

def probe_command(t,first_contact,settings):
    v=np.zeros(2)
    if first_contact is None:return v
    age=t-first_contact;period=settings['direction_duration_s']
    if 0<=age<2*period:
        direction=int(age//period);v[direction]=settings['amplitude_per_m']*np.sin(2*np.pi*(age-direction*period)/period)
    return v

