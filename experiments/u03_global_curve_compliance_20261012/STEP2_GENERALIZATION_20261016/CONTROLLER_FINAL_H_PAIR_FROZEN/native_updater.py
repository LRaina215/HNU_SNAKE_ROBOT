"""Evidence-driven bounded cylinder writes, checked through native Step1 q."""
import numpy as np
from geometry import Geometry,normalize
def guide_signature(G):
    T=np.diff(G,axis=0);T/=np.linalg.norm(T,axis=1)[:,None];K=np.diff(T,axis=0)/.0645
    return G,T,K

class NativeUpdater:
    def __init__(self,settings,enabled):
        self.settings=settings;self.enabled=enabled;self.next_update=0.;self.history=[];self.first_update=None
    def update(self,ref,estimator,S,t):
        if t<self.next_update:return
        p=self.settings;dt=p['interval_s'];self.next_update=t+dt
        if not estimator.last.get('reliable',False):return
        old=ref.geometry.copy();goal=estimator.geometry;delta=ref.delta;ref.delta=0.
        q0,i0=ref.nominal_command(S);sig0=guide_signature(i0['reference_points'])
        dc=goal.entry-old.entry;dc*=min(1.,p['center_rate_m_s']*dt/max(np.linalg.norm(dc),1e-12))*p['gain']
        angle=np.arccos(np.clip(old.axis@goal.axis,-1,1));fraction=min(1.,np.radians(p['axis_rate_deg_s'])*dt/max(angle,1e-12))*p['gain']
        dr=np.clip(goal.radius-old.radius,-p['radius_rate_m_s']*dt,p['radius_rate_m_s']*dt)*p['gain']
        accepted=False;factor=1.;failures=[];metrics={}
        for attempt in range(p['backtracking_attempts']):
            ref.geometry=Geometry(old.entry+factor*dc,normalize((1-factor*fraction)*old.axis+factor*fraction*goal.axis),old.radius+factor*dr)
            q1,i1=ref.nominal_command(S);sig1=guide_signature(i1['reference_points'])
            metrics=dict(command_deg=float(abs(q1-q0).max()*180/np.pi),material_m=float(np.linalg.norm(sig1[0]-sig0[0],axis=1).max()),
                G_position_m=float(np.linalg.norm(sig1[0]-sig0[0],axis=1).max()),G_tangent_deg=float(np.degrees(np.arccos(np.clip(np.sum(sig0[1]*sig1[1],axis=1),-1,1))).max()),
                G_curvature_per_m=float(np.linalg.norm(sig1[2]-sig0[2],axis=1).max()),material_frame_deg=float(np.degrees(np.arccos(np.clip((np.trace(i0['reference_frame'].T@i1['reference_frame'])-1)/2,-1,1)))))
            accepted=bool(metrics['command_deg']<=p['same_S_command_bound_deg'] and metrics['material_m']<=p['material_node_step_m'] and metrics['G_position_m']<=p['G_position_step_m'] and metrics['G_tangent_deg']<=p['G_tangent_step_deg'] and metrics['G_curvature_per_m']<=p['G_curvature_step_per_m'] and metrics['material_frame_deg']<=p['material_frame_step_deg'])
            if accepted:break
            failures.append(dict(attempt=attempt,factor=factor,metrics=metrics.copy()));factor*=.5
        write=bool(self.enabled and accepted and np.linalg.norm(dc)+abs(fraction)+abs(dr)>1e-12)
        if not write:ref.geometry=old
        if write and self.first_update is None:self.first_update=t
        self.history.append(dict(t=t,S=S,written=write,write_permission=self.enabled,continuity_passed=accepted,before=old.as_dict(),after=ref.geometry.as_dict(),candidate=goal.as_dict(),theta_before_rad=0.,theta_after_rad=0.,factor=factor,metrics=metrics,proposal_failures=failures,fit_t=estimator.last['t'],evidence_t=estimator.last['latest_evidence_t'],material_history_reset=False))
        ref.delta=delta
