"""Contact-informed theta/transition/receiver update with G and C limits."""
import numpy as np
from geometry import Geometry,normalize
from estimated_curve import EstimatedCurve

def signature(ref,S):
    delta=ref.delta;ref.delta=0.
    try:
        ref.set_nominal_family(S);u=np.linspace(ref.start_arc,ref.target_arc+.3,129)
        p=ref.spline(u);v=ref.spline(u,1);a=ref.spline(u,2);speed=np.linalg.norm(v,axis=1)
        tangent=v/np.maximum(speed[:,None],1e-12)
        curvature=(a-tangent*np.sum(tangent*a,axis=1)[:,None])/np.maximum(speed[:,None],1e-12)**2
        return p,tangent,curvature
    finally:ref.delta=delta

class GlobalUpdater:
    def __init__(self,settings,enabled):self.settings=settings;self.enabled=enabled;self.next_update=0.;self.history=[];self.first_update=None
    def update(self,ref,estimator,S,t):
        if t<self.next_update:return
        p=self.settings;dt=p['interval_s'];self.next_update=t+dt
        if not estimator.last.get('reliable',False):return
        before=ref.geometry.copy();theta0=ref.theta_offset;goal=estimator.geometry
        dc=goal.entry-before.entry;dc*=min(1.,p['center_rate_m_s']*dt/max(np.linalg.norm(dc),1e-12))*p['gain']
        angle=np.arccos(np.clip(before.axis@goal.axis,-1,1));fraction=min(1.,np.radians(p['axis_rate_deg_s'])*dt/max(angle,1e-12))*p['gain']
        dr=np.clip(goal.radius-before.radius,-p['radius_rate_m_s']*dt,p['radius_rate_m_s']*dt)*p['gain']
        points=np.array([x['point'] for x in estimator.samples]);ax,rad,_=goal.coordinates(points);e,b=goal.frame()
        phases=np.arctan2(rad@b,rad@e)-2*np.pi*ax/ref.pitch
        circular=np.mean(np.exp(1j*phases));coherence=float(abs(circular))
        nominal=EstimatedCurve.receiving_phase(ref,S);theta_goal=theta0
        if coherence>=p['theta_min_coherence']:
            diff=float(np.angle(np.exp(1j*(np.angle(circular)-nominal))))
            theta_goal=float(np.clip(diff,-np.radians(p['theta_bound_deg']),np.radians(p['theta_bound_deg'])))
        dtheta=float(np.clip(theta_goal-theta0,-np.radians(p['theta_rate_deg_s'])*dt,np.radians(p['theta_rate_deg_s'])*dt)*p['gain'])
        q0,i0=ref.command(S);g0=signature(ref,S);failures=[];accepted=False;metrics={};proposal=before.copy();factor=1.
        for attempt in range(p['backtracking_attempts']):
            proposal=Geometry(before.entry+factor*dc,normalize((1-fraction*factor)*before.axis+fraction*factor*goal.axis),before.radius+factor*dr)
            ref.geometry=proposal;ref.theta_offset=theta0+factor*dtheta
            try:
                q1,i1=ref.command(S);g1=signature(ref,S)
                metrics=dict(command_deg=float(np.max(abs(q1-q0))*180/np.pi),material_m=float(np.max(np.linalg.norm(i1['global_reference_points']-i0['global_reference_points'],axis=1))),
                    G_position_m=float(np.max(np.linalg.norm(g1[0]-g0[0],axis=1))),G_tangent_deg=float(np.max(np.arccos(np.clip(np.sum(g1[1]*g0[1],axis=1),-1,1)))*180/np.pi),
                    G_curvature_per_m=float(np.max(np.linalg.norm(g1[2]-g0[2],axis=1))),
                    material_frame_deg=float(np.arccos(np.clip((np.trace(i0['reference_frame'].T@i1['reference_frame'])-1)/2,-1,1))*180/np.pi))
                accepted=bool(np.all(np.isfinite(q1)) and metrics['command_deg']<=p['same_S_command_bound_deg'] and metrics['material_m']<=p['material_node_step_m']
                    and metrics['G_position_m']<=p['G_position_step_m'] and metrics['G_tangent_deg']<=p['G_tangent_step_deg'] and metrics['G_curvature_per_m']<=p['G_curvature_step_per_m'] and metrics['material_frame_deg']<=p['material_frame_step_deg'])
                if accepted:break
                reason='position/tangent/curvature/material/native-command limit'
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:reason=str(exc)
            failures.append(dict(attempt=attempt,factor=factor,reason=reason,metrics=metrics.copy()));factor*=.5
        changed=np.linalg.norm(dc)+abs(fraction)+abs(dr)+abs(dtheta)>1e-12
        write=bool(self.enabled and accepted and changed)
        if not write:ref.geometry=before;ref.theta_offset=theta0
        ref.command(S)
        if write and self.first_update is None:self.first_update=t
        self.history.append(dict(t=float(t),S=float(S),written=write,write_permission=self.enabled,continuity_passed=accepted,factor=factor,
            before=before.as_dict(),candidate=goal.as_dict(),after=ref.geometry.as_dict(),theta_before_rad=theta0,theta_goal_rad=theta_goal,theta_after_rad=ref.theta_offset,
            theta_contact_coherence=coherence,metrics=metrics,proposal_failures=failures,fit_t=estimator.last['t'],evidence_t=estimator.last['latest_evidence_t'],material_history_reset=False))

def probe_command(t,first_contact,settings):
    probe=np.zeros(2)
    if first_contact is None:return probe
    age=t-first_contact;period=settings['direction_duration_s']
    if 0<=age<2*period:
        j=int(age//period);probe[j]=settings['amplitude_per_m']*np.sin(2*np.pi*(age-j*period)/period)
    return probe
