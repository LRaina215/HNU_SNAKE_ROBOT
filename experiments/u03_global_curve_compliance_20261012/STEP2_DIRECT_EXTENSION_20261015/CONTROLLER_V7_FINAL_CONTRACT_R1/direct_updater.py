"""Same confidence/write bounds; explicit negative-handed receiver phase."""
import numpy as np
from geometry import Geometry,normalize

def guide_signature(G):
    T=np.diff(G,axis=0);T/=np.linalg.norm(T,axis=1)[:,None];K=np.diff(T,axis=0)/.0645
    return G,T,K

class DirectUpdater:
    def __init__(self,settings,enabled):self.settings=settings;self.enabled=enabled;self.next_update=0.;self.history=[];self.first_update=None
    def update(self,planner,estimator,alpha,beta,theta,t,position_cap=None,joint_step_rad=None):
        if t<self.next_update:return theta
        p=self.settings;dt=p['interval_s'];self.next_update=t+dt
        if not estimator.last.get('reliable',False):return theta
        old=planner.geometry.copy();goal=estimator.geometry;oldq=planner.q.copy();q0,G0,_=planner.solve(alpha,beta,theta,position_cap=position_cap,joint_step_rad=joint_step_rad);planner.fk(q0);R0=planner.d.xmat[planner.head].reshape(3,3).copy();g0=guide_signature(G0);planner.q=oldq.copy()
        dc=goal.entry-old.entry;dc*=min(1.,p['center_rate_m_s']*dt/max(np.linalg.norm(dc),1e-12))*p['gain']
        angle=np.arccos(np.clip(old.axis@goal.axis,-1,1));fraction=min(1.,np.radians(p['axis_rate_deg_s'])*dt/max(angle,1e-12))*p['gain']
        dr=np.clip(goal.radius-old.radius,-p['radius_rate_m_s']*dt,p['radius_rate_m_s']*dt)*p['gain']
        points=np.array([x['point'] for x in estimator.samples]);ax,rad,_=goal.coordinates(points);e,b=goal.frame()
        phases=np.arctan2(rad@b,rad@e)-planner.p['receiver_winding_sign']*2*np.pi*ax/.18266863819422827
        circular=np.mean(np.exp(1j*phases));coherence=float(abs(circular));theta_goal=theta
        if coherence>=p['theta_min_coherence']:theta_goal=float(np.clip(np.angle(np.exp(1j*(np.angle(circular)-planner.nominal_contact_phase(beta)))),-np.radians(p['theta_bound_deg']),np.radians(p['theta_bound_deg'])))
        dtheta=np.clip(theta_goal-theta,-np.radians(p['theta_rate_deg_s'])*dt,np.radians(p['theta_rate_deg_s'])*dt)*p['gain']
        accepted=False;factor=1.;failures=[];metrics={};newtheta=theta
        for attempt in range(p['backtracking_attempts']):
            proposal=Geometry(old.entry+factor*dc,normalize((1-factor*fraction)*old.axis+factor*fraction*goal.axis),old.radius+factor*dr)
            planner.set_geometry(proposal);planner.q=oldq.copy();q1,G1,fit=planner.solve(alpha,beta,theta+factor*dtheta,position_cap=position_cap,joint_step_rad=joint_step_rad);g1=guide_signature(G1)
            planner.fk(q1);R1=planner.d.xmat[planner.head].reshape(3,3)
            metrics=dict(command_deg=float(abs(q1-q0).max()*180/np.pi),material_m=float(np.linalg.norm(g1[0]-g0[0],axis=1).max()),
                G_position_m=float(np.linalg.norm(g1[0]-g0[0],axis=1).max()),G_tangent_deg=float(np.arccos(np.clip(np.einsum('ij,ij->i',g0[1],g1[1]),-1,1)).max()*180/np.pi),
                G_curvature_per_m=float(np.linalg.norm(g1[2]-g0[2],axis=1).max()),material_frame_deg=float(np.arccos(np.clip((np.trace(R0.T@R1)-1)/2,-1,1))*180/np.pi))
            accepted=bool(fit['accepted'] and metrics['command_deg']<=p['same_S_command_bound_deg'] and metrics['material_m']<=p['material_node_step_m'] and metrics['G_position_m']<=p['G_position_step_m'] and metrics['G_tangent_deg']<=p['G_tangent_step_deg'] and metrics['G_curvature_per_m']<=p['G_curvature_step_per_m'] and metrics['material_frame_deg']<=p['material_frame_step_deg'])
            if accepted:break
            failures.append(dict(attempt=attempt,factor=factor,metrics=metrics.copy()));factor*=.5
        write=bool(self.enabled and accepted and np.linalg.norm(dc)+abs(fraction)+abs(dr)+abs(dtheta)>1e-12)
        if write:newtheta=theta+factor*dtheta
        else:planner.set_geometry(old)
        # A preview never changes accumulated C. Next command follows the new G.
        planner.q=oldq.copy()
        if write and self.first_update is None:self.first_update=t
        self.history.append(dict(t=t,written=write,write_permission=self.enabled,continuity_passed=accepted,factor=factor,before=old.as_dict(),after=planner.geometry.as_dict(),candidate=goal.as_dict(),
            alpha=alpha,beta=beta,S=0.,theta_before_rad=theta,theta_after_rad=newtheta,theta_contact_coherence=coherence,metrics=metrics,proposal_failures=failures,fit_t=estimator.last['t'],evidence_t=estimator.last['latest_evidence_t'],material_history_reset=False))
        return newtheta
    def update_post(self,post,estimator,S,theta,capture_theta,t):
        if t<self.next_update:return theta
        p=self.settings;dt=p['interval_s'];self.next_update=t+dt
        if not estimator.last.get('reliable',False):return theta
        old=post.geometry.copy();goal=estimator.geometry;q0,i0=post.command(S);g0=guide_signature(i0['global_reference_points'])
        dc=goal.entry-old.entry;dc*=min(1.,p['center_rate_m_s']*dt/max(np.linalg.norm(dc),1e-12))*p['gain']
        angle=np.arccos(np.clip(old.axis@goal.axis,-1,1));fraction=min(1.,np.radians(p['axis_rate_deg_s'])*dt/max(angle,1e-12))*p['gain'];dr=np.clip(goal.radius-old.radius,-p['radius_rate_m_s']*dt,p['radius_rate_m_s']*dt)*p['gain']
        points=np.array([x['point'] for x in estimator.samples]);ax,rad,_=goal.coordinates(points);e,b=goal.frame();circular=np.mean(np.exp(1j*(np.arctan2(rad@b,rad@e)-post.receiver_winding_sign*2*np.pi*ax/.18266863819422827)));coherence=float(abs(circular));wanted=theta
        if coherence>=p['theta_min_coherence']:wanted=float(np.clip(np.angle(np.exp(1j*(np.angle(circular)-post.nominal_contact_phase)) ),-np.radians(p['theta_bound_deg']),np.radians(p['theta_bound_deg'])))
        dtheta=np.clip(wanted-theta,-np.radians(p['theta_rate_deg_s'])*dt,np.radians(p['theta_rate_deg_s'])*dt)*p['gain'];factor=1.;accepted=False;failures=[];metrics={}
        for attempt in range(p['backtracking_attempts']):
            proposal=Geometry(old.entry+factor*dc,normalize((1-factor*fraction)*old.axis+factor*fraction*goal.axis),old.radius+factor*dr);post.set_geometry(proposal,theta+factor*dtheta-capture_theta);q1,i1=post.command(S);g1=guide_signature(i1['global_reference_points'])
            metrics=dict(command_deg=float(abs(q1-q0).max()*180/np.pi),material_m=float(np.linalg.norm(g1[0]-g0[0],axis=1).max()),G_position_m=float(np.linalg.norm(g1[0]-g0[0],axis=1).max()),G_tangent_deg=float(np.arccos(np.clip(np.einsum('ij,ij->i',g0[1],g1[1]),-1,1)).max()*180/np.pi),G_curvature_per_m=float(np.linalg.norm(g1[2]-g0[2],axis=1).max()),material_frame_deg=float(np.arccos(np.clip((np.trace(i0['reference_frame'].T@i1['reference_frame'])-1)/2,-1,1))*180/np.pi))
            accepted=bool(all(np.isfinite(q1)) and metrics['command_deg']<=p['same_S_command_bound_deg'] and metrics['material_m']<=p['material_node_step_m'] and metrics['G_position_m']<=p['G_position_step_m'] and metrics['G_tangent_deg']<=p['G_tangent_step_deg'] and metrics['G_curvature_per_m']<=p['G_curvature_step_per_m'] and metrics['material_frame_deg']<=p['material_frame_step_deg'])
            if accepted:break
            failures.append(dict(attempt=attempt,factor=factor,metrics=metrics.copy()));factor*=.5
        write=bool(self.enabled and accepted and np.linalg.norm(dc)+abs(fraction)+abs(dr)+abs(dtheta)>1e-12);newtheta=theta+factor*dtheta if write else theta
        if not write:post.set_geometry(old,theta-capture_theta)
        if write and self.first_update is None:self.first_update=t
        self.history.append(dict(t=t,S=S,written=write,write_permission=self.enabled,continuity_passed=accepted,before=old.as_dict(),after=post.geometry.as_dict(),candidate=goal.as_dict(),theta_before_rad=theta,theta_after_rad=newtheta,theta_contact_coherence=coherence,factor=factor,metrics=metrics,proposal_failures=failures,fit_t=estimator.last['t'],evidence_t=estimator.last['latest_evidence_t'],material_history_reset=False))
        return newtheta
