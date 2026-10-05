"""Directional identifiability from encoder pulse responses and robot dynamics.

The rear body defines an assumed stationary gauge during the short hold. Its
actual floating pose is never read. Inverse dynamics uses known robot masses,
joint damping and encoded derivatives, zero gravity and no physical contacts.
An unknown constant gravity vector of magnitude 9.81 is fitted afterwards.
Virtual CAD contact normals explain changes in joint reaction, with signed
coefficients because these are perturbations, not absolute support forces.

This is a model-consistency diagnostic, not a force-closure certificate. Rear
slip, tangential reactions and finite-difference acceleration can invalidate it.
"""
import copy
import json
import numpy as np
import mujoco
from scipy.optimize import minimize
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from encoder_pipe_proximity import EncoderPipeProximity


def encoded_mechanics(model,records):
    m=copy.copy(model);m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.opt.gravity[:]=0
    d=mujoco.MjData(m);ids=m.actuator_trnid[:,0]
    qadr=m.jnt_qposadr[ids];vadr=m.jnt_dofadr[ids];tail=m.body('tail').id
    q=np.array([r['q'] for r in records]);commands=np.array([r['command'] for r in records]);pose=[]
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qadr]=q[0];mujoco.mj_kinematics(m,d)
    anchor=d.xpos[tail].copy();anchor_rot=d.xmat[tail].reshape(3,3).copy()
    for row in q:
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qadr]=row;mujoco.mj_kinematics(m,d)
        rot=anchor_rot@d.xmat[tail].reshape(3,3).T
        root=anchor-rot@d.xpos[tail];quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rot.reshape(9))
        p=d.qpos.copy();p[:3]=root;p[3:7]=quat;pose.append(p)
    pose=np.asarray(pose);velocity=np.zeros((len(q),m.nv))
    for i in range(len(q)):
        lo=max(0,i-1);hi=min(len(q)-1,i+1)
        mujoco.mj_differentiatePos(m,velocity[i],.02*(hi-lo),pose[lo],pose[hi])
    acceleration=np.gradient(velocity,.02,axis=0);rows=[]
    for i,p in enumerate(pose):
        d.qpos[:]=p;d.qvel[:]=velocity[i];d.qacc[:]=acceleration[i];mujoco.mj_inverse(m,d)
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,tail)
        j=np.vstack((jp,jr));transform=np.zeros((m.nv,len(vadr)))
        transform[vadr,np.arange(len(vadr))]=1.;transform[:6]=-np.linalg.solve(j[:,:6],j[:,vadr])
        act=np.zeros(m.nv)
        act[vadr]=np.clip(m.actuator_gainprm[:,0]*(commands[i]-q[i]),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        gravity=np.zeros((m.nv,3))
        for body in np.flatnonzero(m.body_mass>0):
            jac=np.zeros((3,m.nv));mujoco.mj_jacBodyCom(m,d,jac,None,int(body))
            gravity+=m.body_mass[body]*jac.T
        points=1000*np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(k)).xanchor for k in range(1,21)],d.site('s_tail').xpos])
        rows.append(dict(pose=p.copy(),q=q[i],transform=transform,points=points,
            reaction=transform.T@(d.qfrc_inverse-act),gravity=transform.T@gravity))
    quiet=slice(4,10)
    baseline=np.mean([r['reaction'] for r in rows[quiet]],axis=0)
    baseline_g=np.mean([r['gravity'] for r in rows[quiet]],axis=0)
    for row in rows:row['reaction']-=baseline;row['gravity']-=baseline_g
    return m,d,rows


def normal_columns(query,m,d,row,initial,target,selected):
    corner=np.asarray(initial['corner_mm']);source=np.asarray(initial['source']);radius=initial['pipe_radius_mm']
    query.gaps(row['q'],row['points'],corner,source,target,radius)
    d.qpos[:]=row['pose'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,row['pose'][3:7]);rotation=rotation.reshape(3,3)
    columns=[]
    for side,module in selected:
        slot=query.pipe_slots[side];best=None
        for geom in query.robot_geoms[module]:
            segment=np.zeros(6)
            gap=mujoco.mj_geomDistance(query.m,query.d,int(geom),slot,.2,segment)
            if best is None or gap<best[0]:best=(gap,segment,int(geom))
        gap,segment,geom=best
        normal=segment[:3]-segment[3:]
        if gap<0:normal=-normal
        if np.linalg.norm(normal)<1e-10:
            a=(source,target)[side];point=rotation@segment[3:]+row['pose'][:3]
            normal=rotation.T@(point-corner/1000-((point-corner/1000)@a)*a)
        normal/=max(np.linalg.norm(normal),1e-12)
        point=rotation@segment[:3]+row['pose'][:3];jac=np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,jac,None,point,int(m.geom_bodyid[geom]))
        columns.append((rotation@normal)@jac@row['transform'])
    return np.column_stack(columns) if columns else np.zeros((20,0))


def score(query,m,d,rows,initial,target,selected):
    rhs=[];gravity=[]
    for row in rows[10:51:2]:
        a=normal_columns(query,m,d,row,initial,target,selected)
        projection=np.eye(20)-a@np.linalg.pinv(a,rcond=1e-6)
        rhs.append(projection@row['reaction']);gravity.append(projection@row['gravity'])
    y=np.concatenate(rhs);b=np.vstack(gravity);bb=b.T@b;by=b.T@y
    objective=lambda g:float(g@bb@g-2*by@g+y@y)
    gradient=lambda g:2*(bb@g-by)
    best=None
    for seed in np.r_[np.eye(3),-np.eye(3)]*9.81:
        solved=minimize(objective,seed,jac=gradient,method='SLSQP',
            constraints=[dict(type='eq',fun=lambda g:g@g-9.81**2,jac=lambda g:2*g)],
            options=dict(maxiter=100,ftol=1e-10))
        if best is None or solved.fun<best.fun:best=solved
    return dict(residual_Nm=float(np.sqrt(max(0,best.fun)/len(y))),gravity=best.x,
        gravity_solver_success=bool(best.success),equations=len(y),normal_candidates=len(selected))


def run():
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials'];reports=[]
    for scene in ('horizontal','vertical'):
        name='horizontal_encoder_live_whole_coil' if scene=='horizontal' else 'vertical_encoder_live_whole_coil'
        folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/name
        _,z,model,_,_=context(folder);times=z['time']-z['time'][0];qpos=z['qpos']
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'));_,_,true_axis,_=pipe_frame(model)
        for trial in trials:
            if trial['scene']!=scene or trial['pattern']!='multisine' or trial['target_removed']:continue
            at=trial['snapshot_seconds'];info=min(controls,key=lambda r:abs(r['time']-.2-at));initial=info['geometry_observer']['estimate']
            m,d,rows=encoded_mechanics(model,trial['records']);query=EncoderPipeProximity(model)
            s=np.asarray(initial['source']);t=np.asarray(initial['target']);c=np.asarray(initial['corner_mm']);r=initial['pipe_radius_mm']
            gap=query.gaps(rows[0]['q'],rows[0]['points'],c,s,t,r)
            selected=np.argwhere(gap<5.).tolist();solutions=[]
            for phi in np.radians(np.arange(-12.,12.1,3.)):
                candidate=t*np.cos(phi)+np.cross(s,t)*np.sin(phi)+s*(s@t)*(1-np.cos(phi))
                result=score(query,m,d,rows,initial,candidate,selected)
                result.update(delta_deg=float(np.degrees(phi)),target=candidate);solutions.append(result)
            # Geometry truth is introduced only after all candidate scores.
            i=int(np.argmin(abs(times-at)));rot=np.zeros(9);mujoco.mju_quat2Mat(rot,qpos[i,3:7]);rot=rot.reshape(3,3)
            for row in solutions:row['held_out_axis_error_deg']=float(np.degrees(np.arccos(np.clip((rot@row['target'])@true_axis,-1,1))))
            chosen=min(solutions,key=lambda r:r['residual_Nm']);original=next(r for r in solutions if r['delta_deg']==0)
            reports.append(dict(scene=scene,time=at,solutions=solutions,chosen=chosen,original=original))
            print(scene,at,'error original/chosen',original['held_out_axis_error_deg'],chosen['held_out_axis_error_deg'],'delta',chosen['delta_deg'],'residual',chosen['residual_Nm'],flush=True)
            write_json(root/'wrench_direction.json',dict(rows=reports,
                limitation='Perturbation-normal consistency, approximate held-rear dynamics, no friction-force model. Only a direction probe; no control rollout or support-margin claim.'))


if __name__=='__main__':run()
