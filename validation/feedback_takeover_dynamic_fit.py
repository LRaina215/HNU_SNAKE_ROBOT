"""Fit virtual pipe geometry/gravity to a collected short joint-response window.

No real contact, pose, velocity or gravity orientation enters the objective.
First 0.8 s trains, last 0.4 s scores prediction. True axes score only after
optimization. This diagnostic tests whether dynamics add information beyond
near-surface fitting; it is not yet an online controller.
"""
import argparse
import json
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from feedback_takeover_dynamic_observer_probe import virtual_model,predict
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame,write_json


def run(scene,multistart=False,fit_velocity=False,fit_joint_velocity=False,error_scale=.2,central=False):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    at=0. if scene=='horizontal' else 2.
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,m,_,meta=context(folder)
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    initial=min(controls,key=lambda r:abs(r['time']-.2-at))['geometry_observer']['estimate']
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials']
    records=next(r['records'] for r in trials if r['scene']==scene and r['snapshot_seconds']==at and r['pattern']=='multisine' and not r['target_removed'])
    measured=np.array([r['q'] for r in records]);train=records[:41]
    initial_joint_velocity=(measured[1]-measured[0])/.02
    s=np.asarray(initial['source']);t=np.asarray(initial['target']);v=np.cross(s,t);c=np.asarray(initial['corner_mm'])
    coarse=json.loads((root/'dynamic_gravity_probe.json').read_text(encoding='utf8'))['runs']
    starts=next(r['candidates'] for r in coarse if r['scene']==scene)
    start=min(starts,key=lambda r:r['rms_deg'])
    def decode(x):
        source=s+x[3]*t+x[4]*v;source/=np.linalg.norm(source)
        perpendicular=t-(t@source)*source;perpendicular/=np.linalg.norm(perpendicular)
        target=np.cos(x[5])*perpendicular+np.sin(x[5])*np.cross(source,perpendicular)
        return dict(initial,corner_mm=c+x[:3],source=source,target=target,pipe_radius_mm=initial['pipe_radius_mm']+x[6]),x[7:10]
    calls=0;history=[];begin=time.perf_counter()
    def residual(x):
        nonlocal calls
        estimate,g=decode(x);prediction=predict(virtual_model(m,estimate,g,meta['source_model']),train,x[10:16] if fit_velocity else None,
            initial_joint_velocity+x[16:] if fit_joint_velocity else None)
        error=np.degrees(np.asarray(prediction['predicted'])-measured[:41])
        result=np.r_[error.reshape(-1)/error_scale,x[:3]/5.,x[3:6]/.1,x[6]/2.,(np.linalg.norm(g)-9.81)/.2]
        if fit_velocity:result=np.r_[result,x[10:13]/.03,x[13:16]/.3]
        if fit_joint_velocity:result=np.r_[result,x[16:]/.3]
        calls+=1
        if calls%20==0:
            row=dict(evaluations=calls,elapsed_seconds=time.perf_counter()-begin,rms_deg=float(np.sqrt(np.mean(error**2))),parameters=x.copy())
            history.append(row);print(scene,calls,'RMS',row['rms_deg'],'wall',round(row['elapsed_seconds'],1),flush=True)
        return result
    steps=np.r_[np.full(3,.1),np.full(3,.001),.05,np.full(3,.1)]
    if fit_velocity:steps=np.r_[steps,np.full(3,.001),np.full(3,.01)]
    if fit_joint_velocity:steps=np.r_[steps,np.full(20,.005)]
    if central:steps*=.1
    def jacobian(x):
        base=residual(x);columns=[]
        for i,h in enumerate(steps):
            changed=x.copy();changed[i]+=h
            if central:
                opposite=x.copy();opposite[i]-=h
                columns.append((residual(changed)-residual(opposite))/(2*h))
            else:columns.append((residual(changed)-base)/h)
        return np.column_stack(columns)
    solutions=[]
    seeds=[dict(label=seed['gravity_label']+str(seed['gravity_sign']),x0=np.r_[np.zeros(7),seed['gravity']]) for seed in (starts if multistart else [start])]
    if fit_velocity:
        prior_file=root/('dynamic_fit_'+scene+'_multistart.json')
        if not prior_file.exists():prior_file=root/('dynamic_fit_'+scene+'.json')
        prior_fit=json.loads(prior_file.read_text(encoding='utf8'))
        seeds=[dict(label='previous training-selected fit',x0=np.r_[prior_fit['parameters'],np.zeros(6)])]
    if fit_joint_velocity:
        prior_fit=json.loads((root/('dynamic_fit_'+scene+'_velocity.json')).read_text(encoding='utf8'))
        seeds=[dict(label='previous geometry/velocity fit',x0=np.r_[prior_fit['parameters'],np.zeros(20)])]
        if central:
            prior_fit=json.loads((root/('dynamic_fit_'+scene+'_joint_velocity_precise.json')).read_text(encoding='utf8'))
            seeds=[dict(label='previous joint velocity fit',x0=np.asarray(prior_fit['parameters']))]
    for seed in seeds:
        x0=seed['x0']
        low=np.r_[np.full(3,-10.),np.full(3,-.25),-4.,np.full(3,-10.)]
        high=np.r_[np.full(3,10.),np.full(3,.25),4.,np.full(3,10.)]
        scale=np.r_[np.full(3,3.),np.full(3,.05),1.,np.full(3,3.)]
        if fit_velocity:
            low=np.r_[low,np.full(3,-.1),np.full(3,-1.)];high=np.r_[high,np.full(3,.1),np.full(3,1.)]
            scale=np.r_[scale,np.full(3,.02),np.full(3,.2)]
        if fit_joint_velocity:
            low=np.r_[low,np.full(20,-1.)];high=np.r_[high,np.full(20,1.)];scale=np.r_[scale,np.full(20,.1)]
        fitted=least_squares(residual,x0,jac=jacobian,
            bounds=(low,high),x_scale=scale,
            max_nfev=60,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        solutions.append(fitted)
        print(scene,'seed',seed['label'],'cost',fitted.cost,flush=True)
    fitted=min(solutions,key=lambda r:r.cost)
    estimate,g=decode(fitted.x);final=predict(virtual_model(m,estimate,g,meta['source_model']),records,fitted.x[10:16] if fit_velocity else None,
        initial_joint_velocity+fitted.x[16:] if fit_joint_velocity else None)
    error=np.degrees(np.asarray(final['predicted'])-measured)
    # Simulator pose is used only after the finished fit.
    i=int(np.argmin(abs(z['time']-z['time'][0]-at)));rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
    corner,source,target,_=pipe_frame(m)
    scores=dict(target_error_deg=float(np.degrees(np.arccos(np.clip((rot@estimate['target'])@target,-1,1)))),
        target_line_offset_mm=float(np.linalg.norm(np.cross(rot@estimate['corner_mm']+1000*z['qpos'][i,:3]-corner,target))),
        gravity_error_deg=float(np.degrees(np.arccos(np.clip((rot@g)@m.opt.gravity/(np.linalg.norm(g)*9.81),-1,1)))),
        training_rms_deg=float(np.sqrt(np.mean(error[:41]**2))),held_out_rms_deg=float(np.sqrt(np.mean(error[41:]**2))))
    write_json(root/('dynamic_fit_'+scene+('_joint_velocity' if fit_joint_velocity else '_velocity' if fit_velocity else '_multistart' if multistart else '')+('_precise' if error_scale!=.2 else '')+('_central' if central else '')+'.json'),dict(scene=scene,time=at,estimate=estimate,gravity=g,parameters=fitted.x,fit_velocity=fit_velocity,fit_joint_velocity=fit_joint_velocity,error_scale=error_scale,central=central,
        seed_solutions=[dict(cost=float(r.cost),parameters=r.x,success=bool(r.success),message=r.message) for r in solutions],
        success=bool(fitted.success),message=fitted.message,history=history,prediction=final,scoring=scores,
        limitation='Diagnostic; seed selection uses training objective only. Free initial velocity is fitted only when fit_velocity is true; exact nominal robot/friction assumed. Convergence is not identifiability or control success.'))
    print(scene,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['horizontal','vertical'],required=True)
    p.add_argument('--multistart',action='store_true');p.add_argument('--fit-velocity',action='store_true')
    p.add_argument('--fit-joint-velocity',action='store_true');p.add_argument('--error-scale',type=float,default=.2);p.add_argument('--central',action='store_true');args=p.parse_args()
    if args.fit_joint_velocity and not args.fit_velocity:p.error('--fit-joint-velocity requires --fit-velocity')
    run(args.scene,args.multistart,args.fit_velocity,args.fit_joint_velocity,args.error_scale,args.central)
