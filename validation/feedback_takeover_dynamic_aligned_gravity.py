"""Response identification under the explicit horizontal/vertical pipe prior.

Compare all six gravity signs/axes by joint-response fit. The true gravity
direction, scene label, root pose and contacts never select a candidate. This
is restricted to axis-aligned right-angle test environments, not arbitrary
oriented pipe networks.
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


def run(scene):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    at=0. if scene=='horizontal' else 2.
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,m,_,meta=context(folder)
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    initial=min(controls,key=lambda r:abs(r['time']-.2-at))['geometry_observer']['estimate']
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials']
    records=next(r['records'] for r in trials if r['scene']==scene and r['snapshot_seconds']==at and r['pattern']=='multisine' and not r['target_removed'])
    measured=np.array([r['q'] for r in records]);train=records[:41]
    s=np.asarray(initial['source']);t=np.asarray(initial['target']);v=np.cross(s,t);c=np.asarray(initial['corner_mm'])
    steps=np.r_[np.full(3,.01),np.full(3,.0001),.005,np.full(3,.0001),np.full(3,.001)]
    low=np.r_[np.full(3,-20.),np.full(3,-.35),-5.,np.full(3,-.1),np.full(3,-1.)]
    high=-low;scale=np.r_[np.full(3,3.),np.full(3,.05),1.,np.full(3,.02),np.full(3,.2)]
    previous=json.loads((root/('dynamic_fit_'+scene+('_joint_velocity_precise_central' if scene=='vertical' else '')+'.json')).read_text(encoding='utf8'))
    start=np.r_[previous['parameters'][:7],previous['parameters'][10:16] if previous.get('fit_velocity',False) else np.zeros(6)]
    def geometry(x):
        source=s+x[3]*t+x[4]*v;source/=np.linalg.norm(source)
        perpendicular=t-(t@source)*source;perpendicular/=np.linalg.norm(perpendicular)
        target=np.cos(x[5])*perpendicular+np.sin(x[5])*np.cross(source,perpendicular)
        return dict(initial,corner_mm=c+x[:3],source=source,target=target,pipe_radius_mm=initial['pipe_radius_mm']+x[6])
    solutions=[];begin=time.perf_counter()
    for axis in ('source','target','normal'):
        for sign in (-1,1):
            calls=0
            def decode(x):
                estimate=geometry(x)
                g=9.81*sign*(np.cross(estimate['source'],estimate['target']) if axis=='normal' else estimate[axis])
                return estimate,g
            def residual(x):
                nonlocal calls
                estimate,g=decode(x)
                prediction=predict(virtual_model(m,estimate,g,meta['source_model']),train,x[7:13])
                error=np.degrees(np.asarray(prediction['predicted'])-measured[:41])
                calls+=1
                return np.r_[error.reshape(-1)/.02,x[:3]/5.,x[3:6]/.1,x[6]/2.,x[7:10]/.03,x[10:13]/.3]
            def jacobian(x):
                columns=[]
                for i,h in enumerate(steps):
                    plus=x.copy();plus[i]+=h;minus=x.copy();minus[i]-=h
                    columns.append((residual(plus)-residual(minus))/(2*h))
                return np.column_stack(columns)
            fitted=least_squares(residual,start,jac=jacobian,bounds=(low,high),x_scale=scale,max_nfev=60,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            estimate,g=decode(fitted.x)
            solutions.append(dict(axis=axis,sign=sign,cost=float(fitted.cost),parameters=fitted.x,estimate=estimate,gravity=g,success=bool(fitted.success),evaluations=calls))
            print(scene,axis,sign,'cost',fitted.cost,'calls',calls,'wall',round(time.perf_counter()-begin,1),flush=True)
    best=min(solutions,key=lambda r:r['cost']);estimate=best['estimate'];g=best['gravity'];x=best['parameters']
    prediction=predict(virtual_model(m,estimate,g,meta['source_model']),records,x[7:13]);error=np.degrees(np.asarray(prediction['predicted'])-measured)
    i=int(np.argmin(abs(z['time']-z['time'][0]-at)));rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
    corner,_,target,_=pipe_frame(m)
    scores=dict(target_error_deg=float(np.degrees(np.arccos(np.clip((rot@estimate['target'])@target,-1,1)))),
        target_line_offset_mm=float(np.linalg.norm(np.cross(rot@estimate['corner_mm']+1000*z['qpos'][i,:3]-corner,target))),
        gravity_error_deg=float(np.degrees(np.arccos(np.clip((rot@g)@m.opt.gravity/(9.81**2),-1,1)))),
        training_rms_deg=float(np.sqrt(np.mean(error[:41]**2))),held_out_rms_deg=float(np.sqrt(np.mean(error[41:]**2))))
    write_json(root/('dynamic_fit_'+scene+'_aligned_gravity.json'),dict(scene=scene,time=at,estimate=estimate,gravity=g,
        parameters=np.r_[x[:7],g,x[7:13]],fit_velocity=True,seed_solutions=solutions,prediction=prediction,scoring=scores,
        wall_seconds=time.perf_counter()-begin,limitation='Six gravity candidates from an explicit horizontal/vertical right-angle prior. Training objective alone chooses candidate. Short-window offline diagnostic, exact nominal model, not arbitrary gravity/environment or control success.'))
    print(scene,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['horizontal','vertical'],required=True);run(p.parse_args().scene)
