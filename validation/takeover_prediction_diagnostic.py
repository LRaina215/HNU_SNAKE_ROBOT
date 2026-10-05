"""Read-only forecast audit from the existing encoder/command dynamics belief.

Recorded future commands are supplied for an offline known-input prediction;
this is not an online policy or an autonomous crossing result. Physical root
pose, velocities and contacts are used only in score(), after prediction.
"""
import argparse
import json
from pathlib import Path
import copy
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import write_json
from feedback_takeover_dynamic_tracker import DynamicPipeTracker


def module_centers(model,data):
    points=np.vstack([data.site('s_head').xpos,
        *[data.joint('joint'+str(i)).xanchor for i in range(1,21)],data.site('s_tail').xpos])
    return (points[:-1]+points[1:])/2


def forecast(tracker,commands,period):
    m=tracker.observer.model;d=copy.copy(tracker.observer.data)
    target_slot=tracker.pipe_slots[1];last=module_centers(m,d);speed=[];contacts=[];q=[]
    for command in commands:
        d.ctrl[:]=command
        for _ in range(round(period/.01)):
            mujoco.mj_step(m,d,nstep=round(.01/m.opt.timestep));mujoco.mj_forward(m,d)
            centers=module_centers(m,d);speed.append(float(np.linalg.norm((centers-last)/.01,axis=1).max()));last=centers
            active=False
            for i in range(d.ncon):
                con=d.contact[i]
                if target_slot not in (con.geom1,con.geom2):continue
                other=con.geom2 if con.geom1==target_slot else con.geom1
                if int(m.geom_bodyid[other]) not in tracker.body_to_module:continue
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
                active=active or wrench[0]>.1
            contacts.append(active)
        q.append(d.qpos[tracker.qadr].copy())
    return dict(peak_speed_m_s=max(speed),target_contact_fraction=float(np.mean(contacts)),q=np.array(q))


def run(folder,seconds=140.,horizon=1.,stride=2.,adaptive_flow=False):
    _,z,robot,_,meta=context(folder)
    with z:truth={k:z[k].copy() for k in ('time','qpos','points_mm','normal_force')}
    time=truth['time'];qadr=robot.jnt_qposadr[robot.actuator_trnid[:,0]]
    encoders=truth['qpos'][:,qadr]
    commands=json.loads((folder/'commands.json').read_text(encoding='utf8'))
    probe=json.loads((folder/'response_probe.json').read_text(encoding='utf8'))
    fit=json.loads((folder/'response_calibration.json').read_text(encoding='utf8'))
    tracker=DynamicPipeTracker(robot,meta['source_model'],fit,probe,adaptive_flow=adaptive_flow)
    period=.02;count=round(horizon/period);every=round(stride/period);rows=[]
    centers=(truth['points_mm'][:,:-1]+truth['points_mm'][:,1:])/2000
    actual_speed=np.linalg.norm(np.diff(centers,axis=0)/np.diff(time)[:,None,None],axis=2).max(axis=1)
    exact_updates=0
    for i,command in enumerate(commands):
        if command['time']-time[0]>seconds:break
        # These are exactly the causal inputs available to the original filter.
        if i:
            at=int(np.argmin(abs(time-command['time'])))
            # Diagnostic ablation: linearize the full sampling interval after
            # a poor encoder prediction instead of exponentiating one local
            # 1-ms contact mode. This modifies only the private observer.
            tracker.update(encoders[at],np.array(commands[i-1]['target']),period)
            exact_updates+=int(tracker.info['exact_interval_linearization'])
        if i%every or i+count>len(commands):continue
        predicted=forecast(tracker,[c['target'] for c in commands[i:i+count]],period)
        future=np.array([int(np.argmin(abs(time-(c['time']+period)))) for c in commands[i:i+count]])
        a=int(np.argmin(abs(time-command['time'])));b=future[-1]
        rows.append(dict(seconds=float(command['time']-time[0]),
            predicted_peak_m_s=predicted['peak_speed_m_s'],actual_peak_m_s=float(actual_speed[a:b].max()),
            predicted_target_contact=predicted['target_contact_fraction'],
            actual_target_contact=float(np.mean(np.any(truth['normal_force'][a+1:b+1,1]>.1,axis=1))),
            joint_prediction_rms_deg=float(np.degrees(np.sqrt(np.mean((predicted['q']-encoders[future])**2)))),
            innovation_rms_deg=tracker.info.get('innovation_rms_deg',0.)))
        if i%round(20/period)==0:print(folder.parent.name,round(command['time']-time[0],1),rows[-1],flush=True)
    predictions=np.array([r['predicted_peak_m_s'] for r in rows]);actual=np.array([r['actual_peak_m_s'] for r in rows])
    result=dict(folder=str(folder),horizon_seconds=horizon,stride_seconds=stride,records=rows,
        adaptive_flow=adaptive_flow,exact_interval_updates=exact_updates,
        speed_error_mean_m_s=float(np.mean(abs(predictions-actual))),
        speed_error_max_m_s=float(abs(predictions-actual).max()),
        actual_overspeed_windows=int(np.sum(actual>.5)),predicted_overspeed_windows=int(np.sum(predictions>.5)),
        missed_overspeed_windows=int(np.sum((actual>.5)&(predictions<=.5))),
        limitation='Offline known-future-command audit, causal encoder filter; no controller modification. Nominal exact dynamics assumed.')
    write_json(folder/('prediction_diagnostic'+('_adaptive_flow' if adaptive_flow else '')+'.json'),result)
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--seconds',type=float,default=140.);p.add_argument('--horizon',type=float,default=1.)
    p.add_argument('--stride',type=float,default=2.);p.add_argument('--adaptive-flow',action='store_true')
    a=p.parse_args();run(a.folder,a.seconds,a.horizon,a.stride,a.adaptive_flow)
