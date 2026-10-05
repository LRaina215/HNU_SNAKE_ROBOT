"""Test whether an encoder-conditioned private dynamics model tracks pipe axes.

The initial geometry/gravity comes from the active joint-response fit. Later
input consists solely of measured joint angles and issued commands. Real root
pose and pipe axes are read only by the separate scoring section. This replay
does not command the physical robot or prove closed-loop crossing.
"""
import argparse
import json
import time
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from feedback_takeover_dynamic_observer_probe import virtual_model


def run(scene,stop,condition):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    suffix='_joint_velocity_precise_central' if scene=='vertical' else ''
    fit=json.loads((root/('dynamic_fit_'+scene+suffix+'.json')).read_text(encoding='utf8'))
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,robot,_,meta=context(folder)
    model=virtual_model(robot,fit['estimate'],fit['gravity'],meta['source_model'])
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]];vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
    times=z['time']-z['time'][0];start=int(np.argmin(abs(times-fit['time'])))
    d=mujoco.MjData(model);d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qadr]=z['qpos'][start,qadr]
    if fit.get('fit_velocity',False):d.qvel[:6]=fit['parameters'][10:16]
    d.qvel[vadr]=(z['qpos'][start+1,qadr]-z['qpos'][start,qadr])/.01
    mujoco.mj_forward(model,d)
    previous=z['qpos'][start,qadr].copy();rows=[];begin=time.perf_counter()
    # The true world data below is isolated to scoring; no observation update
    # consumes it. qvel, physical forces and contact labels are never inputs.
    corner,source,target,_=pipe_frame(robot)
    for i in range(start+1,min(len(times),int(round(stop/.01))+1)):
        dt=float(times[i]-times[i-1]);d.ctrl[:]=z['ctrl'][i]
        for _ in range(round(dt/model.opt.timestep)):mujoco.mj_step(model,d)
        measured=z['qpos'][i,qadr];prediction_error=np.degrees(d.qpos[qadr]-measured)
        if condition:
            d.qpos[qadr]=measured
            d.qvel[vadr]=(measured-previous)/dt
            mujoco.mj_forward(model,d)
        previous=measured.copy()
        if i%20:continue
        estimated_rot=np.zeros(9);mujoco.mju_quat2Mat(estimated_rot,d.qpos[3:7]);estimated_rot=estimated_rot.reshape(3,3)
        local_target=estimated_rot.T@np.asarray(fit['estimate']['target'])
        local_corner=estimated_rot.T@(np.asarray(fit['estimate']['corner_mm'])-1000*d.qpos[:3])
        actual_rot=np.zeros(9);mujoco.mju_quat2Mat(actual_rot,z['qpos'][i,3:7]);actual_rot=actual_rot.reshape(3,3)
        error=float(np.degrees(np.arccos(np.clip((actual_rot@local_target)@target,-1,1))))
        offset=float(np.linalg.norm(np.cross(actual_rot@local_corner+1000*z['qpos'][i,:3]-corner,target)))
        rows.append(dict(time=float(times[i]),target_error_deg=error,target_offset_mm=offset,
            joint_prediction_rms_deg=float(np.sqrt(np.mean(prediction_error**2))),private_contacts=d.ncon,
            estimated_qpos=d.qpos.copy(),estimated_qvel=d.qvel.copy()))
        if i%1000==0:print(scene,condition,round(times[i],2),'axis',round(error,3),'offset',round(offset,3),'wall',round(time.perf_counter()-begin,1),flush=True)
    report=dict(scene=scene,condition_on_measured_joints=condition,fit_file='dynamic_fit_'+scene+suffix+'.json',rows=rows,
        wall_seconds=time.perf_counter()-begin,
        limitation='Recorded commands/angles only after response-based calibration; perfect nominal dynamics assumed. Private state corrections are not physical interventions. Offline diagnostic, not closed-loop success.')
    write_json(root/('dynamic_tracking_'+scene+('_conditioned' if condition else '_open')+'.json'),report)
    print(scene,condition,'max axis/offset',max(r['target_error_deg'] for r in rows),max(r['target_offset_mm'] for r in rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['horizontal','vertical'],required=True)
    p.add_argument('--stop',type=float,default=100.);p.add_argument('--condition',action='store_true');a=p.parse_args();run(a.scene,a.stop,a.condition)
