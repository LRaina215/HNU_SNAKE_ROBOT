"""Stage-2 identifiability experiment: small pulses with/without target contact.

Each diagnostic pair restores the SAME recorded q/qvel/command into model
copies. Only the negative copy disables target-pipe collision. This is NOT a
crossing candidate or a modification of the saved environment. Joint angles
and issued commands generate features; actual contacts/pose score them only.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,write_json


OUT=ROOT/'experiments/segmented_pole_20261005/active_takeover'


def features(t,q,u,pattern,reference,gain):
    # Fit constant and drift together with the requested pulse. No simulator
    # force, contact, floating pose, qvel or paired counterfactual enters.
    basis=np.column_stack((np.ones(len(t)),t,u,np.cos(2*np.pi*np.maximum(t-.2,0)/.8)*(t>=.2)*(t<=1.)))
    projected=(q-reference)@pattern/max(float(pattern@pattern),1e-12)
    fit=np.linalg.lstsq(basis,projected,rcond=None)[0]
    angle=(q-reference)@pattern/max(float(np.linalg.norm(pattern)),1e-12)
    load=np.clip(gain*(reference+u[:,None]*pattern-q),-2.355,2.355)
    return dict(pulse_gain=float(fit[2]),quadrature=float(fit[3]),
        pulse_magnitude=float(np.hypot(fit[2],fit[3])),
        residual_deg=float(np.degrees(np.sqrt(np.mean((q[-10:]-reference)**2)))),
        projected_motion_deg=float(np.degrees(np.ptp(angle))),
        mean_absolute_load=float(np.mean(abs(load))))


def simulate(model,start,pattern,missing_target):
    m=copy.copy(model);d=mujoco.MjData(m)
    target=m.geom('pipe_straight_03').id
    if missing_target:
        m.geom_contype[target]=0;m.geom_conaffinity[target]=0
    d.qpos[:]=start['qpos'];d.qvel[:]=start['qvel'];d.ctrl[:]=start['ctrl']
    mujoco.mj_forward(m,d)
    ids=m.actuator_trnid[:,0];qadr=m.jnt_qposadr[ids]
    measured=d.qpos[qadr].copy();kp=m.actuator_gainprm[:,0]
    cap=m.actuator_forcerange[:,1]/kp
    reference=measured+np.clip(start['ctrl']-measured,-cap,cap)
    records=[];duration=1.2;dt=.02
    for k in range(round(duration/dt)+1):
        t=k*dt
        u=float(np.sin(2*np.pi*(t-.2)/.8)) if .2<=t<=1. else 0.
        q=d.qpos[qadr].copy();cmd=reference+u*pattern
        d.ctrl[:]=np.clip(cmd,m.jnt_range[ids,0]+.01,m.jnt_range[ids,1]-.01)
        contact=0;normal=0.
        for j in range(d.ncon):
            item=d.contact[j]
            if target not in (item.geom1,item.geom2):continue
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,j,wrench)
            if wrench[0]>.1:contact+=1;normal+=float(wrench[0])
        records.append(dict(time=t,q=q,command=d.ctrl.copy(),u=u,
            evaluation_only=dict(target_contacts=contact,target_normal_N=normal,head_position=d.qpos[:3].copy())))
        if k<round(duration/dt):
            for _ in range(round(dt/m.opt.timestep)):mujoco.mj_step(m,d)
    t=np.array([r['time'] for r in records]);q=np.array([r['q'] for r in records]);u=np.array([r['u'] for r in records])
    feat=features(t,q,u,pattern,reference,kp)
    return dict(records=records,features=feat,
        scoring=dict(target_contact_fraction=float(np.mean([r['evaluation_only']['target_contacts']>0 for r in records])),
            mean_target_normal_N=float(np.mean([r['evaluation_only']['target_normal_N'] for r in records])),
            head_displacement_m=float(np.linalg.norm(d.qpos[:3]-start['qpos'][:3]))))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'probe_trials.json').exists():raise FileExistsError('Preserve the previous probe experiment; choose a new output directory.')
    shutil.copy2(Path(__file__),OUT/'feedback_takeover_active_probe.py')
    write_json(OUT/'protocol.json',dict(duration=1.2,pulse_duration=.8,settling=.2,amplitude_deg=.6,
        input='joint angle, issued target and calibrated position gain; physical contact/pose scoring only',
        negative_control='Target collision disabled in a separate model copy. No geometry is added, no torque/friction is raised.',
        limitation='Restored-state diagnostic pairs, not a crossing rollout or proof of force closure.',
        training='horizontal checkpoints at 0,16,32,40s',validation='vertical checkpoints at 0,16,32,40s; same pulses and feature formula'))
    rows=[]
    for scene in ('horizontal','vertical'):
        name='horizontal_encoder_live_whole_coil' if scene=='horizontal' else 'vertical_encoder_live_whole_coil'
        folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/name
        _,z,model,_,_=context(folder)
        time=z['time']-z['time'][0];qpos=z['qpos'];qvel=z['qvel'];ctrl=z['ctrl']
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
        for at in (0.,16.,32.,40.):
            i=int(np.argmin(abs(time-at)))
            info=min(controls,key=lambda c:abs(c['time']-.2-at))
            front=min(18,int(info['actual_prefix']))
            start=dict(qpos=qpos[i].copy(),qvel=qvel[i].copy(),ctrl=ctrl[i].copy())
            patterns={}
            for parity,label in ((0,'middle_yaw'),(1,'middle_pitch')):
                index=front+(parity-front)%2
                delta=np.zeros(20);delta[index]=np.radians(.6);patterns[label]=delta
            contraction=start['qpos'][qadr].copy();contraction[front:]=0.
            contraction*=np.radians(.6)/max(abs(contraction).max(),1e-12)
            patterns['front_contraction']=contraction
            patterns['hold']=np.zeros(20)
            for label,delta in patterns.items():
                for missing in (False,True):
                    trial=simulate(model,start,delta,missing)
                    trial.update(scene=scene,snapshot_seconds=at,pattern=label,front=front,target_removed=missing,
                        pulse_joint_indices=np.flatnonzero(delta).tolist())
                    rows.append(trial)
                print(scene,at,label,'gain present/missing',rows[-2]['features']['pulse_gain'],rows[-1]['features']['pulse_gain'],flush=True)
            write_json(OUT/'probe_trials.json',dict(trials=rows))
    compact=[{k:v for k,v in row.items() if k!='records'} for row in rows]
    write_json(OUT/'summary.json',dict(trials=compact))


if __name__=='__main__':run()
