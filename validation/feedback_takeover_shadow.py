"""Score a frozen geometry hypothesis on recorded positive/negative windows.

Encoder joint angles are reconstructed in an identity head frame. Truth is
used only after inference for scoring; no control or physical state changes.
The contact labels describe these observations, not a support certificate.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json,pipe_frame
from snake_control.mujoco_checkpoint import load_checkpoint
from snake_control.encoder_pipe_geometry import right_angle_hypothesis

OUT=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'
BASE=ROOT/'experiments/segmented_pole_20261005/support_transfer'


def run():
    rows=[]
    cases=[('horizontal',HORIZONTAL,'radial_front_relative_braking'),
           ('vertical_down',VERTICAL,'vertical_initial_front_reference'),
           ('vertical_failed',VERTICAL,'vertical_frozen_front_relative')]
    for scene,path,name in cases:
        m,d,meta,_=load_checkpoint(path,verify_hashes=False)
        robot=mujoco.MjData(m);robot.qpos[:7]=[0,0,0,1,0,0,0]
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        with np.load(BASE/name/'trace.npz') as z:
            data={k:z[k] for k in ['time','qpos','ctrl','torque','normal_force','ground']}
        relative=data['time']-data['time'][0]
        indices=np.unique(np.r_[np.arange(0,min(801,len(relative)),40),
            np.arange(max(0,len(relative)-801),len(relative),40)])
        _,source,target,_=pipe_frame(m)
        for index in indices:
            robot.qpos[qadr]=data['qpos'][index,qadr];mujoco.mj_kinematics(m,robot)
            points=1000*np.vstack([robot.site('s_head').xpos,
                *[robot.joint('joint'+str(i)).xanchor for i in range(1,21)],robot.site('s_tail').xpos])
            clock=time.perf_counter();estimate=right_angle_hypothesis(points,meta['settings']['radius'])
            elapsed=time.perf_counter()-clock
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,data['qpos'][index,3:7]);rotation=rotation.reshape(3,3)
            errors=[float(np.degrees(np.arccos(np.clip(abs((rotation@estimate[k])@axis),0,1))))
                    for k,axis in [('source',source),('target',target)]]
            force=data['normal_force'][index]
            row=dict(run=name,time=float(relative[index]),estimate=estimate,inference_seconds=elapsed,
                observed_load_fraction=abs(data['torque'][index])/m.actuator_forcerange[:,1],
                scoring=dict(source_error_deg=errors[0],target_error_deg=errors[1],
                    source_contact_modules=np.flatnonzero(force[0]>.1),
                    target_contact_modules=np.flatnonzero(force[1]>.1),ground=bool(data['ground'][index])))
            rows.append(row)
        subset=rows[-len(indices):]
        print(name,'samples',len(subset),'geometry candidates',sum(r['estimate']['geometric_candidate'] for r in subset),flush=True)
    write_json(OUT/'shadow_geometry.json',dict(rows=rows,
        inference_inputs='Joint encoders, robot geometry, nominal coil radius. Identity head frame.',
        priors='Intersecting right-angle pipe axes, comparable radii, pre-transfer rear coil.',
        distinction='A geometric candidate is not contact, support or takeover permission.',
        negatives='Completed single-target coils and failed-run terminal poses; no natural approach trace yet.',
        limitation='Development data from two scene captures; not held-out classifier validation.'))


if __name__=='__main__':run()
