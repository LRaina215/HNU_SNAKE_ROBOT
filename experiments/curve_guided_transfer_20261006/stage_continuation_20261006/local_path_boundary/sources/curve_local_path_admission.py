"""Fixed-reference timing and held-pose interface checks, without mj_step."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import math
import shutil
import argparse
import numpy as np
import mujoco
from scipy.integrate import cumulative_trapezoid
from curve_local_path import OUT,BASE,LocalPathFit,write
from snake_control.local_material_reference import CoupledMaterialCurve,BoundaryMaterialCurve
from snake_control.local_transfer_score import LocalTransferScore


def main(boundary=False):
    global OUT
    if boundary:OUT=BASE/'local_path_boundary'
    if (OUT/'path_gate.json').exists():raise FileExistsError('Preserve the registered admission result')
    reviewed=json.loads((OUT/'refinement_result.json').read_text(encoding='utf8'))
    if not reviewed['review']['geometry_pass']:raise RuntimeError('Geometry gate closed')
    reference=(BoundaryMaterialCurve if boundary else CoupledMaterialCurve)(OUT/'reference_ready.npz');fit=LocalPathFit()
    dense=np.unique(np.r_[np.arange(0,161.25,.01),reference.feeds,80.625,161.25])
    slope=np.array([np.degrees(abs(reference.joint_derivative(float(d)))).max() for d in dense])
    speed=np.minimum(5.,9./np.maximum(slope,1e-9))
    activeclock=cumulative_trapezoid(1/speed,dense,initial=0)
    middle=float(np.interp(80.625,dense,activeclock));duration=float(activeclock[-1])
    n1=math.ceil(middle/.05);n2=math.ceil((duration-middle)/.05)
    f1=np.interp(np.arange(1,n1+1)/n1*middle,activeclock,dense)
    f2=np.interp(middle+np.arange(1,n2+1)/n2*(duration-middle),activeclock,dense)
    nentry,npause,nend=100,60,100
    feeds=np.r_[np.zeros(nentry),f1,np.full(npause,80.625),f2,np.full(nend,161.25)]
    stages=np.array(['entry']*nentry+['feed_1']*n1+['pause']*npause+['feed_2']*n2+['end_hold']*nend)
    active=np.r_[np.zeros(nentry),np.arange(1,n1+1)*.05,np.full(npause,n1*.05),
                  n1*.05+np.arange(1,n2+1)*.05,np.full(nend,(n1+n2)*.05)]
    times=np.arange(len(feeds)+1)*.05
    # Last row is endpoint metadata; each actual command consumes one interval.
    feeds=np.r_[feeds,feeds[-1]];stages=np.concatenate([stages,stages[-1:]]);active=np.r_[active,active[-1]]
    nominal=np.array([reference.joint_angles(float(d)) for d in feeds])
    rates=np.degrees(abs(np.diff(nominal,axis=0)))/.05
    with np.load(BASE/'R2_C0/trace.npz') as z:
        qpos=z['qpos'][-1].copy();ctrl=z['ctrl'][-1].copy();qvel=np.zeros_like(z['qvel'][-1])
        initialpoints=z['points_mm'][-1];initialforces=z['contact_force_N'][-1]
    # Verify the known prior and preserved acknowledged target. The matching
    # interface C0, not this forward check, must establish dynamic stability.
    fit.d.qpos[:]=qpos;fit.d.qvel[:]=qvel;fit.d.ctrl[:]=ctrl;mujoco.mj_forward(fit.m,fit.d)
    scorer=LocalTransferScore(initialpoints,initialforces)
    residual=np.degrees(ctrl-reference.joint_angles(0.))
    geometric_command_contacts=[]
    for d in np.unique(feeds):
        fit.set(reference.state(float(d)));mujoco.mj_forward(fit.m,fit.d)
        negative=[float(c.dist*1000) for c in fit.d.contact if c.dist<0]
        if negative:geometric_command_contacts.append(dict(feed_mm=float(d),contacts_mm=negative))
    passed=bool(not geometric_command_contacts and rates.max()<=9.001 and
        (n1+n2)*.05<=60 and times[-1]<=78 and np.all(np.isfinite(qpos)) and
        np.all(abs(ctrl)<np.radians(90)))
    plan=dict(geometry_and_command_timing_pass=passed,physics_steps=0,
        active_duration_s=(n1+n2)*.05,total_B_duration_s=float(times[-1]),
        paired_C0_duration_s=max(45.25,float(times[-1])),entry_s=5.,pause_s=3.,end_hold_s=5.,
        maximum_nominal_command_speed_deg_s=float(rates.max()),
        command_geometry_collision_count=len(geometric_command_contacts),command_geometry_collisions=geometric_command_contacts,
        dense_timing_mesh_mm=.01,geometric_check_mesh_mm=.25,
        interpolation='BoundaryMaterialCurve: saved first tangent Hermite, other coupled smoothstep/SLERP' if boundary else 'CoupledMaterialCurve: common smoothstep scalar and SO(3) SLERP',
        projection_direction='segment limiting joint direction at zero-speed nodes; separate true joint_derivative used for timing',
        controller='R2 frozen; tau .5, K .01/.02, original angle/step/force/physical limits',
        interface='actual R2 final qpos/acknowledged ctrl; qvel zero only before first physical step',
        inherited_prior_and_age=bool(boundary),controller_age_at_start_s=45.25 if boundary else 0.,
        first_command_retained_residual_deg=residual.tolist(),
        preserved_state_identity='r=acknowledged_target - new_nominal(0); no latent state and no online measured-reference rebase',
        material_modules=[8,9],initial_scored_membership=[scorer.in_region(initialpoints,initialforces,i) for i in [8,9]],
        freeze_difference='root x +1mm before first step; same reference and controller, same-perturbation C0 required',
        dynamic_interface_holding_pass=False,old_V3_full_path='NOT_READY',
        finite_sampling_limitation='sampled admission only; real shell/contact behavior still requires dynamics')
    write('path_gate.json',plan)
    if passed:
        np.savez_compressed(OUT/'command_schedule.npz',time_s=times,feed_mm=feeds,stage=stages,active_time_s=active)
        prefix='initial_boundary_pose' if boundary else 'initial_path_pose'
        np.savez_compressed(BASE/(prefix+'.npz'),qpos=qpos,qvel=qvel,target=ctrl)
        perturbed=qpos.copy();perturbed[0]+=.001
        np.savez_compressed(BASE/(prefix+'_perturb.npz'),qpos=perturbed,qvel=qvel,target=ctrl)
        for name in ['validation/curve_local_path_refine.py','validation/curve_local_path_admission.py',
                     'snake_control/local_material_reference.py']:
            shutil.copy2(ROOT/name,OUT/'sources'/Path(name).name)
    print(json.dumps({k:plan[k] for k in ['geometry_and_command_timing_pass','active_duration_s','total_B_duration_s',
        'paired_C0_duration_s','maximum_nominal_command_speed_deg_s','command_geometry_collision_count','initial_scored_membership']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--boundary',action='store_true')
    main(parser.parse_args().boundary)
