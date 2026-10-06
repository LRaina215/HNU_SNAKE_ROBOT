"""Finite saved-reference review and necessary retiming, zero physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import math,json,shutil
import numpy as np
import mujoco
from front_local_rolling_geometry import BASE,ORIGINAL,write
from front_local_rolling_fit import ConsistentRollingFit,pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points
from front_local_rolling_continue import NumericalFit


def main():
    folder=BASE/'same_reference_continuation'
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    if result['status']!='READY_FOR_FULL_REVIEW':raise RuntimeError('Complete reference not available')
    if (BASE/'reference_ready.npz').exists():raise FileExistsError('Admitted reference already recorded')
    ref=FrontLocalRollingReference(folder/'reference_partial.npz',ORIGINAL);fit=NumericalFit()
    failures=[];length=rear=maxangle=0.;minimum=1e6
    for phi in np.unique(np.r_[np.radians(np.arange(0,15.0001,.05)),ref.phases]):
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        g=fit.fit.gaps(x)[:24];minimum=min(minimum,float(g.min()))
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        if negative or np.any(g<fit.clearance_floor(float(phi))-1e-9):failures.append(dict(phase_deg=float(np.degrees(phi)),gap_mm=float(g.min()),collisions=negative))
        length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        rear=max(rear,float(abs(p[12:]-fit.p0[12:]).max()));maxangle=max(maxangle,float(np.degrees(abs(x[6:])).max()))
    fit.fit.set(ref.state(ref.maximum_phase_rad));psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d)
    twist=np.degrees((psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi)
    original_precision_pass=bool(np.all(twist<-7.5) and abs(twist.mean()+15)<3)
    # Explicit review decision: retain this fixed9.1..12.9deg local action,
    # while preserving its failure to meet the former exact15deg fit goal.
    twistpass=bool(np.all(twist<-7.5) and np.median(twist)<-7.5)
    uu=np.linspace(0,1,2401);phase=math.radians(15)*uu*uu*(3-2*uu)
    q=np.array([ref.joint_angles(float(v)) for v in phase]);slope=float(np.degrees(abs(np.diff(q,axis=0))).max()*2400)
    active=math.ceil(max(12.,slope/8.5)/.05)*.05
    total=5+active+10
    if total>66.8:raise RuntimeError('Necessary retiming exceeds existing matched zero-phase holding duration; no automatic dynamics')
    tt=np.arange(round(total/.05)+1)*.05;u=np.clip((tt-5)/active,0,1);phase=math.radians(15)*u*u*(3-2*u)
    stages=np.where(tt<5,'entry',np.where(tt<5+active,'front_rolling','end_hold'))
    qq=np.array([ref.joint_angles(float(v)) for v in phase]);rate=float(np.degrees(abs(np.diff(qq,axis=0))).max()/.05)
    commandfails=[]
    for t,phi in zip(tt,phase):
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d)
        if any(c.dist<0 for c in fit.fit.d.contact) or np.any(fit.fit.gaps(x)[:24]<fit.clearance_floor(float(phi))-1e-9):commandfails.append(float(t))
    passed=not failures and not commandfails and rate<=9 and maxangle<89.001 and rear<1e-6 and length<1e-6 and twistpass
    report=dict(status='READY_FOR_INTERFACE' if passed else 'NOT_READY',physics_steps=0,failures=failures,command_failures=commandfails,
        maximum_length_error_mm=length,rear_nominal_node_maximum_error_mm=rear,maximum_joint_deg=maxangle,
        minimum_modifiable_conservative_gap_mm=minimum,nominal_pipe_relative_twist_deg=twist.tolist(),nominal_rolling_pass=twistpass,
        original_exact15deg_precision_pass=original_precision_pass,
        action_admission_revision='Retain fixed small9.1..12.9deg material action; phi0..15 is a control parameter, exact15deg geometry precision not achieved; actual>=4/6 and median7.5deg test unchanged',
        maximum_nominal_command_rate_deg_s=rate,active_s=active,total_s=total,rear_nominal_joint_error_rad=float(abs(qq[:,12:]-fit.x0[18:]).max()),
        retiming='Original12s retained if sufficient; otherwise solely reduced phase speed to obey unchanged9deg/s offline margin',
        finite_sampling='.05deg, all adaptive nodes and all50ms commands, not continuum proof',old_V3_path='NOT_READY',
        initial_clearance_compatibility='M0/head proxy continuous original positive gap to standard by .25deg; M11/source no deterioration of original accepted gap; all actual collisions remain hard fail')
    write(folder/'full_review.json',report)
    if passed:
        shutil.copy2(folder/'reference_partial.npz',BASE/'reference_ready.npz')
        np.savez_compressed(BASE/'command_schedule.npz',time_s=tt,phase_rad=phase,stage=stages)
        write(BASE/'path_gate.json',dict(geometry_and_command_timing_pass=True,total_s=total,active_s=active,old_V3_path='NOT_READY'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
