"""Review repaired source-preserving interpolation, schedule, and interface."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import shutil
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from target_open_close_geometry import BASE,ORIGINAL,write
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.target_open_close_reference import TargetOpenCloseReference


def schedule(t,durations):
    stages=[('entry',5.,0.,0.),('opening',durations[0],0.,35.),
        ('feed_open',durations[1],35.,115.625),('closing',durations[2],115.625,150.625),
        ('post_close_hold',10.,150.625,150.625),('feed_closed',durations[3],150.625,231.25),
        ('end_hold',10.,231.25,231.25)]
    start=0.
    for name,duration,a,b in stages:
        if t<start+duration-1e-9:
            u=np.clip((t-start)/duration,0,1)
            return float(a+(b-a)*u*u*(3-2*u)),name
        start+=duration
    return 231.25,'end_hold'


def main():
    if (BASE/'path_gate.json').exists():raise FileExistsError('Preserve completed review')
    shutil.copy2(BASE/'geometry_result.json',BASE/'geometry_root_interpolation_attempt0.json')
    reference=TargetOpenCloseReference(BASE/'reference_discrete.npz',ORIGINAL)
    fit=LocalPathFit();failures=[];maxangle=0.;length=0.;sourceerror=0.;key=[]
    samples=np.unique(np.r_[np.arange(0,reference.action_end,.25),reference.coordinates,reference.action_end])
    for a in samples:
        x=reference.state(float(a));fit.set(x);mujoco.mj_forward(fit.m,fit.d);p=points(fit.m,fit.d)
        contacts=[dict(geom1=fit.m.geom(c.geom1).name,geom2=fit.m.geom(c.geom2).name,
            body1=fit.m.body(int(fit.m.geom_bodyid[c.geom1])).name,body2=fit.m.body(int(fit.m.geom_bodyid[c.geom2])).name,
            distance_mm=float(c.dist*1000)) for c in fit.d.contact if c.dist<0]
        if contacts:failures.append(dict(action_mm=float(a),contacts=contacts))
        maxangle=max(maxangle,float(np.degrees(abs(x[6:])).max()))
        length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        fit.set(reference.original.state(reference.material_feed(a)));oldp=points(fit.m,fit.d)
        sourceerror=max(sourceerror,float(abs(p[10:]-oldp[10:]).max()))
        if a in [0,35,115.625,150.625,231.25]:
            offset=p[:9]-[1100,0,1000]
            key.append(dict(action_mm=float(a),feed_mm=reference.material_feed(a),opening_mm=reference.opening(a),
                joint_angles_deg=np.degrees(x[6:]).tolist(),points_mm=p.tolist(),
                distance_to_pipe_axis_mm=np.linalg.norm(offset[:,[0,2]],axis=1).tolist(),
                explanation='pipe-axis distances, not a free fitted helix radius'))
    durations=[12.,45.,12.,45.]
    ranges=[(0,35),(35,115.625),(115.625,150.625),(150.625,231.25)]
    for i,(lo,hi) in enumerate(ranges):
        us=np.linspace(0,1,1601);aa=lo+(hi-lo)*us*us*(3-2*us)
        qq=np.array([reference.joint_angles(float(a)) for a in aa])
        # Finite difference w.r.t. normalized phase gives necessary duration.
        slope=float(np.degrees(abs(np.diff(qq,axis=0))).max()*1600)
        durations[i]=math.ceil(max(durations[i],slope/8.)/.05)*.05
    total=25+sum(durations)
    if total>180:raise RuntimeError('Fixed action exceeds predeclared 180s bound; no physics')
    times=np.arange(round(total/.05)+1)*.05
    actions=[];stages=[];feeds=[];opening=[];qs=[];collisions=[]
    for t in times:
        a,stage=schedule(t,durations);x=reference.state(a)
        fit.set(x);mujoco.mj_forward(fit.m,fit.d)
        if any(c.dist<0 for c in fit.d.contact):collisions.append(float(t))
        actions.append(a);stages.append(stage);feeds.append(reference.material_feed(a));opening.append(reference.opening(a));qs.append(x[6:])
    rate=float(np.degrees(abs(np.diff(qs,axis=0))).max()/.05)
    passed=not failures and not collisions and maxangle<=89.001 and length<1e-6 and sourceerror<1e-6 and rate<=9.
    report=dict(status='READY_FOR_ACTION_SIM' if passed else 'NOT_READY',physics_steps=0,
        interpolation_failures=failures,command_collision_times=collisions,
        maximum_joint_deg=maxangle,maximum_length_error_mm=length,maximum_source_reference_node_error_mm=sourceerror,
        maximum_command_speed_deg_s=rate,durations_s=durations,total_s=total,key_poses=key,checked_count=len(samples),
        repair='Root solved from fixed link10 anchor for EVERY nominal interpolated q; source joints exact old material reference',
        original_failure='Independent root/joint interpolation moved source at195 samples, max0.5894mm; old attempt preserved',
        full_V3_path='NOT_READY',finite_sampling='0.25 action mm plus knots and all commands; not continuum proof')
    write(BASE/'geometry_result.json',report)
    if passed:
        shutil.copy2(BASE/'reference_discrete.npz',BASE/'reference_ready.npz')
        np.savez_compressed(BASE/'command_schedule.npz',time_s=times,action_mm=actions,feed_mm=feeds,opening_mm=opening,stage=stages)
        write(BASE/'path_gate.json',dict(geometry_and_command_timing_pass=True,physics_steps=0,
            total_s=total,durations_s=durations,maximum_nominal_command_speed_deg_s=rate,
            source_reference_unchanged=True,source_physical_support_not_proven=True,
            complete_d0_branch_equivalent_to_R2=True,controller_age_at_start_s=45.25,
            stopping='original ground/nonadjacent collision/loss/highspeed stops; during opening target contacts may disappear, rear must retain >=2 actual contacts',
            post_close_and_end='last1s original two-pipe contact/speed screen plus last2s displacement<=6.45mm',
            historical_holding='R2_SPEED3_PAIR_C0 66.8s; no extrapolated matched C0 beyond66.8s claimed',
            full_V3_path='NOT_READY'))
    print(json.dumps({k:report[k] for k in ['status','checked_count','maximum_command_speed_deg_s','total_s','maximum_source_reference_node_error_mm']}),flush=True)


if __name__=='__main__':main()
