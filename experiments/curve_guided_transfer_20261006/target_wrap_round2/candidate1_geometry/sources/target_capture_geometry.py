"""One finite-endpoint capture hypothesis, at most two offline constraints passes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
import json
import math
import shutil
import time
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.target_capture_reference import TargetCaptureReference

BASE = ROOT/'experiments/curve_guided_transfer_20261006/target_wrap_round2'
OUT = BASE/'candidate1_geometry'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    registry = json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    if registry['formal_revisions'] != 0:
        raise RuntimeError('Candidate1 already registered; preserve it')
    registry.update(formal_revisions=1, candidate1='Finite-module capture; offline admission pending')
    (BASE/'revision_registry.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    write('plan.json', dict(physics_steps=0, mechanism_revision=1,
        primary_hypothesis='M9 has no planned finite-shell capture at the old endpoint; coordinate its shape and receiving participation',
        action='x(d)=old x(d)+smooth((d-80.625)/80.625)*fixed finite capture offset; rotations composed on SO3',
        original_first_half_and_initial_unchanged=True, new_radius_search=False,
        constraints='same .05mm shell clearance, 89deg offline joints, 25mm guide RMS; endpoint M8/M9 plus two original target and source anchors within1mm',
        scope='five body-material guide with a local finite-unit adjustment near target entry; not five identified pipe axes',
        maximum_offline_passes=2, stop='second sampled constraint pass still fails or no material progress',
        controller='planned capture phase adds receiving-window compliance only; original axis estimator/gains/ACK/restoring/physics retained',
        hold_equivalence='d0 state/root/tangent and receiving phase identical for every encoder input and ACK state in a constant d0 trial',
        feedback='encoders and ACK history only; no oracle axis in this candidate',
        freeze_perturbation='root x +1mm, matching C0 required if primary full pass'))
    for name in ['validation/target_capture_geometry.py', 'snake_control/target_capture_reference.py']:
        dest=OUT/'sources'/Path(name).name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/name,dest)
    fit=LocalPathFit()
    old=BoundaryMaterialCurve(BASE/'inputs/reference_ready.npz')
    end=old.state(161.25)
    fit.set(end);base_points=points(fit.m,fit.d).copy()
    ell=fit.curve.material_positions(161.25)
    sources=sorted(np.flatnonzero(ell<0), key=lambda i:fit.radial_gap(int(i),0))[:2]
    targets=sorted(np.flatnonzero(ell>fit.curve.bounds[-1]),key=lambda i:fit.radial_gap(int(i),1))[:2]
    grips=[(int(i),0) for i in sources]+[(int(i),1) for i in targets]+[(8,1),(9,1)]
    units=np.r_[np.full(3,10.),np.full(23,.1)]
    seed=np.zeros(26);collocation=[161.25]
    samples=np.unique(np.r_[np.arange(0,161.25+1e-8,.25),old.feeds,(old.feeds[:-1]+old.feeds[1:])/2,161.25])
    schedule=np.load(BASE/'inputs/command_schedule.npz')
    def save_reference(path, offset):
        np.savez_compressed(path,feed_mm=old.feeds,states=old.states,
            start_tangent_rad_per_mm=old.start_tangent,capture_offset=offset)
    def state_at(d, offset):
        x=old.state(float(d));h=TargetCaptureReference.capture_phase(float(d))
        x[:3]+=h*offset[:3]
        x[3:6]=(Rotation.from_rotvec(h*offset[3:6])*Rotation.from_rotvec(x[3:6])).as_rotvec()
        x[6:]+=h*offset[6:]
        return x
    bounds=[(-3,3)]*3+[(-8,8)]*3+[((-math.radians(89)-end[i])/.1,(math.radians(89)-end[i])/.1) for i in range(6,26)]
    for pass_index in range(2):
        notices=[time.monotonic()];iterations=[0]
        def constraint(z):
            offset=units*z;values=[]
            for d in collocation:
                x=state_at(d,offset);fit.set(x)
                values.extend(fit.gaps(x)-.05)
                values.extend(math.radians(89)-abs(x[6:]))
            fit.set(state_at(161.25,offset));p=points(fit.m,fit.d)
            values.extend(1.-np.array([fit.radial_gap(i,s) for i,s in grips]))
            nodes=p[8:11]-[1100,0,1000];radial=np.linalg.norm(nodes[:,[0,2]],axis=1)
            values.extend(nodes[:,1]-75.001);values.extend(radial-56.5);values.extend(119.999-radial)
            values.append(25.-np.sqrt(np.mean(np.sum((p-base_points)**2,axis=1))))
            # Existing front nominal anchors should not be relocated wholesale.
            values.extend(16.125-np.linalg.norm(p[:9]-base_points[:9],axis=1))
            return np.array(values)
        def callback(z):
            iterations[0]+=1
            if time.monotonic()-notices[0]>15:
                print(json.dumps(dict(pass_index=pass_index,iteration=iterations[0],physics_steps=0)),flush=True)
                notices[0]=time.monotonic()
        result=minimize(lambda z:float(z@z),seed,method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=constraint)],callback=callback,
            options=dict(maxiter=80,ftol=1e-8,eps=1e-5))
        offset=units*result.x;archive=OUT/f'capture_pass{pass_index}.npz';save_reference(archive,offset)
        reference=TargetCaptureReference(archive)
        rows=[];failures=[];maxq=0.;maxlength=0.;maxrms=0.
        for d in samples:
            x=reference.state(float(d));fit.set(x);mujoco.mj_forward(fit.m,fit.d);p=points(fit.m,fit.d)
            contacts=[dict(geom1=fit.m.geom(c.geom1).name,geom2=fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.d.contact if c.dist<0]
            maxq=max(maxq,float(np.degrees(abs(x[6:])).max()))
            maxlength=max(maxlength,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
            oldx=old.state(float(d));fit.set(oldx);op=points(fit.m,fit.d);rms=float(np.sqrt(np.mean(np.sum((p-op)**2,axis=1))))
            maxrms=max(maxrms,rms)
            if contacts:failures.append(dict(feed_mm=float(d),contacts=contacts))
            if d in [0,80.625,161.25]:
                fit.set(x);rows.append(dict(feed_mm=float(d),target_gaps_mm=[float(fit.radial_gap(i,1)) for i in [0,6,7,8,9]],joint_angles_deg=np.degrees(x[6:]).tolist(),points_mm=p.tolist()))
        command_feeds=schedule['feed_mm'];qt=np.array([reference.joint_angles(float(d)) for d in command_feeds])
        commandrate=float(np.degrees(abs(np.diff(qt,axis=0))).max()/.05)
        commandcollisions=[]
        for d in np.unique(command_feeds):
            fit.set(reference.state(float(d)));mujoco.mj_forward(fit.m,fit.d)
            if any(c.dist<0 for c in fit.d.contact):commandcollisions.append(float(d))
        fit.set(reference.state(161.25));near=[float(fit.radial_gap(i,s)) for i,s in grips]
        passed=not failures and not commandcollisions and maxq<=89.001 and maxlength<1e-6 and maxrms<=25 and max(near)<=1.001 and commandrate<=9
        report=dict(status='READY_FOR_LOCAL_SIM' if passed else 'NOT_READY',physics_steps=0,pass_index=pass_index,
            solver_success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),
            endpoint_constraint_minimum=float(constraint(result.x).min()),offset=offset.tolist(),
            endpoint_grip_modules=grips,endpoint_grip_gaps_mm=near,failures=failures,
            command_collision_feeds=commandcollisions,maximum_nominal_command_rate_deg_s=commandrate,
            maximum_joint_deg=maxq,maximum_length_error_mm=maxlength,total_length_mm=1354.5,
            maximum_deviation_from_saved_reference_rms_mm=maxrms,checked_count=len(samples),key_poses=rows,
            finite_sampling_limitation='0.25mm plus knots/midpoints and all commands; not a continuum proof',
            full_V3_path='NOT_READY')
        write(f'capture_pass{pass_index}_review.json',report)
        print(json.dumps({k:report[k] for k in ['status','pass_index','iterations','endpoint_constraint_minimum','maximum_nominal_command_rate_deg_s','maximum_joint_deg']}),flush=True)
        if passed:
            shutil.copy2(archive,OUT/'reference_ready.npz')
            shutil.copy2(BASE/'inputs/command_schedule.npz',OUT/'command_schedule.npz')
            gate=json.loads((BASE/'inputs/path_gate.json').read_text(encoding='utf8'))
            gate.update(geometry_and_command_timing_pass=True,maximum_nominal_command_speed_deg_s=commandrate,candidate='finite module capture',checked_count=len(samples))
            write('path_gate.json',gate)
            shutil.copy2(BASE/'inputs/interface_check.json',OUT/'interface_check.json')
            write('result.json',report);return
        if not failures or pass_index==1:
            report['stop']='No self-consistent admitted capture with this fixed offset construct; no dynamics.'
            write('result.json',report);return
        # One meaningful second pass adds actual failed connections, retaining
        # the same geometry hypothesis and parameters, never a radius scan.
        worst=sorted(failures,key=lambda r:min(c['distance_mm'] for c in r['contacts']))
        collocation=sorted(set(collocation+[r['feed_mm'] for r in worst[:12]]))
        seed=result.x.copy()
    raise AssertionError('Unreachable')


if __name__=='__main__':main()
