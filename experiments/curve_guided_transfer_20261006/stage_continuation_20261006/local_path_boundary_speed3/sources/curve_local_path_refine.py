"""Adaptive collocation of the fixed local reference, no physics stepping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import math
import shutil
import numpy as np
import mujoco
from scipy.optimize import minimize,OptimizeResult
from scipy.spatial.transform import Rotation
from curve_local_path import LocalPathFit,OUT,write
from curve_guided_geometry import points
from snake_control.local_material_reference import CoupledMaterialCurve


def inspect(fit,reference):
    f=reference.feeds
    samples=np.unique(np.r_[np.arange(0,f[-1]+1e-8,.25),f,(f[:-1]+f[1:])/2,f[-1]])
    failures=[];slopes=[];worst_length=0.;maxq=0.;maxrms=0.
    for feed in samples:
        x=reference.state(float(feed));fit.set(x);mujoco.mj_forward(fit.m,fit.d)
        negative=[dict(geom1=int(c.geom1),geom2=int(c.geom2),distance_mm=float(c.dist*1000)) for c in fit.d.contact if c.dist<0]
        if negative:failures.append(dict(feed_mm=float(feed),contacts=negative))
        p=points(fit.m,fit.d)
        worst_length=max(worst_length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        maxq=max(maxq,float(np.degrees(abs(x[6:])).max()))
        maxrms=max(maxrms,float(np.sqrt(np.mean(np.sum((p-fit.desired(float(feed)))**2,axis=1)))))
        slopes.append(float(np.degrees(abs(reference.joint_derivative(float(feed)))).max()))
    speed=np.minimum(5.,9./np.maximum(slopes,1e-9))
    return dict(geometry_pass=not failures and maxq<=89.001 and worst_length<1e-6 and maxrms<=25,
        physics_steps=0,interpolation='coupled smoothstep translation/joints, SLERP at the same smoothstep parameter',
        continuity='C1 zero derivatives at all joins; proper rotations, no rotvec-component interpolation',
        failures=failures,failure_count=len(failures),checked_count=len(samples),spacing_mm=.25,
        maximum_penetration_mm=max([-c['distance_mm'] for r in failures for c in r['contacts']],default=0),
        maximum_joint_deg=maxq,maximum_length_error_mm=worst_length,total_length_mm=1354.5,
        maximum_guide_rms_mm=maxrms,maximum_rate_at_5mm_s_deg_s=max(slopes)*5,
        reparameterized_active_duration_s=float(np.trapz(1/speed,samples)),planned_speed_limit_deg_s=9.,
        finite_sampling_limitation='0.25mm plus all knots/midpoints does not prove every arbitrary real position')


def repair(fit,feed,seed):
    # Nearest feasible collocation point to the current coupled connection:
    # avoids independent refits that would introduce new arbitrary branches.
    Rseed=Rotation.from_rotvec(seed[3:6]);units=np.r_[np.full(3,10.),np.full(23,.1)]
    def unpack(z):
        x=seed+units*z
        x[3:6]=(Rotation.from_rotvec(units[3:6]*z[3:6])*Rseed).as_rotvec()
        return x
    fit.last=None;fit.set(seed)
    ell=fit.curve.material_positions(feed)
    grips=[]
    for side,ids in enumerate([np.flatnonzero(ell<0),np.flatnonzero(ell>fit.curve.bounds[-1])]):
        grips += [(int(i),side) for i in sorted(ids,key=lambda i:abs(fit.radial_gap(int(i),side)))[:2]]
    def constraint(z):
        x=unpack(z);fit.set(x)
        return np.r_[fit.gaps(x)-.05,1.-np.array([fit.radial_gap(i,s) for i,s in grips])]
    bounds=[(-3,3)]*3+[(-8,8)]*3
    bounds += [((-math.radians(89)-seed[i])/.1,(math.radians(89)-seed[i])/.1) for i in range(6,26)]
    class Feasible(Exception):pass
    accepted=[None];iterations=[0]
    def early_stop(z):
        iterations[0]+=1
        if constraint(z).min()>=-1e-6:
            mujoco.mj_forward(fit.m,fit.d)
            if not any(c.dist<0 for c in fit.d.contact):
                rms=float(np.sqrt(np.mean(np.sum((points(fit.m,fit.d)-fit.desired(feed))**2,axis=1))))
                if rms<=25:
                    accepted[0]=z.copy();raise Feasible()
    try:
        result=minimize(lambda z:float(z@z),np.zeros(26),method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=constraint)],callback=early_stop,
            options=dict(maxiter=80,ftol=1e-9,eps=1e-5))
    except Feasible:
        result=OptimizeResult(x=accepted[0],success=False,nit=iterations[0],
            message='Geometry-feasible early stop; objective optimum is not claimed')
    x=unpack(result.x);fit.set(x);mujoco.mj_forward(fit.m,fit.d)
    negative=[float(c.dist*1000) for c in fit.d.contact if c.dist<0]
    gaps=fit.gaps(x);near=[float(fit.radial_gap(i,s)) for i,s in grips]
    rms=float(np.sqrt(np.mean(np.sum((points(fit.m,fit.d)-fit.desired(feed))**2,axis=1))))
    passed=not negative and float(gaps.min())>=.049 and max(near)<=1.001 and rms<=25
    row=dict(feed_mm=float(feed),pose_pass=passed,iterations=int(result.nit),solver_success=bool(result.success),
        solver_message=str(result.message),minimum_conservative_gap_mm=float(gaps.min()),
        grip_gaps_mm=near,contacts_mm=negative,guide_rms_mm=rms,
        correction_translation_mm=float(np.linalg.norm(x[:3]-seed[:3])),
        correction_joint_max_deg=float(np.degrees(abs(x[6:]-seed[6:])).max()))
    return x,row


def main():
    if (OUT/'reference_ready.npz').exists():raise FileExistsError('Completed path must remain frozen')
    if not (OUT/'attempt1_result.json').exists():
        shutil.copy2(OUT/'result.json',OUT/'attempt1_result.json')
        shutil.copy2(OUT/'reference_discrete.npz',OUT/'attempt1_reference_discrete.npz')
        shutil.copy2(ROOT/'validation/curve_local_path.py',OUT/'sources/curve_local_path_attempt1.py')
    fit=LocalPathFit()
    source=OUT/'reference_discrete.npz'
    with np.load(source) as z:feeds,states=z['feed_mm'].copy(),z['states'].copy()
    for pass_index in range(5):
        archive=OUT/f'refinement_{pass_index:02d}.npz'
        if archive.exists() and not (pass_index==0 and (OUT/'refinement_00_slow_solver_partial.json').exists()):
            raise FileExistsError('Preserve each refinement pass; explicit resume needed')
        np.savez_compressed(archive,feed_mm=feeds,states=states)
        reference=CoupledMaterialCurve(archive)
        review=inspect(fit,reference)
        write(f'refinement_{pass_index:02d}_review.json',review)
        print(json.dumps({k:v for k,v in review.items() if k!='failures'}),flush=True)
        if review['geometry_pass']:
            shutil.copy2(archive,OUT/'reference_ready.npz')
            write('refinement_result.json',dict(status='INTERFACE_AND_TIMING_CHECK_NEEDED',physics_steps=0,
                final_pass=pass_index,knots=len(feeds),review=review,controller_revision=2,old_V3_gate_unchanged=True))
            return
        if pass_index==4:
            write('refinement_result.json',dict(status='NOT_READY',physics_steps=0,final_pass=pass_index,
                knots=len(feeds),review=review,stop='Residual interpolation collision after four adaptive collocation passes'))
            return
        intervals=sorted(set(min(np.searchsorted(feeds,r['feed_mm'],side='right')-1,len(feeds)-2)
                             for r in review['failures']))
        additions=[];rows=[]
        for index in intervals:
            midpoint=float((feeds[index]+feeds[index+1])/2)
            x,row=repair(fit,midpoint,reference.state(midpoint))
            additions.append((midpoint,x));rows.append(row)
            write(f'refinement_{pass_index:02d}_insertions.json',dict(physics_steps=0,rows=rows))
            np.savez_compressed(OUT/f'refinement_{pass_index:02d}_inserted_states.npz',
                feed_mm=np.array([d for d,x in additions]),states=np.array([x for d,x in additions]))
            print(json.dumps(dict(pass_index=pass_index,**row)),flush=True)
            if not row['pose_pass']:
                write('refinement_result.json',dict(status='NOT_READY',physics_steps=0,failed_insertion=row,
                    stop='Nearest collocation repair failed; no dynamics admitted'))
                return
        merged=sorted(list(zip(feeds,states))+additions,key=lambda v:v[0])
        feeds=np.array([d for d,x in merged]);states=np.array([x for d,x in merged])


if __name__=='__main__':main()
