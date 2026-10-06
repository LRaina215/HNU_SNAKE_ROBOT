"""Pre-physics reviewed admission of the frozen2400s ROLL2 reference.

No optimization or physics; old failed margins preserved. All real outcome
criteria remain unchanged. Proxy clearance and axial fit goals are distinct
from actual penetration, direction and motion.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import math,json,time,shutil,csv
import numpy as np
import mujoco
from front_local_rolling_direction_revision import OUT,OppositeFit
from front_local_rolling_geometry import ORIGINAL,write
from front_local_rolling_fit import pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points


def main():
    if (OUT/'fixed_reference_review.json').exists():raise FileExistsError('Preserve fixed-reference admission')
    source=OUT/'reference_partial.npz';frozen=OUT/'reference_frozen_at_2400.npz'
    shutil.copy2(source,frozen)
    write(OUT/'admission_revision.json',dict(physics_steps=0,reviewer_thread='01a10b5d-0c3d-7772-95ae-a95998f833ea',
        old_admission_pass=False,prior_failed_result='result.json',source=frozen.name,no_further_fitting=True,
        prior_nominal_gap_goal_mm=.05,prior_linear_forward_goal_mm='1.5*parameter/15deg',
        revised_proxy_minimum_mm=0.,proxy_floating_tolerance_mm=1e-8,actual_entity_contact_minimum_mm=0.,
        revised_direction='median nominal front+Y >=0 throughout with.001mm precision, endpoint>1mm; not monotonic',
        exact_initial_and_source_M11_compatibility_retained=True,source_M11_comparison_precision_mm=1e-7,
        nominal_rolling_amplitude_and_physical_gates_unchanged=True,
        reason='Pre-physics margin tradeoff after fixed-reference numerical closure; tiny positive margins/linear fit residuals are not physical impossibility',
        maximum_total_dynamics=21,unique_targeted_revision=True))
    ref=FrontLocalRollingReference(frozen,ORIGINAL);fit=OppositeFit();beg=time.monotonic()
    # Necessary retiming only. No modification of joints, phase amplitude or R/P.
    u=np.linspace(0,1,2401);ph=math.radians(15)*u*u*(3-2*u)
    qq=np.array([ref.joint_angles(float(v)) for v in ph]);slope=float(np.degrees(abs(np.diff(qq,axis=0))).max()*2400)
    active=math.ceil(max(12.,slope/8.5)/.05)*.05;total=15+active
    tt=np.arange(round(total/.05)+1)*.05;u=np.clip((tt-5)/active,0,1);ph=math.radians(15)*u*u*(3-2*u)
    q=np.array([ref.joint_angles(float(v)) for v in ph]);rate=float(np.degrees(abs(np.diff(q,axis=0))).max()/.05)
    c0=(fit.p0[:7]+fit.p0[1:8])/2;rows=[];failures=[];oldfail=[]
    length=rear=rear_joint=angle=0.;minimum=1e6
    for phi in np.unique(np.r_[np.radians(np.arange(0,15.0001,.05)),ref.phases,ph]):
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        g=fit.fit.gaps(x)[:24];minimum=min(minimum,float(g.min()))
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        c=(p[:7]+p[1:8])/2;dy=c[:,1]-c0[:,1];median=float(np.median(dy));linear=float(fit.additional_constraints(phi,np.zeros(14),p)[0])
        psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d);twist=np.degrees((psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi)
        row=dict(parameter_deg=float(np.degrees(phi)),nominal_front_displacement_mm=dy.tolist(),median_nominal_forward_mm=median,
            nominal_director_change_deg=twist.tolist(),minimum_proxy_gap_mm=float(g.min()),collisions=negative,
            source_M11_gap_mm=float(g[22]),original_linear_forward_excess_mm=linear)
        rows.append(row)
        if not fit.clearance_pass(phi,g) or linear<-.001:oldfail.append(row)
        bad=np.any(g<-1e-8) or bool(negative) or median<-.001 or g[22]<fit.initial_gap[22]-1e-7
        if bad:failures.append(row)
        length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()));rear=max(rear,float(abs(p[12:]-fit.p0[12:]).max()))
        rear_joint=max(rear_joint,float(abs(x[18:]-fit.x0[18:]).max()));angle=max(angle,float(np.degrees(abs(x[6:])).max()))
    end=rows[-1];twist=np.array(end['nominal_director_change_deg']);forward=np.array([r['median_nominal_forward_mm'] for r in rows])
    passed=bool(not failures and total<=66.8 and rate<=9 and length<1e-6 and rear<1e-6 and rear_joint<1e-12 and angle<89.001
        and np.all(twist>=7.5) and np.median(twist)>=7.5 and end['median_nominal_forward_mm']>1 and np.array_equal(ref.state(0),fit.x0))
    review=dict(physics_steps=0,geometry_and_command_timing_pass=passed,independent_fixed_reference_check=True,
        old_admission_pass=not oldfail,old_failed_samples=len(oldfail),old_failed_record='fixed_reference_old_admission_failures.json',
        revised_failures=failures,samples=len(rows),minimum_proxy_gap_mm=minimum,maximum_length_error_mm=length,
        rear_nominal_node_maximum_error_mm=rear,rear_nominal_joint_error_rad=rear_joint,maximum_joint_deg=angle,
        maximum_nominal_rate_deg_s=rate,active_s=active,total_s=total,exact_initial=True,
        nominal_director_change_deg=twist.tolist(),exact15deg_precision_pass=bool(abs(twist.mean()-15)<3),
        nominal_median_forward_mm=end['median_nominal_forward_mm'],nominal_maximum_forward_mm=float(forward.max()),
        nominal_maximum_return_mm=float((np.maximum.accumulate(forward)-forward).max()),nominal_is_monotonic=bool(np.min(np.diff(forward))>=-1e-6),
        rolling_sign=1,admission_revision='admission_revision.json',old_V3_path='NOT_READY',check_wall_s=time.monotonic()-beg,
        finite_sampling='.05deg/all saved nodes/all50ms commands; no continuum proof',physical_outcome_gates_unchanged=True)
    write(OUT/'fixed_reference_review.json',review);write(OUT/'fixed_reference_old_admission_failures.json',oldfail)
    write(OUT/'nominal_motion_timeline.json',rows);write(OUT/'path_gate.json',review)
    with (OUT/'key_joint_postures.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['pose','parameter_deg']+['J'+str(i) for i in range(1,21)])
        for degree in [0,5,10,15]:writer.writerow(['NOMINAL_ONLY',degree]+np.degrees(ref.joint_angles(math.radians(degree))).tolist())
        adr=fit.fit.m.jnt_qposadr[fit.fit.m.actuator_trnid[:,0]]
        with np.load(OUT/'initial_boundary_pose.npz') as z:writer.writerow(['SAVED_ACTUAL_R2_INITIAL',0]+np.degrees(z['qpos'][adr]).tolist())
    if passed:
        shutil.copy2(frozen,OUT/'reference_ready.npz')
        stages=np.where(tt<5,'entry',np.where(tt<5+active,'front_rolling','end_hold'))
        np.savez_compressed(OUT/'command_schedule.npz',time_s=tt,phase_rad=ph,stage=stages)
    print(json.dumps(review),flush=True)


if __name__=='__main__':main()
