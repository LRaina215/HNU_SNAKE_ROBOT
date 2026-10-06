"""Independent finite review of saved ROLL2; no optimization/physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json,math,csv,time
import numpy as np
import mujoco
from front_local_rolling_direction_revision import OUT,OppositeFit
from front_local_rolling_geometry import ORIGINAL,write
from front_local_rolling_fit import pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    if result['status']!='READY_FOR_INTERFACE':raise RuntimeError('No complete forward candidate to review')
    start=time.monotonic();ref=FrontLocalRollingReference(OUT/'reference_ready.npz',ORIGINAL);fit=OppositeFit()
    with np.load(OUT/'command_schedule.npz') as z:ph=z['phase_rad'].copy();tt=z['time_s'].copy()
    failures=[];rear=length=maximum=rear_joint=0.;timeline=[];min_forward_excess=1e6
    c0=(fit.p0[:7]+fit.p0[1:8])/2
    sample=np.unique(np.r_[np.radians(np.arange(0,15.0001,.05)),ref.phases,ph])
    for phi in sample:
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        g=fit.fit.gaps(x)[:24];pc=(p[:7]+p[1:8])/2;advance=pc[:,1]-c0[:,1]
        excess=float(fit.additional_constraints(phi,np.zeros(14),p)[0]);min_forward_excess=min(min_forward_excess,excess)
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        if negative or not fit.clearance_pass(phi,g) or excess<-fit.additional_admission_tolerance_mm:
            failures.append(dict(parameter_deg=float(np.degrees(phi)),minimum_gap_mm=float(g.min()),forward_excess_mm=excess,collisions=negative))
        rear=max(rear,float(abs(p[12:]-fit.p0[12:]).max()));rear_joint=max(rear_joint,float(abs(x[18:]-fit.x0[18:]).max()))
        length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()));maximum=max(maximum,float(np.degrees(abs(x[6:])).max()))
        psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d)
        timeline.append(dict(parameter_deg=float(np.degrees(phi)),median_nominal_forward_mm=float(np.median(advance)),
            nominal_front_displacement_mm=advance.tolist(),nominal_director_change_deg=np.degrees((psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi).tolist()))
    q=np.array([ref.joint_angles(float(v)) for v in ph]);rate=float(np.degrees(abs(np.diff(q,axis=0))/np.diff(tt)[:,None]).max())
    end=timeline[-1];twist=np.array(end['nominal_director_change_deg'])
    passed=bool(not failures and rate<=9 and maximum<89.001 and rear<1e-6 and rear_joint<1e-12 and length<1e-6
        and np.count_nonzero(twist>=7.5)>=4 and np.median(twist)>=7.5 and end['median_nominal_forward_mm']>1
        and np.array_equal(ref.state(0),fit.x0) and tt[-1]<=66.8)
    review=dict(physics_steps=0,independent_review_pass=passed,geometry_and_command_timing_pass=passed,
        sample_count=len(sample),failures=failures,maximum_length_error_mm=length,rear_nominal_node_maximum_error_mm=rear,
        rear_nominal_joint_error_rad=rear_joint,maximum_joint_deg=maximum,maximum_nominal_rate_deg_s=rate,
        total_s=float(tt[-1]),active_s=float(tt[-1]-15),nominal_director_change_deg=twist.tolist(),
        nominal_median_forward_mm=end['median_nominal_forward_mm'],minimum_forward_constraint_excess_mm=min_forward_excess,
        exact_initial=True,finite_sampling='.05deg/all fitted knots/all actual50ms command parameters; not a continuous proof',
        rolling_sign=1,old_V3_path='NOT_READY',review_wall_s=time.monotonic()-start)
    write(OUT/'independent_review.json',review);write(OUT/'nominal_motion_timeline.json',timeline)
    with (OUT/'key_joint_postures.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['pose','parameter_deg']+['J'+str(i) for i in range(1,21)])
        for degree in [0,5,10,15]:writer.writerow(['NOMINAL_ONLY',degree]+np.degrees(ref.joint_angles(math.radians(degree))).tolist())
        model=fit.fit.m;adr=model.jnt_qposadr[model.actuator_trnid[:,0]]
        with np.load(OUT/'initial_boundary_pose.npz') as z:writer.writerow(['SAVED_ACTUAL_R2_INITIAL',0]+np.degrees(z['qpos'][adr]).tolist())
    gate=json.loads((OUT/'path_gate.json').read_text(encoding='utf8'));gate['geometry_and_command_timing_pass']=passed
    gate['independent_review']='independent_review.json';write(OUT/'path_gate.json',gate)
    print(json.dumps(review),flush=True)


if __name__=='__main__':main()
