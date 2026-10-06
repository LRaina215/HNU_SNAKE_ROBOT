"""One registered opposite-direction finite fit, from the original R2 start.

No physics. The same R/P, variables, shells and compatibility floors are used.
Nominal +Y is a separate admission condition; it is never scored as motion.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import math,json,time,shutil,argparse
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from front_local_rolling_geometry import BASE,ORIGINAL,write
from front_local_rolling_continue import NumericalFit
from front_local_rolling_fit import pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points
OUT=BASE/'direction_revision_roll2'


class OppositeFit(NumericalFit):
    twist_sign=-1.
    additional_admission_tolerance_mm=.001

    def __init__(self):
        super().__init__()
        extension=OUT/'numerical_completion_extension.json'
        if extension.exists():self.maximum_numerical_work_s=json.loads(extension.read_text(encoding='utf8'))['cumulative_limit_s']

    def clearance_pass(self,phi,g):
        tolerance=np.full(24,1e-9);tolerance[22]=1e-7
        return bool(np.all(g>=self.clearance_floor(phi)-tolerance))

    def additional_constraints(self,phi,z,p):
        c=(p[:7]+p[1:8])/2;c0=(self.p0[:7]+self.p0[1:8])/2
        # A small positive direction floor implements the already registered
        # forward geometry goal. It is not a new material feed or feedback.
        return np.array([np.median(c[:,1]-c0[:,1])-1.5*phi/math.radians(15)])


def main(resume=False):
    active=json.loads((BASE/'active_registry.json').read_text(encoding='utf8'))
    if not active['further_dynamics_authorized']:raise RuntimeError('Current study is closed; preserve its fixed reference and evidence')
    if not resume:
        OUT.mkdir(exist_ok=False)
        for name in ['initial_boundary_pose.npz','initial_boundary_pose_perturb.npz','original_reference.npz']:
            shutil.copy2(BASE/name,OUT/name)
    else:
        number=len(list(OUT.glob('*direction_history*')))
        history=OUT/('unconstrained_direction_history' if number==0 else f'forward_direction_history_{number+1}')
        history.mkdir(exist_ok=False)
        for name in ['progress.json','reference_partial.npz','revision_plan.json','numerical_pause.json','result.json','first_failure_diagnosis.json']:
            if (OUT/name).exists():shutil.copy2(OUT/name,history/name)
        for path in OUT.glob('interpolation_*.json'):shutil.copy2(path,history/path.name)
    write(OUT/'revision_plan.json',dict(prior_dynamics=18,maximum_total_dynamics=21,
        reason='ROLL1 rear axial slip, source contact/normal force retained; front actual -Y and nominal axial negative',
        single_revision='Reverse pipe-relative director target and continuum guide; same finite connector fit',
        parameter_deg=[0,15],desired_director_change_deg=[0,15],prior_desired_director_change_deg=[0,-15],
        radius_mm=95,pitch_mm=220,orbital_bound_deg=[-5,5],axial_bound_mm=[-32.25,32.25],
        initial='Exact original R2 before physics; no saved failure endpoint',rear_nominal_joints=[13,20],
        numerical_work_limit_s=2400 if (OUT/'numerical_completion_extension.json').exists() else 1800,
        no_direction_scan=True,no_new_gripping_rule=True,
        nominal_forward_admission='median M0..6 nominal target-axis displacement >1mm at end; not evidence of real forward motion',
        forward_fit_constraint='median nominal M0..6 +Y displacement >=1.5mm*parameter/15deg; fixes direction throughout with .5mm endpoint margin',
        forward_geometry_numerical_tolerance_mm=.001,actual_forward_gate_unchanged=True,
        near_invariant_source_baseline_comparison_tolerance_mm=1e-7,
        source_precision_reason='Successful projection had only2.175e-9mm source baseline residual; not physical penetration or new grip clearance',
        same_revision_continuation=resume,no_numerical_time_reset=True,
        actual_rolling_gate='>=4/6 and median pipe-relative director increase>=7.5deg',
        actual_forward_gate='median actual M0..6 +Y displacement>1mm, separately required for forward-roll-and-grasp claim',
        rear_anchor_gate='unchanged6.45mm/5deg/>=2source contacts',physical_control_and_end_hold_limits_unchanged=True,
        geometry_floor='existing original-start compatibility; actual penetration is hard fail',old_V3_path='NOT_READY'))
    fit=OppositeFit();saved={0.:np.r_[fit.x0[6:18],0.,0.]};rows=[]
    if resume:
        prior=json.loads((OUT/'progress.json').read_text(encoding='utf8'))
        pause=json.loads((OUT/'numerical_pause.json').read_text(encoding='utf-8-sig'))
        fit.used_s=max(prior['numerical_used_s'],pause['conservative_elapsed_numerical_s']);rows=prior['rows']
        with np.load(OUT/'reference_partial.npz') as z:
            for phi,q,a,o in zip(z['phase_rad'],z['joint_rad'],z['axial_mm'],z['orbital_rad']):saved[float(phi)]=np.r_[q,a,o]
    def save():
        keys=np.array(sorted(saved));values=np.array([saved[k] for k in keys])
        np.savez_compressed(OUT/'reference_partial.npz',phase_rad=keys,joint_rad=values[:,:12],axial_mm=values[:,12],orbital_rad=values[:,13])
        write(OUT/'progress.json',dict(physics_steps=0,numerical_used_s=fit.used_s,rows=rows))
    def refine(phases):
        for iteration in range(80):
            save();keys=np.array(sorted(saved));ip=PchipInterpolator(keys,np.array([saved[k] for k in keys]),axis=0)
            reference=FrontLocalRollingReference(OUT/'reference_partial.npz',ORIGINAL);failures={};beg=time.monotonic()
            for phi in phases:
                x=reference.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d)
                g=fit.fit.gaps(x)[:24]
                pc=points(fit.fit.m,fit.fit.d);extra=fit.additional_constraints(float(phi),ip(phi),pc)
                if any(c.dist<0 for c in fit.fit.d.contact) or not fit.clearance_pass(float(phi),g) or np.any(extra<-fit.additional_admission_tolerance_mm):
                    interval=int(np.clip(np.searchsorted(keys,phi,side='right')-1,0,len(keys)-2))
                    severity=min(float(np.min(g-fit.clearance_floor(float(phi)))),float(extra.min()))
                    if interval not in failures or severity<failures[interval][1]:failures[interval]=(float(phi),severity)
            fit.used_s+=time.monotonic()-beg
            if fit.used_s>=fit.maximum_numerical_work_s:raise TimeoutError('Registered cumulative numerical cap')
            if not failures:return True
            write(OUT/f'interpolation_repair_{iteration:02d}.json',dict(physics_steps=0,failures=failures))
            for phi,severity in failures.values():
                if phi==0:raise ValueError('Original start cannot be refitted')
                z,row=fit.solve_feasible(phi,ip(phi),True);rows.append(row);saved[phi]=z;save()
                print(json.dumps(dict(kind='interpolation_projection',phase_deg=float(np.degrees(phi)),ready=row['ready'],used_s=fit.used_s)),flush=True)
                if not row['ready']:return False
        return False
    status='INCOMPLETE';error=None
    try:
        for degree in np.r_[.25,.5,.75,np.arange(1,16)]:
            phi=math.radians(float(degree))
            if phi in saved and fit.evaluate(phi,saved[phi])['ready']:continue
            lower=[k for k in saved if k<phi];seed=saved[max(lower)]
            z,row=fit.solve_feasible(phi,seed)
            if not row['ready'] and not row['collisions'] and row['front_centerline_maximum_error_mm']<=6.45:
                z,row=fit.solve_feasible(phi,z,True)
            saved[phi]=z;rows.append(row);save()
            print(json.dumps({k:row[k] for k in ['phase_deg','ready','axial_mm','orbital_deg','pipe_relative_front_twist_change_deg','numerical_used_s']}),flush=True)
            if not row['ready']:status='FINITE_POSE_NOT_READY';break
        else:
            dense=np.unique(np.r_[np.radians(np.arange(0,15.0001,.05)),list(saved)])
            status='READY_FOR_REVIEW' if refine(dense) else 'CONTINUOUS_GEOMETRY_NOT_READY'
        if status=='READY_FOR_REVIEW':
            # Recompute the slowest necessary schedule and verify each command.
            # If interpolation changes after projection, timing is recalculated.
            for iteration in range(12):
                save();reference=FrontLocalRollingReference(OUT/'reference_partial.npz',ORIGINAL)
                u=np.linspace(0,1,2401);ph=math.radians(15)*u*u*(3-2*u)
                q=np.array([reference.joint_angles(float(v)) for v in ph])
                slope=float(np.degrees(abs(np.diff(q,axis=0))).max()*2400)
                active=math.ceil(max(12.,slope/8.5)/.05)*.05;total=active+15
                if total>66.8:status='TIMING_EXCEEDS_MATCHED_BASELINE';break
                tt=np.arange(round(total/.05)+1)*.05;u=np.clip((tt-5)/active,0,1);ph=math.radians(15)*u*u*(3-2*u)
                if not refine(np.unique(np.r_[dense,ph])):status='COMMAND_GEOMETRY_NOT_READY';break
                save();ref2=FrontLocalRollingReference(OUT/'reference_partial.npz',ORIGINAL)
                qq=np.array([ref2.joint_angles(float(v)) for v in ph]);rate=float(np.degrees(abs(np.diff(qq,axis=0))).max()/.05)
                if rate>9:continue
                fit.fit.set(ref2.state(ref2.maximum_phase_rad));p=points(fit.fit.m,fit.fit.d)
                psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d);twist=np.degrees((psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi)
                centers=(p[:7]+p[1:8])/2;c0=(fit.p0[:7]+fit.p0[1:8])/2;advance=centers[:,1]-c0[:,1]
                maximum_angle=rear_error=length_error=rear_joint_error=0.;timeline=[]
                beg=time.monotonic()
                for value in np.unique(np.r_[dense,ph]):
                    x=ref2.state(float(value));fit.fit.set(x);pcheck=points(fit.fit.m,fit.fit.d)
                    maximum_angle=max(maximum_angle,float(np.degrees(abs(x[6:])).max()))
                    rear_error=max(rear_error,float(abs(pcheck[12:]-fit.p0[12:]).max()))
                    rear_joint_error=max(rear_joint_error,float(abs(x[18:]-fit.x0[18:]).max()))
                    length_error=max(length_error,float(abs(np.linalg.norm(np.diff(pcheck,axis=0),axis=1)-64.5).max()))
                for value in np.radians(np.arange(0,15.001,.25)):
                    x=ref2.state(float(value));fit.fit.set(x);pcheck=points(fit.fit.m,fit.fit.d)
                    pp=(pcheck[:7]+pcheck[1:8])/2;ps,_=pipe_relative_twist(fit.fit.m,fit.fit.d)
                    timeline.append(dict(parameter_deg=float(np.degrees(value)),joint_deg=np.degrees(x[6:]).tolist(),
                        nominal_front_y_displacement_mm=(pp[:,1]-c0[:,1]).tolist(),
                        nominal_director_change_deg=np.degrees((ps[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi).tolist()))
                fit.used_s+=time.monotonic()-beg;write(OUT/'nominal_motion_timeline.json',timeline)
                ready=bool(np.count_nonzero(twist>=7.5)>=4 and np.median(twist)>=7.5 and np.median(advance)>1
                    and maximum_angle<89.001 and rear_error<1e-6 and length_error<1e-6 and rear_joint_error<1e-12)
                report=dict(physics_steps=0,geometry_and_command_timing_pass=ready,total_s=total,active_s=active,
                    maximum_nominal_rate_deg_s=rate,nominal_twist_change_deg=twist.tolist(),nominal_axial_displacement_mm=advance.tolist(),
                    nominal_median_forward_mm=float(np.median(advance)),control_parameter_deg=[0,15],rolling_sign=1,
                    maximum_joint_deg=maximum_angle,maximum_length_error_mm=length_error,
                    rear_nominal_node_maximum_error_mm=rear_error,rear_nominal_joint_error_rad=rear_joint_error,
                    exact15deg_precision_pass=bool(np.all(twist>=7.5) and abs(twist.mean()-15)<3),
                    initial_exact=np.array_equal(ref2.state(0),fit.x0),finite_sampling='.05deg and every50ms command; not a continuum proof',
                    old_V3_path='NOT_READY')
                write(OUT/'path_gate.json',report)
                if ready:
                    shutil.copy2(OUT/'reference_partial.npz',OUT/'reference_ready.npz')
                    stages=np.where(tt<5,'entry',np.where(tt<5+active,'front_rolling','end_hold'))
                    np.savez_compressed(OUT/'command_schedule.npz',time_s=tt,phase_rad=ph,stage=stages)
                    status='READY_FOR_INTERFACE'
                else:status='NOMINAL_DIRECTION_OR_ROLL_NOT_READY'
                break
            else:status='TIMING_CLOSURE_INCOMPLETE'
    except Exception as e:
        error=repr(e);status='NUMERICAL_WORK_CAP' if isinstance(e,TimeoutError) else 'NUMERICAL_ERROR'
    save();write(OUT/'result.json',dict(status=status,error=error,physics_steps=0,numerical_used_s=fit.used_s,
        saved_node_count=len(saved),largest_saved_parameter_deg=float(np.degrees(max(saved))),total_dynamics=18,
        primary_failure_preserved=True,unique_targeted_revision=True))
    print(json.dumps(dict(status=status,error=error,used_s=fit.used_s)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--continue-forward',action='store_true');main(p.parse_args().continue_forward)
