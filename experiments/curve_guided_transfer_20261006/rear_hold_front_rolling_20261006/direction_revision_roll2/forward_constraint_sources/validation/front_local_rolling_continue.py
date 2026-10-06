"""Resume the same reference; at most1800s actual numerical work, no physics.

The .05mm nominal target uses the already registered .001mm tolerance for
the near-invariant M11/source row. Moving front hull constraints keep .05.
Existing feasible nodes are retained, no restart or new motion candidate.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import time,math,json,shutil
import numpy as np
import mujoco
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator
from front_local_rolling_geometry import BASE,ORIGINAL,write
from front_local_rolling_fit import ConsistentRollingFit,pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points
OUT=BASE/'same_reference_continuation'


class NumericalFit(ConsistentRollingFit):
    # Parameter remains nonnegative; a separately registered direction can
    # reverse the director target and continuum guide without changing R/P.
    twist_sign = 1.
    additional_admission_tolerance_mm=1e-6
    def __init__(self):
        super().__init__();self.used_s=0.;self.initial_gap=self.fit.gaps(self.x0)[:24].copy()

    def additional_constraints(self,phi,z,p):
        return np.empty(0)

    def clearance_floor(self,phi,optimization=False):
        floor=np.full(24,.05 if optimization else .049)
        # Existing immutable start has a positive near-anchor source gap below
        # nominal tolerance. Preserve it, rather than moving phi0.
        floor[22]=self.initial_gap[22]
        # Circumscribed head proxy also has an inherited initial shortfall.
        # Continuously recover normal clearance by the first existing .25deg
        # repair node. Finite actual pipe/self collisions are never relaxed.
        u=np.clip(phi/math.radians(.25),0,1);h=u*u*(3-2*u)
        floor[1]=self.initial_gap[1]+h*(floor[1]-self.initial_gap[1])
        return floor

    def solve_feasible(self,phi,seed,nearest=False):
        start=time.monotonic();remaining=1800-self.used_s
        if remaining<=0:raise TimeoutError('1800s actual numerical work cap')
        units=np.r_[np.full(12,.1),10.,.1]
        low=np.r_[np.full(12,-math.radians(89)),-32.25,-math.radians(5)];high=-low
        seed=np.clip(seed,low,high)
        qguide=self.x0[6:12]+self.old.prototype.joint_angles(0.,self.twist_sign*phi)[:6]-self.old.prototype.joint_angles(0.,0.)[:6]
        def residual(v):
            if time.monotonic()-start>remaining:raise TimeoutError('1800s numerical work cap')
            z=seed+units*v;x=self.state(z);g=self.fit.gaps(x)[:24]
            nominal_floor=self.clearance_floor(phi,True)
            violations=np.minimum(g-nominal_floor,0.)/.001
            if nearest:return np.r_[.001*v,violations]
            p=points(self.fit.m,self.fit.d)
            desired=(self.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0,z[13],0]).as_matrix().T+[1100,z[12],1000]
            psi,_=pipe_relative_twist(self.fit.m,self.fit.d);twist=(psi[:6]-self.psi0[:6]+self.twist_sign*phi+np.pi)%(2*np.pi)-np.pi
            original=np.r_[((p[:8]-desired)/4).ravel(),twist/math.radians(1.5),
                .08*(z[:6]-qguide)/math.radians(5),.025*(z[:12]-seed[:12])/math.radians(5),
                z[12]/32.25*.05,z[13]/math.radians(5)*.05]
            return np.r_[original,violations]
        def jacobian(v):
            step=1e-5;columns=[]
            for i in range(14):
                a=v.copy();b=v.copy();a[i]+=step;b[i]-=step
                columns.append((residual(a)-residual(b))/(2*step))
            return np.array(columns).T
        try:
            if True:
                def constraints(v):
                    if time.monotonic()-start>remaining:raise TimeoutError('1800s numerical work cap')
                    z=seed+units*v;x=self.state(z);g=self.fit.gaps(x)[:24]
                    p=points(self.fit.m,self.fit.d)
                    desired=(self.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0,z[13],0]).as_matrix().T+[1100,z[12],1000]
                    return np.r_[g-self.clearance_floor(phi,True),6.45-np.linalg.norm(p[:8]-desired,axis=1),
                        self.additional_constraints(phi,z,p)]
                last_notice=[time.monotonic()]
                def notice(v):
                    if time.monotonic()-last_notice[0]>20:
                        print(json.dumps(dict(phase_deg=float(np.degrees(phi)),solver='normalized_SLSQP',physics_steps=0)),flush=True)
                        last_notice[0]=time.monotonic()
                objective=(lambda v:float(v@v)) if nearest else (lambda v:float(residual(v)[:-24]@residual(v)[:-24]))
                out=minimize(objective,np.zeros(14),method='SLSQP',
                    bounds=list(zip((low-seed)/units,(high-seed)/units)),constraints=[dict(type='ineq',fun=constraints)],
                    callback=notice,options=dict(maxiter=150,ftol=1e-8,eps=1e-6))
                out.nfev=getattr(out,'nfev',out.nit)
        finally:self.used_s+=time.monotonic()-start
        z=seed+units*out.x
        row=self.evaluate(phi,z)
        row.update(solver_success=bool(out.success),evaluations=int(out.nfev),nearest_projection=nearest,numerical_used_s=self.used_s)
        return z,row

    def evaluate(self,phi,z):
        x=self.state(z);g=self.fit.gaps(x)[:24];mujoco.mj_forward(self.fit.m,self.fit.d);p=points(self.fit.m,self.fit.d)
        negative=[dict(geom1=self.fit.m.geom(c.geom1).name,geom2=self.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in self.fit.d.contact if c.dist<0]
        desired=(self.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0,z[13],0]).as_matrix().T+[1100,z[12],1000]
        psi,_=pipe_relative_twist(self.fit.m,self.fit.d);change=(psi[:6]-self.psi0[:6]+np.pi)%(2*np.pi)-np.pi
        error=float(np.linalg.norm(p[:8]-desired,axis=1).max())
        return dict(phase_deg=float(np.degrees(phi)),minimum_modifiable_gap_mm=float(g.min()),collisions=negative,
            front_centerline_maximum_error_mm=error,pipe_relative_front_twist_change_deg=np.degrees(change).tolist(),
            axial_mm=float(z[12]),orbital_deg=float(np.degrees(z[13])),joint_angles_deg=np.degrees(x[6:]).tolist(),points_mm=p.tolist(),
            ready=bool(np.all(g>=self.clearance_floor(phi)-1e-9) and not negative and error<=6.45+1e-6
                and np.all(self.additional_constraints(phi,z,p)>=-self.additional_admission_tolerance_mm)))


def main():
    if OUT.exists():
        before=json.loads((OUT/'result.json').read_text(encoding='utf8'))
        if before['status'] not in ['NUMERICAL_ERROR','STARTING_CONTINUOUS_GEOMETRY_NOT_READY','FINITE_POSE_NOT_READY','CONTINUOUS_GEOMETRY_NOT_READY']:raise FileExistsError('Preserve completed continuation')
        history_number=len(list(OUT.glob('history_result_*.json')))+1
        shutil.copy2(OUT/'result.json',OUT/f'history_result_{history_number}.json')
    else:OUT.mkdir()
    previous=BASE/'numerical_closure'
    write(OUT/'plan.json',dict(physics_steps=0,actual_numerical_work_limit_s=1800,prior_TIME_CAP_preserved=True,
        material_phase_deg=[0,15],orbital_deg=[-5,5],axial_mm=[-32.25,32.25],radius_mm=95,pitch_mm=220,
        nominal_gap_mm=.05,registered_numeric_tolerance_mm=.001,
        near_invariant_row='M11/source22 optimizer uses existing .049 admission floor, other moving hull rows .05; no physical collision relaxation',
        existing_nodes_reused=True,adaptive_nodes_only_where_saved_interpolation_fails=True,new_dynamics=0,total_dynamics=17))
    fit=NumericalFit();saved={};rows=[]
    resume_reference=OUT/'reference_partial.npz'
    if not resume_reference.exists():resume_reference=previous/'reference_augmented_partial.npz'
    with np.load(resume_reference) as z:
        for phi,q,a,o in zip(z['phase_rad'],z['joint_rad'],z['axial_mm'],z['orbital_rad']):saved[float(phi)]=np.r_[q,a,o]
    old=FrontLocalRollingReference(BASE/'reference_partial.npz',ORIGINAL)
    with np.load(BASE/'reference_partial.npz') as z:
        oldphi=z['phase_rad'].copy();olda=z['axial_mm'].copy();oldo=z['orbital_rad'].copy()
    phi2=math.radians(2);saved[phi2]=np.r_[old.joint_angles(phi2)[:12],np.interp(phi2,oldphi,olda),np.interp(phi2,oldphi,oldo)]
    r3=json.loads((previous/'augmented_nearest3.json').read_text(encoding='utf8'))
    saved[math.radians(3)]=np.r_[np.radians(r3['joint_angles_deg'][:12]),r3['axial_mm'],math.radians(r3['orbital_deg'])]
    if (OUT/'progress.json').exists():
        progress=json.loads((OUT/'progress.json').read_text(encoding='utf8'));fit.used_s=progress['numerical_used_s']
        rows=progress['rows']
    saved[0.]=np.r_[fit.x0[6:18],0.,0.]
    write(OUT/'baseline_compatibility.json',dict(physics_steps=0,initial_source_gap_mm=float(fit.initial_gap[22]),
        initial_head_proxy_gap_mm=float(fit.initial_gap[1]),zero_reference_exact=True,
        source='M11/source must not worsen its accepted original positive gap; nominal .05 target is not a uniform feasible starting bound',
        head='Inherited circumscribed head proxy shortfall ramps continuously to standard clearance by first .25deg node; actual pipe/self collision requirement unchanged',
        other_moving_shells='nominal .05mm with original .001mm numerical tolerance',
        thresholds_all_unchanged=False,holding_and_dynamics_thresholds_unchanged=True))
    def save():
        keys=np.array(sorted(saved));zz=np.array([saved[k] for k in keys])
        np.savez_compressed(OUT/'reference_partial.npz',phase_rad=keys,joint_rad=zz[:,:12],axial_mm=zz[:,12],orbital_rad=zz[:,13])
        write(OUT/'progress.json',dict(physics_steps=0,rows=rows,numerical_used_s=fit.used_s))
    save();status='INCOMPLETE';error=None
    def refine(maximum):
        # Find actual geometry/clearance failures before choosing added knots.
        for refinement in range(100):
            save();reference=FrontLocalRollingReference(OUT/'reference_partial.npz',ORIGINAL)
            keys=np.array(sorted(saved));ip=PchipInterpolator(keys,np.array([saved[k] for k in keys]),axis=0)
            failures={};check_start=time.monotonic()
            for phi in np.radians(np.arange(0,np.degrees(maximum)+.0001,.05)):
                x=reference.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d)
                g=fit.fit.gaps(x)[:24]
                bad=any(c.dist<0 for c in fit.fit.d.contact) or np.any(g<fit.clearance_floor(float(phi))-1e-9)
                if bad:
                    interval=min(max(np.searchsorted(keys,phi,side='right')-1,0),len(keys)-2)
                    severity=float(np.min(g-fit.clearance_floor(float(phi))))
                    if interval not in failures or severity<failures[int(interval)][1]:failures[int(interval)]=(float(phi),severity)
            fit.used_s+=time.monotonic()-check_start
            if fit.used_s>=1800:raise TimeoutError('1800s actual numerical work cap')
            if not failures:return True
            write(OUT/f'refinement_{np.degrees(maximum):g}_{refinement}.json',dict(physics_steps=0,failed_intervals=failures))
            for interval,(phi,severity) in failures.items():
                if phi==0:raise ValueError('Original start must never be projected or modified')
                if min(abs(keys-phi))<1e-9:
                    seed=saved[float(keys[abs(keys-phi).argmin()])]
                else:seed=ip(phi)
                z,row=fit.solve_feasible(phi,seed,True);rows.append(row);saved[phi]=z;save()
                print(json.dumps({k:row[k] for k in ['phase_deg','minimum_modifiable_gap_mm','ready','numerical_used_s']}),flush=True)
                if not row['ready']:return False
        return False
    try:
        if not refine(math.radians(3)):status='STARTING_CONTINUOUS_GEOMETRY_NOT_READY'
        else:
            for degree in range(max(4,int(round(np.degrees(max(saved))))+1),16):
                phi=math.radians(degree);z,row=fit.solve_feasible(phi,saved[max(saved)])
                # A feasible projection fixes solver constraint precision while
                # retaining the same phase and objective candidate.
                if not row['ready'] and not row['collisions'] and row['front_centerline_maximum_error_mm']<=6.45:
                    z,row=fit.solve_feasible(phi,z,True)
                saved[phi]=z;rows.append(row);save()
                print(json.dumps({k:row[k] for k in ['phase_deg','minimum_modifiable_gap_mm','front_centerline_maximum_error_mm','ready','numerical_used_s']}),flush=True)
                if not row['ready']:status='FINITE_POSE_NOT_READY';break
            else:status='READY_FOR_FULL_REVIEW' if refine(math.radians(15)) else 'CONTINUOUS_GEOMETRY_NOT_READY'
    except Exception as e:error=repr(e);status='NUMERICAL_WORK_CAP' if isinstance(e,TimeoutError) else 'NUMERICAL_ERROR'
    save();result=dict(status=status,error=error,physics_steps=0,numerical_used_s=fit.used_s,
        largest_saved_phase_deg=float(np.degrees(max(saved))),saved_node_count=len(saved),new_dynamics=0,total_dynamics=17,
        full_review_required=status=='READY_FOR_FULL_REVIEW',original_STOPPED_and_TIME_CAP_preserved=True)
    write(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
