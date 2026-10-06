"""Same bounded closure: augmented feasibility residual and added path knots.

Original .05mm objective and .001mm numerical admission tolerance remain.
No model/controller/geometry range changes. One600s wall cap shared with the
earlier projection, measured from its plan creation time, not reset here.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import time,math,json,shutil
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from front_local_rolling_geometry import BASE,ORIGINAL,write
from front_local_rolling_fit import ConsistentRollingFit,pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points
OUT=BASE/'numerical_closure'


def main():
    deadline=(OUT/'plan.json').stat().st_mtime+600.
    if (OUT/'augmented_result.json').exists():
        previous=json.loads((OUT/'augmented_result.json').read_text(encoding='utf8'))
        if previous['status']!='NUMERICAL_ERROR' or time.time()>deadline:raise FileExistsError('One bounded close only')
        shutil.copy2(OUT/'augmented_result.json',OUT/'augmented_interface_error_attempt0.json')
    write(OUT/'augmented_plan.json',dict(physics_steps=0,total_shared_wall_limit_s=600,
        seconds_already_elapsed=time.time()-(deadline-600),additional_nodes_deg=[.25,.5,.75,1.25,1.5,1.75,2.25,2.5,2.75],
        method='normalized least squares with original shape objective and augmented negative .05mm gap residual; original final .049mm numerical admission bound unchanged',
        projection_max_evaluations=100,construction_max_evaluations=150,no_new_candidate_or_model_parameter=True,
        original_STOPPED_and_first_projection_failure_preserved=True))
    fit=ConsistentRollingFit();reference=FrontLocalRollingReference(BASE/'reference_partial.npz',ORIGINAL)
    with np.load(BASE/'reference_partial.npz') as z:
        oldphi=z['phase_rad'].copy();oldax=z['axial_mm'].copy();oldorb=z['orbital_rad'].copy()
    units=np.r_[np.full(12,.1),10.,.1]
    low=np.r_[np.full(12,-math.radians(89)),-32.25,-math.radians(5)];high=-low
    records=[];phases=[];variables=[]
    def checktime():
        if time.time()>deadline:raise TimeoutError('Shared600s closure wall cap')
    def solve(phi,seed,nearest):
        checktime()
        qguide=fit.x0[6:12]+fit.old.prototype.joint_angles(0.,phi)[:6]-fit.old.prototype.joint_angles(0.,0.)[:6]
        def residual(v):
            checktime();z=seed+units*v;x=fit.state(z);gap=fit.fit.gaps(x)[:24]
            negative=np.minimum(gap-.05,0.)/.001
            if nearest:return np.r_[.001*v,negative]
            p=points(fit.fit.m,fit.fit.d)
            desired=(fit.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0,z[13],0]).as_matrix().T+[1100,z[12],1000]
            psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d);twist=(psi[:6]-fit.psi0[:6]+phi+np.pi)%(2*np.pi)-np.pi
            original=np.r_[((p[:8]-desired)/4).ravel(),twist/math.radians(1.5),
                .08*(z[:6]-qguide)/math.radians(5),.025*(z[:12]-seed[:12])/math.radians(5),
                z[12]/32.25*.05,z[13]/math.radians(5)*.05]
            return np.r_[original,negative]
        def jacobian(v):
            step=1e-5;cols=[]
            for i in range(14):
                a=v.copy();b=v.copy();a[i]+=step;b[i]-=step
                cols.append((residual(a)-residual(b))/(2*step))
            return np.array(cols).T
        solved=least_squares(residual,np.zeros(14),jac=jacobian,
            bounds=((low-seed)/units,(high-seed)/units),max_nfev=100 if nearest else 150,
            ftol=1e-9,xtol=1e-9,gtol=1e-9)
        z=seed+units*solved.x;x=fit.state(z);g=fit.fit.gaps(x)[:24]
        mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        desired=(fit.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0,z[13],0]).as_matrix().T+[1100,z[12],1000]
        psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d);change=(psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi
        row=dict(phase_deg=float(np.degrees(phi)),nearest_projection=nearest,solver_success=bool(solved.success),evaluations=int(solved.nfev),
            minimum_modifiable_gap_mm=float(g.min()),collisions=negative,front_centerline_maximum_error_mm=float(np.linalg.norm(p[:8]-desired,axis=1).max()),
            pipe_relative_front_twist_change_deg=np.degrees(change).tolist(),axial_mm=float(z[12]),orbital_deg=float(np.degrees(z[13])),
            joint_angles_deg=np.degrees(x[6:]).tolist(),points_mm=p.tolist(),
            ready=bool(g.min()>=.049 and not negative and np.linalg.norm(p[:8]-desired,axis=1).max()<=6.45))
        print(json.dumps({k:row[k] for k in ['phase_deg','minimum_modifiable_gap_mm','front_centerline_maximum_error_mm','ready']}),flush=True)
        return z,row
    status='INCOMPLETE';error=None
    try:
        seed=np.r_[reference.joint_angles(oldphi[-1])[:12],oldax[-1],oldorb[-1]]
        fixed3,row=solve(oldphi[-1],seed,True);write(OUT/'augmented_nearest3.json',row)
        if not row['ready']:status='NEAREST_PROJECTION_STILL_FAILED'
        else:
            # Existing endpoints and required extra nodes repaired within the
            # same construction; no radius/phase-range/initial-pose searches.
            grid=np.unique(np.r_[np.arange(0,3.001,.25),np.arange(4,15.001,1)])
            previous=None
            for phi in np.radians(grid):
                if phi==0:
                    z=np.r_[fit.x0[6:18],0,0];row=dict(phase_deg=0.,exact_initial=True,ready=True)
                elif phi<=oldphi[-1]+1e-10:
                    seed=np.r_[reference.joint_angles(float(phi))[:12],np.interp(phi,oldphi,oldax),np.interp(phi,oldphi,oldorb)]
                    if abs(phi-oldphi[-1])<1e-9:z=fixed3;row=json.loads((OUT/'augmented_nearest3.json').read_text(encoding='utf8'))
                    else:z,row=solve(float(phi),seed,True)
                else:z,row=solve(float(phi),previous,False)
                phases.append(float(phi));variables.append(z);records.append(row);previous=z.copy()
                write(OUT/'augmented_progress.json',dict(physics_steps=0,rows=records))
                np.savez_compressed(OUT/'reference_augmented_partial.npz',phase_rad=phases,joint_rad=np.array(variables)[:,:12],
                    axial_mm=np.array(variables)[:,12],orbital_rad=np.array(variables)[:,13])
                if not row['ready']:status='KNOT_FEASIBILITY_FAILED';break
                if abs(phi-oldphi[-1])<1e-9:
                    partial=FrontLocalRollingReference(OUT/'reference_augmented_partial.npz',ORIGINAL);failures=[]
                    for probe in np.radians(np.arange(0,3.0001,.05)):
                        xx=partial.state(float(probe));fit.fit.set(xx);mujoco.mj_forward(fit.fit.m,fit.fit.d)
                        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
                        if negative:failures.append(dict(phase_deg=float(np.degrees(probe)),collisions=negative))
                    write(OUT/'repaired_prefix_gate.json',dict(physics_steps=0,pass_=not failures,failures=failures))
                    if failures:status='REPAIRED_PREFIX_STILL_COLLIDES';break
            else:status='COMPLETE_REQUIRES_DENSE_REVIEW'
    except Exception as e:error=repr(e);status='TIME_CAP' if isinstance(e,TimeoutError) else 'NUMERICAL_ERROR'
    result=dict(status=status,error=error,physics_steps=0,shared_wall_s=time.time()-(deadline-600),
        new_dynamics=0,total_dynamics=17,old_STOPPED_preserved=True,last_phase_deg=float(np.degrees(phases[-1])) if phases else None,
        complete15deg_reference=status=='COMPLETE_REQUIRES_DENSE_REVIEW')
    write(OUT/'augmented_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
