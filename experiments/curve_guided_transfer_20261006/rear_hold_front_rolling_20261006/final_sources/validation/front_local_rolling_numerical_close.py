"""One bounded same-construction constraint closure, never physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json,math,time,shutil
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from front_local_rolling_geometry import BASE,ORIGINAL,write
from front_local_rolling_fit import ConsistentRollingFit,pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference
from curve_guided_geometry import points


def main():
    out=BASE/'numerical_closure';out.mkdir(exist_ok=False)
    for name in ['geometry_progress.json','geometry_result.json','reference_partial.npz','revision_registry.json','saved_partial_diagnostics.json']:
        shutil.copy2(BASE/name,out/('before_'+name))
    shutil.copy2(ROOT/'validation/front_local_rolling_fit.py',out/'construction_source.py')
    write(out/'plan.json',dict(physics_steps=0,one_bounded_closure=True,nearest_projection_max_iterations=120,
        nearest_projection_max_wall_s=90,total_continuation_max_wall_s=600,subsequent_per_knot_max_iterations=100,
        material_phase_deg=[0,15],orbital_deg=[-5,5],axial_mm=[-32.25,32.25],gap_mm=.05,
        same_initial_objective_model_gain_and_constraints=True,
        order='Project saved3deg to nearest constraint-feasible point, then evaluate known prefix collision gate before any later knots; known prefix failure prevents useful full continuation',
        stop_history='before_revision_registry.json preserved; a numerical completion is not a new physical trial or retrospective success'))
    fit=ConsistentRollingFit()
    with np.load(BASE/'reference_partial.npz') as z:
        phases=z['phase_rad'].copy();qq=z['joint_rad'].copy();axial=z['axial_mm'].copy();orbital=z['orbital_rad'].copy()
    if len(phases)!=4 or abs(np.degrees(phases[-1])-3)>1e-8:raise RuntimeError('Expected exact saved0..3 prefix')
    seed=np.r_[qq[-1],axial[-1],orbital[-1]];units=np.r_[np.full(12,.1),10.,.1]
    low=np.r_[np.full(12,-math.radians(89)),-32.25,-math.radians(5)];high=-low
    start=time.monotonic();nearest_deadline=start+90
    last=[None];calls=[0]
    def constraints(v):
        if time.monotonic()>nearest_deadline:raise TimeoutError('90s nearest projection cap')
        z=seed+units*v;x=fit.state(z);g=fit.fit.gaps(x)[:24]-.05
        last[0]=dict(variables=z.copy(),minimum_gap_mm=float(g.min()+.05))
        return g
    def progress(v):
        calls[0]+=1
        if calls[0]%20==0:print(json.dumps(dict(nearest_iteration=calls[0],minimum_gap_mm=last[0]['minimum_gap_mm'],physics_steps=0)),flush=True)
    status='NEAREST_PROJECTION_FAILED';error=None
    try:
        solved=minimize(lambda v:float(v@v),np.zeros(14),method='SLSQP',
            bounds=list(zip((low-seed)/units,(high-seed)/units)),constraints=[dict(type='ineq',fun=constraints)],
            callback=progress,options=dict(maxiter=120,ftol=1e-10,eps=1e-6))
        z=seed+units*solved.x;x=fit.state(z);g=fit.fit.gaps(x)[:24]
        mujoco.mj_forward(fit.fit.m,fit.fit.d)
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        psi,_=pipe_relative_twist(fit.fit.m,fit.fit.d);change=(psi[:6]-fit.psi0[:6]+np.pi)%(2*np.pi)-np.pi
        ready=bool(g.min()>=.049 and not negative)
        nearest=dict(success=bool(solved.success),message=str(solved.message),iterations=int(solved.nit),wall_s=time.monotonic()-start,
            minimum_modifiable_gap_mm=float(g.min()),actual_finite_collisions=negative,constraint_feasible=ready,
            maximum_joint_adjustment_deg=float(np.degrees(abs(z[:12]-seed[:12])).max()),
            pipe_relative_front_twist_change_deg=np.degrees(change).tolist(),joint_angles_deg=np.degrees(x[6:]).tolist())
        write(out/'nearest_result.json',nearest)
        if ready:
            qq[-1]=z[:12];axial[-1]=z[12];orbital[-1]=z[13]
            np.savez_compressed(out/'reference_partial.npz',phase_rad=phases,joint_rad=qq,axial_mm=axial,orbital_rad=orbital)
            ref=FrontLocalRollingReference(out/'reference_partial.npz',ORIGINAL)
            failures=[]
            # The original0..1 knots are unchanged; completing3deg cannot cure
            # an earlier collision. This is the next required path gate.
            for phi in np.radians(np.arange(0,3.0001,.05)):
                xx=ref.state(float(phi));fit.fit.set(xx);mujoco.mj_forward(fit.fit.m,fit.fit.d)
                negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
                if negative:failures.append(dict(phase_deg=float(np.degrees(phi)),collisions=negative))
            write(out/'prefix_path_gate.json',dict(physics_steps=0,geometry_pass=not failures,spacing_deg=.05,failures=failures,
                unchanged_initial_knots=True,full15deg_path=False))
            status='NEAREST_FEASIBLE_BUT_PREFIX_COLLISION' if failures else 'PREFIX_CLEAR_REQUIRES_CONTINUATION'
    except Exception as e:error=repr(e)
    result=dict(status=status,physics_steps=0,wall_s=time.monotonic()-start,error=error,
        historical_STOPPED_preserved=True,new_dynamics=0,total_dynamics=17,
        further_construction_authorized=status=='PREFIX_CLEAR_REQUIRES_CONTINUATION',
        reason='Same construction has an actual finite interpolation collision before1deg; extending later knots cannot produce an admitted path' if status=='NEAREST_FEASIBLE_BUT_PREFIX_COLLISION' else None)
    write(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
