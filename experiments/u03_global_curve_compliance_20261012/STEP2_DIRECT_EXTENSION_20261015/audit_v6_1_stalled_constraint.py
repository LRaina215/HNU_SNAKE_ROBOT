"""One saved stalled C state, finite-difference active constraints and restore solve."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_1_TRANSMITTED_C_STATE';C=B/'T_UPDATE';sys.path.insert(0,str(B))
import rough_planner
from rough_planner import RoughPlanner
from geometry import Geometry
from scipy.optimize import minimize
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(C/'GIVEN_INITIAL.npz');z=np.load(C/'trace.npz');j=int(np.argmin(abs(z['t']-5.)));plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);plan.q=z['geometric_native_command'][j].copy();plan.q[8:]=z['qpos'][j,plan.qa[8:]];plan.observe(z['qpos'][j],int(z['stance_anchor_material'][j]));records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rec=min(records,key=lambda row:abs(row['t']-z['t'][j]));contacts=rec['contacts']
for ct in contacts:
    for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
plan.observe_contacts(contacts);captured={}
def inspect(fun,x0,**kwargs):
    con=kwargs['constraints'][0];c=con['fun'];jac=con['jac'];c0=c(x0).copy();J=jac(x0).copy();epsilon=1e-4;numeric=np.zeros_like(J)
    for k in range(len(x0)):
        u=np.zeros_like(x0);u[k]=epsilon;numeric[:,k]=(c(x0+u)-c(x0-u))/(2*epsilon)
    active=np.flatnonzero(c0<.1);worst=int(np.argmin(c0));captured.update(initial_violation_mm=float(-c0.min()),worst_constraint_index=worst,active_constraints=[dict(index=int(i),residual_mm=float(c0[i]),analytical_gradient=J[i].tolist(),numerical_gradient=numeric[i].tolist(),maximum_gradient_error=float(abs(J[i]-numeric[i]).max())) for i in active])
    fit=minimize(fun,x0,**kwargs);captured.update(original_fit_success=bool(fit.success),original_fit_message=str(fit.message),original_fit_violation_mm=float(-c(fit.x).min()),original_fit_objective=float(fun(fit.x)),initial_objective=float(fun(x0)))
    def feasibility(u):
        v=np.minimum(0.,c(u)[:-8]);return .5*v@v
    def derivative(u):return jac(u)[:-8].T@np.minimum(0.,c(u)[:-8])
    restore=minimize(feasibility,x0,jac=derivative,bounds=kwargs['bounds'],method='SLSQP',constraints=[dict(type='ineq',fun=lambda u:c(u)[-8:],jac=lambda u:jac(u)[-8:])],options=dict(maxiter=35,ftol=1e-10,disp=False));after=c(restore.x);captured.update(restoration_success=bool(restore.success),restoration_message=str(restore.message),restoration_iterations=int(restore.nit),restoration_violation_mm=float(-after[:-8].min()),restoration_motion_constraint_min=float(after[-8:].min()),restoration_maximum_node_step_mm=float(2*np.sqrt(max(0.,1-after[-8:].min()))),restoration_objective=float(fun(restore.x)),restoration_scaled_step=restore.x.tolist(),restoration_gradient_norm=float(np.linalg.norm(derivative(restore.x))))
    return fit
rough_planner.minimize=inspect
q,G,metric=plan.solve(float(z['alpha'][j]),0.,position_cap=.002,joint_step_rad=np.radians(.5));captured.update(simulation_steps=0,saved_time_s=float(z['t'][j]),scope='One saved stalled V6.1 C state at5s; original model/limits and contact planes, no integration; restoration solve is an isolated numerical diagnostic, not a new path or dynamics run.',returned_metric=metric)
(C/'STALLED_CONSTRAINT_AUDIT.json').write_text(json.dumps(captured,indent=2)+'\n',encoding='utf-8');print(json.dumps(captured,indent=2))
