"""Check the added moment gradient, entry, and contact input without integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(B/'GIVEN_INITIAL.npz');w=np.load(B/'BALANCE_WITNESSES.npz');plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),z['qpos'],z['ctrl'],r['kinematic_solver']);rows=[]
def moment(q):
    plan.fk(q)
    return sum(np.cross(plan.d.xipos[plan.head+i]-[0,0,1],plan.m.body_mass[plan.head+i]*plan.m.opt.gravity)[0] for i in range(21))
for alpha,qpos in zip(w['alpha'],w['qpos']):
    q=qpos[plan.qa].copy();plan.observe(z['qpos'],13);moment(q);n=8
    jpa=np.zeros((3,plan.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(plan.m,plan.d,jpa,jra,plan.snapshot['anchorP'],plan.head+plan.anchor);jpa=jpa[:,plan.va[:n]];jra=jra[:,plan.va[:n]];analytic=np.zeros(n)
    for i in range(n):
        force=plan.m.body_mass[plan.head+i]*plan.m.opt.gravity;J,_=plan.point_J(plan.d.xipos[plan.head+i],plan.head+i,jpa,jra,n);analytic+=force[2]*J[1]-force[1]*J[2]
    numeric=np.zeros(n);epsilon=1e-6
    for j in range(n):
        dq=np.zeros(20);dq[j]=epsilon;numeric[j]=(moment(q+dq)-moment(q-dq))/(2*epsilon)
    rows.append(dict(alpha=float(alpha),maximum_absolute_error_Nm_per_rad=float(abs(analytic-numeric).max()),analytic=analytic.tolist(),central_difference=numeric.tolist()))
plan.q=z['qpos'][plan.qa].copy();plan.observe(z['qpos'],13);before=plan.q.copy();q,G,metric=plan.solve(0.,0.,max_iterations=20,position_cap=.002,joint_step_rad=np.radians(.5))
entry=dict(maximum_joint_change_rad=float(abs(q-before).max()),activation=metric['source_axis_balance_activation'],moment_Nm=metric['source_axis_balance_moment_Nm'],head_error_mm=metric['head_error_m']*1000)
contacts=[dict(material=0,point=np.array([1.,.2,1.]),normal=np.array([0.,0.,1.]),penetration_m=.002,world_force_N=np.array([0.,0.,2.]),normal_N=2.)]
plan.observe_contacts(contacts);input_moment=plan.observed_external_source_X_moment_Nm
out=dict(version=r['version'],simulation_steps=0,robot_mass_kg=float(sum(plan.m.body_mass[plan.head:plan.head+21])),rows=rows,entry=entry,synthetic_unlabeled_external_moment_Nm=input_moment,expected_external_moment_Nm=.4,passed=bool(max(x['maximum_absolute_error_Nm_per_rad'] for x in rows)<1e-7 and entry['maximum_joint_change_rad']<1e-9 and entry['activation']==0 and abs(input_moment-.4)<1e-12),scope='Analytical balance term only, no physical rollout or path proof')
(B/'BALANCE_GRADIENT_ENTRY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
