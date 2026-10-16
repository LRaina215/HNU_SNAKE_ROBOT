"""Necessary geometry-Jacobian diagnosis for repeated constrained-solver stalls."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner
reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));folder=BASE.parent/'INITIAL_V4_CAP_LOOP_LEFT';z=np.load(folder/'CANDIDATE_INITIAL.npz');path=np.load(folder/'CONTACT_REPLAY_DIAGNOSTIC_TRUE_AXIS_T_WRAP.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);n=reg['kinematic_solver']['wrap_hinges'];plan=RoughPlanner(Geometry([1.1,.2,1],[0,1,0],.0565),d.qpos,d.ctrl,reg['kinematic_solver']);plan.observe(d.qpos.copy(),n);q=path['qpos'][-1,plan.qa].copy();records=[]
for material,other in [(0,plan.target),(1,plan.target),(5,plan.target),(6,plan.target),(9,plan.source[1])]:
 plan.fk(q);a=plan.robot[material];b=other;gap,near=plan.distance(a,b);reverse=np.zeros(6);reverse_gap=mujoco.mj_geomDistance(plan.m,plan.d,b,a,.025,reverse)
 jpa=np.zeros((3,plan.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(plan.m,plan.d,jpa,jra,plan.snapshot['anchorP'],plan.head+plan.anchor);jpa=jpa[:,plan.va[:n]];jra=jra[:,plan.va[:n]]
 analytic=((near[:3]-near[3:])/gap)@plan.point_J(near[:3],plan.head+material,jpa,jra,n)[0] if abs(gap)>1e-9 else np.zeros(n)
 numerical=[]
 for j in range(n):
  plus=q.copy();minus=q.copy();plus[j]+=1e-6;minus[j]-=1e-6;plan.fk(plus);hi=plan.distance(a,b)[0];plan.fk(minus);lo=plan.distance(a,b)[0];numerical.append((hi-lo)/2e-6)
 numerical=np.array(numerical);records.append(dict(material=material,other=plan.m.geom(b).name,types=[int(plan.m.geom_type[a]),int(plan.m.geom_type[b])],gap_mm=gap*1000,
  witness_swap_error_m=float(np.linalg.norm(near-np.r_[reverse[3:],reverse[:3]])),witness_identical_error_m=float(np.linalg.norm(near-reverse)),max_analytic_error=float(abs(analytic-numerical).max()),analytic=analytic.tolist(),numerical=numerical.tolist()))
out=dict(simulation_steps=0,records=records);(folder/'WRAP_GRADIENT_DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
