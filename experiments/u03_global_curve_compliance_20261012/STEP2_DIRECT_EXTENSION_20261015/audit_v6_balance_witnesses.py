"""Three coarse-prior kinematic balance witnesses; no full path or dynamics proof."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';sys.path.insert(0,str(B));from rough_planner import RoughPlanner;from geometry import Geometry
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(B/'GIVEN_INITIAL.npz');prefix=np.load(P/'CONTROLLER_V3_POSITIVE_POLAR/PRECONTACT_EVALUATION/PRECONTACT_PREFIX.npz');rows=[];states=[]
for alpha in [.3,.6,.888]:
    plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);j=int(np.argmin(abs(prefix['alpha']-alpha)));plan.q=prefix['qpos'][j,plan.qa].copy();plan.observe(initial['qpos'],13);source_before=plan.fk(plan.q).copy();before=float(sum(np.cross(plan.d.xipos[plan.head+i]-[0,0,1],plan.m.body_mass[plan.head+i]*plan.m.opt.gravity)[0] for i in range(21)))
    iterations=[]
    for _ in range(6):
        q,G,metric=plan.solve(alpha,0.,0.,max_iterations=100);iterations.append(metric)
        if abs(metric['source_axis_balance_moment_Nm'])<.02 and metric['head_error_m']<.001:break
    _,_,Tgoal,_=plan.guide(alpha,0.);p=plan.fk(q);T=plan.d.xmat[plan.head].reshape(3,3)[:,2];after=metric['source_axis_balance_moment_Nm'];row=dict(alpha=alpha,seed='Old no-integration coarse-prior command prefix only, not runtime data',gravity_moment_before_Nm=before,net_source_axis_moment_after_Nm=after,head_error_mm=metric['head_error_m']*1000,tangent_error_deg=float(np.degrees(np.arccos(np.clip(T@Tgoal,-1,1)))),tangent_reference='Actual solver guide chord tangent',source_node_motion_mm=float(np.linalg.norm(p[8:]-source_before[8:],axis=1).max()*1000),maximum_joint_deg=float(abs(q).max()*180/np.pi),clearance_violation_mm=metric['clearance_violation_m']*1000,calls=len(iterations),last_solver_success=metric['solver_success']);rows.append(row);states.append(plan.d.qpos.copy())
out=dict(version=r['version'],simulation_steps=0,scope='Three isolated precontact head goals, no time integration or full-path IK; old coarse-prior command poses used only as offline numerical seeds. Runtime never loads these seeds.',rows=rows,limits='Feasible witnesses do not prove controller continuity, full path, source stability, physical target contact or post-contact balance. A failed local solve is not global infeasibility.')
(B/'BALANCE_WITNESSES.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');np.savez_compressed(B/'BALANCE_WITNESSES.npz',qpos=np.array(states),alpha=np.array([x['alpha'] for x in rows]));print(json.dumps(out,indent=2))
