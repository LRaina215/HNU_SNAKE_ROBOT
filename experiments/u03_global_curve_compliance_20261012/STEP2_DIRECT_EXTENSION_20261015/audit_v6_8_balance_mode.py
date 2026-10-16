"""Bounded saved-state C comparisons and projected body-gradient check."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,importlib.util,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_8_HEAD_NULL_BALANCE_MODE';old=P/'CONTROLLER_V6_7_CONTACT_ENTRY_CHART';C=old/'T_UPDATE';sys.path.insert(0,str(B));from rough_planner import RoughPlanner,skew;from geometry import Geometry
for key in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,key,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No integration')))
spec=importlib.util.spec_from_file_location('old_full_body_cost',old/'rough_planner.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');initial=np.load(B/'GIVEN_INITIAL.npz');records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rows=[];fd_report=None
for target in [20.,25.,30.]:
    i=int(np.argmin(abs(z['t']-target)));rec=min(records,key=lambda row:abs(row['t']-z['t'][i]));contacts=rec['contacts'];n=14
    for ct in contacts:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
    results=[]
    for name,cls in [('old_full_body_cost',legacy.RoughPlanner),('new_head_null_mode',RoughPlanner)]:
        plan=cls(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);plan.observe(z['qpos'][i],int(z['stance_anchor_material'][i]));plan.observe_contacts(contacts);qphysical=plan.snapshot['qpos'].copy();local=sum(z[key][i] for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']);requested=np.clip(z['geometric_candidate_native_command'][i]+plan.coordinates.hinges(local),-np.radians(88),np.radians(88));f=z['command_filter_factor'][i];previous_total=(z['raw_native_command'][i]-f*requested)/(1-f);previous=z['previous_transmitted_geometric_native_command'][i].copy();previous[n:]=z['qpos'][i,plan.qa[n:]];plan.q=previous.copy();plan.command_local_offset=previous_total-previous
        G=z['global_reference_points'][i].copy();Ctarget=z['geometric_reference_points'][i].copy();T=(Ctarget[0]-Ctarget[1])/np.linalg.norm(Ctarget[0]-Ctarget[1]);plan.guide=lambda *a:(G.copy(),Ctarget[0].copy(),T.copy(),n);plan.wrap_guide=lambda *a:(Ctarget.copy(),Ctarget[0].copy(),T.copy(),n)
        if name=='new_head_null_mode' and target==25.:
            projection,mode=plan.body_balance_projection(previous,n);test=previous.copy();test[:n]+=np.linspace(-1,1,n)*1e-4
            def body(q,jac=False):
                points=plan.fk(q)
                if not jac:return projection@(points[1:n]-Ctarget[1:n]).ravel()
                jp=np.zeros((3,plan.m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(plan.m,plan.d,jp,jr,plan.snapshot['anchorP'],plan.head+plan.anchor);jp=jp[:,plan.va[:n]];jr=jr[:,plan.va[:n]];return projection@np.vstack([plan.point_J(points[k],plan.head+k,jp,jr,n)[0] for k in range(1,n)])
            analytic=body(test,True);eps=1e-7;numeric=np.column_stack([(body(test+np.r_[np.eye(n)[j]*eps,np.zeros(20-n)])-body(test-np.r_[np.eye(n)[j]*eps,np.zeros(20-n)]))/(2*eps) for j in range(n)]);fd_report=dict(saved_t_s=float(z['t'][i]),nonzero_test_joint_perturbation_rad=1e-4,fd_step_rad=eps,maximum_projected_body_J_error_m_per_rad=float(abs(numeric-analytic).max()),mode=mode)
        candidate,_,metrics=plan.solve(float(z['alpha'][i]),float(z['beta'][i]),0.,position_cap=.002,joint_step_rad=np.radians(20)*.025);results.append(dict(policy=name,metrics=metrics,maximum_native_step_deg=float(abs(candidate-previous).max()*180/np.pi),snapshot_unchanged=bool(np.array_equal(qphysical,plan.snapshot['qpos']))))
    rows.append(dict(saved_t_s=float(z['t'][i]),results=results))
passed=fd_report['maximum_projected_body_J_error_m_per_rad']<1e-7 and all(row['results'][1]['metrics']['accepted'] and row['results'][1]['metrics']['body_balance_mode_available'] and row['results'][1]['snapshot_unchanged'] for row in rows)
out=dict(version=r['version'],simulation_steps=0,passed=passed,rows=rows,projected_body_gradient=fd_report,scope='One original-bounded solve per policy at three saved actual states with identical saved controller targets and fixed previous local offset. Added projector is fixed during each solve; finite differences check its body Jacobian at a nonzero perturbation. No integration, full path, physical source retention or force-response proof.')
(B/'HEAD_NULL_BALANCE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
