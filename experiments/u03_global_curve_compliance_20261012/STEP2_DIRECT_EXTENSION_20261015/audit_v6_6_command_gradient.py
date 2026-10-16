"""Check only the new command constraint Jacobians at one saved contact state."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';C=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF/T_UPDATE';sys.path.insert(0,str(B));import rough_planner as rp
from geometry import Geometry
for key in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,key,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');initial=np.load(B/'GIVEN_INITIAL.npz');i=int(np.argmin(abs(z['t']-22.3)));rec=min([json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')],key=lambda row:abs(row['t']-z['t'][i]));contacts=rec['contacts']
for ct in contacts:
    for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
plan=rp.RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);plan.observe(z['qpos'][i],int(z['stance_anchor_material'][i]));plan.observe_contacts(contacts);n=14;local=sum(z[key][i] for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']);requested=np.clip(z['geometric_candidate_native_command'][i]+plan.coordinates.hinges(local),-np.radians(88),np.radians(88));fraction=z['command_filter_factor'][i];previous_total=(z['raw_native_command'][i]-fraction*requested)/(1-fraction);plan.q=z['previous_transmitted_geometric_native_command'][i].copy();plan.q[n:]=z['qpos'][i,plan.qa[n:]];plan.command_local_offset=previous_total-plan.q
G=z['global_reference_points'][i].copy();Targ=z['geometric_reference_points'][i].copy();T=(Targ[0]-Targ[1])/np.linalg.norm(Targ[0]-Targ[1]);plan.guide=lambda *a:(G.copy(),Targ[0].copy(),T.copy(),n);plan.wrap_guide=lambda *a:(Targ.copy(),Targ[0].copy(),T.copy(),n)
native_minimize=rp.minimize;captured=[]
def check_then_solve(fun,x0,*args,**kwargs):
    if not captured:
        ct_count=sum(ct['material']<n for ct in contacts);basecount=5*n+sum(max(0,21-(j+2)) for j in range(n))+ct_count;end=basecount+ct_count;con=kwargs['constraints'][0];analytic=con['jac'](x0)[basecount:end];eps=1e-5;fd=np.column_stack([(con['fun'](x0+np.eye(n)[j]*eps)[basecount:end]-con['fun'](x0-np.eye(n)[j]*eps)[basecount:end])/(2*eps) for j in range(n)]);error=abs(fd-analytic);captured.append(dict(command_constraint_rows=end-basecount,maximum_scaled_plane_jacobian_error=float(error.max()),maximum_scaled_jacobian_error=float(error.max()),fd_step_native_rad=float(eps*np.radians(20)*.025),passed=bool(error.max()<.02)))
    return native_minimize(fun,x0,*args,**kwargs)
rp.minimize=check_then_solve;plan.solve(float(z['alpha'][i]),float(z['beta'][i]),0.,position_cap=.002,joint_step_rad=np.radians(20)*.025)
out=dict(version=r['version'],simulation_steps=0,saved_t_s=float(z['t'][i]),checks=captured,scope='Analytic versus finite-difference gradients of only the new measured-plane total-command constraints at one saved state; no integration or force-response proof.')
(B/'COMMAND_CONSTRAINT_GRADIENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
