"""Replay four actual states through local V4 allocator, with stepping disabled."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/(sys.argv[1] if len(sys.argv)>1 else 'CONTROLLER_V4_1_ROBOT_MASS_HOLD');C=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
from contact_sensor import known_source
from source_wrench_hold import SourceWrenchHold
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));old=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');contacts=[json.loads(line) for line in (C/'contact_evidence.jsonl').read_text(encoding='utf-8').splitlines()];plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),z['qpos'][0],z['ctrl'][0],r['kinematic_solver']);hold=SourceWrenchHold(plan,r['source_hold'],r['source_known']);ref=int(np.argmin(abs(z['t']-2.1)));plan.d.qpos[:]=z['qpos'][ref];plan.position();reference=plan.d.xpos[plan.head:plan.head+21].copy();rows=[];previous=2.1
for target in [2.1,5.,10.,15.]:
    j=int(np.argmin(abs(z['t']-target)));row=min(contacts,key=lambda x:abs(x['t']-z['t'][j]));sensor=dict(joint_actual=z['qpos'][j,plan.qa].copy(),joint_torque=z['torque'][j].copy(),contacts=[])
    for ct in row['contacts']:
        for k in ['point','normal','world_force_N']:ct[k]=np.asarray(ct[k],float)
        sensor['contacts'].append(ct)
    loaded=[ct for ct in sensor['contacts'] if ct['material']>=8 and known_source(ct,r['source_known'])];anchor=max(loaded,key=lambda ct:ct['normal_N'])['material'];plan.observe(z['qpos'][j],anchor)
    before=plan.initial_qpos.copy();ff,info=hold.update(sensor,z['geometric_native_command'][j],8,max(.025,target-previous),target);previous=target
    rows.append(dict(t=float(z['t'][j]),valid=bool(info['source_hold_qp_valid']),iterations=info['source_hold_solver_iterations'],base_error=info['source_hold_base_error'],constraint_error=info['source_hold_constraint_error'],feedforward_peak_Nm=float(abs(ff).max()),predicted_total_peak_Nm=float(abs(info['source_hold_total_predicted_torque_Nm']).max()),allocated_normal_N=float(info['source_hold_allocated_normal_N'].sum()),axial_error_mm=info['source_hold_axial_error_m']*1000,polar_error_deg=float(np.degrees(info['source_hold_polar_error_rad'])),corrective_moment_Nm=info['source_hold_corrective_moment_Nm'].tolist(),model_flags=int(plan.m.opt.enableflags)))
same={k:r[k]==old[k] for k in ['conditions','source_known','source_constants','kinematic_solver','direct_extension','probe','acceptance','safety_stop','native_curve_projection']};same['given_initial_bytes']=(B/'GIVEN_INITIAL.npz').read_bytes()==(C/'GIVEN_INITIAL.npz').read_bytes()
robot_mass=float(plan.m.body_mass[plan.head:plan.head+21].sum());robot_weight=float(robot_mass*np.linalg.norm(plan.m.opt.gravity));mass_ok=abs(robot_weight-r['robot_weight_N'])<1e-9
out=dict(version=r['version'],simulation_steps=0,rows=rows,unchanged=same,robot_mass_kg=robot_mass,robot_weight_N=robot_weight,fixed_scene_mass_excluded_kg=float(plan.m.body_mass.sum()-robot_mass),mass_matches_registered_robot=mass_ok,passed=all(x['valid'] and x['predicted_total_peak_Nm']<2.300001 and x['allocated_normal_N']<=4*robot_weight+1e-6 for x in rows) and all(same.values()) and mass_ok,limits='A few preserved actual poses verify local allocation and bounds only. Dynamic contact/preload realization and stationary source, contact, wrap and transfer all remain unverified.')
(B/'SOURCE_HOLD_REPLAY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
