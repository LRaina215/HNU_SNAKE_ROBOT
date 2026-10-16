"""Focused, read-only virtual-work and preserved-contact replay checks."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,importlib.util,numpy as np,mujoco
P=Path(__file__).resolve().parent; B=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS'; old=P/'CONTROLLER_V3_POSITIVE_POLAR'; C=old/'T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
from contact_sensor import known_source
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration permitted by this audit')))
reg=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));prior=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz')
plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),z['qpos'][0],z['ctrl'][0],reg['kinematic_solver']);qa=plan.qa
rows=[]
for target in [2.1,5.]:
    j=int(np.argmin(abs(z['t']-target)));q=z['qpos'][j,qa].copy()
    for n in [8,14]:
        plan.observe(z['qpos'][j],n);torque=plan.gravity_torque(q,n)[:n]
        def potential(v):
            plan.fk(v);return -sum(plan.m.body_mass[plan.head+i]*plan.m.opt.gravity@plan.d.xipos[plan.head+i] for i in range(n))
        grad=[]
        for k in range(n):
            plus=q.copy();minus=q.copy();plus[k]+=1e-6;minus[k]-=1e-6;grad.append((potential(plus)-potential(minus))/2e-6)
        error=float(np.max(abs(torque-np.asarray(grad))))
        rows.append(dict(actual_t=float(z['t'][j]),released_hinges=n,maximum_holding_torque_gradient_error_Nm=error,passed=error<1e-6))
spec=importlib.util.spec_from_file_location('preserved_contact_sensor',old/'contact_sensor.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
changed=[];first_old_unknown=None;first_new_unknown=None;source=reg['source_known']
for line in (C/'contact_evidence.jsonl').read_text(encoding='utf-8').splitlines():
    row=json.loads(line)
    for ct in row['contacts']:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
        a=mod.known_source(ct,source);b=known_source(ct,source)
        if ct['material']<8 and not a and first_old_unknown is None:first_old_unknown=row['t']
        if ct['material']<8 and not b and first_new_unknown is None:first_new_unknown=row['t']
        if a!=b:changed.append(dict(t=row['t'],material=ct['material'],point=ct['point'].tolist(),normal=ct['normal'].tolist(),old_source=a,new_source=b))
unchanged_keys=['conditions','source_known','source_constants','kinematic_solver','direct_extension','probe','local_feedback','acceptance','safety_stop','native_curve_projection']
same={k:reg[k]==prior[k] for k in unchanged_keys}
same['initial_bytes']= (B/'GIVEN_INITIAL.npz').read_bytes()==(old/'GIVEN_INITIAL.npz').read_bytes()
same['calibration_bytes']= (B/'KNOWN_SOURCE_COIL_CALIBRATION.npz').read_bytes()==(old/'KNOWN_SOURCE_COIL_CALIBRATION.npz').read_bytes()
preserved=all(f.read_bytes()==(old/'REVIEW_SNAPSHOT'/f.name).read_bytes() for f in old.glob('*.py'))
runtime_changes=[f.name for f in B.glob('*.py') if f.read_bytes()!=(old/f.name).read_bytes()]
out=dict(version=reg['version'],simulation_steps=0,virtual_work_checks=rows,preserved_actual_contact_replay=dict(first_old_front_unknown_t=first_old_unknown,first_corrected_front_unknown_t=first_new_unknown,changed_contacts=changed),unchanged=same,preserved_V3_2_runtime_matches_frozen_snapshot=preserved,changed_python_files=runtime_changes,
    passed=all(r['passed'] for r in rows) and all(same.values()) and preserved and first_new_unknown is None and bool(changed),
    limits='Replay checks the classification of the old trajectory only; it does not predict the corrected dynamic trajectory or prove stability, true acquisition or wrapping.')
(B/'FOCUSED_CORRECTION_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
