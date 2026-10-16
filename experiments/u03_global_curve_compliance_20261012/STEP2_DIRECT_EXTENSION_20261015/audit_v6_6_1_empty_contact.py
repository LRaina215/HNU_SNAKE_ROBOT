"""Required actual15.2s regression and one nonempty-state command comparison."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,importlib.util,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_6_1_EMPTY_CONTACT_DIAGNOSTIC';old=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';C=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF/T_UPDATE';sys.path.insert(0,str(B));from rough_planner import RoughPlanner;from geometry import Geometry
for key in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,key,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No integration')))
spec=importlib.util.spec_from_file_location('rejected_empty_diagnostic',old/'rough_planner.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');initial=np.load(B/'GIVEN_INITIAL.npz');records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rows=[];solutions={}
for target in [15.2,22.3]:
    i=int(np.argmin(abs(z['t']-target)));rec=min(records,key=lambda row:abs(row['t']-z['t'][i]));contacts=rec['contacts'];n=14 if z['beta'][i]>0 else 8
    for ct in contacts:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
    results=[]
    for name,cls in [('rejected_V6_6',legacy.RoughPlanner),('corrected_V6_6_1',RoughPlanner)]:
        plan=cls(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);plan.observe(z['qpos'][i],int(z['stance_anchor_material'][i]));plan.observe_contacts(contacts);qphysical=plan.snapshot['qpos'].copy();plan.q=z['geometric_native_command'][i].copy();plan.q[n:]=z['qpos'][i,plan.qa[n:]];plan.command_local_offset=z['raw_native_command'][i]-plan.q
        G=z['global_reference_points'][i].copy();Ctarget=z['geometric_reference_points'][i].copy();T=(Ctarget[0]-Ctarget[1])/np.linalg.norm(Ctarget[0]-Ctarget[1]);plan.guide=lambda *a:(G.copy(),Ctarget[0].copy(),T.copy(),n);plan.wrap_guide=lambda *a:(Ctarget.copy(),Ctarget[0].copy(),T.copy(),n)
        try:
            q,_,metrics=plan.solve(float(z['alpha'][i]),float(z['beta'][i]),0.,position_cap=.002,joint_step_rad=np.radians(20)*.025);solutions[(target,name)]=q.copy();results.append(dict(version=name,error=None,metrics=metrics,snapshot_unchanged=bool(np.array_equal(qphysical,plan.snapshot['qpos']))))
        except Exception as exc:results.append(dict(version=name,error=type(exc).__name__+': '+str(exc),snapshot_unchanged=bool(np.array_equal(qphysical,plan.snapshot['qpos']))))
    rows.append(dict(saved_t_s=float(z['t'][i]),front_contact_count=sum(ct['material']<n for ct in contacts),contact_materials=sorted(set(int(ct['material']) for ct in contacts)),results=results))
same=float(abs(solutions[(22.3,'rejected_V6_6')]-solutions[(22.3,'corrected_V6_6_1')]).max());passed=rows[0]['front_contact_count']==0 and 'zero-size array' in rows[0]['results'][0]['error'] and rows[0]['results'][1]['error'] is None and rows[0]['results'][1]['metrics']['command_incremental_plane_count']==0 and rows[0]['results'][1]['metrics']['command_incremental_clearance_violation_m']==0 and same==0
out=dict(version=r['version'],simulation_steps=0,passed=passed,rows=rows,nonempty_maximum_native_command_difference_rad=same,scope='Same actual15.2s pose/contacts as independent rejection; saved sent geometric/total commands define a private initial chart. Same saved controller-generated targets across versions; no integration, actual force response or new physical start. Nonempty22.3s command equality checks that only final diagnostics changed.')
(B/'EMPTY_CONTACT_REGRESSION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
