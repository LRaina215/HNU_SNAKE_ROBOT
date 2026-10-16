"""Saved-state command continuation, no rollout and no altered limits."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,ast,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(C/'GIVEN_INITIAL.npz');z=np.load(C/'trace.npz');records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rows=[]
for target in [17.7,19.,20.,21.,22.,25.,28.2]:
    j=int(np.argmin(abs(z['t']-target)));plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);anchor=int(z['stance_anchor_material'][j]);plan.observe(z['qpos'][j],anchor);rec=min(records,key=lambda row:abs(row['t']-z['t'][j]));contacts=rec['contacts']
    for ct in contacts:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
    plan.observe_contacts(contacts);raw=z['raw_native_command'][j];field=z['geometric_curvature_field'][j].copy()
    for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']:field+=z[key][j]
    desired=np.clip(plan.coordinates.hinges(field),-np.radians(88),np.radians(88));delta=desired-raw;peak=float(np.linalg.norm(plan.coordinates.field(delta),axis=1).max()*1000);factor=min(1.,1.5*.025/max(peak,1e-12),np.radians(20)*.025/max(abs(delta).max(),1e-12));rate_factor=factor;attempts=[]
    for _ in range(16):
        metric=plan.command_step_metrics(raw,raw+factor*delta);attempts.append(dict(factor=factor,**metric))
        if metric['maximum_geometry_step_m']<=.002+1e-10 and max(metric['measured_plane_excess_compression_m'],metric['known_source_command_excess_compression_m'])<=2e-6:break
        factor*=.5
    plan.fk(raw);rawg=plan.legacy_source_contacts(21)[0];plan.fk(z['geometric_native_command'][j]);geog=plan.legacy_source_contacts(21)[0];part={}
    for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']:
        part[key]=float(abs(plan.coordinates.hinges(z[key][j])).max()*180/np.pi)
    rows.append(dict(t=float(z['t'][j]),beta=float(z['beta'][j]),runtime_filter_factor=float(z['command_filter_factor'][j]),prospective_rate_factor=rate_factor,prospective_final_factor=factor,maximum_remaining_command_deg=float(abs(delta).max()*180/np.pi),maximum_raw_geometric_separation_deg=float(abs(raw-z['geometric_native_command'][j]).max()*180/np.pi),minimum_raw_known_source_gaps_mm=(rawg.min(0)*1000).tolist(),minimum_geometric_known_source_gaps_mm=(geog.min(0)*1000).tolist(),local_mode_maximum_joint_equivalent_deg=part,attempts=attempts,planner_metric=ast.literal_eval(str(z['planner_metrics'][j]))))
out=dict(simulation_steps=0,scope='Prospective continuation from each saved already-filtered command; not exact replay of the unsaved previous25ms command. Same private C bearing/contact constraints and original native local-mode terms; no actual state writes.',rows=rows)
(C/'COMMAND_STALL_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in ['attempts','planner_metric']}))
