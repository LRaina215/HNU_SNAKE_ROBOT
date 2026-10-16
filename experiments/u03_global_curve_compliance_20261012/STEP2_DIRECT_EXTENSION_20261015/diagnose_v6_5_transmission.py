"""Inspect saved actual command filtering without integrating any dynamics."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
from contact_sensor import known_source
from physics import MODEL
for key in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,key,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');old=np.load(P/'CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE/T_UPDATE/trace.npz');initial=np.load(B/'GIVEN_INITIAL.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);control=DirectController(m,d,r,'DIRECTION',True);plan=control.planner
records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];earliest=None
for row in records:
    for ct in row['contacts']:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
for row in records:
    groups={}
    for ct in row['contacts']:
        if ct['material']<8 and not known_source(ct,r['source_known']):groups.setdefault(ct['material'],[]).append(ct)
    if any(sum(ct['world_force_N'][2] for ct in cs)<0 and sum(ct['normal_N'] for ct in cs)>r['robot_weight_N']/21 for cs in groups.values()):earliest=row['t'];break
n=min(len(z['t']),len(old['t']));mask=(abs(z['t'][:n]-old['t'][:n])<1e-7)&(z['t'][:n]<earliest-1e-6)
rows=[]
for target in [18.,19.,20.,21.,22.3,23.,24.,25.,26.,27.]:
    i=int(np.argmin(abs(z['t']-target)));t=float(z['t'][i]);factor=float(z['command_filter_factor'][i]);rec=min(records,key=lambda row:abs(row['t']-t));contacts=rec['contacts']
    for ct in contacts:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
    plan.observe(z['qpos'][i],int(z['stance_anchor_material'][i]));plan.observe_contacts(contacts)
    local=sum(z[key][i] for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']);requested=np.clip(z['geometric_candidate_native_command'][i]+plan.coordinates.hinges(local),-np.radians(88),np.radians(88));sent=z['raw_native_command'][i]
    if factor>=1-1e-12:continue
    previous=(sent-factor*requested)/(1-factor);delta=plan.coordinates.field(requested-previous);oldfield=plan.coordinates.field(previous);peak=np.linalg.norm(delta,axis=1).max()*1000;rate=min(1.,1.5*.025/max(peak,1e-12),np.radians(20)*.025/max(abs(requested-previous).max(),1e-12));a=np.sum(delta**2,axis=1);b=2*np.sum(oldfield*delta,axis=1);c=np.sum(oldfield**2,axis=1)-.024**2;roots=(-b+np.sqrt(np.maximum(0,b*b-4*a*c)))/np.maximum(2*a,1e-30);roots[a<1e-20]=1.;before=min(rate,float(np.clip(roots,0,1).min()));unshortened=plan.command_step_metrics(previous,previous+before*(requested-previous));final=plan.command_step_metrics(previous,sent)
    rows.append(dict(t=t,observed_final_factor=factor,initial_rate_curvature_factor=before,shortening_ratio=factor/max(before,1e-20),unshortened_metrics=unshortened,reconstructed_final_metrics=final,logged_final_geometry_step_m=float(z['command_maximum_C_geometry_step_m'][i]),requested_relief_per_m=(z['force_curvature_field'][i]*1000).tolist() if False else float(np.linalg.norm(z['force_curvature_field'][i],axis=1).max()*1000)))
out=dict(simulation_steps=0,first_25ms_threshold_activation_t=earliest,preactivation_prefix_samples=int(mask.sum()),preactivation_max_qpos_difference=float(abs(z['qpos'][:n][mask]-old['qpos'][:n][mask]).max()),preactivation_max_ctrl_difference_rad=float(abs(z['ctrl'][:n][mask]-old['ctrl'][:n][mask]).max()),rows=rows,scope='Previous total native command algebraically recovered from actual sent/candidate fields and common fraction. Geometry recomputed from saved physical pose/contact planes. No counterfactual rollout or force-response proof.')
(C/'TRANSMISSION_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
