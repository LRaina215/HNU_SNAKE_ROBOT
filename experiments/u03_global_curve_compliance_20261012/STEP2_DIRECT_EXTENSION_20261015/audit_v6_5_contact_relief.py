"""Saved actual downward contact: old/new local command and outward kinematics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,copy,importlib.util,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF';old=P/'CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE';C=old/'T_UPDATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
from contact_sensor import known_source,read_sensor
from physics import MODEL
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
spec=importlib.util.spec_from_file_location('reviewed_old_support',old/'support_policy.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(B/'GIVEN_INITIAL.npz');z=np.load(C/'trace.npz');j=int(np.argmin(z['eval_pipe_world_N'][:,2:,:,2].sum((1,2))));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);control=DirectController(m,d,r,'DIRECTION',True);d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.ctrl[:]=z['ctrl'][j];mujoco.mj_forward(m,d);qphysical=d.qpos.copy();records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rec=min(records,key=lambda row:abs(row['t']-z['t'][j]));sensor=read_sensor(m,d);sensor['contacts']=copy.deepcopy(rec['contacts'])
for ct in sensor['contacts']:
    for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
source=control.reg['source_known'];non=[ct for ct in sensor['contacts'] if ct['material']<8 and not known_source(ct,source)];supports=[ct for ct in sensor['contacts'] if ct['material']>=8 and ct['normal_N']>.5 and known_source(ct,source)];anchor=max(supports,key=lambda ct:ct['normal_N'])['material'];rows=[]
for name,contacts,post,old_rule in [('old_downward',sensor['contacts'],None,True),('new_downward',sensor['contacts'],None,False),('known_source_only',[ct for ct in sensor['contacts'] if known_source(ct,source)],None,False),('nonnegative_load',copy.deepcopy(sensor['contacts']),None,False),('after_capture',sensor['contacts'],object(),False)]:
    if name=='nonnegative_load':
        for ct in contacts:
            if ct['material']<8 and not known_source(ct,source):ct['world_force_N'][2]=abs(ct['world_force_N'][2])
    control.front_relief[:]=0.;control.post=post;obs=dict(sensor,contacts=contacts)
    if old_rule:legacy.SourcePolicy.local_front(control,d,obs,.025)
    else:control.local_front(d,obs,.025)
    rows.append(dict(case=name,relief_per_m=control.front_relief.tolist(),norm_per_m=float(np.linalg.norm(control.front_relief))))
    if name=='new_downward':relief=control.front_relief.copy()
bounds=control.info['curvature_bounds_mm'];material=control.info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:]);w=np.sin(np.pi*np.clip(mid/material[8],0,1))**2;w[bounds[1:]>material[8]]=0.;field=w[:,None]*relief[None,:]/1000.;dq=control.planner.coordinates.hinges(field);plan=control.planner;plan.observe(qphysical,anchor);actual=qphysical[plan.qa].copy();plan.fk(actual);before=[]
for ct in non:
    body=plan.head+ct['material'];R=plan.d.xmat[body].reshape(3,3);local=R.T@(ct['point']-plan.d.xpos[body]);before.append((ct,local))
plan.fk(actual+dq);displacements=[]
for ct,local in before:
    body=plan.head+ct['material'];point=plan.d.xpos[body]+plan.d.xmat[body].reshape(3,3)@local;displacements.append(float(ct['normal']@(point-ct['point']))*1000)
passed=rows[0]['norm_per_m']==0 and rows[1]['norm_per_m']>0 and abs(rows[1]['norm_per_m']-.06*.025)<1e-12 and all(row['norm_per_m']==0 for row in rows[2:]) and min(displacements)>0 and abs(dq[8:]).max()<1e-12 and np.array_equal(qphysical,d.qpos)
out=dict(version=r['version'],simulation_steps=0,passed=passed,saved_state_t_s=float(z['t'][j]),source_anchor_material=anchor,aggregate_non_source_fore_normal_N=sum(ct['normal_N'] for ct in non),aggregate_non_source_fore_Fz_N=sum(ct['world_force_N'][2] for ct in non),software_threshold_N=r['robot_weight_N']/21,rows=rows,normal_point_displacements_mm=displacements,maximum_native_offset_deg=float(abs(dq).max()*180/np.pi),retained_native_offset_rad=float(abs(dq[8:]).max()),actual_pose_unchanged=bool(np.array_equal(qphysical,d.qpos)),scope='One saved actual pose/contact observation plus explicit sign/postcapture exclusions. Outward point displacement is private ideal kinematics only, not actual force relief or an integrated trajectory.')
(B/'DOWNWARD_RELIEF_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
