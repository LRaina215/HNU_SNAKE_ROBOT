"""Actual saved known-source contact slip estimate; no dynamics integration.

Material displacement includes C posture and passive rolling. Contact-point
tangential velocity is a separate diagnostic, sampled from the saved10Hz state.
"""
from pathlib import Path
import sys,json,numpy as np
C=Path(sys.argv[1]).resolve();sys.path.insert(0,str(C))
from physics import mujoco,MODEL
from contact_sensor import known_source
r=json.loads((C/'REGISTERED.json').read_text());result=json.loads((C/'result.json').read_text());z=np.load(C/'trace.npz');t=z['t'];end=result['first_capture_t'];selected=np.arange(len(t));loaded_times=[]
records=[json.loads(x) for x in (C/'contact_evidence.jsonl').open()];record_t=np.array([x['t'] for x in records]);m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);head=m.body('head').id;rows=[]
for i in selected:
    j=int(np.argmin(abs(record_t-t[i])));record=records[j];d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    data=[]
    for value in record['contacts']:
        ct=dict(value)
        for key in ['point','normal','world_force_N']:ct[key]=np.array(ct[key])
        if ct['normal_N']<=.5 or not known_source(ct,r['source_known']) or abs(ct['normal'][0])>=.45:continue
        point=ct['point']-.5*ct['penetration_m']*ct['normal'];rho=np.linalg.norm(point[1:]-r['source_known']['center_yz_m'])
        if abs(rho-r['source_known']['radius_m'])>.01:continue
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,point,head+ct['material']);velocity=jp@d.qvel;n=ct['normal'];n=n/max(np.linalg.norm(n),1e-12);tangent=velocity-n*(velocity@n)
        radial=np.r_[0.,point[1:]-r['source_known']['center_yz_m']];radial/=np.linalg.norm(radial);circum=np.cross([1.,0,0],radial)
        data.append((ct['normal_N'],tangent[0],tangent@circum,np.linalg.norm(tangent)))
    if data:
        a=np.array(data);w=a[:,0]/a[:,0].sum();rows.append((t[i],w@a[:,1],w@abs(a[:,1]),w@abs(a[:,2]),w@a[:,3]));loaded_times.append(float(t[i]))
    else:rows.append((t[i],0.,0.,0.,0.))
a=np.array(rows);before=a if end is None else a[a[:,0]<end];integrals=np.trapz(before[:,1:],before[:,0],axis=0)*1000 if len(before)>1 else np.zeros(4);all_integrals=np.trapz(a[:,1:],a[:,0],axis=0)*1000 if len(a)>1 else np.zeros(4)
out=dict(evaluation_only=True,simulation_steps=0,scope='Before first validated bearing, or whole failed acquisition if no capture',end_s=None if not len(a) else float(a[-1,0]),sampling_hz=10,contact_point='Saved contact midpoint recovered from environment point/normal/penetration',normal_load_weighted_signed_axial_slip_mm=float(integrals[0]),normal_load_weighted_absolute_axial_slip_path_mm=float(integrals[1]),normal_load_weighted_absolute_circumferential_slip_path_mm=float(integrals[2]),normal_load_weighted_tangential_slip_path_mm=float(integrals[3]),limitations=['Sampled point-velocity estimate using saved qpos/qvel and nearest saved unlabeled contact, not a continuous exact integral','Known source cylinder classification only; source cap normals excluded','A weighted slip path is not the same quantity as a particular material displacement or total source advance','Intentional posture/normal rolling/feed are not silently counted as pure contact sliding; normal original progression is allowed by the current user contract'])
out.update(end_s=None if not len(before) else float(before[-1,0]),minimum_selected_contact_normal_N=.5,all_run_end_s=None if not len(a) else float(a[-1,0]),last_selected_source_contact_s=None if not loaded_times else loaded_times[-1],all_run_normal_load_weighted_signed_axial_slip_mm=float(all_integrals[0]),all_run_normal_load_weighted_absolute_axial_slip_path_mm=float(all_integrals[1]),all_run_normal_load_weighted_absolute_circumferential_slip_path_mm=float(all_integrals[2]),all_run_normal_load_weighted_tangential_slip_path_mm=float(all_integrals[3]))
out['limitations'][-1]='Contact-relative tangential velocity includes sliding created by the intentional gait. It is distinct from center/material displacement; normal original progression is allowed.'
np.savetxt(C/'SOURCE_CONTACT_SLIP_TIMELINE.csv',a,delimiter=',',header='t_s,weighted_signed_axial_velocity_m_s,weighted_abs_axial_velocity_m_s,weighted_abs_circumferential_velocity_m_s,weighted_tangential_speed_m_s',comments='')
(C/'SOURCE_CONTACT_SLIP_METRICS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
