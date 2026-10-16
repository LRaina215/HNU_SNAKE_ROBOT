"""Post-run shape/pose and receiving diagnostics; never updates controller."""
from physics import *
import json
case=BASE/'A11';z=np.load(case/'trace.npz');t=z['t'];m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);initial=np.load(case/'initial.npz')['qpos'];qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
def shape(native,C):
    x=native[10:];y=C[10:];cx=x.mean(0);cy=y.mean(0);u,sv,v=np.linalg.svd((x-cx).T@(y-cy));r=u@np.diag([1,1,np.linalg.det(u@v)])@v;fit=(native-cx)@r+cy;err=np.linalg.norm(fit-C,axis=1)*1000
    return dict(front_rms_mm=float(np.sqrt(np.mean(err[:7]**2))),front_max_mm=float(max(err[:7])),rear_rms_mm=float(np.sqrt(np.mean(err[10:]**2))),full_rms_mm=float(np.sqrt(np.mean(err**2))))
rows=[]
for sec in [0,10,14.5,20,22,30,38,40,42,45,48,50,60,65,70,75,80,89.275,90,94.925,100,110,120,140,160,180,200,float(t[-1])]:
    if sec>t[-1]:continue
    i=int(np.argmin(abs(t-sec)));C=z['nominal_curve_points'][i];p=z['points'][i];c=z['centers'][i];cc=(C[:-1]+C[1:])/2;part=[]
    for cmd in [z['raw_command'][i],z['ctrl'][i]]:
        d.qpos[:]=initial;d.qpos[qa]=cmd;mujoco.mj_forward(m,d);part.append(shape(nodes(d),C))
    mask=(cc[10:,0]<.95)&(np.linalg.norm(cc[10:,1:]-[0,1],axis=1)<.13)
    ap=np.arctan2(c[10:,2]-1,c[10:,1]);cp=np.arctan2(cc[10:,2]-1,cc[10:,1]);diff=ap[mask]-cp[mask]
    phase=float(np.rad2deg(np.arctan2(np.sin(diff).mean(),np.cos(diff).mean()))) if mask.any() else None
    frontphase=np.unwrap(np.arctan2(c[:6,2]-1,c[:6,0]-1.1))
    rows.append(dict(t=float(t[i]),head_center_m=c[0].tolist(),C_head_center_m=cc[0].tolist(),head_world_error_mm=((c[0]-cc[0])*1000).tolist(),source_rear_axial_error_mm=float((c[10:,0][mask]-cc[10:,0][mask]).mean()*1000) if mask.any() else None,source_rear_polar_error_deg=phase,raw_shape=part[0],ctrl_shape=part[1],actual_shape=shape(p,C),target_front_force_N=float(z['forces'][i,1,:6].sum()),target_front_per_module_N=z['forces'][i,1,:6].tolist(),source_contact_modules=np.flatnonzero(z['forces'][i,0]>.1).tolist(),target_contact_modules=np.flatnonzero(z['forces'][i,1]>.1).tolist(),nominal_source_node_fraction=float(z['source_reference'][i]),nominal_target_node_fraction=float(z['target_reference'][i]),target_total_N=float(z['forces'][i,1].sum()),front_span_deg=float(np.rad2deg(np.ptp(frontphase))),max_raw_to_ctrl_deg=float(np.rad2deg(max(abs(z['raw_command'][i]-z['ctrl'][i])))),max_ctrl_to_actual_deg=float(np.rad2deg(max(abs(z['ctrl'][i]-z['qpos'][i,qa])))),front_joint_torque_Nm=z['torque'][i,:6].tolist()))
first=[]
for n in range(6):
    ix=np.flatnonzero(z['forces'][:,1,n]>.1);first.append(dict(module=n,first_target_contact_s=float(t[ix[0]]) if len(ix) else None,last_target_contact_s=float(t[ix[-1]]) if len(ix) else None,peak_target_force_N=float(z['forces'][:,1,n].max())))
out=dict(case='A11',physical_steps_performed=True,high_level_feedback=False,registration='rear nodes10..21 rigid alignment diagnostic only; no control correction',interpretation='World error combines pose and shape; rear alignment only separates residual shape approximately. Source averages include nominal source-side rear modules only. Force is original normal contact, not a success marker.',front_contacts=first,samples=rows)
(BASE/'A11_actual_diagnosis.json').write_text(json.dumps(out,indent=2));(case/'actual_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(front_contacts=first,last=rows[-1]),indent=2))


