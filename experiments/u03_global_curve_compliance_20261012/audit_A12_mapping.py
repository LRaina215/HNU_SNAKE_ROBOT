"""Post-trial anchor applicability and actual limited-command FK."""
from physics import *
import json
def shape(native,C):
    x=native[10:];y=C[10:];cx=x.mean(0);cy=y.mean(0);u,_,v=np.linalg.svd((x-cx).T@(y-cy));rot=u@np.diag([1,1,np.linalg.det(u@v)])@v;err=np.linalg.norm((native-cx)@rot+cy-C,axis=1)*1000
    return dict(front_max_mm=float(err[:7].max()),front_rms_mm=float(np.sqrt(np.mean(err[:7]**2))),rear_rms_mm=float(np.sqrt(np.mean(err[10:]**2))))
case=BASE/'A12';z=np.load(case/'trace.npz');t=z['t'];loc=z['reference_locations'];source_count=np.rint(z['source_reference']*22).astype(int)
first=np.where(source_count>=6,22-source_count,16)
modified=abs(z['local_native_delta'])>1e-8
tainted=np.array([bool(np.any(modified[k,np.arange(20)>=a])) for k,a in enumerate(first)])
switch=np.flatnonzero(np.diff(first)!=0)+1
m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);initial=np.load(case/'initial.npz')['qpos'];qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
rows=[]
for sec in [30,38,40,50,55,60,65,70,75,80,85,94.925,100,120,float(t[-1])]:
    if sec>t[-1]:continue
    k=int(np.argmin(abs(t-sec)));C=z['nominal_curve_points'][k];parts=[]
    for q in [z['curvature_raw_command'][k],z['raw_command'][k],z['ctrl'][k]]:
        d.qpos[:]=initial;d.qpos[qa]=q;mujoco.mj_forward(m,d);parts.append(shape(nodes(d),C))
    rows.append(dict(t=float(t[k]),anchor_first_node=int(first[k]),anchor_internal_hinges_modified=bool(tainted[k]),curvature_command=parts[0],corrected_command=parts[1],actual_limited_command=parts[2],max_raw_ctrl_deg=float(np.rad2deg(abs(z['raw_command'][k]-z['ctrl'][k]).max())),optimizer_registration_before_mm=float(z['local_native_front_before_mm'][k]),optimizer_registration_after_mm=float(z['local_native_front_after_mm'][k])))
events=[]
for k in switch:
    events.append(dict(t=float(t[k]),before_first_node=int(first[k-1]),after_first_node=int(first[k]),raw_command_change_deg=float(np.rad2deg(abs(z['raw_command'][k]-z['raw_command'][k-1]).max())),ctrl_change_deg=float(np.rad2deg(abs(z['ctrl'][k]-z['ctrl'][k-1]).max())),sample_interval_s=float(t[k]-t[k-1]),tainted=bool(tainted[k])))
out=dict(source='saved nominal commands plus original-model mj_forward; no new dynamics or control changes',registration_for_all_shape_rows='same rear nodes10..21 rigid fit diagnostic',optimizer_applicability='Clean anchor removes only rigid motion; modified internal anchor hinges can absorb shape changes. Optimizer residual alone is never transfer evidence.',first_tainted_s=float(t[np.flatnonzero(tainted)[0]]) if tainted.any() else None,anchor_switches=events,samples=rows)
(case/'mapping_audit.json').write_text(json.dumps(out,indent=2));(BASE/'A12_MAPPING_AUDIT.json').write_text(json.dumps(out,indent=2));print('anchor first taint',out['first_tainted_s'],'switch count',len(events))
