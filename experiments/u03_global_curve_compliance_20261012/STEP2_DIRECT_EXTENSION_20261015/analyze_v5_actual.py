"""Preserve V5 actual grip/shape loss, without another rollout."""
from pathlib import Path
import sys,json,csv,numpy as np,mujoco,cv2
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS';C=B/'T_UPDATE';sys.path.insert(0,str(B));from contact_sensor import known_source
z=np.load(C/'trace.npz');r=json.loads((C/'result.json').read_text(encoding='utf-8'));reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);head=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=int(np.argmin(abs(z['t']-r['first_release_t'])))
def points(q):d.qpos[:]=q;mujoco.mj_kinematics(m,d);return d.xpos[head:head+21].copy()
ps=points(z['qpos'][ref]);p0=points(z['qpos'][0]);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);pf=d.xpos[head:head+21].copy();delta=pf[8:]-ps[8:];given=pf[8:]-p0[8:];records=[json.loads(line) for line in (C/'contact_evidence.jsonl').read_text(encoding='utf-8').splitlines()];rows=[]
for target in [2.1,3.,4.,5.]:
    j=int(np.argmin(abs(z['t']-target)));record=min(records,key=lambda row:abs(row['t']-z['t'][j]));normal=0.
    for ct in record['contacts']:
        for k in ['point','normal','world_force_N']:ct[k]=np.asarray(ct[k],float)
        if ct['material']>=8 and known_source(ct,reg['source_known']):normal+=ct['normal_N']
    rows.append(dict(t=float(z['t'][j]),alpha=float(z['alpha'][j]),mean_source_polar_error_deg=float(np.degrees(z['source_hold_polar_error_rad'][j])),source_Xmotion_mm=float(z['actual_source_boundary_X_motion_m'][j]*1000),actual_retained_normal_N=normal,allocated_normal_N=float(z['source_hold_allocated_normal_N'][j].sum()),maximum_posterior_actual_joint_delta_deg=float(abs(z['qpos'][j,qa[8:]]-z['qpos'][ref,qa[8:]]).max()*180/np.pi),maximum_posterior_ctrl_change_deg=float(abs(z['ctrl'][j,8:]-z['ctrl'][0,8:]).max()*180/np.pi),command_filter_factor=float(z['command_filter_factor'][j]),QP_valid=bool(z['source_hold_qp_valid'][j]),requested_posterior_torque_Nm=z['source_hold_total_predicted_torque_Nm'][j,8:].tolist(),actual_posterior_torque_Nm=z['torque'][j,8:].tolist()))
out=dict(version=reg['version'],actual_result=r,final_given_motion_mm=float(np.linalg.norm(given,axis=1).max()*1000),final_stable_motion_mm=float(np.linalg.norm(delta,axis=1).max()*1000),final_stable_Xmotion_mm=float(abs(delta[:,0]).max()*1000),maximum_sampled_true_receiver_normal_N=float(z['eval_pipe_normal_N'][:,2:].sum((1,2)).max()),selected_rows=rows,step2_complete=False,total_actual_starts=32,interpretation='V5 changed the failure mode: mean polar slip is smaller at5s, but posterior grip/shape targets relax, allocated normal load decreases, and source axial drift exceeds the original guard. This is not a performance improvement or unique causal proof.')
cap=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));out['video']=dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS));frames=[]
for index in [0,20,30,40,50]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
    if not ok:raise RuntimeError('Saved video frame unreadable')
    frames.append(frame)
cap.release();cv2.imwrite(str(C/'actual_video_contact_sheet.png'),np.concatenate(frames,axis=0));(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (C/'EXACT_FINAL_ROD_MOTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['material','guarded_source','t_s','given_motion_mm','stable_motion_mm','stable_Xmotion_mm'])
    for i in range(21):w.writerow([i,i>=8,r['sim_time'],np.linalg.norm(pf[i]-p0[i])*1000,np.linalg.norm(pf[i]-ps[i])*1000,(pf[i,0]-ps[i,0])*1000])
print(json.dumps(out,ensure_ascii=False,indent=2))
