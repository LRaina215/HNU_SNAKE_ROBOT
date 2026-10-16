# -*- coding: utf-8 -*-
"""Actual V4.1 outcome and whether rear requested torque reached the servos."""
from pathlib import Path
import json,csv,numpy as np,mujoco,cv2,sys
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V4_1_ROBOT_MASS_HOLD';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from contact_sensor import known_source
z=np.load(C/'trace.npz');r=json.loads((C/'result.json').read_text(encoding='utf-8'));reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=int(np.argmin(abs(z['t']-r['first_release_t'])))
def poses(q):
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);return d.xpos[h:h+21].copy(),d.xmat[h:h+21].reshape(21,3,3).copy()
ps,rs=poses(z['qpos'][ref]);p0,_=poses(z['qpos'][0]);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);pf=d.xpos[h:h+21].copy();rf=d.xmat[h:h+21].reshape(21,3,3).copy();delta=pf[8:]-ps[8:];given=pf[8:]-p0[8:];polar=np.degrees(np.angle(np.exp(1j*(np.arctan2(pf[8:,2]-1,pf[8:,1])-np.arctan2(ps[8:,2]-1,ps[8:,1])))))
active=(z['alpha']>=.15)&z['source_hold_qp_valid'];error=z['torque'][active,8:]-z['source_hold_total_predicted_torque_Nm'][active,8:];contacts=[json.loads(line) for line in (C/'contact_evidence.jsonl').read_text(encoding='utf-8').splitlines()];force_rows=[]
for target in [5.,8.,10.,11.1]:
    j=int(np.argmin(abs(z['t']-target)));row=min(contacts,key=lambda v:abs(v['t']-z['t'][j]));actual=np.zeros((21,3));normal=np.zeros(21)
    for ct in row['contacts']:
        for k in ['point','normal','world_force_N']:ct[k]=np.asarray(ct[k],float)
        if ct['material']>=8 and known_source(ct,reg['source_known']):actual[ct['material']]+=ct['world_force_N'];normal[ct['material']]+=ct['normal_N']
    selected=(normal>0)|(z['source_hold_allocated_normal_N'][j]>0)
    force_rows.append(dict(t=float(z['t'][j]),actual_retained_normal_N=float(normal.sum()),allocated_normal_N=float(z['source_hold_allocated_normal_N'][j].sum()),per_material_force_error_norm_N=float(np.linalg.norm(actual[selected]-z['source_hold_allocated_force_N'][j,selected])),source_polar_error_deg=float(np.degrees(z['source_hold_polar_error_rad'][j])),requested_corrective_moment_Nm=z['source_hold_corrective_moment_Nm'][j].tolist(),command_filter_factor=float(z['command_filter_factor'][j])))
summary=dict(version=reg['version'],actual_result=r,final_guard=dict(given_motion_mm=float(np.linalg.norm(given,axis=1).max()*1000),stable_motion_mm=float(np.linalg.norm(delta,axis=1).max()*1000),stable_Xmotion_mm=float(abs(delta[:,0]).max()*1000),stable_polar_change_deg=float(abs(polar).max())),maximum_alpha=float(z['alpha'].max()),maximum_beta=float(z['beta'].max()),maximum_S=float(abs(z['S']).max()),maximum_sampled_true_receiver_normal_N=float(z['eval_pipe_normal_N'][:,2:].sum((1,2)).max()),rear_torque_realization_after_alpha0p15=dict(rms_error_Nm=float(np.sqrt(np.mean(error**2))),peak_error_Nm=float(abs(error).max()),filter_full_transmission_fraction=float(np.mean(z['command_filter_factor'][active]>.999999))),force_allocation_vs_observation=force_rows,
    interpretation='Rear actuator torques closely follow the requested values after blending, yet source pose drifts and measured force distribution differs from allocated forces. A feasible static contact wrench is not by itself a realized source-hold controller; this case does not establish a unique cause.',step2_complete=False,total_step2_starts=31)
cap=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));meta=dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)));frames=[]
for index in [0,50,100,meta['frames']-1]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
    if not ok:raise RuntimeError('Actual video read failed')
    frames.append(frame);cv2.imwrite(str(C/f'video_frame_{index:04d}.png'),frame)
cap.release();cv2.imwrite(str(C/'actual_video_contact_sheet.png'),np.concatenate(frames,axis=0));summary['video']=meta;(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (C/'EXACT_FINAL_ROD_MOTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['material','guarded_source','time_s','given_motion_mm','stable_motion_mm','stable_Xmotion_mm'])
    for i in range(21):w.writerow([i,i>=8,r['sim_time'],np.linalg.norm(pf[i]-p0[i])*1000,np.linalg.norm(pf[i]-ps[i])*1000,(pf[i,0]-ps[i,0])*1000])
report=f'''# V4.1 实际单例：源侧保持失败

原始物理t=0的唯一T_UPDATE实际积分至{r['sim_time']:.3f}秒，2.1秒开始释放，alpha最大{summary['maximum_alpha']:.3f}、beta0、S0。停止{r['stop']}，最终给定参考位移{summary['final_guard']['given_motion_mm']:.6f}mm，稳定参考位移{summary['final_guard']['stable_motion_mm']:.6f}mm，最终稳定X位移{summary['final_guard']['stable_Xmotion_mm']:.6f}mm、最大周向变化{summary['final_guard']['stable_polar_change_deg']:.6f}度。无接收接触、拟合、G写回、包绕、接管或整身滚动。完整Step2失败，未优于V3.3的15.125秒早停。

alpha>=0.15的有效分配采样中，后段实际力矩相对请求总力矩的RMS误差{summary['rear_torque_realization_after_alpha0p15']['rms_error_Nm']:.6f}Nm、最大{summary['rear_torque_realization_after_alpha0p15']['peak_error_Nm']:.6f}Nm。请求主要已到达后段舵机；但接触力分布未按静力分配实现，源侧继续周向漂移。不能把“力矩请求有界/送达”当“实际保持成功”，也尚不能唯一归因。

当前静态分配缺少源接触点加速度/位姿与浮动基动力学的共同约束，并省略其他接触载荷和接触力矩。下一步应先在少量已保存状态上检查兼容的源接触保持/动力学命令，实际运行仍须同一初态、原物理及12mm保护。所有旧例、启动器未积分错误、连续录像与逐杆数据保留；累计31次实际启动。V4原质量错误版本0次积分，V4.1只有这1次，不重复。
'''
(P/'V4_1_ACTUAL_DIAGNOSTIC.md').write_text(report,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
