"""Document exact actual source drift, body deformation, and frozen command hold."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,numpy as np,mujoco,cv2
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS';C=B/'T_UPDATE';z=np.load(C/'trace.npz');result=json.loads((C/'result.json').read_text(encoding='utf-8'));reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
def poses(q):
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);return d.xpos[h:h+21].copy(),d.xmat[h:h+21].reshape(21,3,3).copy()
ref=int(np.argmin(abs(z['t']-result['first_release_t'])));ps,rs=poses(z['qpos'][ref]);p0,r0=poses(z['qpos'][0]);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);pf=d.xpos[h:h+21].copy();rf=d.xmat[h:h+21].reshape(21,3,3).copy();finalq=d.qpos.copy();n=8
delta=pf[n:]-ps[n:];given=pf[n:]-p0[n:];polar=np.degrees(np.angle(np.exp(1j*(np.arctan2(pf[n:,2]-1,pf[n:,1])-np.arctan2(ps[n:,2]-1,ps[n:,1])))));orient=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',rs[n:],rf[n:])-1)/2,-1,1)))
a=ps[n:];b=pf[n:];u,s,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)));R=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T;prediction=(a-a.mean(0))@R.T+b.mean(0);shape_rms=np.sqrt(np.mean(np.sum((b-prediction)**2,axis=1)))*1000
summary=dict(version=reg['version'],actual_result=result,source_guard=dict(material_start=n,final_given_motion_mm=float(np.linalg.norm(given,axis=1).max()*1000),final_stable_motion_mm=float(np.linalg.norm(delta,axis=1).max()*1000),final_stable_Xmotion_mm=float(abs(delta[:,0]).max()*1000),final_stable_polar_change_deg=float(abs(polar).max()),final_stable_orientation_deg=float(orient.max()),source_shape_rigid_fit_residual_rms_mm=float(shape_rms)),
    posterior_total_command_max_change_deg=float(abs(z['raw_native_command'][:,n:]-z['ctrl'][0,n:]).max()*180/np.pi),posterior_actual_joint_change_peak_deg=float(abs(z['qpos'][ref:,qa[n:]]-z['qpos'][ref,qa[n:]]).max()*180/np.pi),posterior_actual_joint_change_final_deg=float(abs(finalq[qa[n:]]-z['qpos'][ref,qa[n:]]).max()*180/np.pi),posterior_joint17_saturation_fraction=float(np.mean(abs(z['torque'][ref:,17])>=2.355-1e-4)),source_Fz_range_after_release_N=[float(z['eval_pipe_world_N'][ref:,:2,:,2].sum((1,2)).min()),float(z['eval_pipe_world_N'][ref:,:2,:,2].sum((1,2)).max())],maximum_sampled_receiver_normal_N=float(z['eval_pipe_normal_N'][:,2:].sum((1,2)).max()),maximum_alpha=float(z['alpha'].max()),maximum_beta=float(z['beta'].max()),S_max=float(z['S'].max()),fit_or_global_write_or_capture=False,
    inference='Dominant source motion is common circumferential roll with small source-shape deformation, while posterior total joint targets stay fixed. Fixed joint targets alone do not regulate source world pose. Joint17 is persistently torque saturated; a unique mechanical cause has not been proven.',next='Check source contact-wrench/load allocation on a few saved actual states before a focused source-hold correction; do not loosen drift guards or change initial/physical parameters.',step2_complete=False,total_step2_starts=30)
cap=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));meta=dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)));frames=[]
for index in [0,50,100,meta['frames']-1]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
    if not ok:raise RuntimeError('Unreadable actual frame')
    cv2.imwrite(str(C/f'video_frame_{index:04d}.png'),frame);frames.append(frame)
cap.release();cv2.imwrite(str(C/'actual_video_contact_sheet.png'),np.concatenate(frames,axis=0));summary['video']=meta
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (C/'EXACT_FINAL_ROD_MOTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['material','guarded_source','final_time_s','given_motion_mm','stable_motion_mm','stable_Xmotion_mm','stable_polar_change_deg','stable_orientation_deg'])
    for i in range(21):w.writerow([i,i>=n,result['sim_time'],np.linalg.norm(pf[i]-p0[i])*1000,np.linalg.norm(pf[i]-ps[i])*1000,(pf[i,0]-ps[i,0])*1000,polar[i-n] if i>=n else '',orient[i-n] if i>=n else ''])
report=f'''# V3.3 单例实际诊断

从原始物理 t=0 运行唯一 T_UPDATE，2.100 秒开始释放，15.125 秒在 α≈{summary['maximum_alpha']:.3f}、β=0、S=0 时触发 source_boundary_drift_from_given。无前部非源接触、可靠拟合、G 写回、包绕、接收承载、整身转移或滚动。完整 Step2 未通过，原始冻结代码与连续录像保留。

停止时源段相对给定初态最大位移 {summary['source_guard']['final_given_motion_mm']:.6f} mm，超过原 12 mm 限制；相对 2.1 秒稳定参考最大位移 {summary['source_guard']['final_stable_motion_mm']:.6f} mm。稳定参考的最大 X 位移 {summary['source_guard']['final_stable_Xmotion_mm']:.6f} mm，最大周向变化 {summary['source_guard']['final_stable_polar_change_deg']:.6f}°，主要为源管周向共同转动。M8–M20 的刚体配准残差 RMS {shape_rms:.6f} mm。后段总关节目标始终保持给定值（最大变化 {summary['posterior_total_command_max_change_deg']:.3g}°），实际后段关节变化峰值 {summary['posterior_actual_joint_change_peak_deg']:.6f}°。因此“保持后段关节目标”不能替代“保持源侧世界位姿”。

释放后源侧竖向载荷采样范围 {summary['source_Fz_range_after_release_N']} N，真实接收侧所有已保存采样载荷为零。关节索引17（第18关节）在释放后 {summary['posterior_joint17_saturation_fraction']:.1%} 的采样达到2.355Nm物理力矩限值。饱和是否为主要原因仍需验证，不能因一个饱和关节就直接归因。

下一步只在少量已保存实际状态上核对源侧接触力/力矩分配，判断固定预载是否无法维持所需外部力矩，再实现有依据的源侧保持修正。原初态、真实物理及12mm保护不改；不再进行离线整条包绕寻优后才开实际诊断。最终四次配对仍待完整动作证据。累计实际启动30次，当前直接分支2次，完整动作成功0。
'''
(P/'V3_3_ACTUAL_DIAGNOSTIC.md').write_text(report,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
