"""Read a completed V6 actual run: balance prediction, physical motion and video."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,ast,numpy as np,mujoco,cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from contact_sensor import known_source
z=np.load(C/'trace.npz');r=json.loads((C/'result.json').read_text(encoding='utf-8'));reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);head=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=int(np.argmin(abs(z['t']-r['first_release_t']))) if r['first_release_t'] is not None else 0
def pose(q):
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    points=d.xpos[head:head+21].copy();moment=float(sum(np.cross(d.xipos[head+i]-[0,0,1],m.body_mass[head+i]*m.opt.gravity)[0] for i in range(21)))
    return points,moment
p0,_=pose(z['qpos'][0]);ps,_=pose(z['qpos'][ref]);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);pf,final_moment=pose(d.qpos.copy());n=int(z['source_guard_material_start'][-1]);given=pf-p0;stable=pf-ps;gravity=[];predicted=[];rows=[]
for j,q in enumerate(z['qpos']):
    _,moment=pose(q);gravity.append(moment)
    try:metric=ast.literal_eval(str(z['planner_metrics'][j]))
    except (ValueError,SyntaxError):metric={}
    prediction=metric.get('source_axis_balance_moment_Nm',float('nan'));predicted.append(prediction)
    rows.append(dict(t=float(z['t'][j]),alpha=float(z['alpha'][j]),beta=float(z['beta'][j]),source_guard_start=int(z['source_guard_material_start'][j]),actual_gravity_source_X_moment_Nm=moment,C_predicted_net_source_X_moment_Nm=prediction,given_drift_mm=float(z['actual_source_motion_from_given_m'][j]*1000),stable_drift_mm=float(z['actual_source_boundary_motion_m'][j]*1000),rear_ctrl_change_deg=float(abs(z['ctrl'][j,14:]-z['ctrl'][0,14:]).max()*180/np.pi)))
out=dict(version=reg['version'],actual_result=r,source_guard_start=n,exact_final_given_motion_mm=float(np.linalg.norm(given[n:],axis=1).max()*1000),exact_final_stable_motion_mm=float(np.linalg.norm(stable[n:],axis=1).max()*1000),exact_final_stable_Xmotion_mm=float(abs(stable[n:,0]).max()*1000),exact_final_gravity_source_X_moment_Nm=final_moment,maximum_sampled_true_receiver_normal_N=float(z['eval_pipe_normal_N'][:,2:].sum((1,2)).max()),maximum_alpha=float(z['alpha'].max()),maximum_beta=float(z['beta'].max()),maximum_S_m=float(z['S'].max()),maximum_retained_ctrl_change_deg=float(max(row['rear_ctrl_change_deg'] for row in rows)),step2_complete=False,total_actual_starts=34,interpretation='One actual development run. Geometric moment reduction is not source retention, true contact, capture, or completed action; use the original actual guards and ordered action evaluator.')
cap=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));out['video']=dict(frames=count,fps=cap.get(cv2.CAP_PROP_FPS));frames=[]
for index in np.linspace(0,count-1,5).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
    if not ok:raise RuntimeError('Saved video frame unreadable')
    frames.append(frame)
cap.release();cv2.imwrite(str(C/'actual_video_contact_sheet.png'),np.concatenate(frames,axis=0));(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (C/'SOURCE_AXIS_BALANCE_ACTUAL.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (C/'EXACT_FINAL_ROD_MOTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['material','guarded_source','t_s','given_motion_mm','stable_motion_mm','stable_Xmotion_mm'])
    for i in range(21):writer.writerow([i,i>=n,r['sim_time'],np.linalg.norm(given[i])*1000,np.linalg.norm(stable[i])*1000,stable[i,0]*1000])
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained');axes[0].plot(z['t'],gravity,label='actual gravity moment');axes[0].plot(z['t'],predicted,'--',label='C predicted net moment');axes[0].set_ylabel('Moment about source X (Nm)');axes[0].axhline(0,color='black',lw=.5);axes[0].legend();axes[1].plot(z['t'],z['actual_source_motion_from_given_m']*1000,label='actual source drift from given');axes[1].plot(z['t'],z['actual_source_boundary_motion_m']*1000,label='actual source drift from stable');axes[1].axhline(12,color='black',ls='--',label='original stop guard');axes[1].set_ylabel('Maximum body-origin motion (mm)');axes[1].set_xlabel('Actual simulation time(s)');axes[1].legend()
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('V6 one actual development run: moment reduction and physical source motion\nC net moment includes observed non-source forces; actual curve shows gravity only.10Hz samples.')
fig.savefig(C/'SOURCE_AXIS_BALANCE_ACTUAL.png',dpi=150);plt.close(fig);print(json.dumps(out,ensure_ascii=False,indent=2))
