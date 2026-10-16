"""Plot the saved actual trace and export exact final rod motion, no stepping."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,sys,numpy as np,mujoco,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';C=B/'T_UPDATE';ROOT=next(p for p in P.parents if (p/'snake_control').exists());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
z=np.load(C/'trace.npz');audit=json.loads((C/'SOURCE_DRIFT_AND_GRAVITY_AUDIT.json').read_text());result=json.loads((C/'result.json').read_text());t=z['t'];release=result['first_release_t'];end=result['sim_time']
fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
ax[0].plot(t,z['actual_source_boundary_motion_m']*1000,label='Actual source position motion')
ax[0].plot(t,z['actual_source_boundary_X_motion_m']*1000,label='Actual source X motion')
ax[0].scatter([end],[audit['final_source_guard']['maximum_source_motion_mm']],color='red',zorder=4,label='Exact final state')
ax[0].axhline(12,color='red',ls='--',lw=1);ax[0].set_ylabel('Motion (mm)');ax[0].legend(loc='upper left',ncol=2)
forces=z['eval_pipe_world_N'];ax[1].plot(t,forces[:,:2,:,2].sum((1,2)),label='Source Fz')
ax[1].plot(t,forces[:,2:,:,2].sum((1,2)),label='Receiver Fz')
ax[1].axhline(28.01736,color='gray',ls=':',label='Robot weight');ax[1].set_ylabel('Vertical load (N)');ax[1].legend(loc='center left',ncol=3)
ax[2].plot(t,z['alpha'],label='alpha (front release)');ax[2].plot(t,z['beta'],label='beta (front wrap)')
ax[2].plot(t,z['S'],label='S (transport)');ax[2].set_ylabel('Phase coordinate');ax[2].legend(loc='upper left',ncol=3)
img=ax[3].imshow(abs(z['torque']).T,origin='lower',aspect='auto',extent=[t[0],t[-1],-.5,19.5],vmin=0,vmax=2.355,cmap='magma')
ax[3].set_ylabel('Joint index');ax[3].set_xlabel('Actual simulation time (s)');fig.colorbar(img,ax=ax[3],label='Absolute actuator torque (Nm)')
for a in ax:a.axvline(release,color='green',ls='--',lw=1);a.axvline(result['first_raw_front_contact_t'],color='orange',ls='--',lw=1);a.set_xlim(0,end)
fig.suptitle('V3.2 actual diagnostic: source drift stop; no receiver load\nGreen: release 2.1s | Orange: false source-seam event 4.175s | All data preserved')
fig.savefig(C/'ACTUAL_DIAGNOSTIC_TIMELINE.png',dpi=150);plt.close(fig)
m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;ref=int(np.argmin(abs(t-release)))
d.qpos[:]=z['qpos'][ref];mujoco.mj_kinematics(m,d);stable=d.xpos[h:h+21].copy()
state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);final=d.xpos[h:h+21].copy();centers=.5*(nodes(d)[:-1]+nodes(d)[1:])
with (C/'EXACT_FINAL_ROD_MOTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['material','role_at_stop','final_time_s','body_x_m','body_y_m','body_z_m','stable_reference_time_s','stable_body_x_m','stable_body_y_m','stable_body_z_m','body_motion_mm','body_X_motion_mm','final_center_x_m','final_center_y_m','final_center_z_m','last_trace_source_Fz_N','last_trace_receiver_Fz_N'])
    for i in range(21):w.writerow([i,'released_front' if i<14 else 'guarded_source',end,*final[i],t[ref],*stable[i],np.linalg.norm(final[i]-stable[i])*1000,(final[i,0]-stable[i,0])*1000,*centers[i],forces[-1,:2,i,2].sum(),forces[-1,2:,i,2].sum()])
print(str(C/'ACTUAL_DIAGNOSTIC_TIMELINE.png'))
