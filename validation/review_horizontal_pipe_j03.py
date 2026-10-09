"""Independently review saved J03 actual states without physical stepping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.suspended_supply_transfer import nodes
from validation.suspended_supply_experiment import write
BASE=ROOT/'experiments/unified_curve_transfer_20261007/HORIZONTAL_PIPE_20261009'

def metrics(points):
    c=.5*(points[:-1]+points[1:]);r=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
    phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));dy=np.diff(c[:,1]);da=np.diff(phi);fit=np.polyfit(phi,c[:,1],1)
    pitch=-fit[0]*2*np.pi;res=c[:,1]-np.polyval(fit,phi)
    valid=bool(np.all((r>.065)&(r<.125)&(c[:,1]>.15)) and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045))
        and .126<pitch<.234 and np.sqrt(np.mean(res**2))<.012 and c[0,1]-c[-1,1]>.30)
    return dict(valid=valid,c=c,r=r,phi=phi,dy=dy,da=da,pitch=pitch,axis_rms=float(np.sqrt(np.mean(res**2))),res=res)

def review():
    case=BASE/'J03'
    with np.load(case/'trace.npz') as saved:z={k:saved[k] for k in saved.files}
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_single_horizontal_pipe_long.xml'));d=mujoco.MjData(m)
    t=z['continuation_t_s'];ids=np.flatnonzero(t>=t[-1]-1.);samples=[];difference=0.
    for i in ids:
        d.qpos[:]=z['qpos'][i];mujoco.mj_fwdPosition(m,d);p=nodes(d);a=metrics(p)
        difference=max(difference,float(abs(p*1000-z['points_mm'][i]).max()))
        samples.append(dict(t_s=float(t[i]),valid=a['valid'],pitch_mm=float(a['pitch']*1000),axis_rms_mm=a['axis_rms']*1000,
            tail_axis_step_mm=float(a['dy'][-1]*1000),tail_circum_step_rad=float(a['da'][-1]),
            all12_current_received=bool(z['actual_new_retained_mask'][i,9:].all()),joint20_kp=float(z['kp_nm_per_rad'][i,-1]),
            posterior_bearing_n=float(z['contact_force_N'][i,1,15:].sum()),ground=bool(z['ground'][i])))
    final=metrics(z['points_mm'][-1]/1000);first=metrics(z['points_mm'][0]/1000)
    # The last second is checked sample by sample against saved actual FK.
    accepted=[s['valid'] and s['all12_current_received'] and s['joint20_kp']>120 and s['posterior_bearing_n']>.2 and not s['ground'] for s in samples]
    connection=np.zeros((len(t),21),bool)
    for i,p in enumerate(z['points_mm']/1000):
        a=metrics(p);connection[i,:9]=True
        connection[i,1:]=(a['r'][1:]>.065)&(a['r'][1:]<.125)&(a['r'][:-1]>.065)&(a['r'][:-1]<.125)&(a['c'][1:,1]>.15)&(a['c'][:-1,1]>.15)&(a['da']>.30)&(a['da']<1.1)&(a['dy']<-.002)&(a['dy']>-.045)&(np.abs(a['dy']+.180/(2*np.pi)*a['da'])<.025)
    rows=[]
    for j in range(20):
        kp=z['kp_nm_per_rad'][:,j];drive=z['torque'][:,j];full=np.flatnonzero((kp>126)&(t>2))
        rows.append(dict(joint=j+1,kp_initial=float(kp[0]),kp_final=float(kp[-1]),kp_min=float(kp.min()),kp_max=float(kp.max()),
            first_full_after_2s=float(t[full[0]]) if len(full) else None,drive_torque_max_nm=float(abs(drive).max()),drive_torque_final_nm=float(drive[-1]),
            drive_when_kp_zero_max_nm=float(abs(drive[kp<1e-9]).max()) if np.any(kp<1e-9) else None,
            original_damping=float(z['damping_nm_s_per_rad'][-1,j]),force_range_nm=z['force_range_nm'][-1,j]))
    report=dict(name='J03',evidence='saved real27-qpos states, original frozen physical model forward geometry only; no mj_step',physical_audit_steps=0,
        actual_seconds=float(t[-1]),shift_final_mm=float(z['shift_mm'][-1]),whole_single_pipe_helix_complete=bool(all(accepted)),cross_pipe_complete=False,
        final_one_second_all_geometry_and_bearing_valid=bool(all(accepted)),retention_samples=len(samples),retention_duration_s=float(t[ids[-1]]-t[ids[0]]),
        raw_geometry_reconstruction_difference_mm=difference,final_pitch_mm=float(final['pitch']*1000),final_axis_fit_rms_mm=float(final['axis_rms']*1000),
        final_center_radius_range_mm=[float(final['r'].min()*1000),float(final['r'].max()*1000)],
        final_centre_turns=float(np.ptp(final['phi'])/(2*np.pi)),final_axis_span_mm=float((final['c'][0,1]-final['c'][-1,1])*1000),
        final_tail_axis_step_mm=float(final['dy'][-1]*1000),final_tail_circum_step_rad=float(final['da'][-1]),
        final_joint20_deg=float(np.degrees(z['q'][-1,-1])),final_joint20_kp=float(z['kp_nm_per_rad'][-1,-1]),
        final_tail_bearing_n=float(z['contact_force_N'][-1,1,-1]),final_total_bearing_n=float(z['contact_force_N'][-1,1].sum()),
        final_own_bearing_modules=np.flatnonzero(z['contact_force_N'][-1,1]>.1).tolist(),
        actual_first_front_span_rad=float(z['actual_front_span_rad'][0]),actual_final_front_span_rad=float(z['actual_front_span_rad'][-1]),
        actual_body_y_span_mm=[float(final['c'][:,1].min()*1000),float(final['c'][:,1].max()*1000)],
        pipe_axis_y_m=[.15,1.65],first_to_last_head_axis_mm=float(z['centers_mm'][-1,0,1]-z['centers_mm'][0,0,1]),
        all20_final_kp_127=bool(np.all(z['kp_nm_per_rad'][-1]>126)),actual_new_received=np.flatnonzero(z['actual_new_retained_mask'][-1]).tolist(),
        limits=dict(maximum_actual_joint_deg=float(np.degrees(abs(z['q']).max())),maximum_drive_torque_nm=float(abs(z['torque']).max()),
            maximum_pipe_contact_depth_mm=float(z['pipe_depth_mm'].max()),maximum_self_depth_mm=float(z['self_depth_mm'].max()),ground=bool(z['ground'].any())),
        meaningful_limit='one artificially prewrapped initial state, one successful deterministic rollout on a lengthened horizontal pipe; not cross-pipe transfer or robustness evidence')
    write(case/'actual_body_shape_review.json',report);write(case/'final_one_second_review.json',samples);write(case/'joint_region_dynamics_summary.json',rows)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for name in ['J01','J02','J03']:
        with np.load(BASE/name/'trace.npz') as old:axes[0,0].plot(old['continuation_t_s'][::10],old['actual_front_span_rad'][::10],label=name)
    axes[0,0].set(xlabel='Actual time (s)',ylabel='Actual front span (rad)',title='Actual carrying wrap');axes[0,0].legend();axes[0,0].grid(alpha=.3)
    image=axes[0,1].imshow(z['kp_nm_per_rad'][::10].T,origin='lower',aspect='auto',extent=[0,t[-1],.5,20.5],vmin=0,vmax=127,cmap='viridis')
    axes[0,1].set(xlabel='Actual time (s)',ylabel='Joint number',title='J03 actual stiffness (Nm/rad)');fig.colorbar(image,ax=axes[0,1])
    image=axes[0,2].imshow(z['torque'][::10].T,origin='lower',aspect='auto',extent=[0,t[-1],.5,20.5],vmin=-2.355,vmax=2.355,cmap='coolwarm')
    axes[0,2].set(xlabel='Actual time (s)',ylabel='Joint number',title='J03 actual active drive (Nm)');fig.colorbar(image,ax=axes[0,2])
    image=axes[1,0].imshow(connection[::10,9:].T,origin='lower',aspect='auto',extent=[0,t[-1],8.5,20.5],vmin=0,vmax=1,cmap='Greens')
    axes[1,0].set(xlabel='Actual time (s)',ylabel='Posterior module number',title='Actual adjacent helix connection')
    axes[1,1].plot(final['phi'],final['c'][:,1]*1000,'o-',label='All21 actual centres')
    fit=np.polyfit(final['phi'],final['c'][:,1],1);axes[1,1].plot(final['phi'],np.polyval(fit,final['phi'])*1000,'--',label=f'Pitch {report["final_pitch_mm"]:.1f}mm')
    axes[1,1].scatter(final['phi'][-1],final['c'][-1,1]*1000,c='red',label='Tail follows same pitch')
    axes[1,1].set(xlabel='Unwrapped circumferential angle (rad)',ylabel='Actual axis Y (mm)',title='Final entire body is one helix');axes[1,1].legend();axes[1,1].grid(alpha=.3)
    for seconds in [0,100,200,float(t[-1])]:
        i=int(np.argmin(abs(t-seconds)));axes[1,2].plot(np.arange(21),z['centers_mm'][i,:,2],'-o',ms=3,label=f'{t[i]:.1f}s')
    axes[1,2].axhline(1000,color='k',ls='--',lw=1);axes[1,2].set(xlabel='Module number',ylabel='Actual centre height (mm)',title='Hanging posterior enters pipe progressively');axes[1,2].legend();axes[1,2].grid(alpha=.3)
    fig.savefig(case/'actual_evidence.png',dpi=160);plt.close(fig)
    print(json.dumps(report),flush=True)

if __name__=='__main__':review()
