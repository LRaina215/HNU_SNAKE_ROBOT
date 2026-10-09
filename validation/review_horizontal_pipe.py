"""Saved-record audit only: no dynamics, no replay credit from controller state."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/unified_curve_transfer_20261007/HORIZONTAL_PIPE_20261009'

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def review():
    sets={}
    for name in ['J01','J02']:
        with np.load(BASE/name/'trace.npz') as z:sets[name]={k:z[k] for k in z.files}
    z=sets['J02'];c=z['centers_mm'][-1];r=np.linalg.norm(c[:,[0,2]]-[1100,1000],axis=1)
    phi=np.unwrap(np.arctan2(c[:,2]-1000,c[:,0]-1100));dy=np.diff(c[:,1]);da=np.diff(phi)
    fit=np.polyfit(phi[:20],c[:20,1],1);residual=c[:,1]-np.polyval(fit,phi)
    compatible=(r>65)&(r<125)&(c[:,1]>150)
    pitch=-fit[0]*2*np.pi
    report=dict(evidence='actual saved100Hz qpos-derived centres and contact forces; no mj_step',
        actual_seconds=float(z['continuation_t_s'][-1]),near_pipe_modules=np.flatnonzero(compatible).tolist(),
        near_pipe_new_modules=np.flatnonzero(compatible[9:]).__add__(9).tolist(),
        automatic_current_retained=np.flatnonzero(z['actual_new_retained_mask'][-1]).tolist(),
        coherent_prefix_modules=list(range(20)),coherent_prefix_pitch_mm=float(pitch),
        coherent_prefix_axis_residual_rms_mm=float(np.sqrt(np.mean(residual[:20]**2))),
        centre_radial_range_mm=[float(r.min()),float(r.max())],all_centres_near_pipe=bool(compatible.all()),
        total_centre_angular_span_rad=float(np.ptp(phi)),total_centre_turns=float(np.ptp(phi)/(2*np.pi)),
        tail_pitch_reversal_mm=float(dy[-1]),tail_axis_fit_residual_mm=float(residual[-1]),
        final_joint20_deg=float(np.degrees(z['q'][-1,-1])),final_joint20_kp=float(z['kp_nm_per_rad'][-1,-1]),
        final_joint20_active_torque_nm=float(z['torque'][-1,-1]),
        own_bearing_modules=np.flatnonzero(z['contact_force_N'][-1,1]>.1).tolist(),
        new_own_bearing_modules=(np.flatnonzero(z['contact_force_N'][-1,1,9:]>.1)+9).tolist(),
        automatic_outcome='no_new_retained_intake_90s',whole_uniform_helix_complete=False,cross_pipe_complete=False,
        reception_counter_defect='signed time winding from each initial hanging rod incorrectly rejects M16..20 after they physically reach the pipe; this also prevents stiffness restoration',
        failure_location='tail joint20 remains passive at about-90deg; final tail rod advances axially backwards11.49mm relative to preceding rod',
        J01_instruction_defect='Bishop initial normal is reselected with argmin(abs(head tangent)) each frame while accumulated phase is not gauge compensated; direct-angle and repeated world-frame selection can alter shape',
        J02_separate_instruction='true20-hinge projection replaces direct Bishop angles, fixed initial head frame plus shared roll gauge and limited head tilt reconstruct the actual hinge geometry')
    write(BASE/'J02/actual_body_shape_review.json',report)
    summary={}
    for name,data in sets.items():
        rows=[]
        for j in range(20):
            t=data['continuation_t_s'];kp=data['kp_nm_per_rad'][:,j];force=data['torque'][:,j];role=data['region_codes'][:,j]
            rows.append(dict(joint=j+1,initial_kp=float(kp[0]),final_kp=float(kp[-1]),kp_range=[float(kp.min()),float(kp.max())],
                maximum_drive_torque_nm=float(abs(force).max()),final_drive_torque_nm=float(force[-1]),
                first_full_stiffness_after_2s=float(t[np.flatnonzero((kp>126)&(t>2))[0]]) if np.any((kp>126)&(t>2)) else None,
                first_passive_s=float(t[np.flatnonzero(kp<.5)[0]]) if np.any(kp<.5) else None,
                maximum_drive_when_passive_nm=float(abs(force[kp<.5]).max()) if np.any(kp<.5) else None,
                damping_nm_s_per_rad=float(data['damping_nm_s_per_rad'][-1,j])))
        summary[name]=rows
    write(BASE/'joint_region_dynamics_summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for name in ['J01','J02']:
        data=sets[name];axes[0,0].plot(data['continuation_t_s'][::10],data['actual_front_span_rad'][::10],label=name)
    axes[0,0].set(xlabel='Actual time (s)',ylabel='Actual front span (rad)',title='Front carrying wrap');axes[0,0].legend();axes[0,0].grid(alpha=.3)
    t=z['continuation_t_s'];stride=10
    a=axes[0,1].imshow(z['kp_nm_per_rad'][::stride].T,origin='lower',aspect='auto',extent=[0,t[-1],.5,20.5],vmin=0,vmax=127,cmap='viridis')
    axes[0,1].set(xlabel='Actual time (s)',ylabel='Joint number',title='Actual position stiffness (Nm/rad)');fig.colorbar(a,ax=axes[0,1])
    a=axes[0,2].imshow(z['torque'][::stride].T,origin='lower',aspect='auto',extent=[0,t[-1],.5,20.5],vmin=-2.355,vmax=2.355,cmap='coolwarm')
    axes[0,2].set(xlabel='Actual time (s)',ylabel='Joint number',title='Actual active torque (Nm)');fig.colorbar(a,ax=axes[0,2])
    for seconds in [0,70,150,277.65]:
        i=int(np.argmin(abs(t-seconds)));axes[1,0].plot(np.arange(21),z['centers_mm'][i,:,2],'-o',ms=3,label=f'{t[i]:.1f}s')
    axes[1,0].axhline(1000,color='k',ls='--',lw=1);axes[1,0].set(xlabel='Module number',ylabel='Actual centre height (mm)',title='Posterior lifted to the pipe');axes[1,0].legend();axes[1,0].grid(alpha=.3)
    axes[1,1].plot(phi[:20],c[:20,1],'o-',label='M0..19');axes[1,1].plot(phi,np.polyval(fit,phi),'--',label=f'Prefix pitch {pitch:.1f}mm')
    axes[1,1].scatter(phi[-1],c[-1,1],c='red',label='M20 tail pitch reversal');axes[1,1].set(xlabel='Unwrapped circumferential angle (rad)',ylabel='Actual axis Y (mm)',title='Final body winding and tail defect');axes[1,1].legend();axes[1,1].grid(alpha=.3)
    axes[1,2].plot(np.arange(21),r,'o-',label='Actual centre radius');axes[1,2].axhline(56.5,color='k',label='Pipe radius');axes[1,2].axhline(85.5,color='grey',ls='--',label='Pipe + shell nominal radius')
    axes[1,2].set(xlabel='Module number',ylabel='Distance to pipe axis (mm)',title='All21 rods near pipe; geometry is not force credit');axes[1,2].legend();axes[1,2].grid(alpha=.3)
    fig.savefig(BASE/'actual_evidence.png',dpi=160);plt.close(fig)
    print(json.dumps(report),flush=True)

if __name__=='__main__':review()
