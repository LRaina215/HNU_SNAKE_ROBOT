"""Saved actual physics diagnostics/replay. Never advances the simulator."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import argparse,json
import numpy as np
import mujoco,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from front_local_rolling_geometry import BASE,write
from curve_guided_geometry import MODEL
from target_open_close_diagnostics import projection_panel


def main(name):
    case=BASE/name;z=np.load(case/'trace.npz');t=z['sim_time'];p=z['centers_mm'];f=z['contact_force_N']
    result=json.loads((case/'result.json').read_text(encoding='utf8'));controls=json.loads((case/'control.json').read_text(encoding='utf8'))
    rear=np.linalg.norm(p[:,13:]-p[0,13:],axis=2);counts=np.count_nonzero(f>.1,axis=2)
    rc=np.count_nonzero(f[:,0,13:]>.1,axis=1);fc=np.count_nonzero(f[:,1,:8]>.1,axis=1)
    change=np.degrees((z['pipe_relative_twist_rad'][:,:6]-z['pipe_relative_twist_rad'][0,:6]+np.pi)%(2*np.pi)-np.pi)
    ct=np.array([c['time_s'] for c in controls]);h=np.degrees(np.array([c['retained_shape_rad'] for c in controls]));delta=np.degrees(np.array([c['compliance_delta_rad'] for c in controls]))
    velocity=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
    axial=np.median(p[:,:7,1]-p[0,:7,1],axis=1)
    metrics=dict(physics_steps_added=0,actual_frames=len(t),actual_seconds=float(t[-1]),
        ground_frames=int(np.count_nonzero(z['ground'])),self_contact_frames=int(np.count_nonzero(z['self_contact'])),
        external_force_maximum=float(z['external_applied_force_maximum'].max()),generalized_force_maximum=float(z['generalized_applied_force_maximum'].max()),
        maximum_torque_Nm=float(abs(z['torque']).max()),rear_source_contacts_min=int(rc.min()),front_target_contacts_min=int(fc.min()),
        rear_retained_change_deg=(h[-1,12:]-h[0,12:]).tolist(),rear_compliance_delta_change_deg=(delta[-1,12:]-delta[0,12:]).tolist(),
        actual_front_median_axial_displacement_mm=float(np.median(result['actual_front_axial_displacement_mm'])),
        actual_front_median_director_change_deg=float(np.median(result['actual_front_pipe_relative_twist_change_deg'])),
        actual_front_median_axial_final_frame_mm=float(axial[-1]),actual_front_median_axial_peak_mm=float(axial.max()),
        actual_front_median_axial_maximum_return_mm=float((np.maximum.accumulate(axial)-axial).max()),
        source_force_initial_final_N=[float(f[0,0,13:].sum()),float(f[-1,0,13:].sum())],
        limitation='World/contacts/pipe-frame director are recorded diagnostics only; normal force is not proof of grip')
    write(case/'diagnostic_metrics.json',metrics)
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    axes[0,0].plot(t,rear);axes[0,0].axhline(6.45,color='r',ls='--');axes[0,0].set_ylabel('Rear M13..20 displacement (mm)')
    axes[0,1].plot(t,p[:,13:,0].mean(1)-p[0,13:,0].mean(),label='rear centroid X');axes[0,1].plot(t,axial,label='front median Y (scored)');axes[0,1].legend();axes[0,1].set_ylabel('Actual axial displacement (mm)')
    axes[1,0].plot(t,rc,label='rear/source');axes[1,0].plot(t,fc,label='front/target');axes[1,0].legend();axes[1,0].set_ylabel('Contact modules >0.1N')
    axes[1,1].plot(t,f[:,0,13:].sum(1),label='rear/source');axes[1,1].plot(t,f[:,1,:8].sum(1),label='front/target');axes[1,1].legend();axes[1,1].set_ylabel('Normal force (N)')
    axes[2,0].plot(t,change);axes[2,0].plot(t,np.degrees(z['front_phase_rad']),color='k',ls=':',label='control parameter');axes[2,0].legend();axes[2,0].set_ylabel('Actual director change / parameter (deg)')
    axes[2,1].plot(ct,h[:,12:]);axes[2,1].set_ylabel('Rear retained executable shape (deg)')
    axes[3,0].plot(t,np.degrees(abs(z['q']-z['ctrl'])).max(1),label='all');axes[3,0].plot(t,np.degrees(abs(z['q'][:,6:12]-z['ctrl'][:,6:12])).max(1),label='connector');axes[3,0].legend();axes[3,0].set_ylabel('Tracking error (deg)')
    axes[3,1].plot(t[1:],velocity.max(1));axes[3,1].set_ylabel('Maximum module speed (mm/s)')
    plan=json.loads((case/'plan.json').read_text(encoding='utf8'))
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.axvline(5,color='.5',ls=':')
        if result['schedule_completed']:ax.axvline(5+plan['rolling_s'],color='.5',ls=':')
    for ax in axes[-1]:ax.set_xlabel('Actual simulation time (s)')
    fig.suptitle(name+': actual physics; end hold '+('COMPLETED' if result['schedule_completed'] else 'NOT REACHED')+'; forward '+('PASS' if result['actual_local_forward_pass'] else 'FAIL'));fig.tight_layout();fig.savefig(case/'actual_diagnostics.png',dpi=140);plt.close(fig)
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=480,width=720);camera=mujoco.MjvCamera()
    camera.lookat[:]=[1.08,.13,.98];camera.distance=1.5;camera.azimuth=135;camera.elevation=-25
    writer=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1040,600))
    if not writer.isOpened():raise RuntimeError('Replay writer unavailable')
    indices=np.arange(0,len(t),5)
    if indices[-1]!=len(t)-1:raise ValueError('Expected stop at exact50ms replay boundary')
    for i in indices:
        # Forward evaluation solely reconstructs the saved frame; no mj_step.
        data.qpos[:]=z['qpos'][i];data.qvel[:]=z['qvel'][i];data.ctrl[:]=z['ctrl'][i];mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera);scene=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        stage=str(z['stage'][i]);frame=np.full((600,1040,3),20,np.uint8);frame[:480,:720]=scene;frame[:430,720:]=projection_panel(z['points_mm'][i],0.)
        lines=[f'{name} actual t={t[i]:.2f}s | {stage} | speed=1x',
            f'Control parameter={np.degrees(z["front_phase_rad"][i]):.2f}deg; material feed=0; rear active roll=0',
            f'Rear max displacement={rear[i].max():.3f}mm / limit6.450; contacts rear/front={rc[i]}/{fc[i]}',
            f'Actual front median director={np.median(change[i]):.2f}deg; axial Y={axial[i]:+.2f}mm',
            'Saved physics; dual-pipe support. End hold '+('completed' if result['schedule_completed'] else 'NOT REACHED')]
        for yy,line in zip([24,502,530,557,584],lines):
            cv2.putText(frame,line,(10,yy),cv2.FONT_HERSHEY_SIMPLEX,.47,(245,245,245),1)
        if i==indices[-1]:
            final=('LOCAL ROLL+HOLD: PASS | FORWARD: '+('PASS' if result['actual_local_forward_pass'] else 'FAIL')) if result['combination_pass'] else ('STOP: '+result['outcome'])
            cv2.putText(frame,final,(10,47),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,100,255),1)
        writer.write(frame)
        if i==0:cv2.imwrite(str(case/'initial_frame.png'),frame)
        if abs(t[i]-5)<1e-7:cv2.imwrite(str(case/'entry_frame.png'),frame)
        if i==indices[-1]:cv2.imwrite(str(case/'final_frame.png'),frame)
    writer.release();renderer.close()
    cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if n!=len(indices) or abs(fps-20)>1e-8:raise RuntimeError('Replay metadata mismatch')
    write(case/'replay_metadata.json',dict(fps=fps,frames=n,speed=1,physical_sample_interval_s=.05,
        actual_first_s=float(t[0]),actual_last_s=float(t[-1]),video_duration_s=n/fps,
        endpoint_frame_extra_s=.05,initial_and_stop_frames_included=True,physics_steps_added=0,
        end_hold_reached=result['schedule_completed'],source='trace.npz real qpos; no reference animation'))
    print(json.dumps(dict(case=name,replay_frames=n,actual_s=float(t[-1]),diagnostic=metrics)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',default='ROLL1_PRIMARY');main(p.parse_args().case)
