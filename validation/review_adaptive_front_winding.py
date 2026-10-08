"""Review saved local curve action, observed handover and feedback; no dynamics."""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def review(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];c=z['centers_mm'].copy();force=z['cylindrical_contact_force_N'].copy()
        shift=z['shift_mm'].copy();span=z['actual_front_target_winding_span_rad'].copy();target_mask=z['target_grip_region_mask'].copy()
    tc=np.array([row['t_s'] for row in commands]);action=[row['adaptive_front_winding'] for row in commands]
    values=lambda key:np.array([row.get(key,np.nan) for row in action])
    front=(c[:,:7,1]-c[0,:7,1]).mean(1);rear=(c[:,14:,0]-c[0,14:,0]).mean(1)
    radius=np.linalg.norm(c[:,:13,[0,2]]-[1100.,1000.],axis=2)
    phi=np.unwrap(np.arctan2(-(c[:,:8,2]-1000.),c[:,:8,0]-1100.),axis=0);phi-=phi[0]
    local_reasons=Counter();shift_reasons=Counter()
    for row in action:
        for rejection in row['local_rejections']:local_reasons[rejection['reason'].split(':')[0]]+=1
        for rejection in row['common_shift_rejections']:shift_reasons[rejection['reason'].split(':')[0]]+=1
    report={k:result.get(k) for k in ['name','outcome','complete','physical_seconds','final_shift_mm',
        'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm',
        'actual_front_winding_span_initial_rad','actual_front_winding_span_final_rad','new_sustained_target_modules',
        'max_actual_joint_deg','max_actuator_torque_Nm','max_dispatch_step_deg','first_ctrl_error_rad','save_errors']}
    report.update(maximum_actual_front_mean_y_advance_mm=float(front.max()),maximum_actual_rear_mean_x_advance_mm=float(rear.max()),
        maximum_actual_front_winding_span_rad=float(span.max()),
        maximum_front_integral_tracking_error_deg=float(values('front_integral_tracking_error_deg').max()),
        maximum_rear_integral_tracking_error_deg=float(values('rear_integral_tracking_error_deg').max()),
        final_local_action=action[-1],local_rejection_counts=dict(local_reasons),common_shift_rejection_counts=dict(shift_reasons),
        final_actual_front_temporal_azimuth_change_rad=phi[-1].tolist(),
        final_actual_incoming_M8_M12_radius_mm=radius[-1,8:13].tolist(),
        maximum_actual_new_body_target_cylinder_force_N=force[:,1,8:].max(axis=0).tolist(),
        evidence='existing saved actual trace and commands; local phases, rear displacement and nominal shift alone do not confirm handover; no simulation steps')
    report['new_body_contact_intervals']=[]
    for module in range(8,21):
        raw=force[:,1,module]>.1;valid=raw&target_mask[:,module]
        for kind,active in [('cylinder',raw),('geometry_valid_cylinder',valid)]:
            if not active.any():continue
            edges=np.diff(np.r_[False,active,False].astype(int));start=np.flatnonzero(edges==1);end=np.flatnonzero(edges==-1)-1
            report['new_body_contact_intervals'].append(dict(module=module,kind=kind,
                intervals_s=[[float(t[a]),float(t[b])] for a,b in zip(start,end)]))
    report['checkpoints']=[]
    for at in [0,5,15,30,45,60,90,120,float(t[-1])]:
        if at>t[-1]+1e-6:continue
        i=int(np.argmin(abs(t-at)));j=int(np.argmin(abs(tc-at)))
        report['checkpoints'].append(dict(t_s=float(t[i]),actual_front_mean_y_advance_mm=float(front[i]),
            actual_rear_mean_x_advance_mm=float(rear[i]),actual_front_winding_span_rad=float(span[i]),
            incoming_M8_M12_radius_mm=radius[i,8:13].tolist(),
            incoming_M8_M12_y_mm=c[i,8:13,1].tolist(),
            actual_target_cylinder_modules=np.flatnonzero(force[i,1]>.1).tolist(),
            actual_valid_new_target_modules=np.flatnonzero((force[i,1]>.1)&target_mask[i]& (np.arange(21)>=8)).tolist(),
            local_action={k:action[j].get(k) for k in ['front_local_material_mm','rear_local_material_mm',
                'front_requested_speed_mm_s','front_accepted_speed_mm_s','rear_requested_speed_mm_s','rear_accepted_speed_mm_s',
                'front_integral_tracking_error_deg','rear_integral_tracking_error_deg','existing_supported_target_axis_advance_mm',
                'bounded_common_window_credit_mm','actual_target_cohort_material_estimate_mm','shape_update_fraction',
                'requested_common_shift_speed_mm_s','accepted_common_shift_speed_mm_s','pipe_material_roll_phases_rad','failure']}))
    pressure=[row['curve_pressure_feedback'] for row in commands]
    report['feedback']=dict(accepted_curve_pressure_update_frames=int(np.count_nonzero(values('shape_update_fraction'))),
        maximum_pressure_normal_curve_velocity_mm_s=float(np.max(abs(np.array([row['normal_curve_velocity_mm_s'] for row in pressure])))),
        maximum_loaded_tracking_normal_curve_velocity_mm_s=float(np.linalg.norm(np.array([row['loaded_tracking_normal_velocity_mm_s'] for row in pressure]),axis=-1).max()),
        pressure_station_goals_final_N=pressure[-1]['curve_station_load_goals_N'])
    if any(row.get('target_entry_capture_feedback') for row in action):
        entry=[row.get('target_entry_capture_feedback',{}) for row in action]
        report['feedback']['maximum_target_entry_field_mm']=max(row.get('maximum_entry_field_mm',0.) for row in entry)
        report['feedback']['maximum_target_entry_curve_velocity_mm_s']=max(row.get('maximum_entry_curve_velocity_mm_s',0.) for row in entry)
        report['feedback']['entry_field_note']='local inward request approaches captured bearing radius; overlapping fields do not impose a hard final-radius projection'
    video=case/'replay_1x.mp4'
    if video.exists():
        cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        checks=[]
        for at in sorted(set([0.,min(15.,(frames-1)/fps),min(30.,(frames-1)/fps),(frames-1)/fps])):
            frame=round(at*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=cap.read()
            checks.append(dict(t_s=at,frame=frame,readable=bool(ok)))
            if ok:cv2.imwrite(str(case/f'actual_video_t{at:.2f}.jpg'),pixels)
        cap.release();report['video_readback']=dict(fps=fps,frames=frames,playback_seconds=frames/fps,actual_last_frame_s=(frames-1)/fps,checks=checks)
    (case/'local_winding_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True,layout='constrained')
    axs[0,0].plot(t,front,label='Actual front mean +Y');axs[0,0].plot(t,rear,label='Actual rear mean +X');axs[0,0].set_ylabel('mm');axs[0,0].legend()
    axs[0,1].plot(t,span,label='Actual front wrap span');axs[0,1].axhline(span[0],color='grey',linestyle='--');axs[0,1].set_ylabel('rad');axs[0,1].legend()
    axs[1,0].plot(tc,values('front_local_material_mm'),label='Local front parameter');axs[1,0].plot(tc,values('rear_local_material_mm'),label='Local rear parameter');axs[1,0].plot(t,shift,label='Common window');axs[1,0].set_ylabel('Reference / mm');axs[1,0].legend()
    axs[1,1].plot(tc,values('front_integral_tracking_error_deg'),label='Front integral error');axs[1,1].plot(tc,values('rear_integral_tracking_error_deg'),label='Rear integral error');axs[1,1].set_ylabel('deg');axs[1,1].legend()
    for module in range(8,11):axs[2,0].plot(t,radius[:,module],label=f'M{module}')
    axs[2,0].set_ylabel('Actual incoming radius / mm');axs[2,0].legend()
    axs[2,1].plot(t,force[:,1,8:].sum(1),label='New body cylinder load');axs[2,1].plot(t,(force[:,1,8:]*target_mask[:,8:]).sum(1),label='Geometry-valid new-body load');axs[2,1].set_ylabel('N');axs[2,1].legend()
    for ax in axs.ravel():ax.set_xlabel('Actual elapsed time / s');ax.grid(alpha=.2)
    fig.suptitle(f'{case.name}: {result["outcome"]}; complete={result["complete"]}');fig.savefig(case/'local_winding_review.png',dpi=140);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['name','outcome','physical_seconds','final_shift_mm',
        'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','maximum_front_integral_tracking_error_deg',
        'new_sustained_target_modules','local_rejection_counts','common_shift_rejection_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',type=Path,nargs='+');args=p.parse_args()
    for case in args.cases:review(case)
