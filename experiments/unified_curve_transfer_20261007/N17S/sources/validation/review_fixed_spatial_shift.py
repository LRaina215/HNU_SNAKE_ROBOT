"""Read saved fixed-route motion, material phase and ordered contact evidence."""
from pathlib import Path
from collections import Counter
import argparse,json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def review(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    initial=json.loads((case/'initial_controller_state.json').read_text(encoding='utf8'))
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];c=z['centers_mm'].copy();forces=z['cylindrical_contact_force_N'].copy()
        mask=z['target_grip_region_mask'].copy();span=z['actual_front_target_winding_span_rad'].copy()
        shift=z['shift_mm'].copy();points=z['points_mm'].copy();qpos=z['qpos'].copy()
        torque=z['torque'].copy()
    tc=np.array([r['t_s'] for r in commands]);actions=[r['fixed_spatial_shift'] for r in commands]
    values=lambda key:np.array([r.get(key,np.nan) for r in actions])
    front=(c[:,:7,1]-c[0,:7,1]).mean(1);rear=(c[:,14:,0]-c[0,14:,0]).mean(1)
    radius=np.linalg.norm(c[:,:13,[0,2]]-[1100.,1000.],axis=2)
    reasons={key:dict(Counter(r['reason'].split(':')[0] for a in actions for r in a[key]))
        for key in ['shape_rejections','window_rejections']}
    orientation=np.asarray(initial['controller']['root_orientation'])
    transport_orientation=np.asarray(initial['controller'].get('wrap_transport_orientation',orientation))
    material_head=[]
    for row in commands:
        guide=np.array(row['curve_guide_points_mm']);axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        if row.get('axis_wrap'):
            cross=np.cross(transport_orientation[:,2],axis);sine=np.linalg.norm(cross)
            transport_orientation=Rotation.from_rotvec(cross*np.arctan2(sine,float(transport_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()@transport_orientation
            correction=np.radians(row['axis_wrap']['parameters']['material_phase_deg'])
            orientation=Rotation.from_rotvec(axis*correction).as_matrix()@transport_orientation
        else:
            cross=np.cross(orientation[:,2],axis);sine=np.linalg.norm(cross)
            orientation=Rotation.from_rotvec(cross*np.arctan2(sine,float(orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()@orientation
            correction=np.radians(row['fixed_spatial_shift'].get('head_roll_phase_feedback_step_deg',0.))
            orientation=Rotation.from_rotvec(axis*correction).as_matrix()@orientation
        material_head.append(orientation.copy())
    report={k:result.get(k) for k in ['name','outcome','complete','physical_seconds','final_shift_mm',
        'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','ordered_target_handover_modules',
        'admissions','max_actual_joint_deg','max_actuator_torque_Nm','max_dispatch_step_deg','first_ctrl_error_rad','save_errors']}
    report.update(initial_curve_fit_error_mm=actions[0]['initial_curve_fit_error_mm'],
        maximum_spatial_route_phase_rad=float(np.max(np.abs([a['spatial_route_phase_rad'] for a in actions]))),
        maximum_extra_material_roll_phase_rad=float(np.max(np.abs([a['material_axial_roll_phase_rad'] for a in actions]))),
        measured_material_phase_feedback_enabled=actions[-1].get('continuous_observed_material_phase_feedback',False),
        maximum_observed_head_roll_phase_rad=float(np.max(np.abs([a.get('observed_head_roll_phase_rad',0.) for a in actions]))),
        actual_front_wrap_initial_rad=float(span[0]),actual_front_wrap_final_rad=float(span[-1]),
        actual_front_wrap_maximum_rad=float(span.max()),actual_front_mean_advance_maximum_mm=float(front.max()),
        maximum_full_integral_tracking_error_deg=float(values('full_integral_tracking_error_deg').max()),
        maximum_global_guide_rms_error_mm=float(values('guide_global_rms_error_mm').max()),
        rejection_counts=reasons,final_action=actions[-1],contact_intervals=[],checkpoints=[],
        evidence='saved actual dynamics and commands only; ordered bearing records are local progress, not whole single-helix transfer')
    if actions[-1].get('continuous_observed_material_phase_feedback',False):
        valid=np.array([a['actual_head_normal_projection_length']>.5 for a in actions])
        phase_error=np.array([a.get('actual_head_roll_error_before_feedback_deg',0.) for a in actions],dtype=float)
        steps=np.array([a['head_roll_phase_feedback_step_deg'] for a in actions])
        report['material_phase_feedback']=dict(
            final_cumulative_observed_phase_deg=float(np.degrees(actions[-1]['observed_head_roll_phase_rad'])),
            maximum_valid_absolute_phase_error_deg=float(np.abs(phase_error[valid]).max()),
            observation_invalid_fraction=float(np.mean(~valid)),
            rate_at_5_deg_s_limit_fraction=float(np.mean(np.abs(steps)>=.25-1e-8)),
            final_observation_valid=bool(valid[-1]),
            final_actual_head_tangent_error_deg=actions[-1]['actual_head_tangent_error_deg'],
            placeholder_warning='N13 raw error is zero when projected normal <=0.5 and measurement disabled; this is not phase convergence')
        fig,paxs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
        paxs[0].plot(tc,np.degrees([a['observed_head_roll_phase_rad'] for a in actions]),label='Accumulated observed-frame correction')
        paxs[0].set_ylabel('deg');paxs[0].legend()
        paxs[1].plot(tc,np.where(valid,phase_error,np.nan),label='Measured projected axial error (valid only)')
        paxs[1].plot(tc,[a['actual_head_tangent_error_deg'] for a in actions],label='Actual head tangent error')
        paxs[1].set_ylabel('deg');paxs[1].legend();paxs[1].set_xlabel('Actual elapsed time / s')
        fig.savefig(case/'material_phase_feedback_review.png',dpi=140);plt.close(fig)
    if commands[-1].get('axis_wrap'):
        feedback=[r['axis_wrap'] for r in commands]
        before=np.array([r['physical_FK_goal_residual_before_rms_mm'] for r in feedback])
        after=np.array([r['physical_FK_goal_residual_after_rms_mm'] for r in feedback])
        parameters=feedback[-1]['parameters']
        report['axis_wrap_feedback']=dict(final=feedback[-1],
            parameter_absolute_maxima={k:float(max(abs(r['parameters'][k]) for r in feedback)) for k in parameters},
            rejected_candidates_by_reason=dict(Counter(r['reason'].split(':')[0] for a in feedback for r in a['rejections'])),
            accepted_positive_step_fraction=float(np.mean([r['accepted_fraction']>0 for r in feedback])),
            modeled_residual_improved_fraction=float(np.mean(after<before-1e-9)),
            modeled_residual_worsened_fraction=float(np.mean(after>before+1e-9)),
            prediction_anchor_module_counts=dict(Counter(str(r.get('physical_FK_prediction_registration',{}).get('anchor_module')) for r in feedback)),
            prediction_anchor_changes=[dict(t_s=float(commands[i]['t_s']),module=r.get('physical_FK_prediction_registration',{}).get('anchor_module'))
                for i,r in enumerate(feedback) if i==0 or r.get('physical_FK_prediction_registration',{}).get('anchor_module')!=feedback[i-1].get('physical_FK_prediction_registration',{}).get('anchor_module')],
            first_recovery_t_s=next((float(r['t_s']) for r in commands if r['axis_wrap']['recovery_active']),None),
            actual_final_front_median_radius_mm=float(np.median(radius[-1,:8])),
            warning='FK residual improvement is a predicted action effect, not proof of actual front recovery; actual qpos, wrap and video establish motion')
        fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
        for key in parameters:axs[0].plot(tc,[r['parameters'][key] for r in feedback],label=key)
        axs[0].legend();axs[0].set_ylabel('parameter / deg or mm')
        axs[1].plot(tc,before,label='Physical FK goal residual before');axs[1].plot(tc,after,label='after')
        axs[1].set_ylabel('mm');axs[1].legend()
        axs[2].plot(t,span,label='Actual front span');axs[2].plot(tc,[r['actual_front_span_rad'] for r in feedback],label='Pre-dispatch observed span')
        axs[2].axhline(span[0]*.65,color='gray',linestyle='--');axs[2].set_ylabel('rad');axs[2].legend()
        for ax in axs:ax.grid(alpha=.2);ax.set_xlabel('Actual elapsed time / s')
        fig.savefig(case/'axis_wrap_feedback_review.png',dpi=140);plt.close(fig)
    for module in range(8,21):
        active=(forces[:,1,module]>.1)&mask[:,module]
        if not active.any():continue
        edges=np.diff(np.r_[False,active,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        report['contact_intervals'].append(dict(module=module,maximum_actual_force_N=float(forces[:,1,module].max()),
            intervals_s=[[float(t[a]),float(t[b])] for a,b in zip(starts,ends)],
            longest_geometry_valid_contact_s=float(max(t[b]-t[a]+.01 for a,b in zip(starts,ends)))))
    for at in [0,15,30,45,60,75,90,120,150,180,240,float(t[-1])]:
        if at>t[-1]+1e-6:continue
        i=int(np.argmin(abs(t-at)));j=int(np.argmin(abs(tc-at)))
        actual_rotation=Rotation.from_quat(qpos[i,3:7][[1,2,3,0]]).as_matrix()
        reference_rotation=material_head[j]
        axis_angle=float(np.degrees(np.arccos(np.clip(actual_rotation[:,2]@reference_rotation[:,2],-1,1))))
        frame_angle=float(np.degrees(Rotation.from_matrix(reference_rotation.T@actual_rotation).magnitude()))
        normal=actual_rotation[:,0]-reference_rotation[:,2]*(actual_rotation[:,0]@reference_rotation[:,2])
        normal/=max(np.linalg.norm(normal),1e-12)
        axial_phase=float(np.degrees(np.arctan2(reference_rotation[:,2]@np.cross(reference_rotation[:,0],normal),reference_rotation[:,0]@normal)))
        last=max(commands[j]['transferred_material_modules'],default=7)
        phi=np.unwrap(np.arctan2(-(c[i,:last+1,2]-1000),c[i,:last+1,0]-1100))
        dy=np.diff(c[i,:last+1,1]);da=np.diff(phi);meaningful=np.abs(dy)>2.
        sign=np.sign(initial['curve']['planned_route_spec']['target_right_hand_dphi_daxis'])
        report['checkpoints'].append(dict(t_s=float(t[i]),front_mean_y_mm=float(front[i]),rear_mean_x_mm=float(rear[i]),
            front_wrap_span_rad=float(span[i]),ordered=commands[j]['transferred_material_modules'],
            M8_M12_radius_mm=radius[i,8:13].tolist(),M8_M12_axis_mm=c[i,8:13,1].tolist(),
            M8_M12_force_N=forces[i,1,8:13].tolist(),
            head_actual_vs_transported_reference_axis_deg=axis_angle,
            head_actual_vs_transported_reference_full_frame_deg=frame_angle,
            head_actual_vs_transported_reference_projected_axial_phase_deg=axial_phase,
            actual_entered_body_target_handed_sign_fraction=float(np.mean(np.sign(da[meaningful]*dy[meaningful])==sign)) if meaningful.any() else None,
            actual_entered_body_forward_order_violations=int(np.count_nonzero(dy>3.)),
            maximum_actual_motor_abs_torque_Nm=float(abs(torque[i]).max()),action=actions[j]))
    cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    checks=[]
    if fps>0:
        for at in sorted(set([0.,min(30.,(frames-1)/fps),min(90.,(frames-1)/fps),(frames-1)/fps])):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(at*fps));ok,pixels=cap.read();checks.append(dict(t_s=at,readable=bool(ok)))
            if ok:cv2.imwrite(str(case/f'actual_video_t{at:.2f}.jpg'),pixels)
    cap.release();report['video_readback']=dict(frames=frames,fps=fps,last_actual_frame_s=(frames-1)/fps if fps else None,checks=checks)
    (case/'fixed_spatial_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True,layout='constrained')
    axs[0,0].plot(t,front,label='Actual front mean +Y');axs[0,0].plot(t,rear,label='Actual rear mean +X');axs[0,0].legend();axs[0,0].set_ylabel('mm')
    axs[0,1].plot(t,span,label='Actual front wrap');axs[0,1].axhline(span[0],color='gray',linestyle='--');axs[0,1].legend();axs[0,1].set_ylabel('rad')
    axs[1,0].plot(t,shift,label='Reference window');axs[1,0].plot(tc,values('guide_global_rms_error_mm'),label='Actual-guide RMS');axs[1,0].legend();axs[1,0].set_ylabel('mm')
    axs[1,1].plot(tc,values('full_integral_tracking_error_deg'),label='Integral joint error');axs[1,1].legend();axs[1,1].set_ylabel('deg')
    for module in range(8,12):axs[2,0].plot(t,radius[:,module],label=f'M{module}')
    axs[2,0].set_ylabel('Actual radius / mm');axs[2,0].legend()
    for module in range(8,12):axs[2,1].plot(t,forces[:,1,module]*mask[:,module],label=f'M{module}')
    axs[2,1].set_ylabel('Actual valid cylinder load / N');axs[2,1].legend()
    for ax in axs.ravel():ax.set_xlabel('Actual elapsed time / s');ax.grid(alpha=.2)
    fig.savefig(case/'fixed_spatial_review.png',dpi=140);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['name','outcome','physical_seconds','ordered_target_handover_modules','rejection_counts','maximum_full_integral_tracking_error_deg']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',type=Path);args=p.parse_args()
    for case in args.cases:review(case)
