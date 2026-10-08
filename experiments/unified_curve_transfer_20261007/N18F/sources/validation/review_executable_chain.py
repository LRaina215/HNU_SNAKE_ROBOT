"""Read saved actual motion, exact-chain solver and 1x video; never simulate."""
from pathlib import Path
from collections import Counter
import sys,argparse,json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def review(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    initial=json.loads((case/'initial_controller_state.json').read_text(encoding='utf8'))
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];centers=z['centers_mm'].copy();force=z['cylindrical_contact_force_N'].copy()
        masks=z['target_grip_region_mask'].copy();span=z['actual_front_target_winding_span_rad'].copy()
        qpos=z['qpos'].copy();q=z['q'].copy();ctrl=z['ctrl'].copy();torque=z['torque'].copy()
        pipe=z['pipe_penetration_mm'].copy();selfdepth=z['nonadjacent_self_penetration_mm'].copy();ground=z['ground'].copy()
        trace_shapes={k:list(z[k].shape) for k in ['qpos','qvel','q','ctrl','torque']}
    tc=np.array([r['t_s'] for r in commands]);fit=[r['executable_chain'] for r in commands]
    actions=[r['fixed_spatial_shift'] for r in commands]
    values=lambda key:np.array([r[key] for r in fit])
    geometry=np.asarray([r['joint_geometry_reference_deg'] for r in commands])
    dispatched=np.degrees(np.asarray([r['target_rad'] for r in commands]))
    seed=np.asarray([r['integral_seed_deg'] for r in fit])
    radius=np.linalg.norm(centers[:,:,[0,2]]-[1100.,1000.],axis=2)
    front=(centers[:,:7,1]-centers[0,:7,1]).mean(1);rear=(centers[:,14:,0]-centers[0,14:,0]).mean(1)
    ordered=result['ordered_target_handover_modules']
    report=dict(result=result,trace_shapes=trace_shapes,command_count=len(commands),
        actual_front_wrap_initial_rad=float(span[0]),actual_front_wrap_final_rad=float(span[-1]),actual_front_wrap_max_rad=float(span.max()),
        actual_max_pipe_skin_penetration_mm=float(pipe.max()),actual_max_self_penetration_mm=float(selfdepth.max()),actual_ground_contact=bool(ground.any()),
        contact_intervals=[],checkpoints=[],solver=dict(
            success_fraction=float(np.mean(values('solver_success'))),feasible_candidate_fraction=float(np.mean(values('accepted_feasible_candidate'))),
            status_counts=dict(Counter(str(r['solver_status']) for r in fit)),
            iterations_mean=float(values('iterations').mean()),iterations_max=int(values('iterations').max()),
            total_wall_s=float(values('elapsed_wall_s').sum()),guide_fit_final_rms_mm=float(values('guide_fit_rms_mm')[-1]),
            guide_fit_max_rms_mm=float(values('guide_fit_rms_mm').max()),
            shell_constraint_min_margin_mm=float(values('minimum_shell_constraint_margin_mm').min()),
            predicted_pipe_skin_penetration_max_mm=float(np.max([r['predicted_pipe_skin_penetration_mm'] for r in fit])),
            exact_rod_length_max_error_mm=float(values('maximum_exact_FK_rod_length_error_mm').max()),
            guide_chord_max_error_mm=float(values('maximum_guide_chord_error_mm').max()),
            integral_fit_difference_max_deg=float(np.abs(geometry-seed).max()),
            maximum_reference_joint_change_from_start_deg=float(np.abs(geometry-geometry[0]).max()),
            maximum_full_candidate_joint_step_deg=float(np.abs(np.diff(geometry,axis=0)).max()) if len(geometry)>1 else 0.,
            no_feasible_candidate_count=int(np.count_nonzero(~values('accepted_feasible_candidate').astype(bool))),
            no_feasible_candidate_committed_shift_mm=float(sum(row['shift_speed_mm_s']*.05 for row in commands if not row['executable_chain']['accepted_feasible_candidate'])),
            rate_bound_in_solver_deg=fit[-1].get('joint_candidate_step_bound_deg'),
            anchor_counts=dict(Counter(str((r['anchor_pipe_side'],r['anchor_module'])) for r in fit)),
            anchor_changes=[dict(t_s=float(tc[i]),pipe_side=r['anchor_pipe_side'],module=r['anchor_module']) for i,r in enumerate(fit) if i==0 or r['anchor_changed']],
            integral_prior_used=False,free_root_variables=0,physical_root_assignment=False),
        dispatch=dict(maximum_candidate_minus_dispatched_target_deg=float(np.abs(geometry-dispatched).max()),
            final_candidate_minus_dispatched_target_deg=(geometry[-1]-dispatched[-1]).tolist(),
            maximum_saved_ctrl_deg=float(np.degrees(np.abs(ctrl)).max()),
            maximum_dispatch_step_deg=float(np.abs(np.diff(dispatched,axis=0)).max()) if len(dispatched)>1 else 0.,
            first_saved_ctrl_error_rad=float(np.abs(np.radians(dispatched[0])-ctrl[0]).max()),
            control_chain='feasible whole FK candidate -> torque/kp compensation -> actual-q +/-3deg anti-windup -> +/-88deg and 0.5deg/tick target -> original dynamic actuator'),
        caveats=['registered future FK is a prediction, not actual ctrl or actual body',
            'ever-admitted ordered list does not prove continued target bearing',
            'common shift and fixed group displacement alone are not progress',
            'initial CLOSE material gauge is unchanged; fit uses world pose/contact simulation truth'])
    for module in range(8,21):
        active=(force[:,1,module]>.1)&masks[:,module]
        if not active.any():continue
        edges=np.diff(np.r_[False,active,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        report['contact_intervals'].append(dict(module=module,intervals_s=[[float(t[a]),float(t[b])] for a,b in zip(starts,ends)],
            longest_s=float(max(t[b]-t[a]+.01 for a,b in zip(starts,ends))),maximum_force_N=float(force[:,1,module].max()),
            final_force_N=float(force[-1,1,module]),final_axis_mm=float(centers[-1,module,1]),final_radius_mm=float(radius[-1,module]),final_in_target_mask=bool(masks[-1,module])))
    sign=np.sign(initial['curve']['planned_route_spec']['target_right_hand_dphi_daxis'])
    for at in sorted(set([0.,15.,30.,45.,60.,90.,120.,150.,180.,210.,240.,300.,float(t[-1])])):
        if at>t[-1]+1e-6:continue
        i=int(np.argmin(abs(t-at)));j=int(np.argmin(abs(tc-at)));row=commands[j]
        guide=np.asarray(row['curve_guide_points_mm']);axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        actual_rotation=Rotation.from_quat(qpos[i,3:7][[1,2,3,0]]).as_matrix()
        last=max(row['transferred_material_modules'],default=7)
        phi=np.unwrap(np.arctan2(-(centers[i,:last+1,2]-1000),centers[i,:last+1,0]-1100))
        dy=np.diff(centers[i,:last+1,1]);da=np.diff(phi);meaningful=np.abs(dy)>2.
        report['checkpoints'].append(dict(t_s=float(t[i]),ordered=row['transferred_material_modules'],
            shift_mm=row['shift_mm'],front_wrap_span_rad=float(span[i]),front_mean_y_mm=float(front[i]),rear_mean_x_mm=float(rear[i]),
            actual_target_geometry_bearings=np.flatnonzero((force[i,1]>.1)&masks[i]).tolist(),
            M8_M12_axis_mm=centers[i,8:13,1].tolist(),M8_M12_radius_mm=radius[i,8:13].tolist(),M8_M12_force_N=force[i,1,8:13].tolist(),
            actual_head_vs_guide_tangent_deg=float(np.degrees(np.arccos(np.clip(actual_rotation[:,2]@axis,-1,1)))),
            ever_entered_prefix_handed_fraction=float(np.mean(np.sign(da[meaningful]*dy[meaningful])==sign)) if meaningful.any() else None,
            ever_entered_prefix_axis_reversals=int(np.count_nonzero(dy>3.)),
            chain_fit=fit[j],action=actions[j]))
    cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
    if fps>0:
        for at in sorted(set([0.,min(30.,(frames-1)/fps),min(90.,(frames-1)/fps),(frames-1)/fps])):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(at*fps));ok,pixels=cap.read();checks.append(dict(t_s=at,readable=bool(ok)))
            if ok:cv2.imwrite(str(case/f'actual_video_t{at:.2f}.jpg'),pixels)
    cap.release();report['video_readback']=dict(frames=frames,fps=fps,last_actual_frame_s=(frames-1)/fps if fps else None,checks=checks,size_bytes=(case/'replay_1x.mp4').stat().st_size)
    report['artifact_manifest']={name:dict(exists=(case/name).exists(),size_bytes=(case/name).stat().st_size if (case/name).exists() else None)
        for name in ['final_state.npz','trace.npz','commands.json','controller_state.json','initial_controller_state.json','result.json','replay_1x.mp4','video.json','plan.json']}
    (case/'executable_chain_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(4,2,figsize=(13,13),sharex=True,layout='constrained')
    axs[0,0].plot(t,span,label='Actual front wrap');axs[0,0].axhline(span[0]*.65,color='gray',linestyle='--');axs[0,0].set_ylabel('rad')
    axs[0,1].plot(t,front,label='Actual front +Y');axs[0,1].plot(t,rear,label='Actual rear +X');axs[0,1].set_ylabel('mm')
    axs[1,0].plot(tc,[a['guide_global_rms_error_mm'] for a in actions],label='Actual-guide RMS');axs[1,0].plot(tc,values('guide_fit_rms_mm'),label='Predicted fit-guide RMS');axs[1,0].set_ylabel('mm')
    axs[1,1].plot(tc,[a['chain_geometry_tracking_error_deg'] for a in actions],label='Real joint-fit tracking error');axs[1,1].plot(tc,values('maximum_fit_minus_integral_seed_deg'),label='Fit minus unused integral seed');axs[1,1].set_ylabel('deg')
    axs[2,0].plot(t,pipe,label='Actual shell penetration');axs[2,0].plot(tc,np.max([r['predicted_pipe_skin_penetration_mm'] for r in fit],axis=1),label='Future FK shell penetration');axs[2,0].set_ylabel('mm')
    axs[2,1].plot(tc,[r['anchor_module'] for r in fit],label='Measured bearing anchor');axs[2,1].plot(tc,values('iterations'),label='Solver iterations')
    for module in range(8,12):
        axs[3,0].plot(t,centers[:,module,1],label=f'M{module} axis');axs[3,1].plot(t,force[:,1,module]*masks[:,module],label=f'M{module} valid load')
    axs[3,0].axhline(75.,color='gray',linestyle='--');axs[3,0].set_ylabel('mm');axs[3,1].set_ylabel('N')
    for ax in axs.flat:ax.legend();ax.grid(alpha=.2);ax.set_xlabel('Actual elapsed time / s')
    fig.savefig(case/'executable_chain_review.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(name=result['name'],outcome=result['outcome'],actual_s=result['physical_seconds'],ordered=ordered,span_final=float(span[-1]),solver=report['solver'],video=report['video_readback'])))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);review(parser.parse_args().case)
