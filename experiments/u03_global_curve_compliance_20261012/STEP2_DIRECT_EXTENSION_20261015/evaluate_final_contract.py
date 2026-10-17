"""Offline action evidence under STEP2_FINAL_20261016, no control truth input."""
from pathlib import Path
import argparse,json,sys,numpy as np,cv2
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'snake_control').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.cmu_window_gait import cylinder

def intervals(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))

def evaluate(C):
    C=Path(C).resolve();r=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));result=json.loads((C/'result.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');t=z['t'];centers=z['centers'];points=z['points'];W=r['robot_weight_N'];normal=z['eval_pipe_normal_N'];world=z['eval_pipe_world_N'];source_normal=normal[:,:2].sum((1,2));source_Fz=world[:,:2,:,2].sum((1,2));receive_Fz=world[:,2:,:,2].sum((1,2));straight_normal=normal[:,2].sum(1)
    rho=np.linalg.norm(centers[:,:,[0,2]]-[1.1,1.],axis=2);in_straight=(rho>.07)&(rho<.12)&(centers[:,:,1]>.15)&(centers[:,:,1]<1.2);endpoints_inside=(points[:,:,1]>.15).all(1)&(points[:,:,1]<1.2).all(1);polar=np.unwrap(np.arctan2(centers[:,:,2]-1,centers[:,:,0]-1.1),axis=1);connected_wrap=np.ptp(polar,axis=1)>=2*np.pi;full=in_straight.all(1)&endpoints_inside&(source_normal<.1)&(straight_normal>.2)&connected_wrap
    chosen=None;spin=np.zeros(21);advance=np.zeros(21);candidate_windows=[]
    # The task asks for a complete continuous action, not residence until an
    # arbitrary budget expires. Evaluate the earliest complete sustained window.
    for start,end in intervals(full):
        candidate_windows.append(dict(start_s=float(t[start]),end_s=float(t[end]),duration_s=float(t[end]-t[start])))
        for finish in range(start,end+1):
            if t[finish]-t[start]<r['acceptance']['post_transfer_s']:continue
            displacement=centers[finish,:,1]-centers[start,:,1];rotation=np.degrees(np.trapz(z['actual_axial_spin_rad_s'][start:finish+1],x=t[start:finish+1],axis=0));same=(rotation>0).all() or (rotation<0).all()
            if (displacement>r['acceptance']['minimum_each_axial_advance_m']).all() and (abs(rotation)>=r['acceptance']['minimum_each_same_direction_axial_spin_deg']).all() and same:chosen=(start,finish);advance=displacement;spin=rotation;break
        if chosen is not None:break
    completion_index=chosen[1] if chosen else len(t)-1;prefix=slice(0,completion_index+1);contact=result.get('first_raw_front_contact_t');capture=result.get('first_capture_t');fit=result.get('first_reliable_t');write=result.get('first_G_write_t');release=result.get('first_release_t');before=np.ones(len(t),bool) if capture is None else t<capture-1e-8;ci=None if capture is None else int(np.argmin(abs(t-capture)));first_full=int(np.flatnonzero(full)[0]) if full.any() else None
    initial_rho=np.linalg.norm(centers[0,:,1:]-[0,1],axis=1);initial=bool(t[0]==0 and ((initial_rho>.065)&(initial_rho<.13)).all() and (normal[0,2:].sum()==0))
    no_prefeed=bool((z['S'][before]==0).all() and (z['source_commanded_feed_m'][before]==0).all() and (z['source_commanded_roll_rad'][before]==0).all());motion=bool(ci is not None and z['bearing_motion_evidence'][ci]);capture_load=bool(ci is not None and receive_Fz[ci]>=.39*W and normal[ci,2:].sum()>.2 and motion)
    capture_snapshot_file=C/'CAPTURE_SNAPSHOT.json';capture_snapshot_used=False
    if capture is not None and capture_snapshot_file.exists():
        exact_capture=json.loads(capture_snapshot_file.read_text());assert abs(exact_capture['t']-capture)<1e-8
        capture_force=np.asarray(exact_capture['eval_pipe_world_N']);capture_normal=np.asarray(exact_capture['eval_pipe_normal_N'])
        motion=bool(exact_capture['bearing_motion_evidence']);capture_load=bool(capture_force[2:,:,2].sum()>=.39*W and capture_normal[2:].sum()>.2 and motion);capture_snapshot_used=True
    overlap=(source_Fz>=.2*W)&(receive_Fz>=.2*W);overlap_windows=[dict(start_s=float(t[a]),end_s=float(t[b])) for a,b in intervals(overlap) if t[b]-t[a]>=.5]
    axis_error=None
    try:axis_error=float(np.degrees(np.arccos(np.clip(abs(cylinder(points[completion_index])['axis']@np.array([0,1.,0])),0,1))))
    except (ValueError,np.linalg.LinAlgError):pass
    native=z['curvature_field'];bounds=z['curvature_bounds_mm'][0];nodes=z['material_arc_nodes_mm'][0];A=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]));b=np.einsum('jc,ncd->njd',A,native);recovered=-np.where((np.arange(20)%2)[None,:],b[:,:,0],b[:,:,1]);integral_error=float(abs(recovered-z['raw_native_command']).max());config=next(c for c in r['cases'] if c['case']==C.name)
    ordered=bool(release is not None and contact is not None and capture is not None and first_full is not None and release<=contact<capture<=t[first_full]);physical_gates=dict(full_source_t0=initial,source_stabilized_before_release=release is not None,no_commanded_source_feed_or_roll_before_bearing=no_prefeed,actual_loaded_motion_before_feed=capture_load,contact_bearing_transfer_order=ordered,overlapping_actual_bearing=bool(overlap_windows),all21_finite_straight_connected_wrap=chosen is not None,sustained_true_rolling_and_transport=chosen is not None,axis_under5deg=axis_error is not None and axis_error<5,actual_joints_within90p5deg=float(abs(z['qpos'][prefix,7:]).max()*180/np.pi)<=90.5,torque_within2p355Nm=float(abs(z['torque'][prefix]).max())<=2.355+1e-6,no_self_over3mm=float(z['depth'][prefix,2].max())<=.003,no_floor=float(z['depth'][prefix,3].max())==0.,exact_native_integrals=integral_error<1e-10)
    corrected_rolling=r.get('requirements')=='STEP2_ROLLING_SCOPE_CLARIFIED_20261016'
    if corrected_rolling:
        physical_gates.pop('no_commanded_source_feed_or_roll_before_bearing')
        physical_gates['original_initial_and_settle_respected']=physical_gates.pop('source_stabilized_before_release')
        physical_gates['actual_loaded_motion_handoff_evidence']=physical_gates.pop('actual_loaded_motion_before_feed')
        expected_S=r['source_constants']['feed_per_cycle']*r['nominal_frequency_hz']*np.maximum(0.,t-r['settle_s'])
        physical_gates['original_normal_S_and_roll_coordination']=bool(abs(z['S']-expected_S).max()<1e-8 and abs(z['source_commanded_roll_rad']-2*np.pi*z['S']/r['source_constants']['feed_per_cycle']).max()<1e-8)
    history=[json.loads(line) for line in (C/'global_updates.jsonl').open(encoding='utf-8')];written=[row for row in history if row['written']];g_effect=max([row.get('metrics',{}).get('command_deg',0.) for row in written]+[0.]);g_chain=bool(contact is not None and fit is not None and write is not None and contact<fit<=write<=t[completion_index] and g_effect>1e-6) if config['write_global'] else not written
    force_norm=np.linalg.norm(z['force_curvature_field'],axis=2).max(1)*1000;active=(force_norm>1e-8)&(z['command_filter_factor']>1e-6);physical=all(physical_gates.values());method=physical and g_chain
    cap=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));readback=[]
    for f in [0,frames//2,max(0,frames-1)]:cap.set(cv2.CAP_PROP_POS_FRAMES,f);readback.append(bool(cap.read()[0]))
    cap.release();video_ok=frames==len(t) and abs(fps-10)<.01 and all(readback)
    out=dict(requirements='STEP2_FINAL_20261016',case=C.name,version=r['version'],physical_action_complete=physical,method_chain_complete=method and video_ok,action_chain_complete=method and video_ok,physical_gates=physical_gates,geometry_write_chain=g_chain,first_release_t=release,first_raw_contact_t=contact,first_reliable_t=fit,first_G_write_t=write,first_capture_t=capture,earliest_full_straight_t_s=None if first_full is None else float(t[first_full]),completion_t_s=None if chosen is None else float(t[chosen[1]]),sustained_window=None if chosen is None else dict(start_s=float(t[chosen[0]]),end_s=float(t[chosen[1]]),duration_s=float(t[chosen[1]]-t[chosen[0]])),full_windows=candidate_windows,each_advance_mm=(advance*1000).tolist(),each_spin_deg=spin.tolist(),axis_error_deg=axis_error,source_motion_is_diagnostic_only=True,maximum_source_given_motion_before_capture_mm=float(z['actual_source_motion_from_given_m'][before].max(initial=0)*1000),maximum_source_axis_motion_before_capture_mm=float(z['actual_source_Xmotion_from_given_m'][before].max(initial=0)*1000),bearing_overlap_windows=overlap_windows,minimum_source_vertical_before_capture_N=float(source_Fz[before].min(initial=W)),local_force_adjustment_with_nonzero_transmission_samples=int(active.sum()),maximum_requested_force_curvature_per_m=float(force_norm.max()),geometry_writes=len(written),maximum_same_state_G_effect_command_deg=float(g_effect),native_integral_error_rad=integral_error,all_run_maximum_depth_mm=(z['depth'].max(0)*1000).tolist(),all_run_saturated_joint_s=float(np.trapz((abs(z['torque'])>=2.3549).astype(float).sum(1),t)),video=dict(frames=frames,fps=fps,readback=readback,synchronous_G0_Gt_C_actual=True,passed=video_ok),runner_result=result,limits=['Numeric action windows/envelopes are registered implementation diagnostics, not user hard requirements','Development action and G write evidence do not independently establish C benefit','All-run costs/late failures remain reported even if a required action window occurred earlier'])
    out['exact_capture_snapshot_used']=capture_snapshot_used
    out['normal_precontact_progression_allowed']=corrected_rolling
    if corrected_rolling:
        out['requirements']='STEP2_ROLLING_SCOPE_CLARIFIED_20261016'
        out['source_displacement_interpretation']='Normal rolling/feed after placed original Step1 t0 is allowed; source material displacement is not a no-motion acceptance gate. Actual bearing/rolling/transport and contact slip are reported separately.'
    if 'source_balance_offset_rad' in z:
        out['source_balance_local_C']=dict(peak_native_offset_deg=float(np.degrees(abs(z['source_balance_offset_rad']).max())),peak_moment_residual_Nm=float(abs(z['source_balance_residual_Nm']).max()),active_samples=int((abs(z['source_balance_offset_rad']).max(1)>1e-8).sum()))
    (C/'FINAL_CONTRACT_METRICS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2));return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case');evaluate(ap.parse_args().case)
