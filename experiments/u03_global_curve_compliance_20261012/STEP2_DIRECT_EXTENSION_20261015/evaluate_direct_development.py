"""Offline truth evaluation for a completed direct-extension diagnostic."""
import argparse,json
from pathlib import Path
import sys,numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.cmu_window_gait import cylinder

def first_stable(mask,t,seconds):
    start=None
    for j,on in enumerate(mask):
        if not on:start=None
        elif start is None:start=j
        elif t[j]-t[start]>=seconds:return start
    return None

def run(path):
    C=Path(path).resolve();reg=json.loads((C/'REGISTERED.json').read_text());result=json.loads((C/'result.json').read_text());z=np.load(C/'trace.npz');t=z['t'];c=z['centers'];p=z['points'];W=reg['robot_weight_N']
    split=z['eval_pipe_world_N'];src=split[:,:2,:,2].sum((1,2));target=split[:,2,:,2].sum(1);normal=z['eval_pipe_normal_N']
    sr=np.linalg.norm(c[0,:,1:]-[0,1],axis=1);miter=(c[0,:,0]+c[0,:,1])<1.105
    initial=bool(t[0]==0 and np.all((sr>.065)&(sr<.13)&miter)&np.all(c[0,:,0]>-.2)&(normal[0,2:].sum()==0))
    rho=np.linalg.norm(c[:,:,[0,2]]-[1.1,1.],axis=2);geo=(rho>.07)&(rho<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2)
    polar=np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1);front_span=np.ptp(polar,axis=1)
    front=geo[:,:6].all(1)&(front_span>np.pi)&(normal[:,2,:6].sum(1)>.2);fi=first_stable(front,t,2.)
    full=geo.all(1)&(z['forces'][:,0].sum(1)<.1)&(normal[:,2].sum(1)>.2);full_i=first_stable(full,t,2.)
    not_full=np.flatnonzero(~full);suffix=int(not_full[-1]+1) if len(not_full) else 0;post=float(t[-1]-t[suffix]) if suffix<len(t) else 0.
    advance=c[-1,:,1]-c[suffix,:,1] if suffix<len(t) else np.zeros(21)
    spin=np.degrees(np.trapz(z['actual_axial_spin_rad_s'][suffix:],x=t[suffix:],axis=0)) if suffix<len(t) else np.zeros(21)
    bearing=(src>=.2*W)&(target>=.2*W)&front;bi=first_stable(bearing,t,.5)
    contact=result.get('first_raw_front_contact_t');fit=result.get('first_reliable_t');write=result.get('first_G_write_t');release=result.get('first_release_t');capture=result.get('first_capture_t')
    before=np.ones(len(t),bool) if capture is None else t<capture-1e-10
    commanded_zero=bool(np.all(z['S'][before]==0)&np.all(z['source_commanded_feed_m'][before]==0)&np.all(z['source_commanded_roll_rad'][before]==0))
    cap_i=None if capture is None else int(np.argmin(abs(t-capture)))
    cap_truth=bool(cap_i is not None and front[cap_i] and target[cap_i]>=.4*W)
    # Protect M8 during release/hold and M14 during wrapping; no rear-roll minimum.
    phase=z['controller_phase']
    # Use exactly the controller's body origins and stable reference. Centre
    # positions relative to t0 are a different metric and must not be mixed in.
    max_drift=float(z['actual_source_boundary_motion_m'][before].max(initial=0.));max_xdrift=float(z['actual_source_boundary_X_motion_m'][before].max(initial=0.))
    # The stop can occur between10Hz trace samples. Include the saved final
    # integration state and the controller's stable-body-origin guard metric.
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(C/'model.xml'));data=mujoco.MjData(model);head=model.body('head').id
    stable_index=0 if release is None else int(np.argmin(abs(t-release)));data.qpos[:]=z['qpos'][stable_index];mujoco.mj_kinematics(model,data);stableP=data.xpos[head:head+21].copy();stableR=data.xmat[head:head+21].reshape(21,3,3).copy()
    state=np.load(C/'integration_state.npz');mujoco.mj_setState(model,data,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(model,data)
    guard_start=14 if phase[-1]=='front_wrap' else 8;final_delta=data.xpos[head+guard_start:head+21]-stableP[guard_start:]
    finalP=data.xpos[head:head+21].copy();data.qpos[:]=z['qpos'][0];mujoco.mj_kinematics(model,data);givenP=data.xpos[head:head+21].copy()
    max_given_drift=float(z['actual_source_motion_from_given_m'][before].max(initial=0.));max_given_xdrift=float(z['actual_source_Xmotion_from_given_m'][before].max(initial=0.))
    if capture is None:
        max_drift=max(max_drift,float(np.linalg.norm(final_delta,axis=1).max()),float(z['actual_source_boundary_motion_m'].max()))
        max_xdrift=max(max_xdrift,float(abs(final_delta[:,0]).max()),float(z['actual_source_boundary_X_motion_m'].max()))
        final_given_delta=finalP[guard_start:]-givenP[guard_start:]
        max_given_drift=max(max_given_drift,float(np.linalg.norm(final_given_delta,axis=1).max()))
        max_given_xdrift=max(max_given_xdrift,float(abs(final_given_delta[:,0]).max()))
    ordered=bool(release is not None and contact is not None and capture is not None and full_i is not None and 0<=release<=contact<capture<float(t[full_i]))
    config=next(x for x in reg['cases'] if x['case']==C.name)
    write_chain=bool(write is not None and fit is not None and capture is not None and contact is not None and contact<fit<=write<capture) if config['write_global'] else write is None
    try:axis_error=float(np.degrees(np.arccos(np.clip(abs(cylinder(p[-1])['axis']@np.array([0,1.,0])),0,1))))
    except Exception:axis_error=None
    raw=z['raw_native_command'];field=z['curvature_field'];bounds=z['curvature_bounds_mm'];material=z['material_arc_nodes_mm'];A=np.maximum(0,np.minimum(material[0,2:,None],bounds[0,None,1:])-np.maximum(material[0,:-2,None],bounds[0,None,:-1]));b=np.einsum('jc,ncd->njd',A,field);recovered=-np.where((np.arange(20)%2)[None,:],b[:,:,0],b[:,:,1]);native_error=float(abs(recovered-raw).max())
    import cv2
    video=cv2.VideoCapture(str(C/'actual_dual_1x.mp4'));frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(video.get(cv2.CAP_PROP_FPS));readback=[]
    for frame in sorted(set([0,max(0,frame_count//2),max(0,frame_count-1)])):
        video.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,_=video.read();readback.append(bool(ok))
    video.release()
    dt=float(np.median(np.diff(t))) if len(t)>1 else .1
    saturated_joint_s=float(np.sum(abs(z['torque'])>=2.355-1e-4)*dt)
    gates=dict(full_source_t0=initial,source_stabilized_before_release=release is not None,
        no_commanded_pre_capture_feed_or_roll=commanded_zero,source_boundary_drift_under12mm=max(max_drift,max_given_drift)<=.012+1e-6,
        nondecreasing_alpha=bool(np.diff(z['alpha']).min(initial=0)>=-1e-10),nondecreasing_beta=bool(np.diff(z['beta']).min(initial=0)>=-1e-10),
        contact_before_reliable_geometry=bool(contact is not None and fit is not None and contact<fit),global_write_chain_if_enabled=write_chain,
        new_front_wrap=fi is not None,controller_capture_matches_true_current_wrap_and40pct_load=cap_truth,
        contact_capture_transfer_order=ordered,overlapping_source_receiver_bearing=bi is not None,all21_full_transfer=full_i is not None,
        post_transfer_at_least50s=post>=50,all21_advance_over40mm=bool((advance>.04).all()),
        all21_same_direction_spin_over90deg=bool((abs(spin)>=90).all() and ((spin>0).all() or (spin<0).all())),
        final_axis_under5deg=axis_error is not None and axis_error<5,
        actual_joint_limit=float(abs(z['qpos'][:,7:]).max()*180/np.pi)<=90.5,torque_limit=float(abs(z['torque']).max())<=2.355+1e-6,
        no_self_over3mm=float(z['depth'][:,2].max())<=.003,no_floor=float(z['depth'][:,3].max())==0,
        fixed_material_grid=bool(np.max(abs(material-material[0]))<1e-10 and np.max(abs(bounds-bounds[0]))<1e-10),native_curvature_integral=native_error<1e-10,
        continuous_video_t0_and_readback=bool(t[0]==0 and frame_count==len(t) and abs(fps-10)<.01 and all(readback)),completed_budget=result['stop']=='duration',
        exact_reviewed_initial=(C/'GIVEN_INITIAL.npz').read_bytes()==(C.parent/'GIVEN_INITIAL.npz').read_bytes())
    success=all(gates.values());metrics=dict(case=C.name,role='Development diagnostic; does not replace final four-case comparison',gates=gates,action_chain_complete=success,
        controller_stop=result['stop'],first_release_t=release,first_raw_front_contact_t=contact,first_reliable_t=fit,first_G_write_t=write,controller_capture_t=capture,
        true_stable_front_wrap_t=None if fi is None else float(t[fi]),true_full_transfer_t=None if full_i is None else float(t[full_i]),
        simultaneous_bearing_t=None if bi is None else float(t[bi]),post_transfer_suffix_s=post,each_advance_mm=(advance*1000).tolist(),each_spin_deg=spin.tolist(),
        maximum_pre_capture_source_boundary_motion_mm=max_drift*1000,maximum_pre_capture_source_Xmotion_mm=max_xdrift*1000,
        maximum_pre_capture_source_motion_from_given_mm=max_given_drift*1000,maximum_pre_capture_source_Xmotion_from_given_mm=max_given_xdrift*1000,
        maximum_front6_true_span_deg=float(np.degrees(front_span.max())),maximum_true_receiver_front_materials=int(geo[:,:6].sum(1).max()),
        maximum_sampled_depth_components=['source','receiver','self','floor'],maximum_sampled_depth_mm=(z['depth'].max(0)*1000).tolist(),
        actuator_saturated_joint_s=saturated_joint_s,native_integral_error_rad=native_error,
        final_G_same_material_world_rms_mm=float(np.sqrt(np.mean(np.sum((p[-1]-z['global_reference_points'][-1])**2,axis=1)))*1000),
        geometry_only_success_is_insufficient=True,world_curve_exact_tracking_claim=False,independent_outcome_review_pending=True)
    (C/'ACTION_CHAIN_METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case_directory');run(ap.parse_args().case_directory)
