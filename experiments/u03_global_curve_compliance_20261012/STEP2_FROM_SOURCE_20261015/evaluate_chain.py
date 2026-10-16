"""Offline complete-action gates. Geometry convergence never implies success."""
from physics import *
from snake_control.cmu_window_gait import cylinder
import json,argparse

def first_stable(mask,t,seconds):
    start=None
    for j,on in enumerate(mask):
        if not on:start=None
        elif start is None:start=j
        elif t[j]-t[start]>=seconds:return start
    return None

def evaluate(name):
    case=BASE/name;reg=json.loads((case/'REGISTERED.json').read_text());result=json.loads((case/'result.json').read_text());z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];p=z['points']
    src_r=np.linalg.norm(c[0,:,1:]-[0,1],axis=1);initial=bool(t[0]==0. and z['S'][0]==0. and np.all((src_r>=.065)&(src_r<=.13)&(c[0,:,0]>-.2)&(c[0,:,0]<.95)) and z['eval_pipe_normal_N'][0,2:].sum()==0.)
    rho=np.linalg.norm(c[:,:,[0,2]]-[1.1,1.],axis=2);geo=(rho>.07)&(rho<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2)
    phi=np.unwrap(np.arctan2(c[:,:6,2]-1.,c[:,:6,0]-1.1),axis=1)
    front=geo[:,:6].all(1)&(np.ptp(phi,axis=1)>np.pi)&(z['eval_pipe_normal_N'][:,2,:6].sum(1)>.2)
    fi=first_stable(front,t,2.);full=geo.all(1)&(z['forces'][:,0].sum(1)<.1)&(z['eval_pipe_normal_N'][:,2].sum(1)>.2)
    full_i=first_stable(full,t,2.);not_full=np.flatnonzero(~full);suffix=int(not_full[-1]+1) if len(not_full) else 0
    post=float(t[-1]-t[suffix]) if suffix<len(t) else 0.;advance=c[-1,:,1]-c[suffix,:,1] if suffix<len(t) else np.zeros(21)
    spin=np.degrees(np.trapz(z['actual_axial_spin_rad_s'][suffix:],x=t[suffix:],axis=0)) if suffix<len(t) else np.zeros(21)
    W=reg['robot_weight_N'];src=z['eval_pipe_world_N'][:,:2,:,2].sum((1,2));target=z['eval_pipe_world_N'][:,2,:,2].sum(1)
    bearing=(src>=.20*W)&(target>=.20*W)&front;bi=first_stable(bearing,t,.5)
    contact=result['first_raw_front_contact_t'];fit=result['first_reliable_t'];write=result['first_G_write_t'];release=result['first_release_t']
    capture_t=None if fi is None else float(t[fi]);full_t=None if full_i is None else float(t[full_i])
    acquisition=(t>=contact)&(t<=capture_t) if contact is not None and capture_t is not None else np.zeros(len(t),bool)
    loaded=(z['forces'][:,0,8:]>.1)&acquisition[:,None]
    rear_roll=np.degrees(np.trapz(z['actual_axial_spin_rad_s'][:,8:]*loaded,x=t,axis=0))
    end_axis=float(np.degrees(np.arccos(np.clip(abs(cylinder(p[-1])['axis']@np.array([0.,1.,0.])),0,1))))
    m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    config=next(x for x in reg['cases'] if x['case']==name)
    video=json.loads((case/'video_metadata.json').read_text())
    ordered=bool(release is not None and contact is not None and capture_t is not None and full_t is not None and 0<=release<=contact<capture_t<full_t)
    estimation_order=bool(fit is not None and contact is not None and contact<fit)
    write_chain=bool(write is not None and fit is not None and capture_t is not None and fit<=write<capture_t) if config['write_global'] else write is None
    gates=dict(full_source_t0=initial,source_stabilized_before_release=release is not None,new_front_wrap=fi is not None,
        contact_capture_transfer_order=ordered,estimation_after_contact=estimation_order,global_write_chain_if_enabled=write_chain,overlapping_source_target_bearing=bi is not None,
        existing_rear_source_rolling=bool(np.max(abs(rear_roll))>=15.),all21_full_transfer=full_i is not None,post_transfer_at_least50s=post>=50.,
        all21_advance_over40mm=bool(np.all(advance>.04)),all21_same_direction_spin_over90deg=bool(np.all(abs(spin)>=90) and (np.all(spin>0) or np.all(spin<0))),
        final_axis_under5deg=end_axis<5.,joint_limits=float(np.max(abs(z['qpos'][:,qa]))*180/np.pi)<=90.5,
        torque_limits=float(np.max(abs(z['torque'])))<=2.355+1e-6,no_self_over3mm=float(z['depth'][:,2].max())<=.003,
        no_floor=float(z['depth'][:,3].max())==0.,completed_budget=result['stop']=='duration',initial_file_unchanged=(case/'SOURCE_INITIAL.npz').read_bytes()==(BASE/'SOURCE_INITIAL.npz').read_bytes())
    gates['continuous_video_t0_and_readback']=bool(video['physical_t0_included'] and video['continuous_actual_dynamics'] and video['frames']==len(t) and all(video['readback']))
    success=all(gates.values())
    metrics=dict(case=name,gates=gates,action_chain_complete=success,front_wrap_t=capture_t,full_transfer_t=full_t,
        simultaneous_bearing_t=None if bi is None else float(t[bi]),post_transfer_s=post,each_advance_mm=(advance*1000).tolist(),each_spin_deg=spin.tolist(),
        rear_source_loaded_spin_during_acquisition_deg=rear_roll.tolist(),initial_source_vertical_N=float(src[0]),initial_target_straight_vertical_N=float(target[0]),
        G_same_material_world_rms_mm=float(np.sqrt(np.mean(np.sum((p[-1]-z['global_reference_points'][-1])**2,axis=1)))*1000),
        geometry_only_success_is_insufficient=True,world_curve_exact_tracking_claim=False)
    (case/'ACTION_CHAIN_METRICS.json').write_text(json.dumps(metrics,indent=2));result.update(success=success,action_chain_complete=success,acceptance_status='offline full-chain evaluated; independent review still required')
    (case/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);evaluate(ap.parse_args().case)
