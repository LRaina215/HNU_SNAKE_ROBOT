"""Physical trajectory acceptance, material event order, and post-transfer motion."""
from native_run import *
def stable_start(mask,t,seconds=1):
    run=0
    for i,flag in enumerate(mask):
        run=run+1 if flag else 0
        if run and t[i]-t[i-run+1]>=seconds:return i-run+1
    return None

for name in ['N09','N10']:
    case=BASE/name;z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];forces=z['forces'];r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2)
    front_angle=np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1)
    front_span=np.ptp(front_angle,axis=1)
    front=np.all((r[:,:6]>.065)&(r[:,:6]<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2),axis=1)&(forces[:,1,:6].sum(1)>.2)&(front_span>np.pi)
    front_i=stable_start(front,t,2)
    full=np.all((r>.07)&(r<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2),axis=1)&(forces[:,0].sum(1)<.1)&(forces[:,1].sum(1)>.2)
    full_i=stable_start(full,t,2)
    last_source=np.flatnonzero(forces[:,0].sum(1)>.1)
    tail_exit_time=float(t[last_source[-1]]) if len(last_source) else 0
    # Use the last continuous full-geometry interval; earlier transient masks do
    # not count as stable success. The boundary is observed from actual states.
    last_false=np.flatnonzero(~full);settled_i=last_false[-1]+1 if len(last_false) else 0
    events=[]
    for n in range(21):
        target=(r[:,n]>.065)&(r[:,n]<.13)&(c[:,n,1]>.15)&(np.sum(forces[:,1,max(0,n-1):min(21,n+2)],axis=1)>.2)
        recv=stable_start(target,t,.5)
        source=np.flatnonzero(forces[:,0,n]>.1)
        events.append(dict(module=n,first_retained_target_time=None if recv is None else float(t[recv]),last_source_contact_time=None if not len(source) else float(t[source[-1]]),initial_x=float(c[0,n,0]),final_y=float(c[-1,n,1])))
    end_stats=cylinder(z['points'][-1]);axis=end_stats['axis'];axis_angle=float(np.rad2deg(np.arccos(np.clip(abs(axis[1]),0,1))))
    post=t[-1]-t[settled_i] if settled_i<len(t) else 0
    i50=np.argmin(abs(t-(t[-1]-50)));i20=np.argmin(abs(t-(t[-1]-20)))
    displacement=(c[-1]-c[settled_i]) if settled_i<len(t) else np.zeros((21,3))
    real_joint=np.max(abs(z['qpos'][:,7:]));drive=np.max(abs(z['torque']))
    eligible=bool(full_i is not None and post>=20 and axis_angle<5 and real_joint<np.radians(90.5) and drive<=2.355+1e-6 and np.all(displacement[:,1]>.04) and np.max(z['depth'][:,3])==0)
    out=dict(simulation_success=eligible,front_first_stable_time=None if front_i is None else float(t[front_i]),front_span_at_stable_start_deg=None if front_i is None else float(np.rad2deg(front_span[front_i])),all_body_first_stable_time=None if full_i is None else float(t[full_i]),last_source_contact_time=tail_exit_time,continuous_final_full_target_from=float(t[settled_i]),continuous_final_full_target_seconds=float(post),post_transfer_module_Y_displacement_mm=(displacement[:,1]*1000).tolist(),last50s_module_Y_displacement_mm=((c[-1,:,1]-c[i50,:,1])*1000).tolist(),last20s_module_Y_displacement_mm=((c[-1,:,1]-c[i20,:,1])*1000).tolist(),end_axis_angle_deg=axis_angle,end_center_radius_mm=(r[-1]*1000).tolist(),end_pitch_mm=end_stats['pitch']*1000,end_radial_rms_mm=end_stats['radial_rms']*1000,max_actual_joint_deg=float(np.rad2deg(real_joint)),max_drive_nm=float(drive),max_pipe_penetration_mm=(np.max(z['depth'][:,:2],axis=0)*1000).tolist(),max_self_penetration_mm=float(np.max(z['depth'][:,2])*1000),physical_root_reset=False,source_before_first_front_target_N=None if front_i is None else float(forces[front_i,0].sum()),events=events)
    (case/'verified_success.json').write_text(json.dumps(out,indent=2));print(name,json.dumps({k:v for k,v in out.items() if k not in ['events','end_center_radius_mm','post_transfer_module_Y_displacement_mm','last20s_module_Y_displacement_mm','last50s_module_Y_displacement_mm']},indent=2))
    result=json.loads((case/'result.json').read_text());result.update(success=eligible,verified_success_file='verified_success.json',success_scope='CMU window mechanism adapted on exact original native20/L model; simulation only',completion_actual_time=out['all_body_first_stable_time']);(case/'result.json').write_text(json.dumps(result,indent=2))
