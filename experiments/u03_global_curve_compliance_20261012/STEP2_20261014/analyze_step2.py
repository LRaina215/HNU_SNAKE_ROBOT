"""Saved-state evaluation and causal geometry/command records. No dynamics."""
from physics import *
from snake_control.cmu_window_gait import cylinder
from native_mapping import NativeMapping
import argparse,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stable(mask,t,seconds):
    start=None
    for i,value in enumerate(mask):
        if not value:start=None
        elif start is None:start=i
        elif t[i]-t[start]>=seconds:return start
    return None
def longest(mask,t):
    start=None;best=0.
    for i,on in enumerate(mask):
        if on and start is None:start=i
        if start is not None and (not on or i==len(mask)-1):
            end=i if on else i-1;best=max(best,t[end]-t[start]+.1);start=None
    return float(best)
def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
def scene_truth(m,d):
    geom=m.geom('pipe_straight_04').id;axis=d.geom_xmat[geom].reshape(3,3)[:,2].copy()
    if axis[1]<0:axis=-axis
    center=d.geom_xpos[geom].copy();half=m.geom_size[geom,1]
    return dict(axis=axis,center=center,radius=float(m.geom_size[geom,0]),start=center-half*axis,end=center+half*axis)
def axis_error(axes,truth):return np.degrees(np.arccos(np.clip(np.asarray(axes)@truth['axis'],-1,1)))
def center_error(entries,truth):
    off=np.asarray(entries)-truth['center'];off-=np.sum(off*truth['axis'],axis=-1)[...,None]*truth['axis'];return 1000*np.linalg.norm(off,axis=-1)
def ordered_distance(points,guide):
    result=[]
    for i,p in enumerate(points):
        a=guide[max(0,i-1):min(len(guide)-1,i+2)];b=guide[max(0,i-1)+1:min(len(guide),i+3)];v=b-a
        u=np.clip(np.sum((p-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-12),0,1)
        result.append(float(np.min(np.linalg.norm(p-(a+u[:,None]*v),axis=1))))
    return np.array(result)

def evaluate(name):
    case=BASE/name;z=dict(np.load(case/'trace.npz'));t=z['t'];c=z['centers'];p=z['points'];f=z['forces'];result=json.loads((case/'result.json').read_text());plan=json.loads((case/'plan.json').read_text())
    print('SAVED_ANALYSIS',name,'samples',len(t),flush=True)
    m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);d.qpos[:]=z['qpos'][-1];mujoco.mj_forward(m,d);truth=scene_truth(m,d)
    offset=c-truth['start'];axial=offset@truth['axis'];rad=offset-axial[...,None]*truth['axis'];rho=np.linalg.norm(rad,axis=2);length=float(np.linalg.norm(truth['end']-truth['start']))
    full=np.all((rho>.07)&(rho<.12)&(axial>0)&(axial<length),axis=1)&(f[:,0].sum(1)<.1)&(f[:,1].sum(1)>.2)
    polar=np.unwrap(np.arctan2(c[:,:6,2]-truth['center'][2],c[:,:6,0]-truth['center'][0]),axis=1)
    front=np.all((rho[:,:6]>.065)&(rho[:,:6]<.13)&(axial[:,:6]>0)&(axial[:,:6]<length),axis=1)&(f[:,1,:6].sum(1)>.2)&(np.ptp(polar,axis=1)>np.pi)
    first=stable(full,t,2.);front_first=stable(front,t,2.);false=np.flatnonzero(~full);suffix=int(false[-1]+1) if len(false) else 0
    post=float(t[-1]-t[suffix]) if suffix<len(t) else 0.;advance=(c[-1]-c[suffix])@truth['axis'] if suffix<len(t) else np.zeros(21)
    spin=np.zeros(21)
    if suffix<len(t):
        selected=list(range(suffix,len(t),5))
        if selected[-1]!=len(t)-1:selected.append(len(t)-1)
        rates=[]
        for i in selected:
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_forward(m,d)
            tangent=p[i,:-1]-p[i,1:];tangent/=np.linalg.norm(tangent,axis=1)[:,None];rate=[]
            for n in range(21):
                v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('head').id+n,v,0);rate.append(float(v[:3]@tangent[n]))
            rates.append(rate)
        spin=np.degrees(np.trapz(rates,x=t[selected],axis=0))
    shape=cylinder(p[-1]);end_axis=float(np.degrees(np.arccos(np.clip(abs(shape['axis']@truth['axis']),0,1))))
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]];actual_joint=float(np.degrees(abs(z['qpos'][:,qa]).max()));torque=float(abs(z['torque']).max())
    rolling=bool(np.all(abs(spin)>=90) and (np.all(spin>0) or np.all(spin<0)))
    success=bool(result['stop']=='duration' and first is not None and post>=50 and np.all(advance>.04) and rolling and end_axis<5 and z['depth'][:,3].max()==0 and z['depth'][:,2].max()<.003 and actual_joint<90.5 and torque<=2.355+1e-6)
    update=read_lines(case/'global_updates.jsonl');fits=read_lines(case/'geometry_hypotheses.jsonl');written=[x for x in update if x['written']]
    events=[]
    for n in range(21):
        target=(rho[:,n]>.065)&(rho[:,n]<.13)&(axial[:,n]>0)&(axial[:,n]<length)&(f[:,1,max(0,n-1):min(21,n+2)].sum(1)>.2);recv=stable(target,t,.5);source=np.flatnonzero(f[:,0,n]>.1)
        events.append(dict(material=n,first_retained_true_target_t=None if recv is None else float(t[recv]),last_true_source_contact_t=None if not len(source) else float(t[source[-1]]),post_axial_mm=float(advance[n]*1000),post_axial_spin_deg=float(spin[n]),final_true_target_radius_mm=float(rho[-1,n]*1000)))
    G=z['global_reference_points'];C=z['nominal_curve_points'];same_G=1000*np.linalg.norm(p-G,axis=2);same_C=1000*np.linalg.norm(p-C,axis=2)
    ordered=np.array([ordered_distance(a,b) for a,b in zip(p,G)])*1000
    est_axis=axis_error(z['estimated_axis'],truth);g_axis=axis_error(z['applied_global_axis'],truth)
    est_center=center_error(z['estimated_entry_m'],truth);g_center=center_error(z['applied_global_entry_m'],truth)
    est_radius=1000*(z['estimated_radius_m']-truth['radius']);g_radius=1000*(z['applied_global_radius_m']-truth['radius'])
    saturation=abs(z['torque'])>=2.355-1e-5;role=z['estimated_material_role'];probe=z['probe_curvature_per_m'];probe_delta=[]
    for i in range(len(t)):
        bounds=z['curvature_bounds_mm'][i];material=z['material_arc_nodes_mm'][i];mid=.5*(bounds[:-1]+bounds[1:]);v=np.clip(mid/material[8],0,1);weight=np.sin(np.pi*v)**2;weight[bounds[1:]>material[8]]=0.
        bending=NativeMapping.joint_bending(bounds,weight[:,None]*probe[i,None,:]/1000,material)
        probe_delta.append(-np.where(np.arange(20)%2,bending[:,0],bending[:,1]))
    probe_delta=np.degrees(np.array(probe_delta));estimated_force=z['sensor_force_by_estimated_role_N'].sum(2)
    reliable_time=.1*float(z['geometry_reliable'].sum())
    failures=sum(len(x.get('proposal_failures',[])) for x in update)
    causal=bool(written and result['first_eligible_contact_t']<result['first_reliable_geometry_t']<=result['first_G_write_t'] and all(x['evidence_t']<=x['t'] and x['fit_t']<=x['t'] and x['continuity_passed'] for x in written))
    metrics=dict(case=name,condition=plan['configuration']['condition'],write_global=plan['configuration']['write_global'],stop=result['stop'],sim_time_s=result['sim_time'],action_success=success,
        full_stable_time_s=None if first is None else float(t[first]),front_stable_time_s=None if front_first is None else float(t[front_first]),final_full_target_s=post,
        each_post_axial_mm=(advance*1000).tolist(),minimum_each_post_axial_mm=float(advance.min()*1000),each_post_axial_spin_deg=spin.tolist(),minimum_each_abs_post_spin_deg=float(abs(spin).min()),same_direction_spin=rolling,
        end_true_axis_angle_deg=end_axis,end_actual_vs_final_G_axis_deg=float(np.degrees(np.arccos(np.clip(abs(shape['axis']@z['applied_global_axis'][-1]),0,1)))),end_body_fit_pitch_mm=float(shape['pitch']*1000),planned_pitch_mm=float(z['target_helix_pitch_m'][-1]*1000),
        end_body_fit_radius_mm=float(shape['radius']*1000),planned_helix_radius_mm=float(z['target_helix_radius_m'][-1]*1000),
        end_body_fit_radial_rms_mm=float(shape['radial_rms']*1000),end_body_fit_signed_pitch_mm=float(shape['signed_pitch']*1000),planned_right_handed_signed_pitch_mm=-float(z['target_helix_pitch_m'][-1]*1000),winding_sign_matches_G=bool(shape['signed_pitch']<0),
        actual_joint_max_deg=actual_joint,torque_max_Nm=torque,source_target_self_floor_max_depth_mm=(z['depth'].max(0)*1000).tolist(),
        torque_saturated_joint_time_s=float(.1*saturation.sum()),longest_single_joint_saturation_s=max(longest(saturation[:,n],t) for n in range(20)),
        first_contact_t=result['first_eligible_contact_t'],first_reliable_geometry_t=result['first_reliable_geometry_t'],first_G_write_t=result['first_G_write_t'],G_write_count=len(written),causal_contact_geometry_write_order_passed=causal,
        maximum_written_same_S_command_change_deg=max([x['same_S_raw_change_deg'] for x in written] or [0.]),maximum_written_material_node_step_mm=1000*max([x['global_material_node_change_m'] for x in written] or [0.]),proposal_rejection_attempts=failures,
        reliable_geometry_sampled_time_s=reliable_time,final_estimated_axis_error_deg=float(est_axis[-1]),final_estimated_centerline_error_mm=float(est_center[-1]),final_estimated_radius_error_mm=float(est_radius[-1]),
        final_G_axis_error_deg=float(g_axis[-1]),final_G_centerline_error_mm=float(g_center[-1]),final_G_radius_error_mm=float(g_radius[-1]),
        axial_entry_identified=False,axial_entry_prior_uncertainty_mm=40.,geometry_uncertainty_is_heuristic=True,
        probe_max_abs_curvature_per_m=np.max(abs(probe),axis=0).tolist(),probe_max_abs_native_joint_contribution_deg=float(abs(probe_delta).max()),
        initial_true_source_target_force_N=f[0].sum(1).tolist(),initial_is_given_historical_trajectory_resume=False,
        source_gamma_max_per_m=float(z['source_gamma_per_m'].max()),receiver_delta_range_mm=(1000*np.array([z['compliance_delta_m'].min(),z['compliance_delta_m'].max()])).tolist(),
        local_curve_chord_error_max_mm=float(z['local_curve_chord_error_mm'].max()),
        last_same_index_world_G_rms_mm=float(np.sqrt(np.mean(same_G[-1]**2))),last_same_index_world_C_rms_mm=float(np.sqrt(np.mean(same_C[-1]**2))),last_order_restricted_G_distance_rms_mm=float(np.sqrt(np.mean(ordered[-1]**2))),
        unknown_load_retained=True,estimated_role_is_scene_truth=False,world_curve_exact_tracking_claim=False,
        failure_classification='none' if success else 'execution_or_handoff' if result['first_reliable_geometry_t'] is not None else 'information_insufficient' if result['first_eligible_contact_t'] is not None else 'no_eligible_contact_search_boundary',
        original_model_equal=(case/'model.xml').read_bytes()==MODEL.read_bytes(),given_initial_equal=(case/'given_initial.npz').read_bytes()==(BASE/'GIVEN_INITIAL/integration_state.npz').read_bytes(),
        runtime=plan['runtime'],scope='Given-contact truth-sensor-assisted deterministic development pair; no autonomous initial formation or precontact search',events=events)
    (case/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez(case/'geometry_evaluation.npz',t=t,estimated_axis_error_deg=est_axis,G_axis_error_deg=g_axis,estimated_centerline_error_mm=est_center,G_centerline_error_mm=g_center,estimated_radius_error_mm=est_radius,G_radius_error_mm=g_radius,same_index_world_G_mm=same_G,same_index_world_C_mm=same_C,order_restricted_G_distance_mm=ordered,probe_native_delta_deg=probe_delta)
    with (case/'material_events.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=events[0]);w.writeheader();w.writerows(events)
    with (case/'material_paths.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['t','material','x','y','z','G_x','G_y','G_z','C_x','C_y','C_z','true_source_N','true_target_or_elbow_N','estimated_role','material_observed_progress_m','progress_valid'])
        gc=.5*(G[:,:-1]+G[:,1:]);cc=.5*(C[:,:-1]+C[:,1:])
        for i,sec in enumerate(t):
            for n in range(21):w.writerow([sec,n,*c[i,n],*gc[i,n],*cc[i,n],*f[i,:,n],role[i,n],z['material_progress_observed_m'][i,n],z['material_progress_valid'][i,n]])
    fig,axs=plt.subplots(6,1,figsize=(13,16),sharex=True)
    axs[0].plot(t,est_axis,label='estimate axis error');axs[0].plot(t,g_axis,label='applied G axis error');axs[0].set_ylabel('deg');axs[0].legend()
    axs[1].plot(t,est_center,label='estimate centerline mm');axs[1].plot(t,g_center,label='G centerline mm');axs[1].plot(t,est_radius,label='estimate radius error mm');axs[1].plot(t,g_radius,label='G radius error mm');axs[1].legend(ncol=2)
    axs[2].plot(t,probe[:,0],label='Bishop probe1');axs[2].plot(t,probe[:,1],label='Bishop probe2');axs[2].plot(t,z['compliance_delta_m']*1000,label='local radial delta(mm)');axs[2].legend(ncol=3)
    for n,label in enumerate(['known-source-consistent','receiver-candidate-consistent','bridge/unknown']):axs[3].plot(t,estimated_force[:,n],label=label)
    axs[3].set_ylabel('sensed N');axs[3].legend()
    axs[4].plot(t,f[:,0].sum(1),label='evaluation true source load');axs[4].plot(t,f[:,1].sum(1),label='evaluation target/elbow load');axs[4].plot(t,np.sum((rho>.07)&(rho<.12)&(axial>0)&(axial<length),axis=1),label='true target-envelope material count');axs[4].legend()
    axs[5].plot(t,z['depth'][:,0]*1000,label='true source intrusion');axs[5].plot(t,z['depth'][:,1]*1000,label='true target intrusion');axs[5].plot(t,z['global_update_count'],label='G write count',alpha=.4);axs[5].legend();axs[5].set_xlabel('elapsed physical time(s)')
    for ax in axs:
        for sec,color in [(result['first_reliable_geometry_t'],'orange'),(result['first_G_write_t'],'cyan'),(metrics['full_stable_time_s'],'green')]:
            if sec is not None:ax.axvline(sec,color=color,ls='--',alpha=.6)
    fig.suptitle(name+' : contact -> geometry -> G -> actual transfer');fig.tight_layout();fig.savefig(case/'contact_geometry_action.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(15,9))
    for k,sec in enumerate([0.,15.,30.,60.,120.,float(t[-1])]):
        i=int(np.argmin(abs(t-sec)));ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*G[i].T,color='cyan',label='current G');ax.plot(*C[i].T,color='magenta',label='current local C');ax.plot(*p[i].T,color='gray',lw=2,label='actual body')
        colors={0:'gray',1:'orange',2:'green',3:'purple',4:'gold',5:'lightgreen'}
        for material in range(21):ax.plot(*p[i,material:material+2].T,color=colors[int(role[i,material])],lw=2)
        e=z['estimated_entry_m'][i];a=z['estimated_axis'][i];ax.plot(*np.array([e-.05*a,e+.75*a]).T,color='gold',label='estimated cylinder axis')
        ax.set(xlim=(.6,1.4),ylim=(-.25,1.15),zlim=(.3,1.4),title=f'{name} t={t[i]:.1f}, writes={z["global_update_count"][i]}');ax.view_init(22,140)
        if k==0:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(case/'current_G_C_actual.png',dpi=140);plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ['case','action_success','stop','full_stable_time_s','final_full_target_s','first_reliable_geometry_t','first_G_write_t','final_G_axis_error_deg','final_G_centerline_error_mm','source_target_self_floor_max_depth_mm']},indent=2),flush=True)
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',required=True);args=ap.parse_args()
    results=[evaluate(name) for name in args.cases];(BASE/'latest_analysis.json').write_text(json.dumps(results,indent=2))
