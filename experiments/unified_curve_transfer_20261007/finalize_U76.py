from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from validation.unified_curve_transfer_experiment import video
base=Path(__file__).resolve().parent

def audit(name,failed=False):
    case=base/name
    a=np.load(case/'trace.npz'); commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    t=a['sim_time']-a['sim_time'][0]; f=a['contact_force_N']; cf=a['cylindrical_contact_force_N']; c=a['centers_mm']
    radius=np.linalg.norm(c[:,:,[0,2]]-[1100.,1000.],axis=2)
    def windows(mask):
        edges=np.diff(np.r_[False,mask,False].astype(int))
        return [[float(t[i]),float(t[j-1]),float(t[j-1]-t[i]+.01)] for i,j in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    complete=(c[:,:,1]>80).all(1)&(radius<125).all(1)&((f[:,0]>.1).sum(1)==0)&((f[:,1]>.1).sum(1)>=4)&(f[:,1,17:]>.1).any(1)
    source=json.loads((case/'result.json').read_text(encoding='utf8')) if (case/'result.json').exists() else {}
    result=dict(name=name,outcome=source.get('outcome','controller_exception'),complete=bool(source.get('complete',False)),
        physical_seconds=float(t[-1]),absolute_MuJoCo_time_s=float(a['sim_time'][-1]),dynamics_active=False,
        front_actual_Y_mm=float((c[-1,:7,1]-c[0,:7,1]).mean()),rear_actual_X_mm=float((c[-1,14:,0]-c[0,14:,0]).mean()),
        tail_actual_Y_mm=float(c[-1,20,1]-c[0,20,1]),last_reference_tail_Y_mm=float(a['reference_points_mm'][-1,20:,1].mean()),
        shared_shift_change_mm=float(a['shift_mm'][-1]-a['shift_mm'][0]),
        final_raw_source_modules=np.flatnonzero(f[-1,0]>.1).tolist(),final_cylindrical_source_modules=np.flatnonzero(cf[-1,0]>.1).tolist(),
        final_geometry_filtered_source_modules=np.flatnonzero((cf[-1,0]>.1)&a['source_grip_region_mask'][-1]).tolist(),
        final_raw_target_modules=np.flatnonzero(f[-1,1]>.1).tolist(),final_cylindrical_target_modules=np.flatnonzero(cf[-1,1]>.1).tolist(),
        M20_source_force_range_N=[float(cf[:,0,20].min()),float(cf[:,0,20].max())],M20_final_source_force_N=float(cf[-1,0,20]),
        M20_target_force_range_N=[float(cf[:,1,20].min()),float(cf[:,1,20].max())],M20_final_target_force_N=float(cf[-1,1,20]),
        M20_target_contact_windows_s=windows(cf[:,1,20]>.1),M20_simultaneous_contact_windows_s=windows((cf[:,0,20]>.1)&(cf[:,1,20]>.1)),
        all_raw_source_absent_windows_s=windows((f[:,0]>.1).sum(1)==0),whole_body_target_geometry_windows_s=windows((c[:,:,1]>80).all(1)&(radius<125).all(1)),
        full_completion_windows_s=windows(complete),final_min_body_Y_mm=float(c[-1,:,1].min()),final_max_target_radius_mm=float(radius[-1].max()),
        final_M20_center_mm=c[-1,20].tolist(),max_actual_joint_deg=float(np.degrees(abs(a['q'])).max()),max_actuator_torque_Nm=float(abs(a['torque']).max()),
        max_dispatch_step_deg=float(np.degrees(abs(np.diff(a['ctrl'],axis=0))).max()),max_pipe_penetration_mm=float(a['pipe_penetration_mm'].max()),
        max_self_penetration_mm=float(a['nonadjacent_self_penetration_mm'].max()),max_reference_link_error_mm=float(a['maximum_reference_link_error_mm'].max()),
        final_J17_J20_deg=np.degrees(a['q'][-1,16:20]).tolist(),last_dispatched_J17_J20_deg=np.degrees(a['ctrl'][-1,16:20]).tolist(),
        last_material_region_M18_M20=a['material_region_codes'][-1,18:21].tolist(),
        last_target_tail_curve_step=commands[-1]['localized_yield'].get('target_tail_curve_step'),
        new_events=[x for x in source.get('controller_events',[]) if x['t_s']>=source.get('controller_initial_elapsed_s',1e9)])
    if failed:
        result.update(error='feedback candidate raised original pipe-solid guard before outer rollback; controller export also failed; final_state.npz is complete physical integration+ctrl, final controller unavailable',
            complete_paired_checkpoint='action_t6p80_feed.npz / action_t6p80_feed_controller.json',
            cumulative_actual_seconds_with_U76=.3+float(t[-1]))
        (case/'failure_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        rows=[{key:a[key][i] for key in a.files} for i in range(len(t))]
        video(case,rows)
    else:
        previous=np.load(base/'U76R/trace.npz')
        result['start_preservation_max_difference']={key:float(abs(a[key][0]-previous[key][680]).max()) for key in ['qpos','qvel','ctrl']}
        prior=np.load(base/'U76/trace.npz')
        result['branch_net_front_Y_mm']=float((c[-1,:7,1]-prior['centers_mm'][0,:7,1]).mean())
        result['branch_net_rear_X_mm']=float((c[-1,14:,0]-prior['centers_mm'][0,14:,0]).mean())
        result['branch_net_tail_Y_mm']=float(c[-1,20,1]-prior['centers_mm'][0,20,1])
        result['actual_window_sum_seconds']=.3+7.95+float(t[-1])
        result['selected_branch_physical_seconds']=.3+6.8+float(t[-1])
        if name=='U76P':
            result['actual_window_at_least_seconds']=.3+7.95+10.+float(t[-1])
            result['unrecorded_U76R2_tail_duration']='unknown after saved 10.00s; not counted as known time/result'
            result['reference_registration']=json.loads((base/'U76P_inputs/basis.json').read_text(encoding='utf8'))
            result['runtime_contact_field_note']='runtime actual_*_grip fields also had geometry filter; authoritative raw/cylinder arrays above retain source M20; current source splits actual force and geometrical filter after run frozen'
            ref=(a['reference_points_mm'][:,:-1]+a['reference_points_mm'][:,1:])/2
            result['M18_M20_actual_reference_center_comparison']=[dict(module=j,actual_initial_mm=c[0,j].tolist(),actual_final_mm=c[-1,j].tolist(),actual_delta_mm=(c[-1,j]-c[0,j]).tolist(),reference_initial_mm=ref[0,j].tolist(),reference_final_mm=ref[-1,j].tolist(),reference_delta_mm=(ref[-1,j]-ref[0,j]).tolist()) for j in range(18,21)]
            def orientation(points,centers,index):
                u=points[index,20]-points[index,21];u=u/np.linalg.norm(u)
                radial=centers[index,20]-np.array([1100.,0.,1000.]);radial[1]=0.;radial/=np.linalg.norm(radial)
                azimuth=np.array([-radial[2],0.,radial[0]])
                return dict(unit_toward_head=u.tolist(),target_axis_component=float(u[1]),radial_component=float(u@radial),azimuth_component=float(u@azimuth),angle_to_target_axis_deg=float(np.degrees(np.arccos(np.clip(u[1],-1,1)))))
            result['M20_actual_rod_orientation_initial_final']=[orientation(a['points_mm'],c,0),orientation(a['points_mm'],c,-1)]
            result['M20_reference_rod_orientation_initial_final']=[orientation(a['reference_points_mm'],ref,0),orientation(a['reference_points_mm'],ref,-1)]
            result['J16_negative_torque_saturation_fraction']=float(np.mean(a['torque'][:,15]<-2.3549))
            result['old_source_unload_curvature_max_after_start']=float(max(abs(np.array(x['curve_curvature_release_strength'])).max() for x in commands))
            rejections=[e for e in result['new_events'] if 'solid_guard' in e.get('mode','')]
            result['solid_guard_rejection_count']=len(rejections)
            result['solid_guard_rejection_first_last']=rejections[:1]+rejections[-1:]
            result['final_M18_M20_task_field']=[x for x in commands[-1].get('shared_curve_contact_field',[]) if x['module']>=18]
    (case/'evidence_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='new_events'},ensure_ascii=False,indent=2))

audit(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='failed')
