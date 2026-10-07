"""Read the completed C0/P1 records only; no solver or physical stepping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import numpy as np
from hook_pull_short_physics import OUT,BASE,write


def intervals(t,good):
    result=[];start=None
    for i,value in enumerate(good):
        if value and start is None: start=i
        if start is not None and (not value or i==len(good)-1):
            end=i if value else i-1
            result.append([float(t[start]),float(t[end])]);start=None
    return result


def run():
    traces={}
    for name in ('C0_HOLD','P1_PREFIX'):
        path=OUT/name/'trace.npz'
        if path.exists():
            with np.load(path) as a: traces[name]={k:a[k].copy() for k in a.files}
    if not traces: raise RuntimeError('No raw records to analyze')
    results={}
    for name,a in traces.items():
        t=a['t_s'];c=a['centers_mm'];f=a['contact_force_N'];counts=np.count_nonzero(f>.1,axis=2)
        initial_contacts=[np.flatnonzero(f[0,i]>.1).tolist() for i in range(2)]
        snapshots=[]
        for moment in (2.,12.):
            if t[-1]<moment-1e-9: continue
            i=int(abs(t-moment).argmin());delta=c[i]-c[0];roll=a['axial_roll_deg'][i]
            snapshots.append(dict(t_s=float(t[i]),module_displacement_xyz_mm=delta,
                module_target_y_mm=delta[:,1],module_source_x_mm=delta[:,0],module_axial_self_rotation_deg=roll,
                front_M0_M7_mean_y_mm=float(delta[:8,1].mean()),rear_M13_M20_mean_x_mm=float(delta[13:21,0].mean()),
                front_M0_M7_mean_axial_roll_deg=float(roll[:8].mean()),rear_M13_M20_mean_axial_roll_deg=float(roll[13:21].mean()),
                contact_modules=[np.flatnonzero(f[i,j]>.1).tolist() for j in range(2)],
                actual_joint_deg=np.degrees(a['q'][i]),target_joint_deg=np.degrees(a['ctrl'][i]),
                preload_angle_difference_deg=np.degrees(a['preload_angle_difference_rad'][i])))
        support=[]
        for pipe in range(2):
            for module in range(21):
                good=f[:,pipe,module]>.1
                if np.any(good):
                    support.append(dict(pipe=['source','target'][pipe],module=module,
                        sampled_contact_intervals_s=intervals(t,good),
                        maximum_normal_force_N=float(f[:,pipe,module].max()),
                        sampled_fraction=float(good.mean())))
        target_new=sorted(set(np.flatnonzero(np.any(f[:,1]>.1,axis=0)))-set(initial_contacts[1]))
        region=(np.hypot(c[:,:,0]-1100.,c[:,:,2]-1000.)<=92.)&(c[:,:,1]>=60.)
        newly_in_region=[k for k in range(8,21) if not region[0,k] and np.any(region[:,k])]
        stop_motion=c[t>=2.-1e-9] if t[-1]>=2 else c
        qerr=np.degrees(a['preload_angle_difference_rad'])
        results[name]=dict(physical_seconds=float(t[-1]),raw_frames=len(t),snapshots=snapshots,
            initial_contact_modules=initial_contacts,contact_history=support,new_target_effective_contact_modules=target_new,
            new_M8plus_in_target_region=newly_in_region,
            minimum_source_target_effective_contacts=counts.min(axis=0),
            maximum_module_speed_mm_s=float(a['maximum_module_speed_mm_s'].max()),
            maximum_actual_joint_deg=float(np.degrees(abs(a['q'])).max()),
            maximum_pipe_penetration_mm=float(a['pipe_penetration_mm'].max()),
            ground_frames=int(a['ground'].sum()),nonadjacent_self_contact_frames=int(a['self_contact'].sum()),
            maximum_motor_torque_Nm=float(abs(a['torque']).max()),
            maximum_preload_angle_difference_deg=float(abs(qerr).max()),
            maximum_added_preload_difference_relative_initial_deg=float(abs(qerr-qerr[0]).max()),
            maximum_external_applied_force=float(abs(a['xfrc_applied']).max()),
            maximum_generalized_applied_force=float(abs(a['qfrc_applied']).max()),
            max_post2s_module_displacement_from_t2_mm=float(np.linalg.norm(stop_motion-stop_motion[0],axis=2).max()))
    comparison={}
    if len(traces)==2:
        c0,p1=traces['C0_HOLD'],traces['P1_PREFIX']
        assert np.array_equal(c0['integration_state'][0],p1['integration_state'][0])
        assert np.array_equal(c0['t_s'],p1['t_s'])
        with np.load(BASE/'entry_prefix_path.npz') as a:
            frozen_t,frozen_q=a['dispatch_time_s'].copy(),a['dispatch_ctrl_rad'].copy()
        commands=json.loads((OUT/'P1_PREFIX/commands.json').read_text(encoding='utf8'))
        applied=np.array([c['target_rad'] for c in commands])
        expected=np.vstack([frozen_q[min(i,40)] for i in range(240)])
        assert np.array_equal(applied,expected)
        comparison.update(initial_full_state_max_difference=float(abs(c0['integration_state'][0]-p1['integration_state'][0]).max()),
            full_frozen_command_max_difference_rad=float(abs(applied-expected).max()),
            first_command_is_original_ctrl=bool(np.array_equal(applied[0],c0['ctrl'][0])),
            final_target_first_dispatch_t_s=commands[40]['t_s'],commands=len(commands),
            command_maximum_rate_deg_s=float(np.degrees(abs(np.diff(applied,axis=0))).max()/.05),
            source_of_comparison='Independent runs from identical saved full integration state',snapshot_differences=[])
        for moment in (2.,12.):
            i=int(abs(c0['t_s']-moment).argmin())
            delta=(p1['centers_mm'][i]-p1['centers_mm'][0])-(c0['centers_mm'][i]-c0['centers_mm'][0])
            roll=p1['axial_roll_deg'][i]-c0['axial_roll_deg'][i]
            comparison['snapshot_differences'].append(dict(t_s=float(c0['t_s'][i]),module_displacement_difference_xyz_mm=delta,
                module_target_y_difference_mm=delta[:,1],module_source_x_difference_mm=delta[:,0],module_axial_roll_difference_deg=roll,
                front_mean_y_difference_mm=float(delta[:8,1].mean()),rear_mean_x_difference_mm=float(delta[13:21,0].mean()),
                front_mean_axial_roll_difference_deg=float(roll[:8].mean()),rear_mean_axial_roll_difference_deg=float(roll[13:21].mean())))
        with np.load(BASE/'entry_prefix_path.npz') as a:
            predicted_joints=a['predicted_shape_rad'][-1].copy()
        predicted=json.loads((BASE/'entry_prefix_result.json').read_text(encoding='utf8'))['last_prediction']
        actual=results['P1_PREFIX']['snapshots'][0]
        comparison['prediction_vs_actual_t2']=dict(
            predicted_front_mean_y_mm=predicted['front_all_y_mm'],actual_front_mean_y_mm=actual['front_M0_M7_mean_y_mm'],
            predicted_rear_mean_x_mm=predicted['rear_all_x_mm'],actual_rear_mean_x_mm=actual['rear_M13_M20_mean_x_mm'],
            predicted_front_mean_roll_deg=predicted['front_mean_roll_deg'],actual_front_mean_roll_deg=actual['front_M0_M7_mean_axial_roll_deg'],
            predicted_rear_mean_roll_deg=predicted['rear_mean_roll_deg'],actual_rear_mean_roll_deg=actual['rear_M13_M20_mean_axial_roll_deg'],
            actual_minus_predicted_joint_deg=actual['actual_joint_deg']-np.degrees(predicted_joints),
            predicted_module_axial_roll_deg=predicted['predicted_axial_roll_fk_deg'],actual_module_axial_roll_deg=actual['module_axial_self_rotation_deg'],
            note='t=2 state is sampled before dispatching the final t=2 target, as registered; no reference time shift')
    write(OUT/'diagnostics.json',dict(physics_steps_added_by_analysis=0,results=results,comparison=comparison,
        limitation='No repeated dynamics, no added feedback, no independent front load-bearing or cross-pipe success claim'))
    compact={name:{k:value for k,value in r.items() if k not in ['contact_history','snapshots']} for name,r in results.items()}
    for name,r in results.items():
        compact[name]['snapshot_summary']=[{k:value for k,value in s.items() if k.startswith(('front_','rear_')) or k in ['t_s','contact_modules']} for s in r['snapshots']]
    compact['comparison']={k:v for k,v in comparison.items() if k not in ['snapshot_differences','prediction_vs_actual_t2']}
    compact['difference_summaries']=[{k:v for k,v in x.items() if k.startswith(('front_','rear_')) or k=='t_s'} for x in comparison.get('snapshot_differences',[])]
    print(json.dumps(compact,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),flush=True)


if __name__=='__main__': run()
