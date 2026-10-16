"""Final paired saved-data checks and common-window comparison; no physics steps."""
from physics import *
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_step2 import scene_truth,read_lines

reg=json.loads((BASE/'REGISTERED.json').read_text());names=[x['case'] for x in reg['cases']]
rows=[json.loads((BASE/name/'metrics.json').read_text()) for name in names];pairs=[];artifacts=[]
for frozen,updated in [('T_FROZEN','T_UPDATE'),('P_FROZEN','P_UPDATE')]:
    a=np.load(BASE/frozen/'trace.npz');b=np.load(BASE/updated/'trace.npz');ma=next(r for r in rows if r['case']==frozen);mb=next(r for r in rows if r['case']==updated)
    until=mb['first_G_write_t'];n=min(len(a['t']),len(b['t']));mask=np.ones(n,bool) if until is None else a['t'][:n]<until-1e-6
    keys=['qpos','qvel','ctrl','raw_command','centers','S','probe_curvature_per_m','estimated_entry_m','estimated_axis','estimated_radius_m']
    difference={k:float(np.max(abs(a[k][:n][mask]-b[k][:n][mask]))) for k in keys}
    qa=np.flatnonzero(np.max(abs(a['qpos'][:n]-b['qpos'][:n]),axis=1)>1e-13)
    command=np.flatnonzero(np.max(abs(a['raw_command'][:n]-b['raw_command'][:n]),axis=1)>1e-13)
    cfg_a=json.loads((BASE/frozen/'plan.json').read_text());cfg_b=json.loads((BASE/updated/'plan.json').read_text())
    check=dict(frozen=frozen,updated=updated,only_write_permission_differs=cfg_a['initial_prior']==cfg_b['initial_prior'] and cfg_a['registration']==cfg_b['registration'] and cfg_a['initial_reference_S']==cfg_b['initial_reference_S'],
        original_model_pair_equal=(BASE/frozen/'model.xml').read_bytes()==(BASE/updated/'model.xml').read_bytes(),full_given_initial_pair_equal=(BASE/frozen/'given_initial.npz').read_bytes()==(BASE/updated/'given_initial.npz').read_bytes(),
        before_first_G_write_until_s=until,pre_write_max_differences=difference,pre_write_all_exact=all(v==0 for v in difference.values()),
        first_saved_raw_command_difference_t=None if not len(command) else float(a['t'][command[0]]),first_saved_actual_state_difference_t=None if not len(qa) else float(a['t'][qa[0]]),
        entire_final_S_max_difference=float(np.max(abs(a['S'][:n]-b['S'][:n]))),probe_parameter_max_difference=float(np.max(abs(a['probe_curvature_per_m'][:n]-b['probe_curvature_per_m'][:n]))),
        frozen_G_parameter_max_differences={k:float(np.max(abs(a[k]-a[k][0]))) for k in ['applied_global_entry_m','applied_global_axis','applied_global_radius_m']},
        completion_change_s=None if ma['full_stable_time_s'] is None or mb['full_stable_time_s'] is None else mb['full_stable_time_s']-ma['full_stable_time_s'],
        source_peak_change_mm=mb['source_target_self_floor_max_depth_mm'][0]-ma['source_target_self_floor_max_depth_mm'][0],target_peak_change_mm=mb['source_target_self_floor_max_depth_mm'][1]-ma['source_target_self_floor_max_depth_mm'][1],
        saturated_joint_time_change_s=mb['torque_saturated_joint_time_s']-ma['torque_saturated_joint_time_s'])
    pairs.append(check)
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for name,style in [(frozen,'--'),(updated,'-')]:
        z=np.load(BASE/name/'trace.npz');e=np.load(BASE/name/'geometry_evaluation.npz');t=z['t']
        axs[0].plot(t,e['G_axis_error_deg'],style,label=name+' G axis error');axs[0].plot(t,e['G_centerline_error_mm'],style,label=name+' G centerline mm',alpha=.6)
        axs[1].plot(t,z['centers'][:,20,1],style,label=name+' M20 actualY');axs[1].plot(t,z['centers'][:,0,1],style,label=name+' M0 actualY',alpha=.6)
        axs[2].plot(t,z['depth'][:,0]*1000,style,label=name+' source intrusion');axs[2].plot(t,z['depth'][:,1]*1000,style,label=name+' target intrusion',alpha=.6)
    for ax in axs:ax.legend(ncol=2);ax.axvline(until or 0,color='gray',ls=':')
    axs[2].set_xlabel('elapsed physical time(s)');fig.tight_layout();fig.savefig(BASE/(updated+'_paired_effect.png'),dpi=140);plt.close(fig)

common=[]
for name in names:
    case=BASE/name;z=np.load(case/'trace.npz');t=z['t'];m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);truth=scene_truth(m,d)
    selected=np.flatnonzero((t>=150.-1e-6)&(t<=234.9+1e-6))
    if len(selected) and t[-1]>234.8:
        first,last=int(selected[0]),int(selected[-1]);y=(z['centers'][last]-z['centers'][first])@truth['axis']*1000
        common.append(dict(case=name,window_s=[float(t[first]),float(t[last])],each_axial_mm=y.tolist(),minimum_axial_mm=float(y.min()),mean_axial_mm=float(y.mean())))
    video=json.loads((case/'video_metadata.json').read_text());frozen_files={key:(case/key).read_bytes()==(BASE/reg['freeze_directory']/key).read_bytes() for key in ['step2_runner.py','contact_sensor.py','cylinder_estimator.py','global_update.py','estimated_curve.py','estimated_observer.py','source_mode.py','native_mapping.py','REGISTERED.json']}
    artifacts.append(dict(case=name,frozen_files_equal=frozen_files,video=video,full_initial_equal=(case/'given_initial.npz').read_bytes()==(BASE/'GIVEN_INITIAL/integration_state.npz').read_bytes(),original_model_equal=(case/'model.xml').read_bytes()==MODEL.read_bytes()))
out=dict(results=rows,pair_checks=pairs,artifact_checks=artifacts,common_window=common,superseded_starts=json.loads((BASE/'R1_SUPERSEDED_STARTS/EXCLUSIONS.json').read_text()),
    total_dynamics_starts=8,final_corrected_comparisons=4,controller_inputs_unknown_geometry=True,source_truth_is_known=True,precontact_search_implemented=False,initial_autonomous_formation=False,unique_probe_contribution_identified=False,step1_speed_limitations_resolved=False,next_stage_authorized=False)
(BASE/'COMPARISON.json').write_text(json.dumps(out,indent=2))
columns=['case','condition','write_global','action_success','stop','sim_time_s','first_contact_t','first_reliable_geometry_t','first_G_write_t','full_stable_time_s','final_full_target_s','minimum_each_post_axial_mm','minimum_each_abs_post_spin_deg','final_estimated_axis_error_deg','final_estimated_centerline_error_mm','final_estimated_radius_error_mm','final_G_axis_error_deg','final_G_centerline_error_mm','final_G_radius_error_mm','torque_saturated_joint_time_s','longest_single_joint_saturation_s','source_target_self_floor_max_depth_mm','G_write_count']
with (BASE/'COMPARISON.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=columns);w.writeheader()
    for row in rows:w.writerow({k:row.get(k) for k in columns})
print(json.dumps(dict(actions=[(r['case'],r['action_success']) for r in rows],pairs=pairs,common=common),indent=2))
