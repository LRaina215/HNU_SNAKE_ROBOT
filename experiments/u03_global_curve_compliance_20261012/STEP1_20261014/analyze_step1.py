"""Stage1 saved-state outcomes and separated speed/shape contribution."""
from physics import *
from basic_analysis import evaluate
import argparse,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',required=True);args=ap.parse_args()
outputs=[]
for name in args.cases:
    basic=evaluate(BASE/name);case=BASE/name;z=np.load(case/'trace.npz');t=z['t'];plan=json.loads((case/'plan.json').read_text());force=z['applied_load_force_N'];result=json.loads((case/'result.json').read_text())
    ratio=z['shared_speed_ratio'];gamma=z['source_gamma_per_m'];direction=z['source_curve_direction'];coeff=gamma[:,None]*direction
    missing=z['material_target_expected_unobserved_m'];progress=z['material_progress_observed_m'];chart=z['material_chart'];roles=z['actual_material_role']
    dt=float(np.median(np.diff(t)));source_active=z['source_mode_active'];curve_error=z['local_curve_chord_error_mm']
    baseline=np.load(BASE.parent/'A16/trace.npz');pre=t<32.7
    if not plan['parameters']['compliance']:pre=t<94.
    n=int(pre.sum());expectedload=plan['disturbance'];expected_steps=int(round((expectedload['interval_s'][1]-expectedload['interval_s'][0])/.001)) if np.linalg.norm(expectedload['world_force_N']) else 0
    metrics=dict(basic,controller=dict(speed=plan['parameters']['speed_feedback'],shape=plan['parameters']['compliance']),condition=plan['parameters']['condition'],
        minimum_speed_ratio=float(ratio.min()),mean_speed_ratio=float(np.mean(ratio[t>=.5])),
        speed_below_point99_time_s=float(dt*np.sum(ratio<.99)),speed_below_point01_time_s=float(dt*np.sum(ratio<.01)),
        final_S=float(z['S'][-1]),lost_nominal_S=float(z['lost_nominal_S'][-1]),
        source_gamma_max_per_m=float(gamma.max()),source_gamma_rate_max_per_m_s=float(np.max(abs(z['source_gamma_rate_per_m_s']))),
        source_coefficient_rate_sampled_max_per_m_s=float(np.max(np.linalg.norm(np.diff(coeff,axis=0),axis=1)/np.diff(t))),
        source_mode_active_time_s=float(dt*np.sum(source_active)),local_curve_chord_error_max_mm=float(curve_error.max()),
        receiver_delta_range_mm=[float(z['compliance_delta_m'].min()*1000),float(z['compliance_delta_m'].max()*1000)],
        sourceM20_supported_time_s=float(dt*np.sum(z['actual_source_support'][:,20])),
        max_matched_target_lead_m=float(z['target_progress_lead_m'].max()),
        max_post_anchor_unobserved_prediction_m=float(missing.max()),
        max_all_chart_unobserved_time_s=float(z['material_unobserved_time_s'][-1].max()),
        root_pose_or_joint_physics_reset=False,
        prefix_own_A16_zero_feedback_qpos_difference=float(np.max(abs(z['qpos'][:n]-baseline['qpos'][:n]))),
        actual_load_steps=result['applied_load_steps'],expected_load_steps=expected_steps,
        applied_load_impulse_N_s=result['applied_load_impulse_N_s'],
        direction_visual_review='pending',world_curve_exact_tracking_claim=False,
        observer_scope='geometry-qualified axis charts and matched time masks; unknown separately retained, cumulative history never reset')
    assert metrics['actual_load_steps']==expected_steps,(name,metrics['actual_load_steps'],expected_steps)
    (case/'step1_metrics.json').write_text(json.dumps(metrics,indent=2));outputs.append(metrics)
    fig,axs=plt.subplots(5,1,figsize=(12,13),sharex=True)
    axs[0].plot(t,z['S'],label='S actual reference clock');axs[0].plot(t,z['nominal_time_S'],label='nominal timeS',ls='--');axs[0].legend();axs[0].set_ylabel('S(cells)')
    axs[1].plot(t,ratio,label='shared speed');axs[1].plot(t,z['target_progress_lead_m'],label='matched lead(m)');axs[1].legend()
    axs[2].plot(t,z['compliance_delta_m']*1000,label='receiving delta(mm)');axs[2].plot(t,gamma,label='source gamma(1/m)');axs[2].legend()
    axs[3].plot(t,z['forces'][:,0,20],label='M20 actual sourceN');axs[3].plot(t,z['forces'][:,1].sum(1),label='all targetN');axs[3].legend();axs[3].set_ylabel('normal(N)')
    for i in [0,6,13,20]:axs[4].plot(t,progress[:,i],label=f'M{i} observed chart history')
    axs[4].legend(ncol=2);axs[4].set_ylabel('progress(m)');axs[4].set_xlabel('physical time(s)')
    interval=expectedload['interval_s']
    if expected_steps:
        for ax in axs:ax.axvspan(*interval,color='orange',alpha=.2)
    fig.suptitle(name);fig.tight_layout();fig.savefig(case/'step1_events.png',dpi=140);plt.close(fig)
    with (case/'material_observer.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['t','material','actual_role','chart','observed_progress_m','valid','expected_matched_m','observed_matched_m','unobserved_prediction_m'])
        valid=z['material_progress_valid'];em=z['material_target_expected_matched_m'];om=z['material_target_observed_matched_m']
        for i,sec in enumerate(t):
            for material in range(21):w.writerow([sec,material,roles[i,material],chart[i,material],progress[i,material],valid[i,material],em[i,material],om[i,material],missing[i,material]])
    print(json.dumps({k:metrics[k] for k in ['case','condition','full_transfer_and_rolling_success','full_stable_time_s','minimum_speed_ratio','mean_speed_ratio','source_gamma_max_per_m','local_curve_chord_error_max_mm','max_depth_mm']},indent=2),flush=True)
(BASE/'latest_analysis.json').write_text(json.dumps(outputs,indent=2))
