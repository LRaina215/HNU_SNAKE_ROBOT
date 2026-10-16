"""Saved-trace Stage B outcomes and feedback costs; never runs dynamics."""
from physics import *
from analyze_curve import analyze, stable
import argparse, json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def longest(mask,t):
    best=0.;start=None
    for i,v in enumerate(mask):
        if v and start is None:start=i
        if not v and start is not None:
            best=max(best,t[i-1]-t[start]+.1);start=None
    if start is not None:best=max(best,t[-1]-t[start]+.1)
    return float(best)


def evaluate(case):
    if case.name!='A16':analyze(case)
    plan=json.loads((case/'plan.json').read_text())
    result=json.loads((case/'result.json').read_text())
    audit=json.loads((case/'audit.json').read_text())
    z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];f=z['forces']
    feedback=bool(plan.get('high_level_feedback',False))
    radius=np.linalg.norm(c[:,:,[0,2]]-[1.1,1.],axis=2)
    full=np.all((radius>.07)&(radius<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2),axis=1)&(f[:,0].sum(1)<.1)&(f[:,1].sum(1)>.2)
    false=np.flatnonzero(~full);suffix=int(false[-1]+1) if len(false) else 0
    m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m)
    spin=np.zeros(21);times=[];rates=[]
    saved_qpos=z['qpos'];saved_qvel=z['qvel'];saved_points=z['points']
    if suffix<len(t) and case.name!='A16':
        selected=list(range(suffix,len(t),5))
        if selected[-1]!=len(t)-1:selected.append(len(t)-1)
        for i in selected:
            d.qpos[:]=saved_qpos[i];d.qvel[:]=saved_qvel[i];mujoco.mj_forward(m,d)
            tangents=saved_points[i,:-1]-saved_points[i,1:]
            tangents/=np.linalg.norm(tangents,axis=1)[:,None]
            rate=[]
            for n in range(21):
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('head').id+n,velocity,0)
                rate.append(float(velocity[:3]@tangents[n]))
            times.append(t[i]);rates.append(rate)
        if len(times)>1:spin=np.rad2deg(np.trapz(rates,x=times,axis=0))
    if case.name=='A16':
        supervisor=json.loads((BASE/'SUPERVISION/A16_independent_outcome.json').read_text())
        spin=-np.array(supervisor['post_body_axial_spin_deg_from_saved_velocity']) # common head-directed tangent convention
    spin_ok=bool(np.all(abs(spin)>=90) and (np.all(spin>0) or np.all(spin<0)))
    full_success=bool(audit['physical_full_transfer_success'] and result['stop']=='duration' and spin_ok)
    torque=z['torque'];sat=abs(torque)>=2.355-1e-5
    period=(t>=20)&(t<=110)
    m0=f[:,1,0]>.1
    intervals=[];start=None
    for i,on in enumerate(m0):
        if on and start is None:start=i
        if start is not None and (not on or i==len(t)-1):
            end=i-1 if not on else i
            intervals.append([float(t[start]),float(t[end])]);start=None
    firstgap=intervals[1][0]-intervals[0][1] if len(intervals)>1 else None
    timing=np.array([30.,40.,50.,60.,70.,80.,100.])
    direction=[dict(t=float(t[i]),M0_center=c[i,0].tolist(),M0_target_N=float(f[i,1,0])) for i in [int(np.argmin(abs(t-v))) for v in timing if v<=t[-1]]]
    s_equal=None;prefix_equal=None;prefix_command_equal=None;global_guide_difference=None
    baseline=np.load(BASE/'A16/trace.npz')
    if len(t)==len(baseline['t']):s_equal=bool(np.array_equal(z['S'],baseline['S']) and np.array_equal(z['chi'],baseline['chi']))
    if feedback:
        active=np.flatnonzero(abs(z['compliance_delta_m'])>1e-14)
        stop=int(active[0]) if len(active) else len(t)
        overlap=min(stop,len(baseline['t']))
        prefix_equal=float(np.max(abs(z['qpos'][:overlap]-baseline['qpos'][:overlap]))) if overlap else 0.
        prefix_command_equal=bool(np.array_equal(z['raw_command'][:overlap],baseline['raw_command'][:overlap]))
        if len(t)==len(baseline['t']):
            global_guide_difference=float(1000*np.max(np.linalg.norm(z['global_reference_points']-baseline['reference_points'],axis=2)))
    metrics=dict(case=case.name,high_level_feedback=feedback,full_transfer_and_rolling_success=full_success,
        physical_full_transfer_success=bool(audit['physical_full_transfer_success']),
        direction_visual_review='pending',stop=result['stop'],sim_time_s=result['sim_time'],
        front_stable_time_s=audit['front_stable_time'],full_stable_time_s=audit['full_stable_time'],
        final_full_target_s=audit['final_full_target_seconds'],
        minimum_each_module_post_Y_mm=float(min(audit['module_post_y_mm'])),
        post_each_rod_axial_spin_deg=spin.tolist(),post_minimum_abs_rod_spin_deg=float(min(abs(spin))),
        spin_all_same_direction=bool(np.all(spin>0) or np.all(spin<0)),
        end_axis_angle_deg=audit['end_axis_angle_deg'],
        M0_direct_target_contact_intervals_s=intervals,
        M0_first_contact_break_to_next_s=firstgap,
        front6_zero_direct_target_contact_time_20_110_s=float(.1*np.sum((f[:,1,:6].sum(1)<=.1)&period)),
        front6_longest_zero_direct_target_contact_20_110_s=longest((f[:,1,:6].sum(1)<=.1)&period,t),
        sampled_torque_saturated_joint_time_s=float(.1*sat.sum()),
        sampled_torque_saturated_fraction=float(sat.mean()),
        longest_any_joint_torque_saturation_s=longest(sat.any(1),t),
        sampled_raw_command_limit_joint_time_s=float(.1*np.sum(abs(z['raw_command'])>np.radians(88))),
        max_actual_joint_deg=result['max_actual_joint_deg'],max_torque_Nm=result['max_torque_nm'],
        max_depth_mm=result['maximum_depth_mm'],model_equal_A16=(case/'model.xml').read_bytes()==(BASE/'A16/model.xml').read_bytes(),
        initial_equal_A16=all(np.array_equal(np.load(case/'initial.npz')[key],np.load(BASE/'A16/initial.npz')[key]) for key in ['qpos','qvel','ctrl']),
        applied_pulse_steps=result.get('applied_pulse_steps',0),
        applied_pulse_impulse_N_s=result.get('applied_pulse_impulse_N_s',0.),
        same_reference_S_chi_as_A16=s_equal,zero_feedback_prefix_qpos_max_difference=prefix_equal,
        zero_feedback_prefix_raw_commands_equal=prefix_command_equal,
        nominal_global_guide_max_difference_from_A16_mm=global_guide_difference,
        runtime=plan.get('runtime',{'source':'frozen A16 README: MuJoCo3.2.7 NumPy1.23.5 SciPy1.13.1; threads1'}),
        eligible_feedback_comparison=case.name=='A16' or plan.get('runtime',{}).get('mujoco')=='3.2.7',
        direction_samples=direction,scope='finite paired simulation; saturation durations from0.1s samples; spin saved-state velocity integration, no dynamics')
    if feedback:
        metrics.update(compliance_max_abs_delta_mm=float(1000*np.max(abs(z['compliance_delta_m']))),
            compliance_amplitude_saturated_time_s=float(.1*z['compliance_amplitude_saturated'].sum()),
            compliance_rate_saturated_time_s=float(.1*z['compliance_rate_saturated'].sum()),
            compliance_actual_max_rate_mm_s=float(1000*np.max(abs(z['compliance_rate_m_s']))))
    if not feedback and plan.get('disturbance',{}).get('world_force_N',[0])[0]:
        before=t<72.
        metrics['pre_disturbance_qpos_max_difference_from_A16']=float(np.max(abs(z['qpos'][before]-baseline['qpos'][before])))
    (case/'b_metrics.json').write_text(json.dumps(metrics,indent=2))
    if case.name!='A16':
        audit.update(high_level_feedback=feedback,continued_rolling_verified=spin_ok,
            post_each_rod_axial_spin_deg=spin.tolist(),full_transfer_and_rolling_success=full_success)
        (case/'audit.json').write_text(json.dumps(audit,indent=2))
        result.update(success=full_success,acceptance_status='saved_trace_evaluated_direction_visual_review_pending',verdict_source='audit.json and b_metrics.json')
        (case/'result.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for n in range(6):ax[0].plot(t,radius[:,n]*1000,label=f'M{n}',lw=.7)
    ax[0].axhline(92.5,color='black',ls='--');ax[0].set(ylabel='actual radius(mm)');ax[0].legend(ncol=6)
    ax[1].plot(t,f[:,1,:6].sum(1),label='front6 target');ax[1].plot(t,f[:,0].sum(1),label='all source');ax[1].set(ylabel='normal load(N)');ax[1].legend()
    if feedback:
        ax[2].plot(t,z['compliance_delta_m']*1000,label='local normal coordinate(mm)')
        ax[2].plot(t,z['compliance_gap_m']*1000,label='filtered radius error(mm)',alpha=.6)
        ax[2].plot(t,z['compliance_mean_load_N'],label='filtered mean load(N)',alpha=.6)
    ax[2].set(xlabel='actual time(s)',ylabel='feedback')
    if feedback:ax[2].legend()
    if plan.get('disturbance',{}).get('world_force_N',[0])[0]:
        for a in ax:a.axvspan(72,72.5,color='orange',alpha=.35)
    fig.suptitle(case.name);fig.tight_layout();fig.savefig(case/'feedback_evidence.png',dpi=140);plt.close(fig)
    if feedback:
        g=np.load(case/'global_route.npz')['points'];fig=plt.figure(figsize=(14,8))
        for k,sec in enumerate([30,50,70,100,140,t[-1]]):
            i=int(np.argmin(abs(t-sec)));a=fig.add_subplot(2,3,k+1,projection='3d')
            a.plot(*g.T,color='gray',alpha=.25,label='fixed global G')
            a.plot(*z['global_reference_points'][i].T,color='green',label='nominal G window')
            a.plot(*z['reference_points'][i].T,color='blue',label='local curve')
            a.plot(*z['points'][i].T,'r.-',label='actual body')
            a.set(xlim=(.5,1.4),ylim=(-.25,1.15),zlim=(.3,1.4),title=f'{t[i]:.1f}s delta={z["compliance_delta_m"][i]*1000:.2f}mm')
            a.view_init(22,140)
            if k==0:a.legend(fontsize=7)
        fig.tight_layout();fig.savefig(case/'global_local_actual.png',dpi=140);plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ['case','full_transfer_and_rolling_success','full_stable_time_s','final_full_target_s','minimum_each_module_post_Y_mm','post_minimum_abs_rod_spin_deg','max_depth_mm','zero_feedback_prefix_qpos_max_difference']},indent=2))
    return metrics


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',required=True);args=ap.parse_args()
    outputs=[evaluate(BASE/name) for name in args.cases]
    keys=['case','full_transfer_and_rolling_success','stop','sim_time_s','front_stable_time_s','full_stable_time_s','final_full_target_s','minimum_each_module_post_Y_mm','post_minimum_abs_rod_spin_deg','M0_first_contact_break_to_next_s','front6_zero_direct_target_contact_time_20_110_s','sampled_torque_saturated_joint_time_s','longest_any_joint_torque_saturation_s','max_depth_mm','compliance_amplitude_saturated_time_s','compliance_rate_saturated_time_s']
    with (BASE/'B_comparison.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader()
        for row in outputs:writer.writerow({k:row.get(k) for k in keys})
    (BASE/'B_comparison.json').write_text(json.dumps(outputs,indent=2))
