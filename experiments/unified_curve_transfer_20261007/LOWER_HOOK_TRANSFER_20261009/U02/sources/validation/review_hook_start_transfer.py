"""Read-only physical-state and whole-body review for saved-hook trials."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.suspended_supply_transfer import nodes
from snake_control.hook_pull_review_support import restore_saved_state
from validation.suspended_supply_experiment import write
from validation.hook_start_transfer_experiment import shape_stats
BASE=ROOT/'experiments/unified_curve_transfer_20261007/HOOK_START_TRANSFER_20261009'

def review(case,lower_entry=False):
    sign=-1 if lower_entry else 1;model_name='Snake_Robot_o20_lower_hook_transfer.xml' if lower_entry else 'Snake_Robot_o20_hook_start_transfer.xml'
    z=np.load(case/'trace.npz');t=z['sim_time'];m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model'/model_name));d=mujoco.MjData(m)
    restore_saved_state(m,d,case/('initial_lower_state.npz' if lower_entry else 'initial_saved_state.npz'));initial=dict(qpos=float(abs(d.qpos-z['qpos'][0]).max()),qvel=float(abs(d.qvel-z['qvel'][0]).max()),ctrl=float(abs(d.ctrl-z['ctrl'][0]).max()))
    ks=np.flatnonzero(t>=t[-1]-1.00001);rows=[];error=0.
    for i in ks:
        d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);error=max(error,float(abs(nodes(d)*1000-z['points_mm'][i]).max()))
        row={k:z[k][i] for k in ['centers_mm','contact_force_N']};stats=shape_stats(row,sign)
        rows.append(dict(t_s=t[i],source_absent=bool(not z['source_contact_any'][i].any() and z['contact_force_N'][i,0].sum()<.1),all_kp127=bool((z['kp_nm_per_rad'][i]>=126.999).all()),posterior_bearing=bool(z['contact_force_N'][i,1,18:].sum()>.2),shift_mm=z['shift_mm'][i],**stats))
    plan=json.loads((case/'plan.json').read_text());forces=z['contact_force_N'].sum(2);sourceclear=(~z['source_contact_any'].any(1))&(forces[:,0]<.1)
    edges=np.diff(np.r_[False,sourceclear,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
    intervals=[dict(start_s=t[a],end_s=t[b],duration_s=t[b]-t[a]) for a,b in zip(starts,ends) if t[b]-t[a]>=.5]
    final=rows[-1];verified=all(r['strict_actual_shape'] and r['source_absent'] and r['all_kp127'] and r['posterior_bearing'] and r['shift_mm']>=plan['complete_shift_mm']-1e-6 for r in rows)
    actual_reference_rms=np.sqrt(np.mean(np.sum((z['points_mm']-z['reference_points_mm'])**2,axis=2),axis=1))
    report=dict(case=case.name,read_only=True,physical_steps=0,initial_saved_state_differences=initial,initial_saved_state_preserved=all(x==0 for x in initial.values()),
        last_second_samples=len(ks),last_second_fk_max_point_difference_mm=error,strict_whole_transfer_verified=verified,last_second=rows,
        source_clear_intervals_at_least_half_second=intervals,final_source_contacts=np.flatnonzero(z['source_contact_any'][-1]),
        initial_front8_wrap_deg=float(np.degrees(z['actual_front_span_rad'][0])),final_front8_wrap_deg=float(np.degrees(z['actual_front_span_rad'][-1])),
        final_head_axis_advance_mm=float(z['centers_mm'][-1,0,1]-z['centers_mm'][0,0,1]),final_whole_world_reference_rms_mm=float(actual_reference_rms[-1]),
        final_center_radius_mm=final['center_radius_mm'],last_second_pitch_range_mm=[min(r['pitch_mm'] for r in rows),max(r['pitch_mm'] for r in rows)],
        last_second_axis_rms_max_mm=max(r['axis_rms_mm'] for r in rows),last_second_tail_radius_range_mm=[min(r['center_radius_mm'][-1] for r in rows),max(r['center_radius_mm'][-1] for r in rows)],
        last_second_tail_direct_force_range_n=[min(r['tail_direct_force_n'] for r in rows),max(r['tail_direct_force_n'] for r in rows)],
        maximum_actual_joint_deg=float(np.degrees(abs(z['q']).max())),maximum_drive_torque_nm=float(abs(z['torque']).max()),maximum_pipe_depth_mm=float(z['pipe_depth_mm'].max()),
        ground=bool(z['ground'].any()),yamada_compliance_enabled=False)
    write(case/'actual_evidence_review.json',report)
    if lower_entry:
        report.update(initial_state_is_artificial=True,active_upward_hook_acquisition_demonstrated=False,actual_winding_sign=-1,initial_state_file='initial_lower_state.npz')
        write(case/'actual_evidence_review.json',report)
    depth=z['pipe_depth_mm'];imax=int(depth.argmax());d.qpos[:]=z['qpos'][imax];mujoco.mj_fwdPosition(m,d)
    pairs=[dict(geom_names=[m.geom(int(g)).name for g in [con.geom1,con.geom2]],body_names=[m.body(int(m.geom_bodyid[g])).name for g in [con.geom1,con.geom2]],depth_mm=float(-con.dist*1000),position_mm=con.pos*1000) for con in d.contact if -con.dist*1000>3]
    e=np.diff(np.r_[False,depth>3,False].astype(int));a=np.flatnonzero(e==1);b=np.flatnonzero(e==-1)-1
    deep_intervals=[dict(first_above3_sample_s=t[x],last_above3_sample_s=t[y],sample_span_s=t[y]-t[x],sample_count=int(y-x+1),estimated_sampled_duration_s=(y-x+1)*.01,preceding_below3_sample_s=t[x-1] if x else None,following_below3_sample_s=t[y+1] if y+1<len(t) else None,max_depth_mm=depth[x:y+1].max()) for x,y in zip(a,b)]
    any_source=z['source_contact_any'].any(1);bearing_source=forces[:,0]>.1;c=z['centers_mm']
    wrap=lambda cc,n:float(np.degrees(np.ptp(np.unwrap(np.arctan2(cc[:n,2]-1000,cc[:n,0]-1100)))))
    write(case/'contact_and_wrap_review.json',dict(sample_period_s=.01,maximum_pipe_depth_mm=depth[imax],maximum_depth_time_s=t[imax],maximum_depth_contacts=pairs,
        above3mm_intervals=deep_intervals,largest_estimated_continuous_above3_duration_s=max((r['estimated_sampled_duration_s'] for r in deep_intervals),default=0),
        last_actual_source_contact_time_s=t[np.flatnonzero(any_source)[-1]] if any_source.any() else None,last_actual_source_bearing_time_s=t[np.flatnonzero(bearing_source)[-1]] if bearing_source.any() else None,
        initial_front8_wrap_deg=wrap(c[0],8),final_front8_wrap_deg=wrap(c[-1],8),initial_front9_wrap_deg=wrap(c[0],9),final_front9_wrap_deg=wrap(c[-1],9),final_all21_winding_deg=wrap(c[-1],21)))
    joints=[dict(joint=j+1,actual_angle_min_deg=float(np.degrees(z['q'][:,j].min())),actual_angle_max_deg=float(np.degrees(z['q'][:,j].max())),
        maximum_drive_torque_nm=float(abs(z['torque'][:,j]).max()),kp_min=float(z['kp_nm_per_rad'][:,j].min()),kp_final=float(z['kp_nm_per_rad'][-1,j]),
        maximum_passive_torque_nm=float(abs(z['passive_torque_nm'][:,j]).max()),original_damping=float(z['damping_nm_s_per_rad'][0,j])) for j in range(20)]
    write(case/'joint_region_dynamics_summary.json',dict(joints=joints))
    fig,axs=plt.subplots(2,2,figsize=(12,7));axs[0,0].plot(t,forces[:,0],label='Source');axs[0,0].plot(t,forces[:,1],label='Target');axs[0,0].set(ylabel='Actual bearing force(N)');axs[0,0].legend()
    axs[0,1].plot(t,np.degrees(z['actual_front_span_rad']),label='Actual front8 wrap');axs[0,1].set(ylabel='Degrees');axs[0,1].legend()
    axs[1,0].plot(t,z['shift_mm'],label='Shared reference shift');axs[1,0].plot(t,z['centers_mm'][:,0,1]-z['centers_mm'][0,0,1],label='Actual head axial advance');axs[1,0].set(ylabel='mm');axs[1,0].legend()
    axs[1,1].plot(t,actual_reference_rms,label='Actual world/reference RMS');axs[1,1].plot(t,z['pipe_depth_mm'],label='Actual penetration');axs[1,1].set(ylabel='mm');axs[1,1].legend()
    for ax in axs.ravel():ax.set_xlabel('Actual trial time(s)')
    fig.tight_layout();fig.savefig(case/'actual_feedback_and_reference_error.png',dpi=150);plt.close(fig)
    ids=[0,len(t)//2,len(t)-1];all_points=np.concatenate([z['points_mm'][ids].reshape(-1,3),z['reference_points_mm'][ids].reshape(-1,3)])/1000
    limits=[(min(low,all_points[:,j].min()-.08),max(high,all_points[:,j].max()+.08)) for j,(low,high) in enumerate([(.4,1.4),(-.25,1.7),(.4,1.2)])]
    fig=plt.figure(figsize=(16,6))
    for panel,i in enumerate(ids):
        ax=fig.add_subplot(1,3,panel+1,projection='3d');actual=z['points_mm'][i]/1000;ref=z['reference_points_mm'][i]/1000
        ax.plot([-.2,1.1],[0,0],[1,1],color='gray',lw=10,alpha=.25);ax.plot([1.1,1.1],[0,2],[1,1],color='gray',lw=10,alpha=.25)
        ax.plot(actual[:,0],actual[:,1],actual[:,2],'o-',ms=3,label='Actual FK');ax.plot(ref[:,0],ref[:,1],ref[:,2],'--',lw=1.4,label='Executed reference')
        ax.text(*actual[0],'Head N0',fontsize=8);ax.text(*actual[-1],'Tail N21',fontsize=8)
        ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)',title=f'Actual {t[i]:.2f}s / reference S={z["shift_mm"][i]:.1f}mm',xlim=limits[0],ylim=limits[1],zlim=limits[2]);ax.tick_params(labelsize=8,pad=0);ax.view_init(27,140);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(case/'actual_and_executed_reference.png',dpi=150);plt.close(fig)
    if lower_entry:
        windows=json.loads((case/'reference_initial_middle_final.json').read_text(encoding='utf-8'));fig=plt.figure(figsize=(16,6));marks=[]
        for panel,w in enumerate(windows):
            ax=fig.add_subplot(1,3,panel+1,projection='3d');p=np.asarray(w['reference_points_mm'])/1000;shift=float(w['material_coordinate_mm'][0])-float(windows[0]['material_coordinate_mm'][0]);reached=shift<=z['shift_mm'].max()+1e-6
            label=f'Planned reference S={shift:.1f}mm'+(' / initial seed' if panel==0 else ' / NOT REACHED' if not reached else ' / planned window')
            ax.plot([-.2,1.1],[0,0],[1,1],color='gray',lw=10,alpha=.25);ax.plot([1.1,1.1],[0,2],[1,1],color='gray',lw=10,alpha=.25);ax.plot(p[:,0],p[:,1],p[:,2],'o--',ms=3,color='darkgreen')
            ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)',title=label,xlim=(.4,1.4),ylim=(-.25,1.7),zlim=(.7,1.2));ax.tick_params(labelsize=8,pad=0);ax.view_init(27,140)
            marks.append(dict(planned_shift_mm=shift,reference_window_reached=reached,actual_whole_shape_not_inferred_from_reference=True))
        fig.tight_layout();fig.savefig(case/'reference_initial_middle_final.png',dpi=150);plt.close(fig)
        write(case/'reference_window_status.json',dict(maximum_actual_reference_shift_mm=float(z['shift_mm'].max()),windows=marks,whole_actual_transfer_verified=verified))
    print(json.dumps({k:v for k,v in report.items() if k!='last_second'},default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':
    lower_entry='--lower-entry' in sys.argv
    if lower_entry:BASE=ROOT/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009'
    for name in [x for x in sys.argv[1:] if x!='--lower-entry'] or (['U01'] if lower_entry else ['H01A','H02','H03']):
        if (BASE/name/'trace.npz').exists():review(BASE/name,lower_entry=lower_entry)
