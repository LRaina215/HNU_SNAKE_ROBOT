"""Independent trace review; no dynamics, resets, or new formal trials."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np,mujoco,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.suspended_supply_experiment import write
from validation.hook_start_transfer_experiment import shape_stats
from snake_control.suspended_supply_transfer import nodes

BASE=ROOT/'experiments/unified_curve_transfer_20261007/FULL_SOURCE_INNER_20261009'
MODEL='Snake_Robot_o20_inner_hook_transfer.xml'


def review(case):
    z=np.load(case/'trace.npz');t=z['sim_time'];p=z['points_mm']/1000;c=z['centers_mm']/1000;forces=z['contact_force_N'];commands=json.loads((case/'commands.json').read_text(encoding='utf-8'));result=json.loads((case/'result.json').read_text(encoding='utf-8'))
    model_path=case/'sources/mujoco_model'/MODEL
    if not model_path.exists():model_path=case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'
    m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
    ids=sorted(set([0,len(t)-1,int(np.argmax(p[:,0,0])),int(np.argmax(forces[:,1].sum(1)))]));fk_error=0.
    for i in ids:
        d.qpos[:]=z['qpos'][i];mujoco.mj_fwdPosition(m,d);fk_error=max(fk_error,float(np.max(np.linalg.norm(nodes(d)-p[i],axis=1))*1000))
    src_radius=np.linalg.norm(c[:,:,[1,2]]-[0,1],axis=2);target_radius=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2)
    geom_exit=(src_radius>.130)&(~z['source_contact_any'])&(forces[:,0]<.1)
    target_bearing=forces[:,1].sum(1)>.2
    first_bearing=float(t[np.flatnonzero(target_bearing)[0]]) if target_bearing.any() else None
    head_error=np.linalg.norm(p[:,0]-z['reference_points_mm'][:,0]/1000,axis=1)*1000
    source_phi=np.unwrap(np.arctan2(c[0,:,2]-1,c[0,:,1]));fit=np.polyfit(source_phi,c[0,:,0],1)
    projection=np.array([x['projection'].get('curve_fit_rms_mm',x['projection'].get('whole_fit_rms_mm',0)) for x in commands]);ct=np.array([x['t_s'] for x in commands])
    anchors=np.array([x['projection'].get('anchor_module',-1) for x in commands])
    actual_connected=z['actual_received_mask'].sum(1)
    kp=z['kp_nm_per_rad'];first_outside=np.flatnonzero((p[:,0,1]<0)&(t>0))
    audit=dict(name=case.name,physical_steps_in_review=0,outcome=result['outcome'],whole_transfer_complete=result['whole_transfer_complete'],active_acquisition_demonstrated=result['active_hook_acquisition_demonstrated'],actual_seconds=float(t[-1]),shift_mm=float(z['shift_mm'][-1]),first_target_bearing_s=first_bearing,maximum_target_bearing_n=float(forces[:,1].sum(1).max()),final_source_target_bearing_n=forces[-1].sum(1),maximum_actual_connected_target=int(actual_connected.max()),final_actual_connected_target=int(actual_connected[-1]),initial_source_center_radius_mm=src_radius[0]*1000,initial_source_wrap_deg=float(np.degrees(np.ptp(source_phi))),initial_source_pitch_mm=float(fit[0]*2*np.pi*1000),initial_source_axis_rms_mm=float(np.std(c[0,:,0]-np.polyval(fit,source_phi))*1000),initial_target_bearing_n=float(forces[0,1].sum()),initial_source_bearing_n=float(forces[0,0].sum()),actual_head_x_range_mm=[float(p[:,0,0].min()*1000),float(p[:,0,0].max()*1000)],actual_head_y_range_mm=[float(p[:,0,1].min()*1000),float(p[:,0,1].max()*1000)],first_actual_head_outside_positive_y_s=float(t[first_outside[0]]) if len(first_outside) else None,head_world_reference_error_max_mm=float(head_error.max()),head_world_reference_error_final_mm=float(head_error[-1]),private_projection_rms_max_mm=float(projection.max()),anchor_changes=int(np.count_nonzero(np.diff(anchors))),anchor_modules=np.unique(anchors).tolist(),minimum_actual_kp=float(kp.min()),zero_kp_joint_samples=int(np.count_nonzero(kp<.5)),maximum_actual_joint_deg=float(np.degrees(abs(z['q']).max())),maximum_drive_torque_nm=float(abs(z['torque']).max()),maximum_pipe_depth_mm=float(z['pipe_depth_mm'].max()),maximum_self_depth_mm=float(z['self_depth_mm'].max()),ground=bool(z['ground'].any()),frozen_model_fk_node_error_mm=fk_error,reviewed_actual_sample_ids=ids,final_geometry_source_exit_modules=np.flatnonzero(geom_exit[-1]),controller_final_exit_modules=np.flatnonzero(z['actual_exit_mask'][-1]),source_exit_definition_limitation='Controller exit is contact-neighborhood clearance; it can label a still nearby source rod. Radius>130mm is separately reviewed and neither count proves support acquisition.',last_progress_command_s=commands[-1]['last_actual_progress_s'],last_command_s=commands[-1]['t_s'],seconds_since_last_actual_progress=commands[-1]['t_s']-commands[-1]['last_actual_progress_s'],completion_hold_s=result['actual_completion_hold_seconds'],physical_root_reset_during_run=result['physical_root_reset_during_run'],yamada_compliance_enabled=result['yamada_compliance_enabled'])
    write(case/'actual_review.json',audit)
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    axes[0].plot(t,z['shift_mm'],label='Shared material shift');axes[0].plot(t,head_error,label='Actual head / world guide distance');axes[0].set(ylabel='mm');axes[0].legend()
    axes[1].plot(t,forces[:,0].sum(1),label='Actual source bearing');axes[1].plot(t,forces[:,1].sum(1),label='Actual target bearing');axes[1].set(ylabel='N');axes[1].legend()
    axes[2].plot(ct,projection,label='Private FK curve projection RMS');axes[2].plot(t,actual_connected,label='Actual connected target rods');axes[2].plot(t,kp.min(1),label='Minimum joint Kp');axes[2].set(xlabel='Actual dynamics time (s)',ylabel='mm / rods / Kp');axes[2].legend()
    fig.suptitle(f'{case.name}: {result["outcome"]}; actual acquisition={result["active_hook_acquisition_demonstrated"]}');fig.tight_layout();fig.savefig(case/'actual_progress_review.png',dpi=155);plt.close(fig)
    stage_ids=[0,int(np.argmax(p[:,0,0])),int(np.argmax(forces[:,1].sum(1))) if target_bearing.any() else int(len(t)//2),len(t)-1]
    labels=['Artificial all-source start','Farthest actual head X','Strongest actual target contact' if target_bearing.any() else 'Probe midpoint; target not reached','Actual stop']
    fig,axes=plt.subplots(2,4,figsize=(19,9));stage_data=[]
    for k,(i,label) in enumerate(zip(stage_ids,labels)):
        actual=p[i];reference=z['reference_points_mm'][i]/1000
        for row,(a,b) in enumerate([(0,1),(0,2)]):
            ax=axes[row,k]
            if row==0:
                ax.fill([-.2,1.1,1.1,-.2],[0,0,2,2],color='gold',alpha=.10);ax.plot([-.2,1.1],[0,0],color='gray',lw=12,alpha=.3);ax.plot([1.1,1.1],[0,2],color='gray',lw=12,alpha=.3)
            else:
                theta=np.linspace(0,2*np.pi,100);ax.fill(1.1+.0565*np.cos(theta),1+.0565*np.sin(theta),color='gray',alpha=.3);ax.axhline(1,color='gray',ls=':')
            ax.plot(actual[:,a],actual[:,b],'o-',ms=2.5,label='Actual body');ax.plot(reference[:,a],reference[:,b],'--',alpha=.7,label='Executed world guide');ax.scatter(actual[0,a],actual[0,b],color='crimson',label='Actual head',zorder=5)
            ax.set(xlabel='World X(m)',ylabel='World Y(m)' if row==0 else 'World Z UP(m)',title=f'{label}\nt={t[i]:.2f}s / S={z["shift_mm"][i]:.1f}mm',xlim=(.35,1.3),ylim=(-.35,.65) if row==0 else (0,1.3),aspect='equal');ax.legend(fontsize=6)
        stage_data.append(dict(label=label,t_s=float(t[i]),shift_mm=float(z['shift_mm'][i]),actual_points_mm=z['points_mm'][i],executed_reference_mm=z['reference_points_mm'][i],target_force_n=float(forces[i,1].sum()),actual_target_connected=int(actual_connected[i])))
    fig.suptitle('No stable acquisition in this trial: contact frames are not a successful hook stage');fig.tight_layout();fig.savefig(case/'actual_stage_comparison.png',dpi=150);plt.close(fig);write(case/'actual_stage_comparison.json',stage_data)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,(a,b) in zip(axes,[(0,1),(0,2)]):
        ax.plot(p[:,0,a],p[:,0,b],color='steelblue',label='Actual head path');ax.plot(z['reference_points_mm'][:,0,a]/1000,z['reference_points_mm'][:,0,b]/1000,'--',color='orange',label='Executed reference head');ax.scatter(p[0,0,a],p[0,0,b],c='green',label='Start');ax.scatter(p[-1,0,a],p[-1,0,b],c='red',label='Stop');ax.set(xlabel='World X(m)',ylabel='World Y(m)' if b==1 else 'World Z UP(m)',aspect='equal');ax.axvline(1.1,color='gray',ls=':');ax.axhline(0 if b==1 else 1,color='gray',ls=':');ax.legend(fontsize=8)
    fig.suptitle(case.name+' actual head trajectory and world reference');fig.tight_layout();fig.savefig(case/'actual_head_path.png',dpi=160);plt.close(fig)
    # Correct the reused renderer metadata without replaying any dynamics.
    video=json.loads((case/'video.json').read_text(encoding='utf-8'));video['source']='saved actual qpos from artificial all21-source helix, then one continuous original-gravity run; no I01 state inserted';video['acquisition_demonstrated']=result['active_hook_acquisition_demonstrated'];video['whole_transfer_complete']=result['whole_transfer_complete'];write(case/'video.json',video)
    return audit


if __name__=='__main__':
    audits=[review(BASE/name) for name in ['F01','F02','F03','F04','F05']]
    write(BASE/'branch_review.json',dict(registered_first_candidate_trials=3,authorized_second_candidate_trials=2,formal_trials_completed=5,first_result_within_40_minutes=True,stable_actual_acquisition_count=0,whole_transfer_success_count=0,trials=audits,conclusion='All five authorized trials ended. Source approach was achieved; active acquisition and whole transfer remain incomplete. No further dynamics trial.'))
    print(json.dumps([{k:a[k] for k in ['name','outcome','actual_seconds','first_target_bearing_s','maximum_actual_connected_target','head_world_reference_error_max_mm','zero_kp_joint_samples','anchor_changes','frozen_model_fk_node_error_mm']} for a in audits],ensure_ascii=False),flush=True)
