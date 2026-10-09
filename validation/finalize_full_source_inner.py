"""Save stage, support, and physics evidence without stepping dynamics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,shutil
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.suspended_supply_experiment import write

BASE=ROOT/'experiments/unified_curve_transfer_20261007/FULL_SOURCE_INNER_20261009'


def macro_review(name):
    case=BASE/name;z=np.load(case/'trace.npz');t=z['sim_time'];p=z['points_mm']/1000;c=z['centers_mm']/1000;f=z['contact_force_N'];cmd=json.loads((case/'commands.json').read_text(encoding='utf-8'));arrival=json.loads((case/'actual_junction_arrival.json').read_text(encoding='utf-8'));j=arrival['released_front_modules'];at=int(np.argmin(abs(t-arrival['t_s'])));rear_motion=np.linalg.norm(p[:,j:]-p[at,j:],axis=2)*1000
    brace=f[:,0,j:].sum(1);after=t>=arrival['t_s'];hold_commands=np.array([x['ctrl_rad'][j:] for x in cmd if x['stage'].startswith('rear_braced')]);hold_delta=float(np.max(abs(hold_commands-hold_commands[0])))
    fields=['body_mass','body_inertia','geom_friction','geom_solref','geom_solimp','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','jnt_range']
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'));baseline=mujoco.MjModel.from_xml_path(str(ROOT/'experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009/I01/sources/mujoco_model/Snake_Robot_o20_inner_hook_transfer.xml'))
    physics={field:bool(np.array_equal(getattr(m,field),getattr(baseline,field))) for field in fields};physics['gravity']=bool(np.array_equal(m.opt.gravity,baseline.opt.gravity));physics['timestep']=m.opt.timestep==baseline.opt.timestep
    below=np.flatnonzero(after&(p[:,0,0]<1.1)&(p[:,0,1]>0)&(p[:,0,2]<.95));ids=[0,at,int(below[0]) if len(below) else len(t)//2,len(t)-1];labels=['Artificial all-source start','Actual junction vicinity reached','Actual inner-side low head; no grip','Actual stop / incomplete']
    fig=plt.figure(figsize=(18,7));stages=[]
    for k,(i,label) in enumerate(zip(ids,labels)):
        ax=fig.add_subplot(1,4,k+1,projection='3d');ax.plot(*p[i].T,'o-',ms=2,label='Actual body');ax.plot(*(z['reference_points_mm'][i]/1000).T,'--',alpha=.7,label='Executed guide');ax.scatter(*p[i,0],c='crimson',label='Actual head');ax.plot([-.2,1.1],[0,0],[1,1],lw=8,color='gray',alpha=.3);ax.plot([1.1,1.1],[0,2],[1,1],lw=8,color='gray',alpha=.3);ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)',title=f'{label}\n{t[i]:.2f}s',xlim=(.4,1.3),ylim=(-.3,.55),zlim=(.6,1.2));ax.view_init(27,140);ax.legend(fontsize=6)
        stages.append(dict(label=label,t_s=float(t[i]),actual_points_mm=z['points_mm'][i],executed_reference_mm=z['reference_points_mm'][i],source_brace_force_n=float(brace[i]),target_force_n=float(f[i,1].sum())))
    fig.suptitle(name+' continuous physical stages; no stable target acquisition');fig.tight_layout();fig.savefig(case/'actual_macro_stages.png',dpi=155);plt.close(fig);write(case/'actual_macro_stages.json',stages)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True);axes[0].plot(t,brace,label=f'Actual held-source M{j}..M20 bearing');axes[0].plot(t,f[:,0,:j].sum(1),label='Actual front source bearing');axes[0].plot(t,f[:,1].sum(1),label='Actual target bearing');axes[0].set(ylabel='N');axes[0].legend();axes[1].plot(t,rear_motion.max(1),label='Largest posterior-node motion from arrival');axes[1].plot(t,z['tracking_deg'],label='Actual joint tracking deg');axes[1].set(xlabel='Actual time(s)',ylabel='mm / deg');axes[1].legend();fig.tight_layout();fig.savefig(case/'source_brace_review.png',dpi=160);plt.close(fig)
    write(case/'macro_action_review.json',dict(formal_trial=name,physical_steps_in_review=0,initial_state_is_artificial=True,source_approach_achieved=True,junction_arrival_s=arrival['t_s'],arrival_trigger='actual head X>=1.025m; no hard-block contact claimed',initial_head_x_mm=p[0,0,0]*1000,arrival_head_x_mm=p[at,0,0]*1000,actual_source_approach_head_delta_mm=(p[at,0]-p[0,0])*1000,source_held_modules=list(range(j,21)),posterior_hold_command_change_rad=hold_delta,posterior_kp_min=float(z['kp_nm_per_rad'][after,j:].min()),posterior_bearing_min_n=float(brace[after].min()),posterior_bearing_final_n=float(brace[-1]),posterior_node_motion_max_mm=float(rear_motion[after].max()),actual_target_connected_max=int(z['actual_received_mask'].sum(1).max()),first_low_inner_head_s=float(t[below[0]]) if len(below) else None,first_low_inner_head_is_acquisition=False,stage_counts={stage:int(np.count_nonzero(z['stage']==stage)) for stage in np.unique(z['stage'])},all_original_physics_equal=all(physics.values()),physics_equal_fields=physics,target_axis_end_y_m=2.,actual_transfer_handoff_occurred=False,target_polar_post_review_patch_executed=False))
    # The pre-run curve figure is a planned transfer template, not the full macro start.
    meta=json.loads((case/'complete_reference_curve.json').read_text(encoding='utf-8'));meta['scope']='pre-run transfer template only; actual approach/unfold/coil references are in every trace and commands row';meta['initial_curve_overlay_is_actual_start']=False;write(case/'complete_reference_curve.json',meta)


if __name__=='__main__':
    for name in ['F04','F05']:macro_review(name)
    dest=BASE/'POST_REVIEW_UNRUN';dest.mkdir(exist_ok=True);shutil.copy2(ROOT/'snake_control/braced_inner_probe.py',dest/'braced_inner_probe.py')
    write(dest/'STATUS.json',dict(physical_steps=0,executed_in_formal_trials=False,source_file='snake_control/braced_inner_probe.py',status='target-polar node interpolation only; rigid link feasibility remains unresolved; not a successful gait'))
    print('Saved actual macro stages, posterior brace evidence, physics equivalence, and explicitly unrun code.',flush=True)
