"""Read-only actual whole-helix audit of the inner-side hook transfer."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from validation.review_hook_start_transfer import review
from validation.suspended_supply_experiment import write,default
BASE=ROOT/'experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009'


def audit(case):
    review(case,model_name='Snake_Robot_o20_inner_hook_transfer.xml',initial_filename='initial_state.npz',initial_state_artificial=True)
    z=np.load(case/'trace.npz');a=json.loads((case/'commands.json').read_text(encoding='utf-8'));g=json.loads((case/'initial_geometry.json').read_text(encoding='utf-8'))
    original=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_inner_hook_transfer.xml'));head=m.body('head').id
    keys=['dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','geom_friction','geom_solref','geom_solimp','jnt_range']
    same={k:bool(np.array_equal(getattr(original,k),getattr(m,k))) for k in keys};same.update(robot_mass=bool(np.array_equal(original.body_mass[head:head+21],m.body_mass[head:head+21])),robot_inertia=bool(np.array_equal(original.body_inertia[head:head+21],m.body_inertia[head:head+21])),gravity=bool(np.array_equal(original.opt.gravity,m.opt.gravity)))
    write(case/'physics_equivalence_review.json',dict(original_physics_equal=all(same.values()),arrays=same,gravity=m.opt.gravity,physics_steps_in_review=0,target_pipe_end_y_m=2.))
    i=next((i for i,x in enumerate(a) if x['free_supply_mode_latched']),None)
    if i is not None:
        x=a[i];step=np.asarray(x['reference_points_mm'])-np.asarray(a[max(0,i-1)]['reference_points_mm']);full=next((x for x in a if x['free_handoff_fraction']>=.999),None)
        write(case/'handoff_continuity_review.json',dict(first_supply_handoff_s=x['t_s'],first_supply_handoff_shift_mm=x['shift_mm'],first_full_roll_s=full['t_s'] if full else None,first_full_roll_shift_mm=full['shift_mm'] if full else None,front10_node_reference_step_rms_mm=float(np.sqrt(np.mean(np.sum(step[:10]**2,axis=1)))),all22_reference_step_rms_mm=float(np.sqrt(np.mean(np.sum(step**2,axis=1)))),continuous_reform_handoff=x['continuous_reform_handoff'],event_based_progress=x['event_based_progress']))
    p=z['points_mm'][0]/1000;n=g['front_modules'];j=g['source_start_module'];bridge=p[n:j+1];outlet=p[j]-p[j+1];entry=p[n-1]-p[n]
    nominal=np.asarray(json.loads((case/'complete_reference_curve.json').read_text(encoding='utf-8'))['points_mm'])/1000
    write(case/'inner_direction_review.json',dict(physical_steps=0,real_recorded_initial_pose=True,inner_region='X<1.1m and Y>0m; finite source-X and target+Y pipe segments',
        source_outlet_node=j,source_outlet_point_mm=p[j]*1000,source_tail_to_head_outlet_vector=outlet,source_outlet_toward_positive_y=bool(outlet[1]>0 and p[j,1]>0),
        target_entry_node=n,target_entry_point_mm=p[n]*1000,target_tail_to_head_entry_vector=entry,target_entry_inward_and_upward=bool(entry[0]<0 and entry[2]>0 and p[n,0]<1.1 and p[n,2]<1),
        all_actual_transition_nodes_inside=bool(np.all((bridge[:,0]<1.1)&(bridge[:,1]>0))),actual_entry_nodes_tail_to_head=[n+1,n,n-1],actual_entry_points_mm=p[[n+1,n,n-1]]*1000,
        actual_front8_wrap_deg=g['initial_front8_wrap_deg'],actual_front9_wrap_deg=g['initial_front9_wrap_deg'],initial_state_artificial=True,active_hook_acquisition_demonstrated=False,
        geometry_fit_rms_mm=g['initial_geometry_fit']['guide_rms_mm'],bridge_reference_length_mm=g['bridge_modules']*.0645*1000,nominal_curve_is_not_exact_initial_fk=True))
    report=json.loads((case/'actual_evidence_review.json').read_text(encoding='utf-8'));print(json.dumps(dict(strict_whole_transfer_verified=report['strict_whole_transfer_verified'],last_second_samples=report['last_second_samples'],initial_pose_preserved=report['initial_saved_state_preserved'],physics_equal=all(same.values()),final_tail_radius_mm=report['final_center_radius_mm'][-1],source_clear_intervals=report['source_clear_intervals_at_least_half_second']),default=default),flush=True)


if __name__=='__main__':
    for name in sys.argv[1:] or ['I01']:audit(BASE/name)
