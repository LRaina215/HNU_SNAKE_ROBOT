"""Read-only audit of lower-hook reference handoff and actual progress events."""
from pathlib import Path
import sys,json
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.suspended_supply_experiment import write
from validation.review_hook_start_transfer import review
BASE=ROOT/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009'


def audit(case):
    review(case,lower_entry=True)
    commands=json.loads((case/'commands.json').read_text(encoding='utf-8'));z=np.load(case/'trace.npz')
    result=json.loads((case/'result.json').read_text(encoding='utf-8'))
    received=set(range(8));exited=set();best=-1;lastcount=0.;events=[]
    for x in commands:
        rr=set(x['actual_received_modules']);ee=set(x['actual_source_exited_modules']);newr=rr-received;newe=ee-exited
        if newr or newe:events.append(dict(t_s=x['t_s'],new_actual_intake_modules=sorted(newr),new_actual_exit_modules=sorted(newe)))
        received.update(rr);exited.update(ee);count=len(rr)+len(ee)
        if count>best:best=count;lastcount=x['t_s']
    event_mode=bool(commands[-1].get('event_based_progress',False))
    last=commands[-1].get('last_actual_progress_s',events[-1]['t_s'] if events else 0.)
    write(case/'progress_stop_review.json',dict(event_based_rule_used=event_mode,
        registered_stop='first distinct actual received/cleared module, or actual10mm axial advance while first8 connected and bearing>1N' if event_mode else 'increase in maximum sum of current received and exited counts',
        last_maximum_count_time_s=lastcount,maximum_sum_count=best,last_distinct_new_module_event_s=events[-1]['t_s'] if events else None,
        last_actual_progress_s=last,actual_seconds_since_last_actual_progress=result['actual_seconds']-last,
        distinct_new_module_events=events,continuous_intake_progress_times_s=[x['t_s'] for x in commands if x.get('actual_continuous_intake_progress',False)],
        stop_name=result['outcome'],physical_transfer_complete=False,
        limitation='U01/U03 count stagnation was not45s without a new event; U04 event rule corrected without extending registered overall limits'))
    indices=[i for i,x in enumerate(commands) if x['free_supply_mode_latched']]
    if indices:
        i=indices[0];x=commands[i];old=commands[max(0,i-1)]
        step=np.asarray(x['reference_points_mm'])-np.asarray(old['reference_points_mm']);s=x['shift_mm']/200;blend=min(s,1)**2*(3-2*min(s,1))
        full=next((c for c in commands if c['free_handoff_fraction']>=.999),None)
        write(case/'handoff_continuity_review.json',dict(first_supply_handoff_s=x['t_s'],first_supply_handoff_shift_mm=x['shift_mm'],hook_morph_blend_at_supply_handoff=blend,
            first_full_roll_s=None if full is None else full['t_s'],first_full_roll_shift_mm=None if full is None else full['shift_mm'],
            front9_node_reference_step_rms_mm=float(np.sqrt(np.mean(np.sum(step[:9]**2,axis=1)))),front11_node_reference_step_range_mm=[float(np.linalg.norm(step[:11],axis=1).min()),float(np.linalg.norm(step[:11],axis=1).max())],
            all22_reference_step_rms_mm=float(np.sqrt(np.mean(np.sum(step**2,axis=1)))),current_received_at_handoff=x['actual_received_modules'],source_exited_at_handoff=x['actual_source_exited_modules'],
            continuous_reform_handoff=bool(x.get('continuous_reform_handoff',False)),recorded_commands_only=True,no_additional_dynamics=True))
    original=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    actual=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_lower_hook_transfer.xml'));head=actual.body('head').id
    keys=['dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','geom_friction','geom_solref','geom_solimp','jnt_range']
    equal={k:bool(np.array_equal(getattr(original,k),getattr(actual,k))) for k in keys}
    equal.update(robot_body_mass=bool(np.array_equal(original.body_mass[head:head+21],actual.body_mass[head:head+21])),robot_body_inertia=bool(np.array_equal(original.body_inertia[head:head+21],actual.body_inertia[head:head+21])),gravity=bool(np.array_equal(original.opt.gravity,actual.opt.gravity)))
    write(case/'physics_equivalence_review.json',dict(original_physics_equal=all(equal.values()),arrays=equal,gravity=actual.opt.gravity,target_pipe_end_y_m=2.,initial_state_is_artificial=True,physical_steps_in_review=0))


if __name__=='__main__':
    for name in sys.argv[1:] or ['U01','U02','U03','U04']:audit(BASE/name)
