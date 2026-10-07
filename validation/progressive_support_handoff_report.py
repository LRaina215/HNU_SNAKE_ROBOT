"""Summarize the two saved handoff runs without advancing physics."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/alternating_helix_transfer_20261007/handoff_round2'


def main():
    summaries = []
    for name in ['B01', 'B02']:
        case = BASE/name
        result = json.loads((case/'result.json').read_text(encoding='utf8'))
        with np.load(case/'trace.npz') as trace:
            t = trace['sim_time']-trace['sim_time'][0]
            c, q, force = trace['centers_mm'], np.degrees(trace['q']), trace['contact_force_N']
            commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
            modules = []
            for module in range(21):
                touching = force[:, 0, module] > .1
                ids = np.flatnonzero(touching)
                if not len(ids):
                    continue
                modules.append(dict(module=module, first_contact_s=float(t[ids[0]]),
                    last_contact_s=float(t[ids[-1]]), contact_fraction=float(touching.mean())))
            observed_return = []
            seen = set()
            for event in result['events']:
                if event['event'] == 'source_support_observed':
                    if event['module'] in seen:
                        observed_return.append(dict(t_s=event['t_s'], module=event['module']))
                    seen.add(event['module'])
            summaries.append(dict(name=name, outcome=result['outcome'], duration_s=float(t[-1]),
                newly_admitted_modules=result['newly_admitted_modules'],
                maximum_target_force_M10_M20_N=force[:, 1, 10:].max(axis=0).tolist(),
                M10_start_center_mm=c[0, 10].tolist(), M10_final_center_mm=c[-1, 10].tolist(),
                M9_y_displacement_mm=float(c[-1, 9, 1]-c[0, 9, 1]),
                fixed_front_M0_M7_y_displacement_mm=float((c[-1, :8, 1]-c[0, :8, 1]).mean()),
                fixed_rear_M13_M20_x_displacement_mm=float((c[-1, 13:, 0]-c[0, 13:, 0]).mean()),
                candidate_goal_start_mm=commands[0]['candidate_goal_mm'],
                candidate_goal_final_mm=commands[-1]['candidate_goal_mm'],
                candidate_goal_distance_start_mm=commands[0]['candidate_distance_mm'],
                candidate_goal_distance_final_mm=commands[-1]['candidate_distance_mm'],
                final_distance_to_initial_candidate_goal_mm=float(np.linalg.norm(
                    c[-1, 10]-commands[0]['candidate_goal_mm'])),
                initial_joint_degrees=q[0].tolist(), final_joint_degrees=q[-1].tolist(),
                tail_start_mm=c[0, -1].tolist(), tail_final_mm=c[-1, -1].tolist(),
                final_target_modules=result['final_target_modules'], final_source_modules=result['final_source_modules'],
                source_contacts=modules, source_support_reacquisitions=observed_return,
                protected_source_counts=dict(minimum=int(np.count_nonzero(trace['protected_source_mask'][1:], axis=1).min()),
                                             maximum=int(np.count_nonzero(trace['protected_source_mask'], axis=1).max())),
                rear_first_module_values=sorted(set(x['rear_first_module'] for x in commands if x['rear_first_module'] is not None)),
                protected_set_empty_commands=sum(not x['protected_source_modules'] for x in commands),
                max_command_step_deg=result['max_dispatch_step_deg'],
                max_actual_joint_deg=result['max_actual_joint_deg'],
                max_actuator_torque_Nm=result['max_actuator_torque_Nm'],
                max_pipe_penetration_mm=result['max_pipe_penetration_mm'],
                max_nonadjacent_self_penetration_mm=result['max_self_penetration_mm'],
                ground_observed=bool(trace['ground'].any()),
                first_command_difference_rad=result['first_ctrl_error_rad']))
    output = dict(complete=False, new_physical_runs=2, further_runs_authorized=False,
        first_round_commit='71632657', initial='A07/final_state.npz', trials=summaries,
        interpretation='Both are independent continuous runs from the same saved state. No new target-side module contacted the pipe. Rear support redistribution is not target takeover.',
        LocalSection='fixed-parameter shape objective in the same support solve; not a force-adaptive loop')
    (BASE/'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
