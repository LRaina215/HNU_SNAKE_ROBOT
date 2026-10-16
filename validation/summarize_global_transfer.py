"""Summarize recorded actual movement separately from global/local references."""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from snake_control.global_transfer_route import route_body


def summarize(case):
    trace=np.load(case/'trace.npz')
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    state=json.loads((case/'controller_state.json').read_text(encoding='utf8'))
    spec=state['curve']['planned_route_spec']
    def sample(index,command):
        nodes=trace['points_mm'][index]
        pitch=spec['axial_sine'];rate=np.array([spec['source_polar_rate'],spec['target_polar_rate']])
        shift=command['shift_mm']/1000
        phase=-np.sign(rate)*shift/(spec['radius']*np.sqrt(1-pitch*pitch))
        global_nodes,params=route_body(spec,shift,phase)
        right=np.unwrap(np.arctan2(-(nodes[:,2]-1000),nodes[:,0]-1100))
        front=np.arange(0,9)
        return dict(actual_global_goal_rms_mm=float(np.sqrt(np.mean((nodes-global_nodes*1000)**2))),
            actual_local_reference_rms_mm=float(np.sqrt(np.mean((nodes-trace['reference_points_mm'][index])**2))),
            actual_front_N0_N8_delta_y_mm=float(nodes[8,1]-nodes[0,1]),
            actual_front_N0_N8_right_hand_delta_phi_rad=float(right[8]-right[0]),
            actual_front_N0_N8_axial_fold_count=int(np.count_nonzero(np.diff(nodes[:9,1])>3.)),
            actual_tail_N21_mm=nodes[-1].tolist(),global_node_parameters_mm=(params*1000).tolist())
    g=[x.get('global_route_guidance') or {} for x in commands]
    report=dict(name=case.name,initial='CLOSE1B' if result['controller_initial_elapsed_s']==0 else 'saved continuation',
        actual_seconds=result['physical_seconds'],outcome=result['outcome'],complete=result['complete'],
        front_target_axis_net_mm=result['fixed_front_M0_M6_mean_y_advance_mm'],
        rear_source_axis_net_mm=result['fixed_rear_M14_M20_mean_x_advance_mm'],
        new_target_material=result['new_sustained_target_modules'],
        actual_initial=sample(0,commands[0]),actual_final=sample(-1,commands[-1]),
        guidance_acceptance_counts={str(f):sum(x.get('accepted_fraction')==f for x in g) for f in sorted(set(x.get('accepted_fraction',-1) for x in g))},
        local_reference_to_global_rms_mm=[g[0].get('global_goal_rms_mm'),g[-1].get('global_goal_rms_mm')],
        root_models=sorted(set(x['root_execution_model'] for x in commands)),
        final_source_raw=result['final_source_modules'],final_target_raw=result['final_target_modules'],
        max_actual_joint_deg=result['max_actual_joint_deg'],max_drive_torque_Nm=result['max_actuator_torque_Nm'],
        max_ctrl_step_deg=result['max_dispatch_step_deg'],first_ctrl_error_rad=result['first_ctrl_error_rad'],
        planned_source_right_hand_dphi_daxis=spec['source_right_hand_dphi_daxis'],
        planned_target_right_hand_dphi_daxis=spec['target_right_hand_dphi_daxis'],
        limitation='Reference changes, accepted proposals and predicted rollouts are not actual transfer or proof of a same-handed final whole body.')
    (case/'global_motion_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    summarize(Path(sys.argv[1]))
