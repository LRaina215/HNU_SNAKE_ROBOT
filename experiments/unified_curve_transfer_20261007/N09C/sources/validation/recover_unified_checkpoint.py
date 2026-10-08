"""Restore an intermediate physical/controller checkpoint from an actual run.

Original ctrl commands are replayed in the unchanged model. Curve history is
recovered from recorded observations and frozen feedback, without rerunning the
optimizers or trajectory observer. This is checkpoint recovery, not a new trial.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from snake_control.hook_pull_review_support import restore_saved_state
from validation.recover_unified_curve_state import save_state


def recover_current(case, seconds, plan, commands, trace, frozen):
    """Reconstruct current contact/yield history with its frozen controller."""
    destination = case/f'resume_t{seconds:.2f}'.replace('.', 'p')
    destination.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    physical, observed = mujoco.MjData(model), mujoco.MjData(model)
    restore_saved_state(model, physical, ROOT/plan['initial'])
    restore_saved_state(model, observed, ROOT/plan['initial'])
    controller = frozen.UnifiedCurveTransfer(model, observed)
    offset = controller.load_state(ROOT/plan['controller_state'])
    initial = json.loads((case/'initial_controller_state.json').read_text(encoding='utf8'))
    controller.soft_positive_clearance_enabled = initial['controller']['soft_positive_clearance_enabled']
    controller.curve.soft_positive_clearance = initial['curve']['soft_positive_clearance']
    controller.curve.arc_cache = None
    controller.curve.shape_revision += 1
    controller.localized_yield_enabled = initial['controller']['localized_yield_enabled']
    controller.alternating_yield_enabled = initial['controller']['alternating_yield_enabled']
    count = round(seconds/frozen.DT)
    if abs(count*frozen.DT-seconds) > 1e-8 or count > len(commands):
        raise ValueError('Checkpoint must lie on a recorded command boundary')
    maximum_reference_error = maximum_ctrl_error = 0.
    for tick, command in enumerate(commands[:count]):
        index = tick*5
        # Observations are diagnostic copies. The physical replay below only
        # receives the original dispatched controls after its full restoration.
        observed.qpos[:] = trace['qpos'][index]
        observed.qvel[:] = trace['qvel'][index]
        observed.ctrl[:] = trace['ctrl'][index]
        command_data = physical if getattr(controller,'physics_execution_prediction_enabled',False) else observed
        target, info = controller.command(command_data, command['controller_t_s'],
            trace['contact_force_N'][index], trace['torque'][index],
            trace['cylindrical_contact_force_N'][index])
        maximum_ctrl_error = max(maximum_ctrl_error,
            float(np.max(np.abs(target-command['target_rad']))))
        maximum_reference_error = max(maximum_reference_error,
            float(np.max(np.abs(np.asarray(info['reference_points_mm'])-command['reference_points_mm']))))
        physical.ctrl[:] = command['target_rad']
        for _ in range(50):
            mujoco.mj_step(model, physical)
        if tick % 500 == 0:
            print(f'Restored {tick*frozen.DT:.2f}s; ctrl error {maximum_ctrl_error:.3g}; reference error {maximum_reference_error:.3g} mm', flush=True)
    boundary = count*5
    errors = dict(qpos=float(np.max(np.abs(physical.qpos-trace['qpos'][boundary]))),
        qvel=float(np.max(np.abs(physical.qvel-trace['qvel'][boundary]))),
        ctrl=float(np.max(np.abs(physical.ctrl-trace['ctrl'][boundary]))))
    if maximum_reference_error > 1e-5 or maximum_ctrl_error > 1e-10 or max(errors.values()) > 1e-10:
        raise RuntimeError(f'Checkpoint reconstruction differs: {errors}; ctrl {maximum_ctrl_error}; reference {maximum_reference_error}')
    integration = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, physical, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(destination/'final_state.npz', integration_state=integration,
        ctrl=physical.ctrl.copy(), sim_time=physical.time)
    save_state(destination/'controller_state.json', controller, offset+seconds)
    report = dict(source=str(case), checkpoint_s=seconds,
        physics='original dispatched ctrl replay in unchanged model; no physical root assignment',
        controller='frozen controller on original recorded observations, including contact/yield/progress history',
        maximum_reference_error_mm=maximum_reference_error, maximum_dispatched_ctrl_error_rad=maximum_ctrl_error,
        physical_boundary_errors=errors, shift_mm=controller.shift*1000,
        localized_yield=controller.yield_state,
        source_contacts=np.flatnonzero(trace['contact_force_N'][boundary, 0] > .1).tolist(),
        target_contacts=np.flatnonzero(trace['contact_force_N'][boundary, 1] > .1).tolist())
    (destination/'recovery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ['checkpoint_s', 'maximum_reference_error_mm',
        'maximum_dispatched_ctrl_error_rad', 'physical_boundary_errors', 'source_contacts', 'target_contacts']}), flush=True)


def recover(case, seconds):
    plan = json.loads((case/'plan.json').read_text(encoding='utf8'))
    commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
    trace = np.load(case/'trace.npz')
    specification = importlib.util.spec_from_file_location('frozen_checkpoint_curve', case/'sources/snake_control/unified_curve_transfer.py')
    frozen = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(frozen)
    if commands and 'shared_curve_contact_field' in commands[0]:
        return recover_current(case, seconds, plan, commands, trace, frozen)
    destination = case/f'resume_t{seconds:.2f}'.replace('.', 'p')
    destination.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    physical, observed = mujoco.MjData(model), mujoco.MjData(model)
    restore_saved_state(model, physical, ROOT/plan['initial'])
    observed.qpos[:] = trace['qpos'][0]
    observed.ctrl[:] = trace['ctrl'][0]
    controller = frozen.UnifiedCurveTransfer(model, observed)
    offset = controller.load_state(ROOT/plan['controller_state'])
    count = round(seconds/frozen.DT)
    if abs(count*frozen.DT-seconds) > 1e-8:
        raise ValueError('Checkpoint must lie on a dispatched-command boundary')
    projection_rows = []
    maximum_reference_error = 0.
    for tick, command in enumerate(commands[:count]):
        index, t = tick*5, command['controller_t_s']
        observed.qpos[:] = trace['qpos'][index]
        forces, torque = trace['contact_force_N'][index], trace['torque'][index]
        controller.current_control_time = t
        controller.observer_rebased = False
        p, centers = controller.observe(observed)
        q = observed.qpos[controller.qa]
        nodes, reference, locations = controller.curve.reference(controller.shift)
        if t-controller.last_route_registration_time > 20.:
            bad = controller.pipe_geometry.link_minimum(nodes).min() < .0145
            wrong = np.linalg.norm(reference-centers, axis=1).max() > .12
            if (bad or wrong) and controller.repair_registered_route(p, centers, t, forces):
                nodes, reference, locations = controller.curve.reference(controller.shift)
        controller.filtered_load += .1*(np.abs(torque)-controller.filtered_load)
        if t-controller.last_observer_time >= .5:
            controller.last_observer_time = t
        if t-controller.last_shape_time >= .5:
            curve = controller.curve
            original_deposit = curve.deposit
            def sample_pair(u):
                enabled = curve.clearance_enabled
                curve.clearance_enabled = False
                raw = curve.sample(u)
                curve.clearance_enabled = enabled
                return raw, curve.sample(u)
            def measured_deposit(u, changes, rate, dt):
                if len(u) != 22:
                    original_deposit(u, changes, rate, dt)
                    return
                raw0, projected0 = sample_pair(u)
                original_deposit(u, changes, rate, dt)
                raw1, projected1 = sample_pair(u)
                for side, code, xy, origin in [(0, 0, [1, 2], [0., 1.]), (1, 2, [0, 2], [1.1, 1.])]:
                    normal = raw0[:, xy]-origin
                    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-10)
                    raw_inward = -np.sum((raw1-raw0)[:, xy]*normal, axis=1)
                    actual_inward = -np.sum((projected1-projected0)[:, xy]*normal, axis=1)
                    selected = (curve.regions(u) == code) & (raw_inward > 1e-6)
                    supported_node = np.r_[forces[side, 0], np.maximum(forces[side, :-1], forces[side, 1:]), forces[side, -1]] > .1
                    held = selected & supported_node
                    for label, mask in [('pipe_region', selected), ('near_actual_contact', held)]:
                        if mask.any():
                            projection_rows.append(dict(t_s=t-offset, side=side, group=label, nodes=int(mask.sum()),
                                raw_inward_sum_um=float(raw_inward[mask].sum()*1e6),
                                projected_inward_sum_um=float(actual_inward[mask].sum()*1e6),
                                suppressed_nodes=int(np.count_nonzero(actual_inward[mask] < .2*raw_inward[mask]))))
            curve.deposit = measured_deposit
            controller.feedback_shape(p, q, reference, locations, np.abs(controller.last-q), forces, .5)
            del curve.__dict__['deposit']
            controller.last_shape_time = t
        ds = command['shift_speed_mm_s']/1000*frozen.DT
        controller.curve.increment(ds)
        controller.shift += ds
        wanted, _, _ = controller.curve.reference(controller.shift)
        maximum_reference_error = max(maximum_reference_error, float(np.max(np.abs(wanted*1000-command['reference_points_mm']))))
        controller.last = np.asarray(command['target_rad'])
        controller.speed = command['shift_speed_mm_s']/1000
        controller.observed_shift = controller.last_body_progress = command['actual_material_estimate_mm']/1000
        controller.projected_locations = np.asarray(command['observed_projected_locations_mm'])/1000
        physical.ctrl[:] = command['target_rad']
        for _ in range(50):
            mujoco.mj_step(model, physical)
        if tick % 500 == 0:
            print(f'Restored {tick*frozen.DT:.2f}s; reference error {maximum_reference_error:.3g} mm', flush=True)
    last = commands[count-1]
    controller.last_mode = last['stage']
    controller.progress_history = [(x['controller_t_s'], x['actual_material_estimate_mm']/1000)
                                  for x in commands[:count] if x['controller_t_s'] >= last['controller_t_s']-10.]
    result = json.loads((case/'result.json').read_text(encoding='utf8'))
    controller.events = [x for x in result['controller_events'] if x['t_s'] < offset+seconds]
    boundary = count*5
    errors = dict(qpos=float(np.max(np.abs(physical.qpos-trace['qpos'][boundary]))),
                  qvel=float(np.max(np.abs(physical.qvel-trace['qvel'][boundary]))),
                  ctrl=float(np.max(np.abs(physical.ctrl-trace['ctrl'][boundary]))))
    if maximum_reference_error > 1e-5 or max(errors.values()) > 1e-10:
        raise RuntimeError(f'Checkpoint reconstruction differs: {errors}; reference {maximum_reference_error}')
    integration = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, physical, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(destination/'final_state.npz', integration_state=integration, ctrl=physical.ctrl.copy(), sim_time=physical.time)
    save_state(destination/'controller_state.json', controller, offset+seconds)
    summaries = []
    for side in [0, 1]:
        for group in ['pipe_region', 'near_actual_contact']:
            rows = [x for x in projection_rows if x['side'] == side and x['group'] == group]
            raw = sum(x['raw_inward_sum_um'] for x in rows)
            projected = sum(x['projected_inward_sum_um'] for x in rows)
            summaries.append(dict(side=side, group=group, raw_inward_sum_um=raw, projected_inward_sum_um=projected,
                retained_fraction=projected/raw if raw > 0 else None,
                affected_nodes=sum(x['nodes'] for x in rows), suppressed_nodes=sum(x['suppressed_nodes'] for x in rows)))
    report = dict(source=str(case), checkpoint_s=seconds, physics='original dispatched ctrl replay only; unchanged model; no root assignment',
        replayed_physical_seconds=seconds, maximum_reference_error_mm=maximum_reference_error, physical_boundary_errors=errors,
        physical_time=float(physical.time), shift_mm=controller.shift*1000,
        source_contacts=np.flatnonzero(trace['contact_force_N'][boundary, 0] > .1).tolist(),
        target_contacts=np.flatnonzero(trace['contact_force_N'][boundary, 1] > .1).tolist(), projection_summary=summaries)
    (destination/'recovery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (destination/'pressure_projection_history.json').write_text(json.dumps(projection_rows, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path)
    parser.add_argument('--time', type=float, required=True)
    args = parser.parse_args()
    recover(args.case, args.time)
