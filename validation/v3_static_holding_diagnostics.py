"""Analyze the saved single static trial. No physical evolution or new controls."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from curve_guided_geometry import MODEL, points
from snake_control.curve_grasp_revision import FittedMaterialCurve, RetainedGraspController

BASE = ROOT / 'experiments/curve_guided_transfer_20261006/v3_static_holding_20261006'
CASE = BASE / 'V3_STATIC_C0'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    trace = np.load(CASE / 'trace.npz')
    controls = json.loads((CASE / 'control.json').read_text(encoding='utf8'))
    commands = json.loads((CASE / 'commands.json').read_text(encoding='utf8'))
    result = json.loads((CASE / 'result.json').read_text(encoding='utf8'))
    t, c, f = trace['sim_time'], trace['centers_mm'], trace['contact_force_N']
    velocity = np.diff(c, axis=0) / np.diff(t)[:, None, None]
    speed = np.linalg.norm(velocity, axis=2)
    centroid = c.mean(axis=1)
    centroid_drift = np.linalg.norm(centroid - centroid[0], axis=1)
    module_drift = np.linalg.norm(c - c[0], axis=2)
    root_drift = np.linalg.norm(trace['qpos'][:, :3] - trace['qpos'][0, :3], axis=1) * 1000
    root_rotation = Rotation.from_quat(trace['qpos'][:, [4, 5, 6, 3]])
    root_rotation_deg = np.degrees((root_rotation * root_rotation[0].inv()).magnitude())
    contact_count = np.count_nonzero(f > .1, axis=2)
    tracking = np.degrees(trace['ctrl'] - trace['q'])
    ct = np.array([r['time_s'] for r in controls])
    fields = ['compliance_delta_rad', 'retained_shape_rad', 'effective_residual_rad', 'shape_update_rad']
    values = {key: np.degrees(np.array([r[key] for r in controls])) for key in fields}
    raw = np.degrees(np.array([r['nominal_rad'] for r in controls])) + values['effective_residual_rad']
    blend = np.clip(ct / 5, 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    initial_deg = np.degrees(trace['ctrl'][0])
    raw = (1 - blend[:, None]) * initial_deg + blend[:, None] * raw
    targets = np.degrees(np.array([r['target_rad'] for r in commands]))
    body_ok = (contact_count[:, 0] >= 2) & (contact_count[:, 1] >= 2)
    instantaneous_ok = body_ok[1:] & (speed.max(axis=1) < 50)
    after_entry = t[1:] >= 5 - 1e-9
    longest = current = 0
    start_index = end_index = current_start = 0
    for index, good in enumerate(instantaneous_ok):
        if good:
            if current == 0:
                current_start = index
            current += 1
            if current > longest:
                longest, start_index, end_index = current, current_start, index + 1
        else:
            current = 0
    selected = []
    for moment in [0., .1, .5, 1., 3., 4., 5., 6., 7., float(t[-1])]:
        index = int(abs(t - moment).argmin())
        selected.append(dict(time_s=float(t[index]),
            source_modules=np.flatnonzero(f[index, 0] > .1).tolist(),
            target_modules=np.flatnonzero(f[index, 1] > .1).tolist(),
            source_total_normal_N=float(f[index, 0].sum()),
            target_total_normal_N=float(f[index, 1].sum()),
            maximum_module_speed_mm_s=float(speed[max(0, index - 1)].max()) if index else 0.,
            root_displacement_mm=float(root_drift[index]), centroid_displacement_mm=float(centroid_drift[index])))

    # The failed compute call raised before returning control fields. Reconstruct
    # it offline from the final angles and last successful h/target. Verify the
    # previous call agrees first. This never dispatches or invokes mj_step.
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    initial, kinematic = mujoco.MjData(model), mujoco.MjData(model)
    initial.qpos[:] = trace['qpos'][0]
    mujoco.mj_forward(model, initial)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    def encoder_fk(angles):
        kinematic.qpos[:] = 0
        kinematic.qpos[3] = 1
        kinematic.qpos[qadr] = angles
        mujoco.mj_kinematics(model, kinematic)
        return points(model, kinematic)
    curve = FittedMaterialCurve(BASE / 'inputs/fitted_reference.npz')
    controller = RetainedGraspController(curve,
        json.loads((CASE / 'config.json').read_text(encoding='utf8')),
        trace['ctrl'][0], encoder_fk, initial.body('head').xmat.reshape(3, 3).copy())
    controller.shape = np.array(controls[-2]['retained_shape_rad'])
    controller.last = np.array(commands[-2]['target_rad'])
    i = int(abs(t - controls[-1]['time_s']).argmin())
    reproduced, reproduced_fields = controller.compute(trace['q'][i], 0., controls[-1]['time_s'], .05)
    target_error = float(abs(reproduced - np.array(commands[-1]['target_rad'])).max())
    state_error = float(abs(controller.shape - np.array(controls[-1]['retained_shape_rad'])).max())
    controller.shape = np.array(controls[-1]['retained_shape_rad'])
    controller.last = np.array(commands[-1]['target_rad'])
    before = controller.shape.copy()
    error = ''
    try:
        controller.compute(trace['q'][-1], 0., float(t[-1]), .05)
    except ValueError as exc:
        error = str(exc)
    weights = np.array(controls[-1]['grasp_weights'])
    rejected_raw = np.degrees(curve.joint_angles(0.) + weights * controller.shape)
    failure = dict(
        classification='OFFLINE_RECONSTRUCTION, not a returned failed-call control record',
        failed_call_time_s=float(t[-1]), physics_steps=0,
        previous_call_target_maximum_difference_rad=target_error,
        previous_call_state_maximum_difference_rad=state_error,
        prior_call_reproduced=bool(target_error < 1e-10 and state_error < 1e-10),
        error_reproduced=error, rejected_raw_target_deg=rejected_raw.tolist(),
        retained_shape_after_rejected_compute_deg=np.degrees(controller.shape).tolist(),
        retained_shape_update_at_rejection_deg=np.degrees(controller.shape - before).tolist(),
        exceeded_joint_indices_1based=(np.flatnonzero(abs(rejected_raw) > 90) + 1).tolist(),
        last_actually_applied_target_deg=targets[-1].tolist(),
        final_actual_joint_deg=np.degrees(trace['q'][-1]).tolist(),
        no_rejected_target_dispatched=True,
        input_provenance='saved final physical joint angles; bridge publishes degree conversion without rounding')
    write(BASE / 'rejected_control_reconstruction.json', failure)
    report = dict(
        physics_steps_added_by_diagnostics=0, actual_frames=len(t), actual_commands=len(commands),
        actual_successful_control_updates=len(controls), full_path_status='NOT_READY',
        outcome=result['outcome'], first_stop_event_time_s=float(t[-1]),
        entry_screen_pass=bool(result.get('entry_maximum_speed_mm_s', 1e9) < 50 and
            body_ok[(t > 4 - 1e-9) & (t <= 5 + 1e-9)].all()),
        planned_duration_s=45.25, recorded_duration_s=float(t[-1]), completed_duration=False,
        after_entry_observed_seconds=float(t[-1] - 5),
        after_entry_every_sample_satisfies_contacts_and_speed=bool(instantaneous_ok[after_entry].all()),
        after_entry_minimum_contacts_per_pipe=contact_count[t >= 5 - 1e-9].min(axis=0).tolist(),
        after_entry_maximum_module_speed_mm_s=float(speed[after_entry].max()),
        final_maximum_module_speed_mm_s=float(speed[-1].max()),
        instantaneous_valid_longest_interval_s=[float(t[start_index]), float(t[end_index])],
        instantaneous_valid_longest_duration_s=float(t[end_index] - t[start_index]),
        root_final_drift_mm=float(root_drift[-1]), root_final_rotation_deg=float(root_rotation_deg[-1]),
        geometric_centroid_final_drift_mm=float(centroid_drift[-1]),
        maximum_module_final_displacement_mm=float(module_drift[-1].max()),
        after_entry_centroid_displacement_mm=float(np.linalg.norm(centroid[-1] - centroid[int(abs(t - 5).argmin())])),
        maximum_tracking_error_deg=float(abs(tracking).max()),
        maximum_applied_torque_Nm=float(abs(trace['torque']).max()),
        ground_contact_frames=int(np.count_nonzero(trace['ground'])),
        nonadjacent_self_contact_frames=int(np.count_nonzero(trace['self_contact'])),
        full_support_loss_frames=int(np.count_nonzero(contact_count.sum(axis=1) == 0)),
        added_external_force_maximum=float(trace['external_applied_force_maximum'].max()),
        added_generalized_force_maximum=float(trace['generalized_applied_force_maximum'].max()),
        correction_errors=[dict(time_s=r['time_s'], errors=r['correction_errors']) for r in controls if r['correction_errors']],
        rate_limited_updates=int(np.count_nonzero([r['rate_limited_joints'] for r in controls])),
        last_raw_minus_applied_deg=(raw[-1] - targets[-1]).tolist(),
        last_retained_shape_deg=values['retained_shape_rad'][-1].tolist(),
        selected_actual_samples=selected,
        interpretation='Brief observed two-pipe support passes entry; control-state limit stops before full-duration holding qualification',
        controller_benefit='No paired comparison; no superiority conclusion')
    for key in fields:
        report['maximum_' + key.replace('_rad', '_deg')] = float(abs(values[key]).max())
    write(BASE / 'diagnostic_metrics.json', report)
    np.savez_compressed(BASE / 'derived_diagnostics.npz', time_s=t, velocity_time_s=t[1:],
        module_velocity_mm_s=velocity, module_speed_mm_s=speed, module_displacement_mm=module_drift,
        root_displacement_mm=root_drift, root_rotation_deg=root_rotation_deg,
        geometric_centroid_mm=centroid, geometric_centroid_displacement_mm=centroid_drift,
        contact_count=contact_count, contact_normal_force_sum_N=f.sum(axis=2),
        tracking_error_deg=tracking, control_time_s=ct, raw_target_deg=raw,
        applied_target_deg=targets, **values)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes[0, 0].plot(t, contact_count[:, 0], label='Source pipe')
    axes[0, 0].plot(t, contact_count[:, 1], label='Target pipe')
    axes[0, 0].axhline(2, color='k', ls=':', lw=1)
    axes[0, 0].set_ylabel('Modules with normal force >0.1 N')
    axes[0, 0].legend()
    axes[0, 1].plot(t[1:], speed.max(axis=1))
    axes[0, 1].axhline(50, color='r', ls=':', label='Holding screening limit')
    axes[0, 1].set_ylabel('Maximum module speed (mm/s)')
    axes[0, 1].legend()
    axes[1, 0].plot(t, root_drift, label='Head free-root translation')
    axes[1, 0].plot(t, centroid_drift, label='Geometric body centroid')
    axes[1, 0].set_ylabel('Displacement from saved initial (mm)')
    axes[1, 0].legend()
    axes[1, 1].plot(t, abs(tracking).max(axis=1))
    axes[1, 1].set_ylabel('Maximum joint tracking error (deg)')
    axes[2, 0].plot(ct, abs(values['compliance_delta_rad']).max(axis=1), label='Local A-D difference')
    axes[2, 0].plot(ct, abs(values['retained_shape_rad']).max(axis=1), label='Retained h state')
    axes[2, 0].set_ylabel('Maximum absolute adjustment (deg)')
    axes[2, 0].legend()
    axes[2, 1].plot(ct, raw[:, 1], label='J2 raw target')
    axes[2, 1].plot(ct, targets[:, 1], label='J2 applied target')
    axes[2, 1].scatter([t[-1]], [rejected_raw[1]], color='r', label='Rejected (offline reconstruction)')
    axes[2, 1].axhline(-90, color='r', ls=':')
    axes[2, 1].set_ylabel('Joint 2 angle (deg)')
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.axvline(5, color='.5', ls='--', lw=1)
        ax.grid(alpha=.2)
    axes[2, 0].set_xlabel('Physical time (s)')
    axes[2, 1].set_xlabel('Physical time (s)')
    fig.suptitle('V3 d=0 single static diagnostic: actual physics 7.55 s, then control limit stop')
    fig.tight_layout()
    fig.savefig(BASE / 'actual_holding_diagnostics.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print(json.dumps(failure, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
