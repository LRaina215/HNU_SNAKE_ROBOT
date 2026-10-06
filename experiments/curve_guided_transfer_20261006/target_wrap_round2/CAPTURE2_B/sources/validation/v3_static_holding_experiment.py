"""One separately authorized d=0 diagnostic; the full V3 path gate stays closed.

--check-only reconstructs the saved first pose without mj_step. --run uses that
same pose once. No initial-pose solver, parameter search, B or release phase.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation'), str(ROOT / 'visualizer')]
import argparse
import csv
import json
import shutil
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import curve_guided_experiment as infrastructure
from curve_grasp_revision_experiment import holding
from snake_control.curve_grasp_revision import FittedMaterialCurve, RetainedGraspController

PARENT = ROOT / 'experiments/curve_guided_transfer_20261006'
V3 = PARENT / 'structural_revision_v3'
BASE = PARENT / 'v3_static_holding_20261006'
CASE = BASE / 'V3_STATIC_C0'
LEDGER = PARENT / 'trial_ledger.csv'
infrastructure.BASE = BASE
write = infrastructure.write


def previous_trials():
    with LEDGER.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if len(rows) != 2 or [r['trial_id'] for r in rows] != ['V1_C0', 'V2_C0']:
        raise RuntimeError('This authorization requires exactly the two preserved earlier trials')
    return fields


def check_initial():
    """Forward evaluation of exactly the saved d=0; never fit or step physics."""
    previous_trials()
    if BASE.exists():
        raise FileExistsError('Preserve this diagnostic directory; do not reconstruct another candidate')
    BASE.mkdir()
    geometry = json.loads((V3 / 'geometry_result.json').read_text(encoding='utf8'))
    if geometry['design_status'] != 'NOT_READY':
        raise RuntimeError('Unexpected full-path status; this diagnostic does not authorize the path')
    with np.load(V3 / 'fitted_reference.npz') as fitted:
        indices = np.flatnonzero(fitted['feed_mm'] == 0.)
        if len(indices) != 1:
            raise ValueError('Exactly one saved d=0 is required')
        state = fitted['states'][indices[0]].copy()
    if state.shape != (26,) or not np.isfinite(state).all():
        raise ValueError('Saved d=0 must contain 26 finite state coordinates')
    row = next(r for r in geometry['rows'] if r['feed_mm'] == 0.)
    model = mujoco.MjModel.from_xml_path(str(infrastructure.MODEL))
    if model.nu != 20 or model.nq != 27 or model.opt.timestep != .001:
        raise ValueError('Original 20-joint 1 ms model required')
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[joints]
    data.qpos[:3] = state[:3] / 1000
    quat = Rotation.from_rotvec(state[3:6]).as_quat()[[3, 0, 1, 2]]
    data.qpos[3:7] = quat
    data.qpos[qadr] = state[6:]
    data.qvel[:] = 0.
    data.ctrl[:] = state[6:]
    mujoco.mj_forward(model, data)
    points = infrastructure.points(model, data)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    reference = FittedMaterialCurve(V3 / 'fitted_reference.npz')
    collision = [dict(geom1=model.geom(c.geom1).name, geom2=model.geom(c.geom2).name,
                      distance_mm=float(c.dist * 1000)) for c in data.contact]
    checks = dict(
        qpos_saved_maximum_difference=float(abs(data.qpos - np.array(row['qpos'])).max()),
        nodes_saved_maximum_difference_mm=float(abs(points - np.array(row['actual_points_mm'])).max()),
        reference_joint_maximum_difference_rad=float(abs(reference.joint_angles(0.) - state[6:]).max()),
        maximum_length_error_mm=float(abs(lengths - 64.5).max()),
        within_joint_limits=bool(np.all(state[6:] >= model.jnt_range[joints, 0]) and
                                 np.all(state[6:] <= model.jnt_range[joints, 1])),
        collision_contacts=collision,
        qvel_zero=bool(np.all(data.qvel == 0)),
        physics_steps=0,
    )
    passed = (checks['qpos_saved_maximum_difference'] < 1e-10 and
              checks['nodes_saved_maximum_difference_mm'] < 1e-7 and
              checks['reference_joint_maximum_difference_rad'] < 1e-12 and
              checks['maximum_length_error_mm'] < 1e-6 and
              checks['within_joint_limits'] and not collision)
    checks.update(initial_pose_check='PASS' if passed else 'FAIL', full_transfer_path='NOT_READY',
                  maximum_joint_angle_deg=float(np.degrees(abs(state[6:])).max()),
                  joint_angles_deg=np.degrees(state[6:]).tolist(),
                  total_length_mm=float(lengths.sum()), saved_grip_modules=row['grip_modules'],
                  source_reference=(V3 / 'fitted_reference.npz').relative_to(ROOT).as_posix(),
                  interpretation='Geometry only: no initial contact force or established grasp')
    write(BASE / 'initial_check.json', checks)
    if not passed:
        raise ValueError('Saved initial pose/interface check failed; no dynamics authorized')
    np.savez_compressed(BASE / 'initial_pose.npz', qpos=data.qpos, qvel=data.qvel, target=data.ctrl,
                        saved_state=state, feed_mm=0., roll_rad=0.)
    write(BASE / 'plan.json', dict(
        scope='Saved V3 d=0 static holding only; no B, rear expansion or release',
        total_seconds=45.25, entry_seconds=5., control_period_seconds=.05,
        sample_period_seconds=.01, physics_period_seconds=.001,
        feed_mm=0., roll_rad=0., full_transfer_path='NOT_READY',
        holding_speed_limit_mm_s=50., holding_modules_per_pipe=2, normal_force_threshold_N=.1,
        holding_screen='Every sample in last 1 s at entry t=5 and completion t=45.25',
        stop_conditions=dict(ground=True, nonadjacent_self_contact=True,
            whole_body_support_loss_after_1s_for_4_control_intervals=True,
            maximum_module_speed_above_500_mm_s_after_1s_for_4_control_intervals=True,
            invalid_feedback_or_target=True),
        physics_parameters_changed=False, controller_changed=False,
        gains=[.01, .02], target_rate_limit_deg_s=10., target_joint_limit_deg=90.,
        feedback=['fresh_encoder_angles', 'last_sent_targets', 'retained_shape_state'],
        known_prior='Saved initial rigid registration and nominal pipe axes; no online world pose',
        diagnostic_only=['exact_contacts', 'world_pose', 'module_speed', 'body_drift'],
        previous_dynamics_trials=2, this_round_maximum_dynamics_trials=1,
        initial_placement='Saved d=0 qpos/ctrl, zero qvel before first physical step only',
        after_physics_start='No reset, no external support or additional initial search',
        runtime=dict(python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__),
    ))
    write(BASE / 'model_parameters.json', dict(
        model=infrastructure.MODEL.relative_to(ROOT).as_posix(), timestep=float(model.opt.timestep),
        gravity=model.opt.gravity.tolist(), body_mass=model.body_mass.tolist(),
        geom_names=[model.geom(i).name for i in range(model.ngeom)],
        geom_friction=model.geom_friction.tolist(), geom_condim=model.geom_condim.tolist(),
        joint_limits_rad=model.jnt_range[joints].tolist(),
        damping=model.dof_damping.tolist(), armature=model.dof_armature.tolist(),
        actuator_gain=model.actuator_gainprm.tolist(), actuator_bias=model.actuator_biasprm.tolist(),
        actuator_force_range=model.actuator_forcerange.tolist(),
        parameters_read_only=True))
    sources = ['validation/v3_static_holding_experiment.py',
               'validation/curve_grasp_revision_experiment.py', 'validation/curve_guided_experiment.py',
               'validation/curve_guided_geometry.py', 'snake_control/curve_grasp_revision.py',
               'snake_control/curve_guided_transfer.py', 'snake_control/curve_transfer_controller.py',
               'snake_control/pole_compliance.py', 'snake_control/three_segment_compliance.py',
               'snake_control/mujoco_execution_bridge.py', 'snake_control/mujoco_visualizer_bridge.py',
               'visualizer/python_client.py', 'validation/fixtures/app-state.json',
               'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml',
               'mujoco_model/asset/c_link_H.stl', 'mujoco_model/asset/c_link_V.stl']
    for name in sources:
        dest = BASE / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    # Save input data locally for reproduction without changing any V3 originals.
    for name in ['fitted_reference.npz', 'geometry_result.json', 'motion_revision.md', 'result_report.md']:
        dest = BASE / 'inputs' / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(V3 / name, dest)
    print(json.dumps(checks, ensure_ascii=False), flush=True)


class StaticTrial(infrastructure.Trial):
    def __init__(self):
        self.exact_contacts = []
        super().__init__('C0', 'V3_STATIC', 'initial_pose.npz')

    def initialize_relay(self):
        super().initialize_relay()
        self.controller = RetainedGraspController(
            FittedMaterialCurve(V3 / 'fitted_reference.npz'), self.controller.config,
            self.d.ctrl.copy(), self.encoder_fk, self.d.body('head').xmat.reshape(3, 3).copy())
        write(self.case / 'config.json', self.controller.config)

    def snapshot(self):
        super().snapshot()
        self.rows[-1].update(external_applied_force_maximum=float(abs(self.d.xfrc_applied).max()),
                             generalized_applied_force_maximum=float(abs(self.d.qfrc_applied).max()))
        contacts = []
        for index, c in enumerate(self.scratch.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.scratch, index, force)
            contacts.append(dict(geom1=self.m.geom(c.geom1).name, geom2=self.m.geom(c.geom2).name,
                body1=self.m.body(self.m.geom_bodyid[c.geom1]).name,
                body2=self.m.body(self.m.geom_bodyid[c.geom2]).name,
                position_mm=(c.pos * 1000).tolist(), distance_mm=float(c.dist * 1000),
                contact_frame_force_N_Nm=force.tolist()))
        self.exact_contacts.append(dict(time_s=float(self.d.time), contacts=contacts))

    def close(self):
        super().close()
        write(self.case / 'exact_contacts.json', self.exact_contacts)


def run_once():
    fields = previous_trials()
    if CASE.exists() or (BASE / 'physics_started.json').exists():
        raise FileExistsError('This sole diagnostic was already attempted; never restart it')
    check = json.loads((BASE / 'initial_check.json').read_text(encoding='utf8'))
    if check['initial_pose_check'] != 'PASS':
        raise RuntimeError('Saved-pose check required before this one diagnostic')
    trial = StaticTrial()
    shutil.copy2(BASE / 'plan.json', trial.case / 'plan.json')
    result = dict(condition='static_C0', version='V3', outcome='engineering_failure',
                  static_holding_pass=False, initial_usable=False, local_transfer=False,
                  full_transfer_path='NOT_READY', body_feed_mm=0., roll_rad=0.,
                  real_hardware=False, B_run=False, F_run=False)
    start = time.monotonic()
    try:
        trial.initialize_relay()
        lost = fast = 0
        for k in range(905):
            t = k * .05
            trial.feed = 0.
            trial.stage = 'entry' if t < 5 else 'static_hold'
            geometry = trial.fresh()
            measured = np.radians(np.asarray(geometry['angles_deg']) * trial.bridge.signs)
            target, control = trial.controller.compute(measured, 0., t, .05)
            control.update(time_s=t, stage=trial.stage)
            trial.controls.append(control)
            trial.dispatch(target, geometry)
            if k == 0:
                if abs(target - trial.initial_qpos[trial.qadr]).max() > 1e-12:
                    raise ValueError('t=0 interface changed the saved initial joint target')
                write(BASE / 'physics_started.json', dict(
                    first_step_about_to_start=True, previous_trials=2, reserved_cumulative_trial=3,
                    attempted_trial='V3_STATIC_C0', never_restart=True, wall_unix_ms=time.time() * 1000))
            for step in range(50):
                mujoco.mj_step(trial.m, trial.d)
                if (step + 1) % 10 == 0:
                    trial.snapshot()
            recent = trial.rows[-6:]
            last = recent[-1]
            speed = float(np.linalg.norm((recent[-1]['centers_mm'] - recent[0]['centers_mm']) / .05, axis=1).max())
            lost = lost + 1 if not np.any(last['contact_force_N'] > .1) else 0
            fast = fast + 1 if speed > 500 else 0
            if last['ground'] or last['self_contact']:
                result['outcome'] = 'ground_or_nonadjacent_self_contact'
                break
            if t >= 1 and (lost >= 4 or fast >= 4):
                result.update(outcome='support_loss_or_uncontrolled_motion',
                              stopping_speed_mm_s=speed, lost_intervals=lost, fast_intervals=fast)
                break
            if k == 99:
                usable, maximum = holding(trial.rows)
                result.update(initial_usable=usable, entry_maximum_speed_mm_s=maximum)
                if not usable:
                    result['outcome'] = 'initial_cannot_hold_on_both_pipes'
                    break
            if k % 100 == 99:
                print(json.dumps(dict(time_s=float(trial.d.time), static_C0=True, speed_mm_s=speed)), flush=True)
        else:
            usable, maximum = holding(trial.rows)
            result.update(static_holding_pass=bool(usable), initial_usable=bool(usable),
                          final_maximum_speed_mm_s=maximum,
                          outcome='completed_stable_static_C0' if usable else 'final_C0_cannot_hold')
    except ValueError as error:
        result.update(outcome='controller_target_or_geometry_stop', error=str(error))
    except Exception as error:
        result.update(error=repr(error))
    finally:
        result.update(physical_seconds=float(trial.d.time), wall_seconds=time.monotonic() - start,
                      new_dynamics_trials=int(trial.d.time > 0),
                      cumulative_dynamics_trials=2 + int(trial.d.time > 0))
        if not result['static_holding_pass']:
            result['initial_usable'] = False
        trial.close()
    centers = np.array([r['centers_mm'] for r in trial.rows])
    intervals = np.diff([r['sim_time'] for r in trial.rows])
    speeds = np.linalg.norm(np.diff(centers, axis=0) / intervals[:, None, None], axis=2) if len(intervals) else np.zeros((1, 21))
    q = np.array([r['q'] for r in trial.rows])
    ctrl = np.array([r['ctrl'] for r in trial.rows])
    result.update(maximum_module_speed_mm_s=float(speeds.max()),
                  maximum_tracking_error_deg=float(np.degrees(abs(q - ctrl)).max()),
                  root_displacement_mm=float(np.linalg.norm(trial.d.qpos[:3] - trial.initial_qpos[:3]) * 1000),
                  commands=len(trial.commands), all_commands_applied=all(r['applied'] for r in trial.commands),
                  completion_scope='Static holding only; no full-path release or controller superiority claim')
    write(trial.case / 'result.json', result)
    if result['new_dynamics_trials']:
        # Append only: the two earlier failures and their names/budget stay intact.
        row = {field: '' for field in fields}
        row.update(trial_id='V3_STATIC_C0', stage='independent_static_holding', version='V3', condition='C0',
            initial_usable=result['static_holding_pass'], physical_seconds=result['physical_seconds'],
            wall_seconds=result['wall_seconds'], outcome=result['outcome'],
            evidence=trial.case.relative_to(ROOT).as_posix(), initial_file='v3_static_holding_20261006/initial_pose.npz',
            comparability='saved_V3_d0_and_retained_control_no_paired_comparison',
            maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],
            entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s', 'NOT_REACHED'),
            maximum_tracking_error_deg=result['maximum_tracking_error_deg'], root_displacement_mm=result['root_displacement_mm'],
            commanded_feed_mm=0., metrics_evidence='v3_static_holding_20261006/V3_STATIC_C0/result.json')
        with LEDGER.open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        infrastructure.replay(trial.case)
    write(BASE / 'simulation_gate.json', dict(
        full_transfer_path='NOT_READY', static_holding=result['static_holding_pass'],
        new_dynamics_trials=result['new_dynamics_trials'], cumulative_dynamics_trials=result['cumulative_dynamics_trials'],
        B_run=False, F_run=False, real_trials=0, next_step='STOP: this authorization ends after the single static diagnostic'))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check_initial()
    else:
        run_once()


if __name__ == '__main__':
    main()
