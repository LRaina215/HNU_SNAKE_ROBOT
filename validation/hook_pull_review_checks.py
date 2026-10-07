"""CLOSE1B units/state/zero-input checks only. No physical evolution."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import json
import time
from types import SimpleNamespace
import mujoco
import numpy as np
import hook_pull_phase_aprime_v2 as v2
import hook_pull_phase_aprime_v3 as v3
import hook_pull_phase_b as b
from snake_control.hook_pull_motion import RollSolver
from snake_control.hook_pull_review_support import axis_progress_mm, continuous_target

OUT = ROOT / 'experiments/hook_pull_review_20261007'


def main():
    # Fail closed if a future code change accidentally evolves physics here.
    attempts = []
    original_step = mujoco.mj_step
    def forbidden_step(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('Physical evolution is forbidden in this check')
    mujoco.mj_step = forbidden_step
    try:
        expected_turn_m = float(np.hypot(2 * np.pi * .095, .220))
        expected_rotation = 2 * np.pi / (expected_turn_m * 1000)
        expected_axial = .220 / expected_turn_m
        for module in (v2, v3, b):
            assert np.isclose(module.ROT_PER_MM, expected_rotation, rtol=1e-14)
            assert np.isclose(module.AX_PER_MM, expected_axial, rtol=1e-14)
        unit_rows = [dict(arc_mm=u, rotation_rad=expected_rotation * u,
                         rotation_deg=float(np.degrees(expected_rotation * u)),
                         axial_mm=expected_axial * u,
                         independent_arc_mm=float(np.hypot(95 * expected_rotation * u, expected_axial * u)),
                         old_unwrapped_rotation_rad=2 * np.pi / expected_turn_m * u)
                     for u in (2., 8.)]
        assert all(abs(r['independent_arc_mm'] - r['arc_mm']) < 1e-12 for r in unit_rows)

        probe = b.Runner('CLOSE1B', record=False, case=OUT / 'state_restore_probe')
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        reference = mujoco.MjData(probe.m)
        with np.load(b.BASE / 'CLOSE1B/final_state.npz') as saved:
            saved_integration = saved['integration_state'].copy()
            mujoco.mj_setState(probe.m, reference, saved_integration, spec)
            separate_ctrl = saved['ctrl'].copy()
        mujoco.mj_forward(probe.m, reference)
        restored = np.empty(mujoco.mj_stateSize(probe.m, spec))
        mujoco.mj_getState(probe.m, probe.d, restored, spec)
        q0, ctrl0 = reference.qpos[probe.qadr].copy(), reference.ctrl.copy()
        before_error = ctrl0 - q0
        restored_error = probe.d.ctrl - probe.d.qpos[probe.qadr]
        restoration = dict(
            integration_max_difference=float(abs(restored - saved_integration).max()),
            qpos_max_difference=float(abs(probe.d.qpos - reference.qpos).max()),
            qvel_max_difference=float(abs(probe.d.qvel - reference.qvel).max()),
            ctrl_max_difference=float(abs(probe.d.ctrl - separate_ctrl).max()),
            preload_angle_error_max_difference=float(abs(restored_error - before_error).max()),
            actuator_force_max_difference=float(abs(probe.d.qfrc_actuator - reference.qfrc_actuator).max()),
            saved_preload_angle_deg=np.degrees(before_error).tolist(),
            maximum_saved_preload_angle_deg=float(abs(np.degrees(before_error)).max()),
            saved_qvel_maximum=float(abs(reference.qvel).max()),
            restored_time_s=float(probe.d.time), saved_controller_history=probe.start_controller_history)
        for key in ('integration_max_difference', 'qpos_max_difference', 'qvel_max_difference',
                    'ctrl_max_difference', 'preload_angle_error_max_difference', 'actuator_force_max_difference'):
            assert restoration[key] < 1e-12, (key, restoration[key])

        # Exactly one zero-motion residual evaluation/solve; no nonzero candidate.
        # This exercises the actual plan_step on private kinematic data.
        planner = RollSolver(probe.m, probe.d.qpos[:3] * 1000, probe.d.qpos[3:7], q0, list(range(10, 21)))
        probe.planner, probe.x_plan = planner, planner.x0.copy()
        probe.p0_plan = planner.p0.copy()
        centers = probe.centers()
        order = np.argsort(centers[:9, 1])
        theta = np.unwrap(np.arctan2(centers[:9, 2] - 1000, centers[:9, 0] - 1100)[order])
        probe.kappa = float(np.polyfit(centers[:9, 1][order], theta, 1)[0])
        probe.arc_per_rad = 8 * 64.5 / max(abs(theta[-1] - theta[0]), 1e-6)
        probe.w_plan = np.full(22, 15.)
        probe.w_plan[:9] = probe.w_plan[10:] = 60.
        probe.arc_before = {n: (n - 10) * 64.5 + 32.25 for n in range(10, 22)}
        original_solve = b.least_squares
        solver_info = {}
        def measured_solve(*args, **kwargs):
            started = time.monotonic()
            result = original_solve(*args, **kwargs)
            solver_info.update(nfev=int(result.nfev), cost=float(result.cost),
                               wall_s=time.monotonic() - started)
            return result
        b.least_squares = measured_solve
        try:
            zero_plan = probe.plan_step(0., 0., False)
        finally:
            b.least_squares = original_solve
        after_plan = restored.copy()
        mujoco.mj_getState(probe.m, probe.d, after_plan, spec)
        assert np.array_equal(restored, after_plan), 'Planner mutated physical integration state'
        first = continuous_target(ctrl0, zero_plan[6:], b.RATE, b.TICK)
        assert abs(first - ctrl0).max() <= b.RATE * b.TICK + 1e-12
        zero_input = dict(
            input_front_arc_mm=0., input_rear_arc_mm=0., solver=solver_info,
            plan_minus_measured_max_deg=float(abs(np.degrees(zero_plan[6:] - q0)).max()),
            physical_state_max_difference_after_plan=float(abs(after_plan - restored).max()),
            first_target_minus_saved_ctrl_deg=np.degrees(first - ctrl0).tolist(),
            maximum_first_target_change_deg=float(abs(np.degrees(first - ctrl0)).max()),
            zero_motion_preserves_preload=bool(np.allclose(first, ctrl0, rtol=0, atol=1e-12)),
            first_target_not_dispatched=True,
            interpretation='Loading is lossless; absolute measured-shape target still unloads preload at zero input')

        # Actual B scoring method, with controlled synthetic mm displacements.
        scoring_rows = []
        for front, rear in ((.025, .025), (24.999, 25.), (25., 25.)):
            synthetic = np.zeros((21, 3))
            synthetic[:8, 1], synthetic[13:21, 0] = front, rear
            evaluator = object.__new__(b.Runner)
            evaluator.start = 'synthetic_mm_boundary'
            evaluator.centers = lambda: synthetic
            forces = np.zeros((2, 21)); forces[1, 8] = .11
            evaluator.module_forces = lambda: forces
            evaluator.wrap_deg = lambda: 200.
            evaluator.transition_torque = lambda: 0.
            scored = evaluator.evaluate([dict(event='cross_onto_target', module=8)], [], 'not_run', 0., 0.)
            assert scored['B2']['front_y_mm'] == front and scored['B2']['rear_x_mm'] == rear
            assert scored['B2']['passed'] == (front >= 25 and rear >= 25)
            assert scored['verdict'] == 'UNVALIDATED_SCORING'
            scoring_rows.append(dict(input_mm=[front, rear], B2=scored['B2'], verdict=scored['verdict']))

        # Relevant threshold boundaries using the actual B danger method.
        body_geom = {int(probe.m.geom_bodyid[g]): g for g in range(probe.m.ngeom)}
        robot_geom, distant_robot = body_geom[probe.head], body_geom[probe.head + 2]
        pipe_geom = next(iter(probe.pipe_source))
        safety_rows = []
        for kind, distance_m in (('pipe', -.003), ('pipe', -.0031), ('self', -.00049), ('self', -.00051)):
            dummy = object.__new__(b.Runner)
            dummy.__dict__.update(probe.__dict__)
            dummy.d = SimpleNamespace(contact=[SimpleNamespace(
                geom1=robot_geom, geom2=pipe_geom if kind == 'pipe' else distant_robot, dist=distance_m)])
            depth_mm, ground, selfhit = dummy.danger()
            assert selfhit == (kind == 'self' and distance_m < -.0005)
            assert (depth_mm < -3) == (kind == 'pipe' and distance_m < -.003)
            safety_rows.append(dict(kind=kind, input_m=distance_m, pipe_depth_mm=depth_mm,
                                    pipe_penetration_stop=depth_mm < -3, self_contact_stop=selfhit))
        forces = probe.module_forces()
        contacts = dict(source_modules_above_0p1_N=np.flatnonzero(forces[0] > .1).tolist(),
                        target_modules_above_0p1_N=np.flatnonzero(forces[1] > .1).tolist(),
                        target_total_normal_N=float(forces[1].sum()),
                        old_target_position_gate_count=int(sum(c.pos[1] > 75. for c in probe.d.contact)),
                        interpretation='Single saved-state forward diagnosis; not a new holding test')
        target_gate = [dict(y_m=y, counted=y * 1000 > 75.) for y in (.0749, .075, .0751)]
        assert [r['counted'] for r in target_gate] == [False, False, True]
        assert not b.DYNAMICS_ADMITTED
        report = dict(scope='CLOSE1B A-prime/B source and zero-input interface only',
                      physics_steps=0, physics_step_attempts=len(attempts), new_dynamics_trials=0,
                      nonzero_action_candidates_solved=0, units=unit_rows,
                      restoration=restoration, zero_motion_first_target=zero_input,
                      scoring_boundaries=scoring_rows, safety_boundaries=safety_rows,
                      target_contact_position_gate=target_gate, saved_contact_diagnosis=contacts,
                      status='NUMERIC_REPAIRS_CHECKED; BLOCKED_PRELOAD_MAPPING_AND_ACTION_ADMISSION')
        assert not attempts
        (OUT / 'numerical_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        mujoco.mj_step = original_step


if __name__ == '__main__':
    main()
