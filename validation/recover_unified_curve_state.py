"""Recover omitted curve history from recorded observations, without dynamics.

Uses the run's frozen feedback implementation and authoritative dispatched
shift/speed/targets. It neither fits a new route nor steps MuJoCo. Future runs
save this controller checkpoint directly.
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


def export_state(controller, elapsed_s):
    values = {k: v for k, v in controller.__dict__.items()
              if k not in {'m', 'd', 'qa', 'va', 'head', 'curve', 'first_dispatch', 'pipe_geometry', 'execution_tasks', 'capture_execution_tasks'}}
    curve = {k: v for k, v in controller.curve.__dict__.items()
             if k not in {'base', 'arc_cache', 'clearance'}}
    curve['base_x'] = controller.curve.base.x
    curve['base_c'] = controller.curve.base.c
    reference, _, _ = controller.curve.reference(controller.shift)
    return dict(version=1, elapsed_s=elapsed_s, controller=values, curve=curve,
                reference_points_mm=reference*1000)


def save_state(path, controller, elapsed_s):
    path.write_text(json.dumps(export_state(controller, elapsed_s), ensure_ascii=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf8')


def recover(case):
    spec = importlib.util.spec_from_file_location('frozen_curve', case/'sources/snake_control/unified_curve_transfer.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    data = mujoco.MjData(model)
    trace = np.load(case/'trace.npz')
    commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
    data.qpos[:] = trace['qpos'][0]
    data.ctrl[:] = trace['ctrl'][0]
    controller = frozen.UnifiedCurveTransfer(model, data)
    controller.support_force_goals = np.maximum(trace['contact_force_N'][0].sum(1), 1.)
    maximum_error = 0.
    for tick, command in enumerate(commands):
        t, index = command['t_s'], tick*5
        data.qpos[:] = trace['qpos'][index]
        p, _ = controller.observe(data)
        q = data.qpos[controller.qa]
        _, reference, locations = controller.curve.reference(controller.shift)
        controller.filtered_load += .1*(np.abs(trace['torque'][index])-controller.filtered_load)
        if t-controller.last_shape_time >= .5:
            controller.feedback_shape(p, q, reference, locations, np.abs(controller.last-q),
                                      trace['contact_force_N'][index], .5)
            controller.last_shape_time = t
        ds = frozen.DT*command['shift_speed_mm_s']/1000
        controller.curve.increment(ds)
        controller.shift += ds
        controller.curve.reference(controller.shift)
        controller.last = np.asarray(command['target_rad'])
        if t-controller.last_observer_time >= .5:
            controller.last_observer_time = t
        actual = controller.curve.parameters()
        expected = command['curve_parameters']
        for key in actual:
            maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(actual[key])-expected[key]))))
        if tick % 1000 == 0:
            print(f'Recovered observations through {t:.2f}s; curve parameter error {maximum_error:.3g}', flush=True)
    last = commands[-1]
    controller.speed = last['shift_speed_mm_s']/1000
    controller.observed_shift = controller.last_body_progress = last['actual_material_estimate_mm']/1000
    controller.projected_locations = np.asarray(last['observed_projected_locations_mm'])/1000
    controller.last_mode = last['stage']
    controller.progress_history = [(x['t_s'], x['actual_material_estimate_mm']/1000)
                                  for x in commands if x['t_s'] >= last['t_s']-10.]
    result = json.loads((case/'result.json').read_text(encoding='utf8'))
    controller.events = result['controller_events']
    elapsed = float(trace['sim_time'][-1]-trace['sim_time'][0])
    if maximum_error > 1e-7 or controller.n_adaptations != result['curve_feedback_updates']:
        raise RuntimeError(f'Curve history disagrees with original run: {maximum_error}')
    if not np.array_equal(controller.last, trace['ctrl'][-1]):
        raise RuntimeError('Recovered last command differs from saved physical ctrl')
    save_state(case/'controller_state.json', controller, elapsed)
    report = dict(source=str(case), reconstruction='frozen feedback on original 100 Hz observations; no physics steps or curve fit',
                  maximum_recorded_curve_parameter_error=maximum_error,
                  feedback_updates=controller.n_adaptations, elapsed_s=elapsed,
                  restored_shift_mm=controller.shift*1000, last_ctrl_matches=True)
    (case/'controller_recovery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path)
    recover(parser.parse_args().case)
