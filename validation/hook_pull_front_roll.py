"""钩挂回拉 第4步（第一段）：前部抱住目标管后，前部滚动沿目标管远离接头，后部保持。

接续第3步终态，不重置。离线先求“前部滚动 + 后部保持 + 过渡段自由”的关节表（SplitRollSolver）：
前部 M0–M7 绕自身纵轴滚动，并按目标管上理想不打滑螺旋运动整体移动（+Y，远离接头）；
后部 M10–M20 节点与姿态保持；过渡段 M8–M9 自由伸展。表在约束节点误差超过 5 mm 或关节到限时截止。
动力学：front_roll（沿表滚动，目标变化 ≤10°/s）→ hold 5 s。只回答一个问题：
抱住目标管的前部能否沿目标管真正向前滚、同时保持抱紧。后部跟进不在这一段。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse
import csv
import json
import shutil
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from curve_guided_geometry import MODEL, CORNER
from snake_control.hook_pull_motion import SplitRollSolver, RollTable
from hook_pull_experiment import BASE, DT, RATE, Sim, replay_video, write, smooth
from hook_pull_close import wrap_angle_deg, evaluate

FRONT, REAR = list(range(0, 8)), list(range(10, 21))
TARGET_POINT, TARGET_AXIS = np.array([1100., 0., 1000.]), np.array([0., 1., 0.])


def front_screw_factory(axial_mm_per_rad, rotation_deg_per_rad):
    def screw(psi, nodes):
        rotation = Rotation.from_rotvec(TARGET_AXIS * np.radians(rotation_deg_per_rad) * psi)
        moved = TARGET_POINT + rotation.apply(nodes - TARGET_POINT) + TARGET_AXIS * axial_mm_per_rad * psi
        return moved, rotation
    return screw


def build_table(previous, case, psi_limit_deg):
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    with np.load(BASE / previous / 'final_state.npz') as state:
        mujoco.mj_setState(m, d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        ctrl = state['ctrl'].copy()
    mujoco.mj_forward(m, d)
    direction = json.loads((BASE / 'offline/forward_roll_direction.json').read_text(encoding='utf8'))['front_on_target']
    sign = 1. if direction['axial_mm_per_rad'] > 0 else -1.  # 取让前部沿 +Y 远离接头的滚动方向
    screw = front_screw_factory(direction['axial_mm_per_rad'], direction['rotation_deg_per_rad'])
    solver = SplitRollSolver(m, d.qpos[:3] * 1000, d.qpos[3:7], ctrl, FRONT, REAR, screw)
    psis = sign * np.radians(np.arange(0., psi_limit_deg + .1, 2.5))
    rows = solver.table(psis)
    report, keep = [], 0
    for k, r in enumerate(rows):
        p, _ = solver.forward(r['x'])
        moved, _ = screw(r['psi'], solver.p0[solver.front_nodes])
        front_error = float(np.linalg.norm(p[solver.front_nodes] - moved, axis=1).max())
        rear_error = float(np.linalg.norm(p[solver.rear_nodes] - solver.p0[solver.rear_nodes], axis=1).max())
        joint = float(np.degrees(abs(r['x'][6:])).max())
        ok = front_error <= 5. and rear_error <= 5. and joint <= 88.
        report.append(dict(psi_deg=float(np.degrees(r['psi'])), front_node_error_mm=front_error,
                           rear_node_error_mm=rear_error, max_joint_deg=joint, admitted=ok,
                           planned_front_advance_mm=float(direction['axial_mm_per_rad'] * r['psi'])))
        if not ok:
            break
        keep = k + 1
    psi = psis[:keep]
    joints = np.array([r['x'][6:] for r in rows[:keep]])
    order = np.argsort(psi)
    np.savez(case / 'front_roll_table.npz', psi_rad=psi[order], joint_rad=joints[order], sign=sign)
    write(case / 'front_roll_table.json', dict(direction=direction, roll_sign=sign, admitted_rows=keep, rows=report))
    return RollTable(psi[order], joints[order]), float(psi[keep - 1]), sign, report[:keep]


def run(previous, name, psi_limit_deg, seconds):
    case = BASE / name
    case.mkdir(parents=True, exist_ok=False)
    for src in ['snake_control/hook_pull_motion.py', 'validation/hook_pull_experiment.py',
                'validation/hook_pull_close.py', 'validation/hook_pull_front_roll.py']:
        dest = case / 'sources' / src
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, dest)
    table, psi_end, sign, admitted = build_table(previous, case, psi_limit_deg)
    plan = dict(start=f'{previous} final integration state and ctrl, no reset', front_modules='M0-M7', rear_modules='M10-M20',
                transition='M8-M9 free', admitted_front_roll_deg=float(np.degrees(psi_end)),
                planned_front_advance_mm=admitted[-1]['planned_front_advance_mm'], roll_seconds=seconds, hold_s=5.,
                rate_limit_deg_s=10., control='joint position targets from offline table only; no truth feedback',
                question='does the grasping front actually roll forward along the target pipe while keeping the wrap')
    write(case / 'plan.json', plan)
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    sim = Sim(case)
    with np.load(BASE / previous / 'final_state.npz') as state:
        mujoco.mj_setState(sim.m, sim.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        sim.d.ctrl[:] = state['ctrl']
    mujoco.mj_forward(sim.m, sim.d)
    sim.psi, sim.stage = 0., 'start'
    sim.snapshot()
    started = time.monotonic()
    result = dict(name=name, previous=previous, outcome='engineering_failure')
    last = sim.d.ctrl.copy()
    try:
        lost = fast = 0
        for stage, duration in [('front_roll', seconds), ('hold', 5.)]:
            sim.stage = stage
            for k in range(int(round(duration / DT))):
                if stage == 'front_roll':
                    sim.psi = psi_end * smooth((k + 1) * DT / duration)
                target = np.clip(table(sim.psi), last - RATE, last + RATE)
                target = np.clip(target, -np.pi / 2, np.pi / 2)
                sim.commands.append(dict(time_s=float(sim.d.time), stage=stage, psi_rad=float(sim.psi), target_rad=target.tolist()))
                sim.step_command(target)
                last = target
                r = sim.rows[-1]
                recent = sim.rows[-6:]
                speed = float(np.linalg.norm((recent[-1]['centers_mm'] - recent[0]['centers_mm']) / DT, axis=1).max())
                fast = fast + 1 if speed > 500 else 0
                lost = lost + 1 if np.count_nonzero(r['contact_force_N'][0] > .1) < 2 else 0
                if r['ground'] or r['self_contact']:
                    raise StopIteration('ground_or_nonadjacent_self_contact')
                if lost >= 4:
                    raise StopIteration('source_support_below_two_modules_0p2s')
                if fast >= 4:
                    raise StopIteration('uncontrolled_motion')
                if len(sim.commands) % 40 == 0:
                    print(json.dumps(dict(t=round(float(sim.d.time), 2), stage=stage, psi=round(float(np.degrees(sim.psi)), 1),
                        src=int(np.count_nonzero(r['contact_force_N'][0] > .1)),
                        tgt=np.flatnonzero(r['contact_force_N'][1] > .1).tolist(), enclosed=r['pipe_enclosed'],
                        wrap=round(wrap_angle_deg(r['centers_mm']), 1),
                        front_y=round(float(r['centers_mm'][:8, 1].mean() - sim.rows[0]['centers_mm'][:8, 1].mean()), 1))), flush=True)
        result['outcome'] = 'completed'
    except StopIteration as stop:
        result['outcome'] = str(stop)
    except Exception as error:
        result.update(outcome='controller_or_interface_stop', error=repr(error))
    finally:
        sim.save()
    rows = sim.rows
    c0, c1 = rows[0]['centers_mm'], rows[-1]['centers_mm']
    y = np.array([r['centers_mm'][:8, 1].mean() for r in rows]) - c0[:8, 1].mean()
    err = np.degrees(abs(np.array([r['ctrl'] - r['q'] for r in rows])))
    torque = abs(np.array([r['torque'] for r in rows]))
    result.update(physical_seconds=float(rows[-1]['sim_time'] - rows[0]['sim_time']), wall_seconds=time.monotonic() - started,
                  planned_front_advance_mm=admitted[-1]['planned_front_advance_mm'],
                  actual_front_mean_y_advance_mm=float(y[-1]), actual_front_max_y_advance_mm=float(y.max()),
                  front_module_y_advance_mm=np.round(c1[:8, 1] - c0[:8, 1], 1).tolist(),
                  rear_module_displacement_mm=np.round(np.linalg.norm(c1[10:] - c0[10:], axis=1), 1).tolist(),
                  start_wrap_deg=wrap_angle_deg(c0), final_wrap_deg=wrap_angle_deg(c1),
                  maximum_tracking_error_deg=float(err.max()),
                  torque_saturation_fraction=float(np.mean(torque > 2.355 * .999)),
                  maximum_module_speed_mm_s=float(np.linalg.norm(np.diff([r['centers_mm'] for r in rows], axis=0), axis=2).max() / .01))
    if result['outcome'] == 'completed':
        result['final_hold'] = evaluate(rows, rows[0]['sim_time'])
    write(case / 'result.json', result)
    with (BASE / 'trial_ledger.csv').open('a', encoding='utf8', newline='') as f:
        csv.DictWriter(f, fieldnames=['trial_id', 'step', 'outcome', 'physical_seconds_after_branch', 'final_psi_deg',
                                      'evidence']).writerow(
            dict(trial_id=name, step='4a_front_roll', outcome=result['outcome'], physical_seconds_after_branch=result['physical_seconds'],
                 final_psi_deg=float(np.degrees(sim.psi)), evidence=case.relative_to(ROOT).as_posix()))
    replay_video(case, rows)
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', default='CLOSE1')
    parser.add_argument('--name', default='FRONT1')
    parser.add_argument('--psi-limit-deg', type=float, default=180.)
    parser.add_argument('--seconds', type=float, default=30.)
    args = parser.parse_args()
    run(args.previous, args.name, args.psi_limit_deg, args.seconds)
