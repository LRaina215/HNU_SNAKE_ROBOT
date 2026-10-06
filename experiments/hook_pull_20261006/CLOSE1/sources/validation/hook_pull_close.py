"""钩挂回拉 第3步：钩子挂住后收紧，再放松回拉，检查能否独立抱住目标管。

接续第2步终态（完整积分状态 + ctrl），不重置。
收紧（沿用 2023 顺应性“按实测形状收紧 K”的思路，直接作用在关节角上）：
钩子关节 J1–J6 的目标 = 实测角 ×(1+K)，每 50 ms 按当前实测重算，单关节最多卷到起始角的 MAX_CURL 倍；
没碰到管子时钩子逐步卷紧，被管子挡住后停在管子的实际形状上，只保持 K 对应的压紧量。
其余关节保持第2步回拉终了的目标（后部继续拉住）。
时序：close 10 s → hold 5 s → release（整身滚动表向前回退 release_deg，约松掉回拉）→ hold 10 s。
控制只用编码器角与已下发目标；接触真值只用于评分与安全停止。
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

from curve_guided_geometry import CORNER
from snake_control.hook_pull_motion import RollTable
from hook_pull_experiment import BASE, DT, RATE, HOOK, Sim, replay_video, write, smooth

CLOSE_JOINTS = list(range(0, 6))  # J1–J6：M6 已贴在管子右上方，以它为根让 M0–M5 卷向管子
JOINT_LIMIT = np.radians(88.)
MAX_CURL = 2.0  # 每个关节最多卷到起始角的 2 倍（开口半径约 157 mm → 贴管约 84 mm，曲率比约 1.9）


def wrap_angle_deg(centers):
    """M8→M0 在目标管截面内绕管轴累计转过的角度。"""
    uv = (centers[:9] - CORNER)[:, [0, 2]][::-1]
    a = np.arctan2(uv[:, 1], uv[:, 0])
    return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def evaluate(rows, start_time):
    window = [r for r in rows if r['sim_time'] >= rows[-1]['sim_time'] - 10 - 1e-9 and r['sim_time'] > start_time]
    end = rows[-1]['sim_time']
    p = np.array([r['centers_mm'] for r in window])
    t = np.array([r['sim_time'] for r in window])
    last2 = p[t >= end - 2 - 1e-9]
    last5 = t >= end - 5 - 1e-9
    last1 = [r for r in window if r['sim_time'] >= end - 1 - 1e-9]
    speed5 = np.linalg.norm(np.diff(p[last5], axis=0) / np.diff(t[last5])[:, None, None], axis=2).max()
    metrics = dict(
        last2s_drift_mm=float(np.linalg.norm(last2 - last2[0], axis=2).max()),
        last5s_max_module_speed_mm_s=float(speed5),
        last1s_min_source_contacts=int(min(np.count_nonzero(r['contact_force_N'][0] > .1) for r in last1)),
        last1s_min_target_contacts=int(min(np.count_nonzero(r['contact_force_N'][1] > .1) for r in last1)),
        last1s_pipe_enclosed_fraction=float(np.mean([r['pipe_enclosed'] for r in last1])),
        final_wrap_deg=wrap_angle_deg(rows[-1]['centers_mm']),
        final_target_contact_modules=np.flatnonzero(rows[-1]['contact_force_N'][1] > .1).tolist(),
        final_source_contact_modules=np.flatnonzero(rows[-1]['contact_force_N'][0] > .1).tolist(),
        final_hook_gap_mm=np.round(np.linalg.norm((rows[-1]['centers_mm'][:9] - CORNER)[:, [0, 2]], axis=1) - 84, 1).tolist())
    metrics['independent_grasp'] = bool(metrics['last2s_drift_mm'] <= 6.45 and metrics['last5s_max_module_speed_mm_s'] < 5
                                        and metrics['last1s_min_target_contacts'] >= 3
                                        and metrics['last1s_pipe_enclosed_fraction'] >= .95
                                        and metrics['last1s_min_source_contacts'] >= 2
                                        and metrics['final_wrap_deg'] > 180)
    return metrics


def run(previous, name, gain, release_deg):
    prev = BASE / previous
    case = BASE / name
    case.mkdir(parents=True, exist_ok=False)
    for src in ['snake_control/hook_pull_motion.py', 'validation/hook_pull_experiment.py', 'validation/hook_pull_close.py']:
        dest = case / 'sources' / src
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, dest)
    table = RollTable.load(prev / 'roll_table.npz')
    with np.load(prev / 'roll_table.npz') as data:
        sign = float(data['roll_sign_for_pull'])
    plan = dict(start=f'{previous} final integration state and ctrl, no reset', close_s=10., hold_s=5.,
                release_deg=release_deg, release_s=3., final_hold_s=10., gain=gain,
                close_rule='J1-J6 target = measured*(1+K) each 50 ms, each joint <= 2x its start angle and 88 deg, rate <=10 deg/s; other joints keep pull target',
                release_rule='non-hook joints follow roll table toward smaller pull; hook keeps compliant tightening',
                success='last2s drift<=6.45mm, last5s speed<5mm/s, >=3 target contacts, pipe enclosed>=95% last1s, source>=2, wrap>180deg',
                control_inputs='encoder angles and issued targets only; contact truth for scoring/safety')
    write(case / 'plan.json', plan)
    sim = Sim(case)
    with np.load(prev / 'final_state.npz') as state:
        mujoco.mj_setState(sim.m, sim.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        sim.d.ctrl[:] = state['ctrl']
        psi0 = float(state['psi_rad'])
    mujoco.mj_forward(sim.m, sim.d)
    sim.psi = psi0
    sim.stage = 'start'
    sim.snapshot()
    started = time.monotonic()
    result = dict(name=name, previous=previous, outcome='engineering_failure')
    last = sim.d.ctrl.copy()
    pull_target = last.copy()
    curl_limit = np.minimum(JOINT_LIMIT, MAX_CURL * abs(sim.d.qpos[sim.qadr][CLOSE_JOINTS]))
    schedule = [('close', 10.), ('hold', 5.), ('release', 3.), ('final_hold', 10.)]
    psi_release = psi0 - sign * np.radians(release_deg)
    try:
        lost = fast = 0
        for stage, duration in schedule:
            sim.stage = stage
            for k in range(int(round(duration / DT))):
                measured = sim.d.qpos[sim.qadr].copy()
                if stage == 'release':
                    sim.psi = psi0 + (psi_release - psi0) * smooth((k + 1) * DT / duration)
                body = table(sim.psi) if stage in ('release', 'final_hold') else pull_target
                target = body.copy()
                target[CLOSE_JOINTS] = np.clip(measured[CLOSE_JOINTS] * (1 + gain), -curl_limit, curl_limit)
                target = np.clip(target, last - RATE, last + RATE)
                target = np.clip(target, -JOINT_LIMIT, JOINT_LIMIT)
                sim.commands.append(dict(time_s=float(sim.d.time), stage=stage, psi_rad=float(sim.psi),
                                         measured_rad=measured.tolist(), target_rad=target.tolist()))
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
                        wrap=round(wrap_angle_deg(r['centers_mm']), 1), speed=round(speed, 1))), flush=True)
            if stage == 'hold':
                result['after_close_hold'] = evaluate(sim.rows, sim.rows[0]['sim_time'])
        result['outcome'] = 'completed'
    except StopIteration as stop:
        result['outcome'] = str(stop)
    except Exception as error:
        result.update(outcome='controller_or_interface_stop', error=repr(error))
    finally:
        sim.save()
    rows = sim.rows
    err = np.degrees(abs(np.array([r['ctrl'] - r['q'] for r in rows])))
    torque = abs(np.array([r['torque'] for r in rows]))
    result.update(physical_seconds=float(rows[-1]['sim_time'] - rows[0]['sim_time']),
                  wall_seconds=time.monotonic() - started,
                  start_wrap_deg=wrap_angle_deg(rows[0]['centers_mm']),
                  maximum_wrap_deg=max(wrap_angle_deg(r['centers_mm']) for r in rows),
                  maximum_tracking_error_deg=float(err.max()),
                  torque_saturation_fraction=float(np.mean(torque > 2.355 * .999)),
                  hook_torque_saturation_fraction=float(np.mean(torque[:, CLOSE_JOINTS] > 2.355 * .999)),
                  maximum_module_speed_mm_s=float(np.linalg.norm(np.diff([r['centers_mm'] for r in rows], axis=0), axis=2).max() / .01))
    if result['outcome'] == 'completed':
        result['final'] = evaluate(rows, rows[0]['sim_time'])
    write(case / 'result.json', result)
    ledger = BASE / 'trial_ledger.csv'
    with ledger.open('a', encoding='utf8', newline='') as f:
        csv.DictWriter(f, fieldnames=['trial_id', 'step', 'outcome', 'physical_seconds_after_branch', 'final_psi_deg',
                                      'evidence']).writerow(
            dict(trial_id=name, step='3_close_release', outcome=result['outcome'],
                 physical_seconds_after_branch=result['physical_seconds'], final_psi_deg=float(np.degrees(sim.psi)),
                 evidence=case.relative_to(ROOT).as_posix()))
    replay_video(case, rows)
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', default='PULL1')
    parser.add_argument('--name', default='CLOSE1')
    parser.add_argument('--gain', type=float, default=.02)
    parser.add_argument('--release-deg', type=float, default=15.)
    args = parser.parse_args()
    run(args.previous, args.name, args.gain, args.release_deg)
