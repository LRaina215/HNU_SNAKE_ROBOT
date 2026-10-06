"""钩挂回拉：接续某一步的终态，保持指令不变若干秒，用于对照与延长保持检验。

HOLD_CONTROL1（FRONT1 的不滚动对照）与 EXTHOLD_1B / EXTHOLD_2B（收紧后延长保持）当时用与本脚本相同的内联代码运行。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse
import csv
import json

import mujoco
import numpy as np

from hook_pull_experiment import BASE, Sim, write
from hook_pull_close import wrap_angle_deg


def run(previous, name, seconds):
    case = BASE / name
    case.mkdir(exist_ok=False)
    sim = Sim(case)
    with np.load(BASE / previous / 'final_state.npz') as state:
        mujoco.mj_setState(sim.m, sim.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        ctrl = state['ctrl'].copy()
        sim.psi = float(state['psi_rad'])
    sim.d.ctrl[:] = ctrl
    mujoco.mj_forward(sim.m, sim.d)
    sim.stage = 'extended_hold'
    sim.snapshot()
    outcome = 'completed'
    for _ in range(int(round(seconds / .05))):
        sim.step_command(ctrl.copy())
        r = sim.rows[-1]
        if r['ground'] or r['self_contact']:
            outcome = 'ground_or_self'
            break
        if np.count_nonzero(r['contact_force_N'][0] > .1) < 1 and np.count_nonzero(r['contact_force_N'][1] > .1) < 2:
            outcome = 'support_lost'
            break
    sim.save()
    rows = sim.rows
    c = np.array([r['centers_mm'] for r in rows])
    f = np.array([r['contact_force_N'] for r in rows])
    t = np.array([r['sim_time'] for r in rows]) - rows[0]['sim_time']
    src, tgt = (f[:, 0] > .1).sum(1), (f[:, 1] > .1).sum(1)
    result = dict(previous=previous, outcome=outcome, seconds=float(t[-1]),
                  front_y_creep_mm=float(c[-1, :8, 1].mean() - c[0, :8, 1].mean()),
                  rear_x_creep_mm=float(c[-1, 10:, 0].mean() - c[0, 10:, 0].mean()),
                  last10s_front_y_creep_mm=float(c[-1, :8, 1].mean() - c[t >= t[-1] - 10][0, :8, 1].mean()),
                  source_contacts_min=int(src.min()), source_contacts_last1s_min=int(src[t >= t[-1] - 1].min()),
                  target_contacts_min=int(tgt.min()), target_contacts_last1s_min=int(tgt[t >= t[-1] - 1].min()),
                  wrap_start_deg=wrap_angle_deg(c[0]), wrap_end_deg=wrap_angle_deg(c[-1]),
                  enclosed_last1s=float(np.mean([r['pipe_enclosed'] for r in rows if r['sim_time'] >= rows[-1]['sim_time'] - 1])),
                  last5s_max_speed_mm_s=float(np.linalg.norm(np.diff(c[t >= t[-1] - 5], axis=0), axis=2).max() / .01))
    write(case / 'result.json', result)
    with (BASE / 'trial_ledger.csv').open('a', encoding='utf8', newline='') as handle:
        csv.writer(handle).writerow([name, 'hold', outcome, result['seconds'], float(np.degrees(sim.psi)),
                                     case.relative_to(ROOT).as_posix()])
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    args = parser.parse_args()
    run(args.previous, args.name, args.seconds)
