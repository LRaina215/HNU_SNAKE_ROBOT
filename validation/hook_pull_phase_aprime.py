"""阶段A′离线验证：交替试探脉冲（用户设想）。不推进物理时间。

设想（用户，2026-10-07）：前部在目标管上 rolling 一小步、后部沿源管向接头 rolling 一小步，
交替试探，用过渡段的关节角/力矩判断卡住与解锁，让过渡段逐渐移到前部。

本脚本在"接触不打滑"运动学里检验脉冲交替的几何可行性。力矩反馈不在模型里，
"卡住"以脉冲收益近似：一个脉冲内端部前进 <1 mm 视为该端受阻。判据（运行前登记）：
  A2' 前部沿目标管累计净前进 ≥25 mm 且 后部沿源管累计净前进 ≥25 mm；
  A3' 至少 1 个 M≥8 模块落到目标管斜切区以外（垂足 y≥60 mm）并保持到结束；
  通用门槛：关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触。
  连续 2 个脉冲（一前一后）收益均 <1 mm 判为停滞停止。
政策：脉冲目标 8 mm；FRONT 与 REAR 交替；某端受阻则另一端获得下一次机会；
某端连续受阻 2 次后跳过其 1 次。脉冲内部每步关节变化 ≤0.2°。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse

import mujoco
import numpy as np

import hook_pull_step4_rolling as H

BASE = H.BASE
OUT = BASE / 'phase_aprime'
CRITERION = dict(pulse_target_mm=8., pulse_gain_min_mm=1., stall_cycles=2,
                 A2_front_y_min_mm=25., A2_rear_x_min_mm=25., A3_module_min=8,
                 gates='关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触')
PULSE_STEPS_CAP = 900


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def run(start):
    case = OUT / start
    case.mkdir(parents=True, exist_ok=False)
    model = H.Model()
    with np.load(BASE / start / 'final_state.npz') as state:
        mujoco.mj_setState(model.m, model.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    model.d.qvel[:] = 0
    model.update()
    pairs, _, _, _ = model.touching()
    contacts = sorted(k for k, dd in pairs.items() if dd < 1e-3)
    nodes = model.nodes()
    rho0 = {}
    for (k, pipe) in contacts:
        x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
        rho0[(k, pipe)] = H.radial(x, foot)[0]
    landed = set((k, pipe) for k, pipe in contacts)
    start_front_y = float(((nodes[:-1] + nodes[1:]) / 2)[:8, 1].mean())
    start_rear_x = float(((nodes[:-1] + nodes[1:]) / 2)[13:21, 0].mean())
    log, pulses, events, miter_modules = [], [], [], []
    stop = 'step_cap'
    H.FREE_FLOOR.clear()
    front_y = rear_x = 0.
    next_end = 'FRONT'
    blocked = {'FRONT': 0, 'REAR': 0}
    failed_cycles = 0

    def gates(now):
        failed = []
        q = np.degrees(model.d.qpos[model.qadr])
        if abs(q).max() > np.degrees(H.LIMIT_GATE):
            failed.append(f'joint_limit J{int(np.argmax(np.abs(q))) + 1}')
        nodes_now = model.nodes()
        pairs_now, deepest, ground, selfhit = model.touching()
        if selfhit:
            failed.append('nonadjacent_self_contact')
        if ground:
            failed.append('ground')
        if deepest < -3.:
            failed.append('pipe_penetration')
        wrap = H.wrap_deg(nodes_now)
        if wrap < 180.:
            failed.append('front_wrap_below_180')
        n_t = sum(p == 'target' for _, p in contacts)
        n_s = sum(p == 'source' for _, p in contacts)
        if n_t < 1:
            failed.append('no_target_contact')
        if n_s < 1:
            failed.append('no_source_contact')
        return failed, wrap, n_t, n_s, deepest

    pulse_index = 0
    while pulse_index < 60:
        end = next_end
        mode = (1., None) if end == 'FRONT' else (None, 1.)
        gain = 0.
        pulses.append(dict(index=pulse_index, end=end, gain_mm=None, steps=0, blocked=False))
        for s in range(PULSE_STEPS_CAP):
            qvel, info = H.solve_active(model, contacts, rho0, mode)
            jmax = float(np.abs(qvel[model.vadr]).max())
            if jmax < 1e-12:
                pulses[-1]['blocked'] = True
                break
            dt = min(H.DQ_MAX / jmax, .02)
            nodes = model.nodes()
            centers = (nodes[:-1] + nodes[1:]) / 2
            if end == 'FRONT':
                step_gain = float(np.mean([model.jac(centers[k], k)[0][1] for k in
                                           sorted({k for k, p in contacts if p == 'target'})], axis=0) @ qvel) * dt * 1000
            else:
                step_gain = float(np.mean([model.jac(centers[k], k)[0][0] for k in
                                           sorted({k for k, p in contacts if p == 'source'})], axis=0) @ qvel) * dt * 1000
            gain += step_gain
            qpos = model.d.qpos.copy()
            mujoco.mj_integratePos(model.m, qpos, qvel, dt)
            model.d.qpos[:] = qpos
            model.update()
            nodes = model.nodes()
            centers = (nodes[:-1] + nodes[1:]) / 2
            pairs, deepest, ground, selfhit = model.touching()
            for key, dist in pairs.items():
                if key not in landed and dist < 3e-4:
                    k, pipe = key
                    x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
                    if not H.MITER[pipe](foot):
                        landed.add(key)
                        events.append(dict(pulse=pulse_index, event='land', module=k, pipe=pipe))
                        if H.LANDING == 'noslip':
                            contacts.append(key)
                            rho0[key] = H.radial(x, foot)[0]
            for key in list(contacts):
                k, pipe = key
                x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
                if H.MITER[pipe](foot):
                    contacts.remove(key)
                    events.append(dict(pulse=pulse_index, event='release_at_miter', module=k, pipe=pipe))
            contacts.sort()
            for key in pairs:
                k, pipe = key
                if pipe == 'target' and k >= CRITERION['A3_module_min'] and ('cross', k) not in landed:
                    x, foot = H.closest(nodes[k], nodes[k + 1], 'target')
                    if foot[1] >= .060:
                        landed.add(('cross', k))
                        miter_modules.append(dict(pulse=pulse_index, module=k))
                        events.append(dict(pulse=pulse_index, event='cross_miter_onto_target', module=k))
            front_y = float(centers[:8, 1].mean()) - start_front_y
            rear_x = float(centers[13:21, 0].mean()) - start_rear_x
            failed, wrap, n_t, n_s, deepest = gates(None)
            log.append(dict(pulse=pulse_index, end=end, front_y_mm=front_y * 1000, rear_x_mm=rear_x * 1000,
                            max_joint_deg=float(np.degrees(abs(model.d.qpos[model.qadr])).max()),
                            wrap_deg=wrap, n_target=n_t, n_source=n_s, deepest_mm=deepest * 1000))
            if failed:
                stop = ','.join(failed)
                pulses[-1]['gain_mm'] = gain
                pulses[-1]['steps'] = s + 1
                break
            if gain >= CRITERION['pulse_target_mm']:
                break
        else:
            pulses[-1]['gain_mm'] = gain
            pulses[-1]['steps'] = PULSE_STEPS_CAP
        if stop != 'step_cap':
            break
        pulses[-1]['gain_mm'] = gain
        blocked[end] = 0 if gain >= CRITERION['pulse_gain_min_mm'] else blocked[end] + 1
        if blocked[end] >= 2:
            other = 'REAR' if end == 'FRONT' else 'FRONT'
            if blocked[other] >= 2:
                stop = 'stall_both_ends'
                break
            next_end = other
        else:
            next_end = 'REAR' if end == 'FRONT' else 'FRONT'
        if blocked[end] >= 1 and blocked['FRONT'] >= 1 and blocked['REAR'] >= 1:
            failed_cycles += 1
            if failed_cycles >= CRITERION['stall_cycles']:
                stop = 'stall_cycles'
                break
        front_y = None
        nodes = model.nodes()
        centers = (nodes[:-1] + nodes[1:]) / 2
        front_y = float(centers[:8, 1].mean()) - start_front_y
        rear_x = float(centers[13:21, 0].mean()) - start_rear_x
        pulses[-1]['front_y_mm'] = front_y * 1000
        pulses[-1]['rear_x_mm'] = rear_x * 1000
        print(f"pulse {pulse_index:2d} {end:5s}: 收益 {gain:5.2f} mm | 累计 前 +Y {front_y * 1000:6.1f} "
              f"后 +X {rear_x * 1000:6.1f} mm | 受阻 F{blocked['FRONT']}/R{blocked['REAR']}", flush=True)
        pulse_index += 1

    nodes = model.nodes()
    centers = (nodes[:-1] + nodes[1:]) / 2
    front_y = float(centers[:8, 1].mean()) - start_front_y
    rear_x = float(centers[13:21, 0].mean()) - start_rear_x
    wrap = H.wrap_deg(nodes)
    result = dict(
        start=start, criterion=CRITERION, stop_reason=stop,
        A2=dict(front_y_mm=front_y * 1000, rear_x_mm=rear_x * 1000,
                passed=bool(front_y * 1000 >= CRITERION['A2_front_y_min_mm'] and rear_x * 1000 >= CRITERION['A2_rear_x_min_mm']),
                note='累计净前进；运动学参数化中端部目标只进不退，回退风险只能在动力学中检验'),
        A3=dict(crossed=[m['module'] for m in miter_modules], passed=bool(miter_modules)),
        pulses=pulses, events=events,
        final_max_joint_deg=float(np.degrees(abs(model.d.qpos[model.qadr]).max())), final_wrap_deg=wrap)
    result['verdict'] = 'PASS' if (result['A2']['passed'] and result['A3']['passed']) else 'FAIL'
    write(case / 'report.json', result)
    write(case / 'log.json', log[::10] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in result.items() if k not in ('pulses', 'events')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', nargs='+', default=['CLOSE1B', 'CLOSE2B'])
    args = parser.parse_args()
    summary = {s: run(s) for s in args.starts}
    write(OUT / 'phase_aprime_summary.json', summary)
