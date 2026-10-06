"""钩挂回拉 后续执行方案·阶段A（离线，不推进物理时间）。

判据（预先登记，运行中不修改）：
  A1 放松：运行中存在某步 |q|max ≤80°，且当时包绕 ≥180°、目标管 ≥3 接触；
     全程包绕 ≥180°、目标管 ≥3 接触、不钻管、不触地、无自碰。
  A2 送身：前部 +Y ≥25 mm 且后部 +X ≥25 mm（以实际几何为准）。
  A3 过接头：至少 1 个 M≥8 模块落到目标管斜切区以外（垂足 y≥60 mm）并保持接触。
  通用门槛：关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、两管各 ≥1 接触、不停滞。

修订1（已登记）：先做"原地放松"（solve_A1_step / run_A1，报告存 *_A1/），
两个起点都被接触约束卡死：CLOSE1B 包绕 268°→237° 后钻管；CLOSE2B 松 J6 时
J19 顶到 88°。故 run_coupled 把放松与送身耦合为单次行进：接触不打滑为硬约束，
进度目标（前+Y、后+X 各 1 mm/s）与放松目标（超限关节向 80° 收拢，越超拉得越紧）
在接触零空间内按权重折中；防钻管/防自碰/限位回推用主动集；最后最小化关节速度。
A1/A2/A3 判据独立记录，互不放宽。接触切换：自由模块碰到管即落管（noslip 假设
则加入不打滑约束），接触点走到接头斜切区即释放。受力不在模型里。
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
OUT = BASE / 'phase_a'
CRITERION = dict(A1_max_joint_deg=80., A1_wrap_min_deg=180., A1_target_contacts_min=3,
                 A2_front_y_min_mm=25., A2_rear_x_min_mm=25., A3_module_min=8,
                 gates='关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、目标管 ≥3 接触、两管各 ≥1 接触、不停滞')
RELAX_MAX_STEPS = 4000
# 修订2：跨接区模块的抬起引导点（世界坐标 m）：拐角 (1.1, 0, 1.0) 的外上方
WAYPOINTS = {8: np.array([1.09, .07, 1.10]), 9: np.array([1.08, .04, 1.10]),
             10: np.array([1.05, .01, 1.10]), 11: np.array([1.01, -.03, 1.10])}
W_LIFT = 30.


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def contact_set(model):
    pairs, _, _, _ = model.touching()
    return sorted(k for k, dd in pairs.items() if dd < 1e-3)


def relaxed_inequalities(model, contacts):
    """A 系列共用的不等式：自由模块不往管里钻（以起始半径为底）+ 非相邻不自碰。"""
    nodes = model.nodes()
    centers = (nodes[:-1] + nodes[1:]) / 2
    G, h, tags = [], [], []
    in_contact = {k for k, _ in contacts}
    for k in range(21):
        if k in in_contact:
            continue
        for pipe in H.PIPES:
            x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
            if H.MITER[pipe](foot):
                continue
            r, n = H.radial(x, foot)
            floor = min(.084, H.FREE_FLOOR.setdefault((k, pipe), r)) - .0005
            if r < floor + H.CLEAR_MARGIN:
                G.append(n @ model.jac(x, k)[0])
                h.append(-H.K_R * max(r - floor, 0.))
                tags.append(f'clear M{k}-{pipe}')
    for i in range(21):
        for j in range(i + 2, 21):
            dvec = centers[i] - centers[j]
            dist = float(np.linalg.norm(dvec))
            if dist < .062:
                u = dvec / dist
                G.append(u @ (model.jac(centers[i], i)[0] - model.jac(centers[j], j)[0]))
                h.append(-H.K_R * (dist - .058))
                tags.append(f'self M{i}-M{j}')
    return np.asarray(G).reshape(-1, model.m.nv), np.asarray(h), tags


def load_start(start):
    model = H.Model()
    with np.load(BASE / start / 'final_state.npz') as state:
        mujoco.mj_setState(model.m, model.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    model.d.qvel[:] = 0
    model.update()
    contacts = contact_set(model)
    nodes = model.nodes()
    rho0 = {}
    for (k, pipe) in contacts:
        x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
        rho0[(k, pipe)] = H.radial(x, foot)[0]
    return model, contacts, rho0


def run_A1(start):
    """原地放松（修订前的首次尝试，保留作为修订1的依据）。"""
    case = OUT / f'{start}_A1'
    case.mkdir(parents=True, exist_ok=False)
    model, contacts, rho0 = load_start(start)
    q_start = np.degrees(model.d.qpos[model.qadr])
    q_target = np.clip(q_start, -CRITERION['A1_max_joint_deg'], CRITERION['A1_max_joint_deg'])
    log, events = [], []
    stop = 'step_cap'
    H.FREE_FLOOR.clear()
    landed = set((k, pipe) for k, pipe in contacts)
    for step in range(RELAX_MAX_STEPS):
        nodes = model.nodes()
        nv = model.m.nv
        E, e = [], []
        for (k, pipe) in contacts:
            x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
            r, n = H.radial(x, foot)
            E.append(model.jac(x - H.R_B * n, k)[0])
            e.append(-H.K_R * (r - rho0[(k, pipe)]) * n)
        E = np.vstack(E) if E else np.zeros((0, nv))
        e = np.concatenate(e) if len(e) else np.zeros(0)
        q = model.d.qpos[model.qadr]
        over = [j for j in range(20) if abs(q[j]) > q_target[j] + 1e-9]
        P = np.zeros((len(over), nv))
        p = np.zeros(len(over))
        for i, j in enumerate(over):
            P[i, model.vadr[j]] = np.sign(q[j])
            p[i] = -np.radians(5.)
        G, h, tags = relaxed_inequalities(model, contacts)
        active, active_tags = [], []
        for _ in range(30):
            Aeq = np.vstack([E, G[active]]) if active else E
            beq = np.concatenate([e, h[active]]) if active else e
            qvel = H.priority_solve(Aeq, beq, P, p, nv, model.vadr)
            slack = G @ qvel - h if len(G) else np.zeros(0)
            candidates = [i for i in range(len(G)) if i not in active and slack[i] < -1e-9]
            if not candidates:
                break
            k_ = min(candidates, key=lambda i: slack[i])
            active.append(k_)
            active_tags.append(tags[k_])
        jvel = float(np.abs(qvel[model.vadr]).max())
        if jvel < 1e-10:
            stop = 'converged'
            break
        dt = min(H.DQ_MAX / jvel, .02)
        qpos = model.d.qpos.copy()
        mujoco.mj_integratePos(model.m, qpos, qvel, dt)
        model.d.qpos[:] = qpos
        model.update()
        nodes = model.nodes()
        pairs, deepest, ground, selfhit = model.touching()
        for key, dist in pairs.items():
            if key not in contacts and key not in landed and dist < 3e-4:
                landed.add(key)
                events.append(dict(step=step, event='land', module=key[0], pipe=key[1]))
        for key in list(contacts):
            k, pipe = key
            x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
            if H.MITER[pipe](foot):
                contacts.remove(key)
                events.append(dict(step=step, event='release_at_miter', module=k, pipe=pipe))
        contacts.sort()
        q = np.degrees(model.d.qpos[model.qadr])
        wrap = H.wrap_deg(nodes)
        n_target = sum(p == 'target' for _, p in contacts)
        row = dict(step=step, max_joint_deg=float(abs(q).max()), wrap_deg=wrap, n_target=n_target,
                   deepest_mm=deepest * 1000, active=active_tags)
        log.append(row)
        failed = []
        if abs(q).max() > np.degrees(H.LIMIT_GATE):
            failed.append(f'joint_limit J{int(np.argmax(np.abs(q))) + 1}')
        if selfhit or ground:
            failed.append('self_contact_or_ground')
        if deepest < -3e-3:
            failed.append('pipe_penetration')
        if wrap < CRITERION['A1_wrap_min_deg']:
            failed.append('wrap_below_180')
        if n_target < CRITERION['A1_target_contacts_min']:
            failed.append('target_contacts_below_3')
        if failed:
            stop = ','.join(failed)
            break
        if abs(q).max() <= CRITERION['A1_max_joint_deg'] + 1e-6:
            stop = 'A1_reached'
            break
        if step % 200 == 0:
            print(f'{start} A1 step {step:4d}: |q|max {row["max_joint_deg"]:5.1f}° 包绕 {wrap:5.1f}° '
                  f'目标管接触 {n_target} 起作用 {active_tags[:3]}', flush=True)
    final_q = np.degrees(model.d.qpos[model.qadr])
    wrap = H.wrap_deg(model.nodes())
    n_target = sum(p == 'target' for _, p in contacts)
    deepest = min((row['deepest_mm'] for row in log), default=0.)
    checks = dict(max_joint_deg=float(abs(final_q).max()), wrap_deg=wrap, n_target=n_target, deepest_mm=deepest)
    passed = (stop == 'A1_reached')
    report = dict(start=start, criterion=CRITERION, steps=len(log), stop_reason=stop,
                  q_start_max_deg=float(abs(q_start).max()), final=checks, passed=bool(passed),
                  events=events, contacts=[f'M{k}-{p}' for k, p in contacts])
    write(case / 'a1_report.json', report)
    write(case / 'a1_log.json', log[::5] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, ensure_ascii=False), flush=True)
    return report


def run_coupled(start):
    """修订1：放松与送身耦合（依据见模块 docstring 与 *_A1 报告）。"""
    case = OUT / f'{start}_coupled'
    case.mkdir(parents=True, exist_ok=False)
    model, contacts, rho0 = load_start(start)
    landed = set((k, pipe) for k, pipe in contacts)
    q_target = np.full(20, CRITERION['A1_max_joint_deg'])
    nodes = model.nodes()
    roll = np.zeros(21)
    start_front_y = float(((nodes[:-1] + nodes[1:]) / 2)[:8, 1].mean())
    start_rear_x = float(((nodes[:-1] + nodes[1:]) / 2)[13:21, 0].mean())
    log, events, miter_modules = [], [], []
    stop, a1_step = 'step_cap', None
    H.FREE_FLOOR.clear()
    window = []
    v_relax = np.radians(5.)
    for step in range(H.MAX_STEPS):
        nodes = model.nodes()
        centers = (nodes[:-1] + nodes[1:]) / 2
        nv = model.m.nv
        E, e = [], []
        for (k, pipe) in contacts:
            x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
            r, n = H.radial(x, foot)
            E.append(model.jac(x - H.R_B * n, k)[0])
            e.append(-H.K_R * (r - rho0[(k, pipe)]) * n)
        E = np.vstack(E) if E else np.zeros((0, nv))
        e = np.concatenate(e) if len(e) else np.zeros(0)
        front = sorted({k for k, p in contacts if p == 'target'})
        rear = sorted({k for k, p in contacts if p == 'source'})
        rowsP, pvec = [], []
        if front:
            rowsP.append(H.W_PROG * 1000. * np.mean([model.jac(centers[k], k)[0][1] for k in front], axis=0))
            pvec.append(H.W_PROG * 1.)
        if rear:
            rowsP.append(H.W_PROG * 1000. * np.mean([model.jac(centers[k], k)[0][0] for k in rear], axis=0))
            pvec.append(H.W_PROG * 1.)
        q = model.d.qpos[model.qadr]
        for j in range(20):
            over = float(abs(q[j]) - q_target[j])
            if over > 1e-9:
                row = np.zeros(nv)
                row[model.vadr[j]] = np.sign(q[j])
                rowsP.append(min(over, .35) * row)
                pvec.append(-v_relax)
        # 修订2（登记在案）：主动抬起过渡段。滚动验证两次都显示拐角附近的自由模块
        # 被压回源管而不是绕上目标管，"不许钻管"只是被动防线。这里给跨接区自由模块
        # （M8–M11，尚未落管）一个朝拐角上方引导点的速度目标（2 mm/s）。
        for k in range(8, 12):
            if (k, 'target') in contacts or (k, 'source') in contacts:
                continue
            if centers[k][1] > .09 or centers[k][0] < .98:
                continue  # 已过拐角或还深入源管缠绕区的模块不引导
            waypoint = WAYPOINTS.get(k, WAYPOINTS[9])
            u_dir = waypoint - centers[k]
            norm = float(np.linalg.norm(u_dir))
            if norm < 1e-6:
                continue
            u_dir /= norm
            rowsP.append(W_LIFT * 1000. * (u_dir @ model.jac(centers[k], k)[0]))
            pvec.append(W_LIFT * 2.)
        P = np.asarray(rowsP).reshape(-1, nv)
        p = np.asarray(pvec)
        G, h, tags = relaxed_inequalities(model, contacts)
        active, active_tags = [], []
        for _ in range(40):
            Aeq = np.vstack([E, G[active]]) if active else E
            beq = np.concatenate([e, h[active]]) if active else e
            qvel = H.priority_solve(Aeq, beq, P, p, nv, model.vadr)
            slack = G @ qvel - h if len(G) else np.zeros(0)
            candidates = [i for i in range(len(G)) if i not in active and slack[i] < -1e-9]
            if not candidates:
                break
            k_ = min(candidates, key=lambda i: slack[i])
            active.append(k_)
            active_tags.append(tags[k_])
        jvel = float(np.abs(qvel[model.vadr]).max())
        if jvel < 1e-12:
            stop = 'no_motion'
            break
        dt = min(H.DQ_MAX / jvel, .02)
        fy_rate = 1000. * float(np.mean([model.jac(centers[k], k)[0][1] for k in front], axis=0) @ qvel) if front else 0.
        rx_rate = 1000. * float(np.mean([model.jac(centers[k], k)[0][0] for k in rear], axis=0) @ qvel) if rear else 0.
        for k in range(21):
            t = model.d.xmat[model.head + k].reshape(3, 3)[:, 2]
            _, jr = model.jac(model.d.xpos[model.head + k], k)
            roll[k] += float(t @ (jr @ qvel)) * dt
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
                    events.append(dict(step=step, event='land', module=k, pipe=pipe, model=H.LANDING))
                    if H.LANDING == 'noslip':
                        contacts.append(key)
                        rho0[key] = H.radial(x, foot)[0]
        for key in list(contacts):
            k, pipe = key
            x, foot = H.closest(nodes[k], nodes[k + 1], pipe)
            if H.MITER[pipe](foot):
                contacts.remove(key)
                events.append(dict(step=step, event='release_at_miter', module=k, pipe=pipe))
        contacts.sort()
        for key in pairs:
            k, pipe = key
            if pipe == 'target' and k >= CRITERION['A3_module_min'] and ('cross', k) not in landed:
                x, foot = H.closest(nodes[k], nodes[k + 1], 'target')
                if foot[1] >= .060:
                    landed.add(('cross', k))
                    miter_modules.append(dict(step=step, module=k))
                    events.append(dict(step=step, event='cross_miter_onto_target', module=k))
        q_deg = np.degrees(model.d.qpos[model.qadr])
        front_y = float(centers[:8, 1].mean()) - start_front_y
        rear_x = float(centers[13:21, 0].mean()) - start_rear_x
        wrap = H.wrap_deg(nodes)
        n_target = sum(p == 'target' for _, p in contacts)
        if a1_step is None and abs(q_deg).max() <= CRITERION['A1_max_joint_deg'] + 1e-6:
            a1_step = dict(step=step, max_joint_deg=float(abs(q_deg).max()), wrap_deg=wrap, n_target=n_target)
        log.append(dict(step=step, front_y_mm=front_y * 1000, rear_x_mm=rear_x * 1000,
                        max_joint_deg=float(abs(q_deg).max()), deepest_mm=deepest * 1000, wrap_deg=wrap,
                        n_target=n_target, n_source=sum(p == 'source' for _, p in contacts),
                        active=active_tags, front_rate_mm_s=fy_rate, rear_rate_mm_s=rx_rate,
                        roll_front_deg=float(np.degrees(roll[:8].mean())),
                        roll_rear_deg=float(np.degrees(roll[13:21].mean()))))
        window.append((front_y, rear_x))
        if len(window) > 300:
            window.pop(0)
        failed = []
        if abs(q_deg).max() > np.degrees(H.LIMIT_GATE):
            failed.append(f'joint_limit J{int(np.argmax(np.abs(q_deg))) + 1}')
        if selfhit:
            failed.append('nonadjacent_self_contact')
        if ground:
            failed.append('ground')
        if deepest < -3e-3:
            failed.append('pipe_penetration')
        if wrap < 180.:
            failed.append('front_wrap_below_180')
        if n_target < CRITERION['A1_target_contacts_min']:
            failed.append('target_contacts_below_3')
        if not any(p == 'source' for _, p in contacts):
            failed.append('no_source_contact')
        if len(window) == 300 and (window[-1][0] - window[0][0]) < 2e-4 and (window[-1][1] - window[0][1]) < 2e-4:
            failed.append('stall')
        if failed:
            stop = ','.join(failed)
            break
        if front_y * 1000 >= CRITERION['A2_front_y_min_mm'] and rear_x * 1000 >= CRITERION['A2_rear_x_min_mm']:
            stop = 'A2_reached'
            break
        if step % 250 == 0:
            print(f'{start} coupled step {step:4d}: 前 +Y {front_y * 1000:6.1f} 后 +X {rear_x * 1000:6.1f} mm '
                  f'|q|max {abs(q_deg).max():5.1f}° 包绕 {wrap:5.1f}° 目标管接触 {n_target} 起作用 {active_tags[:3]}', flush=True)
    final_q = np.degrees(model.d.qpos[model.qadr])
    wrap = H.wrap_deg(model.nodes())
    n_target = sum(p == 'target' for _, p in contacts)
    integration = np.empty(mujoco.mj_stateSize(model.m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model.m, model.d, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez(case / 'state.npz', integration_state=integration, ctrl=model.d.ctrl.copy(),
             psi_rad=0., sim_time=model.d.time)
    a1 = dict(passed=a1_step is not None, first_reach=a1_step,
              final_max_joint_deg=float(abs(final_q).max()), wrap_deg=wrap, n_target=n_target)
    a2 = dict(front_y_mm=float(window[-1][0] * 1000) if window else 0.,
              rear_x_mm=float(window[-1][1] * 1000) if window else 0.,
              passed=bool(window and window[-1][0] * 1000 >= CRITERION['A2_front_y_min_mm']
                          and window[-1][1] * 1000 >= CRITERION['A2_rear_x_min_mm']))
    a3 = dict(crossed=[m['module'] for m in miter_modules], passed=bool(miter_modules),
              note='落上目标管斜切区以外（y≥60mm）的 M≥8 模块')
    report = dict(start=start, revision='coupled relax+feed（修订1：原地放松两起点均卡死，详见 CLOSE1B_A1/CLOSE2B_A1 报告）',
                  criterion=CRITERION, steps=len(log), stop_reason=stop, A1=a1, A2=a2, A3=a3,
                  events=events, contacts=[f'M{k}-{p}' for k, p in contacts])
    write(case / 'report.json', report)
    write(case / 'log.json', log[::10] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', nargs='+', default=['CLOSE1B', 'CLOSE2B'])
    parser.add_argument('--in-place-a1', action='store_true', help='只跑原地放松（修订依据）')
    args = parser.parse_args()
    summary = dict(criterion=CRITERION, starts={})
    for start in args.starts:
        if args.in_place_a1:
            summary['starts'][start] = dict(A1=run_A1(start))
        else:
            summary['starts'][start] = run_coupled(start)
    write(OUT / 'phase_a_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
