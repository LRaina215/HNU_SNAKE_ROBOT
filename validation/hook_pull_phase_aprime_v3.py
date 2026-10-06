"""阶段A′离线 v3：前部连续螺旋 + 后部供料的联合行进，按 A′ 判据评估，并导出动力学参考表。

机制与 step4_feasibility 相同（轴坐标毫米、整身形状约束、过渡节点渐退剥离），
但按用户设想组合：前部每步沿目标管螺旋推进（连续），后部按供料节奏推进。
两种供料节奏都验证：
  mode=simultaneous：前后每步同时各进 2 mm（节奏上界）；
  mode=alternate ：前部每步推进、后部每 4 步推进 8 mm（间歇供料）。
判据（运行前登记）：
  A2' 前部沿目标管轴向净前进 ≥25 mm 且 后部沿源管轴向净前进 ≥25 mm；
  A3' 至少 1 个 M≥8 模块落到目标管斜切区以外（垂足 y≥60 mm，中心距管轴 ≤92 mm）并保持；
  通用门槛：关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触、停滞。
通过则导出 q(u) 参考表供阶段B动力学跟踪。
起点关节取实测 qpos（不是指令 ctrl——CLOSE 终态有 9–19° 饱和偏差）。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from curve_guided_geometry import MODEL, points
from snake_control.hook_pull_motion import RollSolver

BASE = ROOT / 'experiments/hook_pull_20261006'
OUT = BASE / 'phase_aprime'
AXIS_T = np.array([1100., 0., 1000.])
AXIS_S = np.array([0., 0., 1000.])
R95, PITCH = .095, .220
L_TURN = float(np.hypot(2 * np.pi * R95, PITCH))
ROT_PER_MM = 2 * np.pi / L_TURN
AX_PER_MM = PITCH / L_TURN
STEP_MM = 2.
W_STRONG, W_MED, W_CONT = 60., 15., 20.
U_MAX = 300.
CRITERION = dict(A2_front_y_min_mm=25., A2_rear_x_min_mm=25., A3_module_min=8,
                 gates='关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触、停滞=300步无进展')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def wrap_angle(centers):
    uv = (centers[:9] - AXIS_T)[:, [0, 2]][::-1]
    a = np.arctan2(uv[:, 1], uv[:, 0])
    return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def run(start, mode):
    case = OUT / f'{start}_{mode}'
    case.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    head = m.body('head').id
    floor = m.geom('floor').id
    with np.load(BASE / start / 'final_state.npz') as state:
        qpos0 = state['integration_state'][1:28].copy()
        joints0 = state['integration_state'][8:28].copy()
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    P0 = points(m, d)
    C0 = (P0[:-1] + P0[1:]) / 2
    th = np.arctan2(C0[:9, 2] - 1000., C0[:9, 0] - 1100.)
    y = C0[:9, 1]
    order = np.argsort(y)
    th_u = np.unwrap(th[order])
    kappa = float(np.polyfit(y[order], th_u, 1)[0])
    arc_per_rad = float(8 * 64.5 / max(abs(th_u[-1] - th_u[0]), 1e-6))
    print(f'{start}/{mode}: κ={kappa:.5f} arc/rad={arc_per_rad:.1f}', flush=True)

    solver = RollSolver(m, qpos0[:3] * 1000, qpos0[3:7], joints0, list(range(10, 21)))
    solver.weights = np.full(22, W_MED)
    solver.weights[:9] = W_STRONG
    x = solver.x0.copy()
    u_f = u_r = 0.
    arc_before = {n: (n - 10) * 64.5 + 32.25 for n in range(10, 22)}
    log, events, miter = [], [], []
    landed = set()
    wrap = wrap_angle(C0)
    stop = 'u_cap'
    step = 0
    no_progress = 0
    table_psi, table_q = [], []
    while u_f < U_MAX and u_r < U_MAX:
        step += 1
        u_f += STEP_MM
        feed = mode == 'simultaneous' or step % 4 == 0
        u_r += STEP_MM if feed else 0.
        axial_f = u_f / (abs(kappa) * arc_per_rad)
        rot_y_f = -kappa * axial_f
        tgt = solver.p0.copy()
        rotF = Rotation.from_rotvec([0., rot_y_f, 0.])
        tgt[:9] = rotF.apply(solver.p0[:9] - AXIS_T) + AXIS_T + np.array([0., axial_f, 0.])
        rotR = Rotation.from_rotvec([-u_r * ROT_PER_MM, 0., 0.])
        tgt[10:] = rotR.apply(solver.p0[10:] - AXIS_S) + AXIS_S + np.array([u_r * AX_PER_MM, 0., 0.])
        # 过渡/后部剥离：距剥离点弧长进入 48 mm 窗口后权重渐退
        for n in range(10, 22):
            fade = float(np.clip((arc_before[n] - u_r) / 48. + 1., 0., 1.))
            solver.weights[n] = (W_STRONG if n >= 10 else W_MED) * fade
        solver.weights[9] = W_MED

        def residual(xx, prev):
            p, _ = solver.forward(xx)
            return np.concatenate([(solver.weights[:, None] * (p - tgt)).ravel(), W_CONT * (xx[6:] - prev[6:])])

        # 每步硬边界：关节 ≤3°、根位置 ≤3 mm、根姿态 ≤2°（与动力学 10°/s 跟踪一致，防零空间翻筋斗）
        d_rot = np.radians(2.)
        low = np.r_[x[:3] - 3., x[3:6] - d_rot, np.maximum(x[6:] - np.radians(3.), -np.pi / 2 + 1e-3)]
        high = np.r_[x[:3] + 3., x[3:6] + d_rot, np.minimum(x[6:] + np.radians(3.), np.pi / 2 - 1e-3)]
        sol = least_squares(residual, x, bounds=(low, high), args=(x,),
                            x_scale=np.r_[np.full(3, 10.), np.full(3, .1), np.full(20, .1)],
                            diff_step=1e-6, xtol=1e-10, ftol=1e-10, max_nfev=400)
        x = sol.x
        p, _ = solver.forward(x)
        centers = (p[:-1] + p[1:]) / 2
        wrap = wrap_angle(centers)
        q_deg = np.degrees(abs(x[6:]).max())
        front_y = float(centers[:8, 1].mean()) - float(C0[:8, 1].mean())
        rear_x = float(centers[13:21, 0].mean()) - float(C0[13:21, 0].mean())
        deepest, ground, selfhit = 0., False, False
        d.qpos[:3] = x[:3] / 1000
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[qadr] = x[6:]
        mujoco.mj_forward(m, d)
        for c in d.contact:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            ground |= (c.geom1 == floor and b2 >= head) or (c.geom2 == floor and b1 >= head)
            selfhit |= b1 >= head and b2 >= head and abs(b1 - b2) > 1 and c.dist < -3e-3
            deepest = min(deepest, float(c.dist))
        n_t = sum(1 for k in range(21) if 79.5 <= np.hypot(centers[k, 0] - 1100., centers[k, 2] - 1000.) <= 92. and centers[k, 1] >= -20.)
        n_s = sum(1 for k in range(21) if 79.5 <= np.hypot(centers[k, 1], centers[k, 2] - 1000.) <= 92. and centers[k, 0] <= 1112.)
        for k in range(8, 21):
            rt = float(np.hypot(centers[k, 0] - 1100., centers[k, 2] - 1000.))
            if rt <= 92. and centers[k, 1] >= 60. and (k, 'cross') not in landed:
                landed.add((k, 'cross'))
                miter.append(dict(u_front_mm=u_f, module=k))
                events.append(dict(u_front_mm=u_f, event='cross_onto_target', module=k))
        log.append(dict(step=step, u_front_mm=u_f, u_rear_mm=u_r, front_y_mm=front_y,
                        rear_x_mm=rear_x, max_joint_deg=q_deg, wrap_deg=wrap,
                        n_target=n_t, n_source=n_s, deepest_mm=deepest * 1000))
        table_psi.append(u_f)
        table_q.append(x[6:].copy())
        failed = []
        if q_deg > 88.:
            failed.append(f'joint_limit J{int(np.argmax(abs(x[6:]))) + 1}')
        if selfhit or ground:
            failed.append('self_or_ground')
        if deepest < -3.:
            failed.append('pipe_penetration')
        if wrap < 180.:
            failed.append('front_wrap_below_180')
        if n_t < 1:
            failed.append('no_target_contact')
        if n_s < 1:
            failed.append('no_source_contact')
        if failed:
            stop = ','.join(failed)
            break
        if step > 20 and abs(front_y) < .5 and abs(rear_x) < .5 and u_f > 20.:
            no_progress += 1
            if no_progress > 5:
                stop = 'stall'
                break
        else:
            no_progress = 0
        if step % 25 == 0:
            print(f'{start}/{mode} u_f={u_f:6.1f} u_r={u_r:6.1f} 前+Y {front_y:7.2f} 后+X {rear_x:7.2f} mm '
                  f'包绕 {wrap:5.1f}° |q| {q_deg:5.1f}° 接触T{n_t}/S{n_s} 迁移 {sorted(set(mm["module"] for mm in miter))}', flush=True)
    p, _ = solver.forward(x)
    centers = (p[:-1] + p[1:]) / 2
    front_y = float(centers[:8, 1].mean()) - float(C0[:8, 1].mean())
    rear_x = float(centers[13:21, 0].mean()) - float(C0[13:21, 0].mean())
    a2_pass = bool(front_y >= CRITERION['A2_front_y_min_mm'] and rear_x >= CRITERION['A2_rear_x_min_mm'])
    a3_pass = bool(miter)
    table = np.array(table_q)
    np.savez(case / 'reference_table.npz', u_front_mm=np.array(table_psi), joint_rad=table,
             kappa=kappa, arc_per_rad=arc_per_rad, start_qpos=qpos0)
    result = dict(start=start, mode=mode, criterion=CRITERION, stop_reason=stop,
                  u_front_mm=u_f, u_rear_mm=u_r, steps=len(log),
                  A2=dict(front_y_mm=front_y, rear_x_mm=rear_x, passed=a2_pass),
                  A3=dict(crossed=sorted(set(mm['module'] for mm in miter)), passed=a3_pass),
                  final_wrap_deg=wrap_angle(centers), final_max_joint_deg=float(np.degrees(abs(x[6:]).max())),
                  events=events)
    result['verdict'] = 'PASS' if (a2_pass and a3_pass) else 'FAIL'
    write(case / 'report.json', result)
    write(case / 'log.json', log[::5] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in result.items() if k not in ('events',)}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', nargs='+', default=['CLOSE1B', 'CLOSE2B'])
    parser.add_argument('--modes', nargs='+', default=['simultaneous', 'alternate'])
    args = parser.parse_args()
    summary = {}
    for s in args.starts:
        for md in args.modes:
            summary[f'{s}_{md}'] = run(s, md)
    write(OUT / 'phase_aprime_v3_summary.json', summary)
