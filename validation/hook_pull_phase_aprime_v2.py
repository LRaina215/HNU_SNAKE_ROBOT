"""阶段A′离线验证 v2：交替试探脉冲（用户设想），修正管轴单位后的正确参数化。

用户设想：前部在目标管上 rolling 一小步、后部沿源管向接头 rolling 一小步，交替进行，
让过渡段逐渐移到前部。力矩反馈不在运动学模型里，"卡住"以脉冲收益近似（<1 mm 受阻）。

参数化（与 step4_feasibility 相同、轴坐标为毫米）：每个 2 mm 步只推进一端——
  FRONT 步：前部 M0–M7 全部节点沿目标管螺旋刚体推进（保持钩形，包绕不散），
            后部 M10–M20 节点固定，过渡段自由；
  REAR  步：后部沿源管螺旋向接头推进，前部固定，过渡段自由。
判据（运行前登记）：A2' 前部累计 ≥25 mm 且后部累计 ≥25 mm；A3' 至少 1 个 M≥8 模块
落到目标管斜切区以外（垂足 y≥60 mm）并保持；通用门槛：关节 ≤88°、无非相邻自碰、
不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触（模块中心距管轴 80–92 mm 计）；
连续 2 个脉冲（一前一后）收益均 <1 mm 判为停滞。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from curve_guided_geometry import MODEL, CORNER, points
from snake_control.hook_pull_motion import RollSolver
from snake_control.hook_pull_review_support import (
    R95_MM, PITCH_MM, L_TURN_MM, ROT_PER_MM, AX_PER_MM,
    restore_saved_state, axis_progress_mm)

BASE = ROOT / 'experiments/hook_pull_20261006'
OUT = BASE / 'phase_aprime'
AXIS_T = np.array([1100., 0., 1000.])
AXIS_S = np.array([0., 0., 1000.])
W_STRONG, W_MED, W_CONT = 60., 15., 20.
STEP_MM = 2.
PULSE_MM = 8.
GAIN_MIN_MM = 1.
MAX_PULSES = 60
CRITERION = dict(A2_front_y_min_mm=25., A2_rear_x_min_mm=25., A3_module_min=8,
                 gates='关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、包绕 ≥180°、两管各 ≥1 接触、停滞=连续2脉冲收益<1mm')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def wrap_angle(centers):
    uv = (centers[:9] - AXIS_T)[:, [0, 2]][::-1]
    a = np.arctan2(uv[:, 1], uv[:, 0])
    return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def run(start):
    case = OUT / f'{start}_v2'
    case.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    head = m.body('head').id
    source = {m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
    target = {m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
    floor = m.geom('floor').id
    restore_saved_state(m, d, BASE / start / 'final_state.npz')
    qpos0 = d.qpos.copy()
    joints0 = d.qpos[qadr].copy()  # Planning shape only; not an executed target.
    P0 = points(m, d)
    C0 = (P0[:-1] + P0[1:]) / 2
    # 前部钩形螺旋参数（dθ/dy 与每弧度弧长）
    th = np.arctan2(C0[:9, 2] - 1000., C0[:9, 0] - 1100.)
    y = C0[:9, 1]
    order = np.argsort(y)
    th_u = np.unwrap(th[order])
    kappa = float(np.polyfit(y[order], th_u, 1)[0])
    arc_per_rad = float(8 * 64.5 / max(abs(th_u[-1] - th_u[0]), 1e-6))
    src_contacts = [k for k in range(21) if 79.5 <= np.hypot(C0[k, 1], C0[k, 2] - 1000.) <= 92. and C0[k, 0] <= 1112.]
    print(f'{start}: κ={kappa:.5f} rad/mm arc/rad={arc_per_rad:.1f} 源管接触 {src_contacts}', flush=True)

    def forward(x):
        d.qpos[:3] = x[:3] / 1000
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[qadr] = x[6:]
        mujoco.mj_kinematics(m, d)
        p = points(m, d)
        frames = d.xmat[head:head + 21].reshape(-1, 3, 3).copy()
        return p, frames

    solver = RollSolver(m, qpos0[:3] * 1000, qpos0[3:7], joints0, list(range(10, 21)))
    x = solver.x0.copy()

    def targets(end, u_f, u_r):
        """FRONT: 前部沿目标管螺旋推进 u_f，后部固定；REAR: 后部沿源管推进 u_r，前部固定。"""
        t = solver.p0.copy()
        if end == 'FRONT':
            axial = u_f / (abs(kappa) * arc_per_rad)
            rot_y = -kappa * axial
            rot = Rotation.from_rotvec([0., rot_y, 0.])
            t[:9] = rot.apply(solver.p0[:9] - AXIS_T) + AXIS_T + np.array([0., axial, 0.])
        else:
            phi = u_r * ROT_PER_MM
            rot = Rotation.from_rotvec([-phi, 0., 0.])
            t[10:] = rot.apply(solver.p0[10:] - AXIS_S) + AXIS_S + np.array([u_r * AX_PER_MM, 0., 0.])
        return t

    def residual(x, tgt, previous):
        p, frames = solver.forward(x)
        err = [(solver.weights[:, None] * (p - tgt)).ravel(), 20. * (x[6:] - previous[6:])]
        return np.concatenate(err)

    solver.weights = np.full(22, W_MED)
    solver.weights[:9] = W_STRONG
    solver.weights[10:] = W_STRONG
    u_f = u_r = 0.
    log, pulses, events, miter = [], [], [], []
    landed = set()
    for k in src_contacts:
        landed.add((k, 'source'))
    next_end = 'FRONT'
    blocked = {'FRONT': 0, 'REAR': 0}
    stop = 'pulse_cap'
    pulse_gain = 0.
    pulse_start_front = pulse_start_rear = 0.
    for pulse in range(MAX_PULSES):
        end = next_end
        pulse_gain = 0.
        pulse_start_front = u_f
        pulse_start_rear = u_r
        pulse_points, _ = forward(x)
        pulse_centers = (pulse_points[:-1] + pulse_points[1:]) / 2
        pulse_front_y = float(pulse_centers[:8, 1].mean())
        pulse_rear_x = float(pulse_centers[13:21, 0].mean())
        pulse_log = 0
        while pulse_log < int(PULSE_MM / STEP_MM) + 6:
            if end == 'FRONT':
                u_f += STEP_MM
            else:
                u_r += STEP_MM
            tgt = targets(end, u_f, u_r)
            low = np.r_[np.full(6, -np.inf), np.full(20, -np.pi / 2 + 1e-3)]
            high = np.r_[np.full(6, np.inf), np.full(20, np.pi / 2 - 1e-3)]
            sol = least_squares(residual, x, bounds=(low, high), args=(tgt, x),
                                x_scale=np.r_[np.full(3, 10.), np.full(3, .1), np.full(20, .1)],
                                diff_step=1e-6, xtol=1e-10, ftol=1e-10, max_nfev=400)
            x = sol.x
            p, frames = forward(x)
            q = np.degrees(abs(x[6:]).max())
            centers = (p[:-1] + p[1:]) / 2
            wrap = wrap_angle(centers)
            node_err = float(np.linalg.norm(p - tgt, axis=1).max())
            deepest = 0.
            ground = selfhit = False
            d.qpos[:3] = x[:3] / 1000
            d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
            d.qpos[qadr] = x[6:]
            mujoco.mj_forward(m, d)
            for c in d.contact:
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                ground |= (c.geom1 == floor and b2 >= head) or (c.geom2 == floor and b1 >= head)
                selfhit |= b1 >= head and b2 >= head and abs(b1 - b2) > 1 and c.dist < -3e-3
                if (c.geom1 in source | target and b2 >= head) or (c.geom2 in source | target and b1 >= head):
                    deepest = min(deepest, float(c.dist))
            n_t = sum(1 for k in range(21) if 79.5 <= np.hypot(centers[k, 0] - 1100., centers[k, 2] - 1000.) <= 92. and centers[k, 1] >= -20.)
            n_s = sum(1 for k in range(21) if 79.5 <= np.hypot(centers[k, 1], centers[k, 2] - 1000.) <= 92. and centers[k, 0] <= 1112.)
            for k in range(8, 21):
                rt = float(np.hypot(centers[k, 0] - 1100., centers[k, 2] - 1000.))
                if rt <= 92. and centers[k, 1] >= 60. and (k, 'cross') not in landed:
                    landed.add((k, 'cross'))
                    miter.append(dict(pulse=pulse, module=k))
                    events.append(dict(pulse=pulse, event='cross_onto_target', module=k))
            log.append(dict(pulse=pulse, end=end, u_front_mm=u_f, u_rear_mm=u_r,
                            front_y_mm=(float(centers[:8, 1].mean()) - float(C0[:8, 1].mean())),
                            rear_x_mm=(float(centers[13:21, 0].mean()) - float(C0[13:21, 0].mean())),
                            max_joint_deg=q, wrap_deg=wrap, node_err_mm=node_err,
                            n_target=n_t, n_source=n_s, deepest_mm=deepest * 1000))
            pulse_log += 1
            failed = []
            if q > 88.:
                failed.append(f'joint_limit J{int(np.argmax(abs(x[6:]))) + 1}')
            if selfhit or ground:
                failed.append('self_or_ground')
            if deepest * 1000 < -3.:
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
        front_now, rear_now = axis_progress_mm(centers, pulse_front_y, pulse_rear_x)
        gain = front_now if end == 'FRONT' else rear_now
        pulses.append(dict(index=pulse, end=end, gain_mm=gain, front_y_mm=front_now, rear_x_mm=rear_now,
                           commanded_arc_gain_mm=(u_f - pulse_start_front) if end == 'FRONT' else (u_r - pulse_start_rear),
                           u_front_mm=u_f, u_rear_mm=u_r))
        print(f"pulse {pulse:2d} {end:5s}: 收益 {gain:5.2f} mm | 累计弧长 前 {u_f:6.1f} 后 {u_r:6.1f} mm "
              f"包绕 {wrap:5.1f}° |q|max {q:5.1f}° 迁移 {sorted(set(mm['module'] for mm in miter))}", flush=True)
        if stop != 'pulse_cap':
            break
        blocked[end] = 0 if gain >= GAIN_MIN_MM else blocked[end] + 1
        if blocked[end] >= 2 and blocked[('REAR' if end == 'FRONT' else 'FRONT')] >= 2:
            stop = 'stall_both_ends'
            break
        next_end = 'REAR' if end == 'FRONT' else 'FRONT'
    p, _ = forward(x)
    centers = (p[:-1] + p[1:]) / 2
    front_y = float(centers[:8, 1].mean()) - float(C0[:8, 1].mean())
    rear_x = float(centers[13:21, 0].mean()) - float(C0[13:21, 0].mean())
    result = dict(start=start, criterion=CRITERION, stop_reason=stop,
                  A2=dict(front_y_mm=front_y, rear_x_mm=rear_x,
                          passed=bool(front_y >= CRITERION['A2_front_y_min_mm'] and rear_x >= CRITERION['A2_rear_x_min_mm']),
                          u_front_mm=u_f, u_rear_mm=u_r),
                  A3=dict(crossed=sorted(set(mm['module'] for mm in miter)), passed=bool(miter)),
                  pulses=pulses, events=events,
                  final_max_joint_deg=float(np.degrees(abs(x[6:]).max())), final_wrap_deg=wrap_angle(centers))
    result.update(physics_steps=0, evidence_scope='offline geometry, not physical contact/holding or active rolling',
                  dynamic_admission='NOT_ADMITTED', gates_passed=stop == 'pulse_cap')
    result['verdict'] = 'PASS' if (result['A2']['passed'] and result['A3']['passed'] and result['gates_passed']) else 'FAIL'
    write(case / 'report.json', result)
    write(case / 'log.json', log[::10] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in result.items() if k not in ('pulses', 'events')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', nargs='+', default=['CLOSE1B', 'CLOSE2B'])
    args = parser.parse_args()
    summary = {s: run(s) for s in args.starts}
    write(OUT / 'phase_aprime_v2_summary.json', summary)
