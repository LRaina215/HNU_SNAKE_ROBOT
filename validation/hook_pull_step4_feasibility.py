"""第4步可行性（离线，不推进物理时间）：后部向接头滚、前部沿目标管远去、过渡段送身。

状态：CLOSE1B 终态（钩子抱住目标管、后部在源管）。
参数化：以穿过拐角的身体弧长 u 为唯一进度。后部按 R95/P220 理想螺旋刚体螺旋转动
（每弧长转 2π/637.7，轴向前进 220/637.7）；前部按当前钩形自身拟合的螺旋转动
（每步从上一解重新拟合 dθ/dy 与每弧度弧长）；过渡节点 M8/M9 之间放开由链自行求形。
每步对 根位姿+20关节 做加权最小二乘，门限：接触节残差、关节限位、关节速率、
非相邻节间距（自碰）、穿管（有限长度圆柱代理）、离地、前部包绕角。
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

BASE = ROOT / 'experiments/hook_pull_20261006'
OUT = BASE / 'step4_feasibility'
CLOSE1B = BASE / 'CLOSE1B/final_state.npz'
AXIS_T = np.array([1100., 0., 1000.])   # 目标管轴：+Y
AXIS_S = np.array([0., 0., 1000.])      # 源管轴：+X
R_NOMINAL, PITCH = 95., 220.
L_TURN = float(np.hypot(2 * np.pi * R_NOMINAL, PITCH))
PHI_PER_MM = 2 * np.pi / L_TURN          # 后部：每弧长的转角
AXIAL_PER_MM = PITCH / L_TURN            # 后部：每弧长的轴向前进
HEAD = 0                                  # body id offset
FRONT_CONTACT = [0, 3, 7]                 # CLOSE1B 终态目标管接触模块
REAR_CONTACT = [14, 17, 20]               # 源管接触模块
FRONT_NODES = list(range(0, 9))           # M0–M7 → 节点 0..8
REAR_NODES = list(range(10, 22))          # M10–M20 → 节点 10..21
W_STRONG, W_MED, W_CONT = 60., 15., 20.
W_GUIDE, Z_FLOOR, GUIDE_Y = 60., 1078., 80.   # 跨接区弱引导：拐角附近不低于管顶体高，防内侧抄近路
U_MAX, U_STEP = 300., 2.
PIPE_R, BODY_R = 55., 29.
TOUCH = PIPE_R + BODY_R                   # 84：贴管中心线半径


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def unwrap_deg(deg):
    return np.degrees(np.unwrap(np.radians(deg)))


class Chain:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.head = self.m.body('head').id

    def forward(self, x):
        d = self.d
        d.qpos[:3] = x[:3] / 1000
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[self.qadr] = x[6:]
        mujoco.mj_kinematics(self.m, d)
        p = points(self.m, d)
        frames = d.xmat[self.head:self.head + 21].reshape(-1, 3, 3).copy()
        return p, frames


def screw_front(p, rot_y, axial):
    """前部刚体螺旋：绕目标管轴转 rot_y（绕+y 的旋转矢量分量），轴向 +Y 前进 axial。

    Rot_y(ψ) 使截面角 θ−ψ；拟合 dθ/dy=κ<0（+Y 前进时 θ 减小），调用方传 rot_y=−κ·axial>0。
    """
    rot = Rotation.from_rotvec([0., rot_y, 0.])
    return rot.apply(p - AXIS_T) + AXIS_T + [0., axial, 0.]


def screw_rear(p, phi, axial):
    """后部刚体螺旋：绕源管轴转 phi（θ 减小方向 = 向接头 +X），轴向 +X 前进 axial。"""
    rot = Rotation.from_rotvec([-phi, 0., 0.])   # Rot_x(ψ) 使 θ+ψ，故取 −φ
    return rot.apply(p - AXIS_S) + AXIS_S + [axial, 0., 0.]


def fit_front_screw(centers):
    """当前钩形的 dθ/dy（rad/mm）与每弧度（绕管轴）弧长。θ 先按轴向排序再展开。"""
    th = np.arctan2(centers[:9, 2] - 1000., centers[:9, 0] - 1100.)
    y = centers[:9, 1]
    order = np.argsort(y)
    th_u = np.unwrap(th[order])
    kappa = np.polyfit(y[order], th_u, 1)[0]
    span = abs(th_u[-1] - th_u[0])
    arc = 8 * 64.5
    arc_per_rad = arc / max(span, 1e-6)
    return float(kappa), float(arc_per_rad)


def radial(p, axis_point, mode):
    if mode == 'target':
        return np.hypot(p[:, 0] - axis_point[0], p[:, 2] - axis_point[2])
    return np.hypot(p[:, 1] - axis_point[1], p[:, 2] - axis_point[2])


def wrap_angle(centers):
    """M8→M0 绕目标管轴累计转角（与项目 pipe_inside 判据同义）。"""
    uv = (centers[:9] - AXIS_T)[:, [0, 2]][::-1]
    a = np.arctan2(uv[:, 1], uv[:, 0])
    return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chain = Chain()
    with np.load(CLOSE1B) as state:
        mujoco.mj_setState(chain.m, chain.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        ctrl0 = state['ctrl'].copy()
        phi_start = float(state['psi_rad'])
    mujoco.mj_forward(chain.m, chain.d)
    P0 = points(chain.m, chain.d)
    C0 = (P0[:-1] + P0[1:]) / 2
    qadr = chain.qadr
    q_actual = chain.d.qpos[qadr].copy()

    kappa0, arc_per_rad0 = fit_front_screw(C0)
    print(f'初始前部钩形：dθ/dy={kappa0:.5f} rad/mm，每弧度弧长={arc_per_rad0:.1f} mm'
          f'（后部标准值 {1 / PHI_PER_MM:.1f} mm/rad）', flush=True)

    contact_front_nodes = sorted({n for m_ in FRONT_CONTACT for n in (m_, m_ + 1)})
    contact_rear_nodes = sorted({n for m_ in REAR_CONTACT for n in (m_, m_ + 1)})
    med_front_nodes = [n for n in FRONT_NODES if n not in contact_front_nodes]
    med_rear_nodes = [n for n in REAR_NODES if n not in contact_rear_nodes]
    # 每个后部节点材料距剥离点（源管→拐角过渡）的弧长：node10≈半模块，依次 +64.5。
    arc_before_departure = {n: (n - 10) * 64.5 + 32.25 for n in REAR_NODES}
    weights_template = np.zeros(22)
    weights_template[contact_front_nodes] = W_STRONG
    weights_template[contact_rear_nodes] = W_STRONG
    weights_template[med_front_nodes] = W_MED
    weights_template[med_rear_nodes] = W_MED

    x = np.r_[chain.d.qpos[:3] * 1000,
              Rotation.from_quat(chain.d.qpos[3:7][[1, 2, 3, 0]]).as_rotvec(), q_actual]

    def residual(x, phi_f, axial_f, phi_r, axial_r, previous, weights, rear_active):
        p, _ = chain.forward(x)
        tf = screw_front(P0[:9], phi_f, axial_f)
        err = [(weights[:9, None] * (p[:9] - tf)).ravel()]
        tr = screw_rear(P0[10:], phi_r, axial_r)[rear_active]
        w = weights[10:][rear_active]
        err.append((w[:, None] * (p[10:][rear_active] - tr)).ravel()
                   if rear_active.any() else np.zeros(0))
        # 跨接区（节点9/10/11）防内切引导：拐角附近且未被螺旋完全约束时，
        # z 低于管顶体高才惩罚；固定长度3，条件不满足时该项为0。
        sag = np.zeros(3)
        for j, n in enumerate((9, 10, 11)):
            if weights[n] < W_MED - 1e-6 and p[n][1] < GUIDE_Y:
                sag[j] = max(0., Z_FLOOR - p[n][2])
        err.append(W_GUIDE * sag)
        err.append(W_CONT * (x[6:] - previous[6:]))
        return np.concatenate(err)

    rows = []
    admitted = 0
    u = 0.
    kappa, arc_per_rad = kappa0, arc_per_rad0
    note = ''
    for step in range(int(U_MAX / U_STEP) + 1):
        u = step * U_STEP
        phi_r = u * PHI_PER_MM
        axial_r = u * AXIAL_PER_MM
        axial_f = u / (abs(kappa0) * arc_per_rad0)        # 前部每弧长的轴向前进（参数固定）
        rot_y_f = -kappa0 * axial_f                       # +Y 前进时 θ 减小：Rot_y(+|κ|·axial)
        phi_f = abs(rot_y_f)
        weights = weights_template.copy()
        # 渐进剥离：距剥离点弧长进入 48mm 窗口后权重线性退到 0，目标仍沿源管螺旋，
        # 避免二元释放造成解突跳；完全释放后由链在前部牵引下自行求形。
        for n in REAR_NODES:
            fade = float(np.clip((arc_before_departure[n] - u) / 48. + 1., 0., 1.))
            weights[n] = weights_template[n] * fade
        rear_active = weights[10:] > 1e-6
        low = np.r_[np.full(6, -np.inf), np.full(20, -np.pi / 2 + 1e-3)]
        high = np.r_[np.full(6, np.inf), np.full(20, np.pi / 2 - 1e-3)]
        solution = least_squares(residual, x, bounds=(low, high),
                                 args=(rot_y_f, axial_f, phi_r, axial_r, x, weights, rear_active),
                                 x_scale=np.r_[np.full(3, 10.), np.full(3, .1), np.full(20, .1)],
                                 diff_step=1e-6, xtol=1e-10, ftol=1e-10, max_nfev=600)
        x = solution.x
        p, frames = chain.forward(x)
        q = x[6:]
        centers = (p[:-1] + p[1:]) / 2
        cq = np.zeros(22)
        cq[contact_front_nodes] = np.linalg.norm(p[contact_front_nodes] - screw_front(P0[contact_front_nodes], rot_y_f, axial_f), axis=1)
        cq[contact_rear_nodes] = np.linalg.norm(p[contact_rear_nodes] - screw_rear(P0[contact_rear_nodes], phi_r, axial_r), axis=1)
        gap_pairs = [(i, j) for i in range(21) for j in range(i + 2, 21)]
        min_pair = min(float(np.linalg.norm(centers[i] - centers[j])) for i, j in gap_pairs)
        rad_t = radial(centers, AXIS_T, 'target')
        rad_s = radial(centers, AXIS_S, 'source')
        pen_t = rad_t[(centers[:, 1] >= -5.)].min() if np.any(centers[:, 1] >= -5.) else 999.
        pen_s = rad_s[(centers[:, 0] <= 1105.)].min() if np.any(centers[:, 0] <= 1105.) else 999.
        row = dict(u_mm=u, phi_front_deg=float(np.degrees(abs(rot_y_f))), phi_rear_deg=float(np.degrees(phi_r)),
                   axial_front_mm=float(axial_f), axial_rear_mm=float(axial_r),
                   contact_residual_max_mm=float(cq.max()),
                   contact_residual_front_mm=[float(v) for v in cq[contact_front_nodes]],
                   contact_residual_rear_mm=[float(v) for v in cq[contact_rear_nodes]],
                   max_joint_deg=float(np.degrees(abs(q)).max()),
                   step_joint_change_deg=float(np.degrees(abs(q - rows[-1]['q_rad'])).max()) if rows else 0.,
                   step_joint_rate_deg_per_mm=float(np.degrees(abs(q - rows[-1]['q_rad'])).max()) / U_STEP if rows else 0.,
                   min_nonadjacent_center_mm=min_pair,
                   min_radial_target_mm=float(pen_t), min_radial_source_mm=float(pen_s),
                   min_center_z_mm=float(centers[:, 2].min()),
                   wrap_deg=wrap_angle(centers),
                   q_rad=q.tolist())
        rows.append(row)
        gates = dict(contact=row['contact_residual_max_mm'] > 6., limit=row['max_joint_deg'] > 88.,
                     rate=rows and row['step_joint_rate_deg_per_mm'] > 10., self_col=min_pair < 56.,
                     pen_t=pen_t < 78., pen_s=pen_s < 78., ground=centers[:, 2].min() < 40.,
                     wrap=wrap_angle(centers) < 180.)
        failed = [k for k, v in gates.items() if v]
        row['gates_failed'] = failed
        if failed:
            note = f'u={u:.0f}mm 失败门限：{failed}'
            rows.pop()
            np.savez(OUT / 'failing_state.npz', x=x, u_mm=u, failed=','.join(failed))
            break
        row['x_backup'] = x.tolist()
        admitted += 1
        if u and u % 50 == 0:
            print(f"u={u:5.0f}mm φ_f={row['phi_front_deg']:6.1f}° φ_r={row['phi_rear_deg']:6.1f}° "
                  f"接触残差={row['contact_residual_max_mm']:4.1f}mm |q|max={row['max_joint_deg']:5.1f}° "
                  f"步变={row['step_joint_change_deg']:4.1f}° 自碰间距={min_pair:5.1f}mm "
                  f"径向min T/S={pen_t:5.1f}/{pen_s:5.1f} 包绕={row['wrap_deg']:5.1f}°", flush=True)
        if u >= U_MAX:
            note = f'达到扫描上限 u={U_MAX:.0f}mm 仍全部通过'
            break

    for r in rows:
        r.pop('x_backup', None)
    head_advance = float(rows[-1]['axial_front_mm']) if rows else 0.
    rear_advance = float(rows[-1]['axial_rear_mm']) if rows else 0.
    max_rate = max((r['step_joint_rate_deg_per_mm'] for r in rows), default=0.)
    result = dict(
        verdict=None, note=note,
        start=dict(state='CLOSE1B final', kappa_front_rad_per_mm=kappa0, front_arc_per_rad_mm=arc_per_rad0,
                   rear_arc_per_rad_mm=1 / PHI_PER_MM),
        admitted_steps=admitted, u_admitted_mm=admitted * U_STEP,
        max_joint_rate_deg_per_mm=max_rate,
        implied_max_feed_mm_s=(10. / max_rate if max_rate > 1e-9 else None),
        front_axial_advance_mm=head_advance, rear_axial_advance_mm=rear_advance,
        modules_through_crossing=round(admitted * U_STEP / 64.5, 2),
        rate_note='步进 2mm 弧长；后部转角按 R95 理想螺旋，前部螺旋参数固定为初始拟合；'
                  'implied_max_feed=10°/s 关节速率下允许的最大前进速度',
        rows=[{k: v for k, v in r.items() if k != 'q_rad'} for r in rows])
    result['verdict'] = 'EXISTS' if admitted * U_STEP >= 120. else 'NOT_ESTABLISHED'
    write(OUT / 'step4_feasibility.json', result)
    write(OUT / 'step4_rows.json', rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
