"""钩挂回拉 第1步：离线运动学检查，不推进物理时间。

起点：开合 A2 第 17 s（张开完成、尚未向前送入）的记录姿态与指令。
1. 求整身滚动表：每个滚动角下保持全身形状、各连杆绕自身纵轴转同一角度（最小二乘）。
2. 理想不打滑：源管接触模块随材料滚动，解出整身绕源管轴的刚体螺旋运动（回拉 + 转动）。
3. 沿螺旋路径与“只平移不转动”两种路径移动整条身体，检查钩子内壁（M0–M6）何时碰到目标管、
   目标管轴是否被钩子包住，以及有无穿透源管、触地或非相邻自碰撞。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from curve_guided_geometry import MODEL, CORNER, points
from snake_control.hook_pull_motion import RollSolver

A2 = ROOT / 'experiments/curve_guided_transfer_20261006/target_open_close_20261006/OPEN_CLOSE_A2_ACTION'
OUT = ROOT / 'experiments/hook_pull_20261006/offline'
T0 = 17.0  # 可用命令行 --t0 / --out 覆盖
BODY_RADIUS, PIPE_RADIUS = 29., 55.
SOURCE_POINT = np.array([0., 0., 1000.])
REAR = list(range(10, 21))
HOOK = list(range(0, 7))
PSI = np.radians(np.arange(0., 200.1, 2.5))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def encloses(polygon, point=(0., 0.)):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    head = m.body('head').id
    source = {m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
    target = {m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
    floor = m.geom('floor').id

    with np.load(A2 / 'trace.npz') as trace:
        i = int(np.argmin(abs(trace['sim_time'] - T0)))
        qpos0, ctrl0, q0 = trace['qpos'][i].copy(), trace['ctrl'][i].copy(), trace['q'][i].copy()
        source_force = trace['contact_force_N'][i, 0].copy()
        target_force = trace['contact_force_N'][i, 1].copy()
        time0 = float(trace['sim_time'][i])

    # ---- 2. 理想不打滑螺旋运动（每弧度后部材料滚动；先用小角度定方向） ----
    probe = RollSolver(m, qpos0[:3] * 1000, qpos0[3:7], ctrl0, REAR).table(PSI[:2])
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    pts = points(m, d)
    centers = (pts[:-1] + pts[1:]) / 2
    contacting = [k for k in range(21) if source_force[k] > .1]
    x_axis = np.array([1., 0., 0.])
    rate = [(probe[1]['link_roll_rad'][k] - probe[0]['link_roll_rad'][k]) / (PSI[1] - PSI[0]) for k in range(21)]
    A, b = [], []
    for k in contacting:
        radial = centers[k] - SOURCE_POINT
        radial[0] = 0.
        R = np.linalg.norm(radial)
        n = radial / R
        t = d.xmat[head + k].reshape(3, 3)[:, 2]
        A.append(np.column_stack([x_axis, (R - BODY_RADIUS) * np.cross(x_axis, n)]))
        b.append(BODY_RADIUS * rate[k] * np.cross(t, n))
    A, b = np.vstack(A), np.concatenate(b)
    (dz, dphi), *_ = np.linalg.lstsq(A, b, rcond=None)
    fit_residual = float(np.linalg.norm(A @ [dz, dphi] - b) / np.linalg.norm(b))
    sign = -1. if dz > 0 else 1.  # 取让整身沿 -X 离开接头的滚动方向
    dz, dphi = sign * dz, sign * dphi
    screw = dict(pull_mm_per_rad=float(-dz), rotation_deg_per_rad=float(np.degrees(dphi)),
                 rotation_deg_per_10mm_pull=float(np.degrees(dphi) / -dz * 10), roll_sign_for_pull=sign,
                 relative_fit_residual=fit_residual, source_contact_modules=contacting)

    # ---- 1. 回拉方向的整身滚动表（以当前指令形状为准） ----
    psi = sign * PSI
    rows = RollSolver(m, qpos0[:3] * 1000, qpos0[3:7], ctrl0, REAR).table(psi)
    joints = np.array([r['x'][6:] for r in rows])
    order = np.argsort(psi)
    np.savez(OUT / 'roll_table.npz', psi_rad=psi[order], joint_rad=joints[order], start_ctrl_rad=ctrl0,
             start_qpos=qpos0, start_time_s=time0, roll_sign_for_pull=sign)
    dq = np.abs(np.diff(joints, axis=0)).max(1) / np.diff(PSI)
    wrap = lambda a: np.angle(np.exp(1j * a))
    shape = dict(
        max_point_error_mm=float(max(r['point_error_mm'].max() for r in rows)),
        max_rms_point_error_mm=float(max(np.sqrt((r['point_error_mm'] ** 2).mean()) for r in rows)),
        rear_roll_error_deg=float(np.degrees(max(abs(wrap(r['link_roll_rad'][REAR] - r['psi'])).max() for r in rows))),
        free_roll_error_deg=float(np.degrees(max(abs(wrap(r['link_roll_rad'][:10] - r['psi'])).max() for r in rows))),
        max_joint_deg=float(np.degrees(abs(joints).max())),
        max_joint_rate_deg_per_roll_deg=float(dq.max()),
        min_time_for_200deg_at_10deg_s=float(np.radians(200) * dq.max() / np.radians(10)),
        note='离散交替关节链滚动时形状本身会“呼吸”：标准 R95/P220 螺旋用 Yamada 解析滚动 60–120° 时误差也有 10–15 mm')

    # ---- 3. 沿两种理想路径检查 ----
    root_p = qpos0[:3] * 1000
    root_rot = Rotation.from_quat(qpos0[3:7][[1, 2, 3, 0]])
    pre_touch = {k for k in range(21) if target_force[k] > .1}

    def pose(pull, phi):
        rx = Rotation.from_rotvec([phi, 0., 0.])
        d.qpos[:] = qpos0
        d.qpos[:3] = (SOURCE_POINT + rx.apply(root_p - SOURCE_POINT) + [-pull, 0., 0.]) / 1000
        d.qpos[3:7] = (rx * root_rot).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        p = points(m, d)
        return (p[:-1] + p[1:]) / 2

    paths = {}
    for mode in ('screw', 'translate'):
        path = []
        for pull in np.arange(0., 140.1, 1.):
            phi = dphi * pull / -dz if mode == 'screw' else 0.
            cen = pose(pull, phi)
            hook_touch, source_pen, other_pen, ground, selfhit = set(), 0., 0., False, False
            for c in d.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                for pipe, body in [(g1, b2), (g2, b1)]:
                    module = body - head
                    if body >= head and pipe in target and c.dist < 0:
                        if module in HOOK:
                            hook_touch.add(module)
                        else:
                            other_pen = min(other_pen, float(c.dist * 1000))
                    if body >= head and pipe in source and c.dist < 0:
                        source_pen = min(source_pen, float(c.dist * 1000))
                if c.dist < 0:
                    ground |= floor in (g1, g2)
                    selfhit |= b1 >= head and b2 >= head and abs(b1 - b2) > 1
            uv = (cen[:9] - CORNER)[:, [0, 2]]
            gaps = np.linalg.norm(uv[HOOK], axis=1) - PIPE_RADIUS - BODY_RADIUS
            path.append(dict(pull_mm=float(pull), rotation_deg=float(np.degrees(phi)),
                             hook_touch=sorted(hook_touch), min_hook_gap_mm=float(gaps.min()),
                             closest_hook_module=int(np.argmin(gaps)),
                             pipe_axis_enclosed_by_M0_M8=encloses(uv),
                             source_penetration_mm=source_pen, M7_on_target_penetration_mm=other_pen,
                             ground=bool(ground), nonadjacent_self=bool(selfhit),
                             hook_uv_mm=np.round(uv, 1).tolist()))
        first = next((r for r in path if r['hook_touch']), None)
        clean = [r for r in path if first is None or r['pull_mm'] <= first['pull_mm']]
        paths[mode] = dict(first_hook_contact=first and {k: v for k, v in first.items() if k != 'hook_uv_mm'},
                           enclosed_at_first_contact=first['pipe_axis_enclosed_by_M0_M8'] if first else None,
                           source_penetration_before_contact_mm=min(r['source_penetration_mm'] for r in clean),
                           non_hook_target_penetration_before_contact_mm=min(r['M7_on_target_penetration_mm'] for r in clean),
                           ground_or_self_before_contact=any(r['ground'] or r['nonadjacent_self'] for r in clean))
        write(OUT / f'path_{mode}.json', path)

    result = dict(start=dict(trial=A2.relative_to(ROOT).as_posix(), time_s=time0,
                             source_contact_modules=contacting, target_contact_modules=sorted(pre_touch),
                             start_tracking_error_deg=float(np.degrees(abs(q0 - ctrl0)).max())),
                  roll_table=shape, no_slip_screw=screw, paths=paths,
                  note='理想刚体路径只说明几何是否能进钩；真实滚动会打滑，接触后的受力和变形要靠动力学回答。')
    write(OUT / 'offline_check.json', result)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, mode in zip(ax, ('screw', 'translate')):
        a.add_patch(plt.Circle((0, 0), PIPE_RADIUS, color='tab:green', alpha=.4))
        a.add_patch(plt.Circle((0, 0), PIPE_RADIUS + BODY_RADIUS, fill=False, ls='--', color='tab:green'))
        path = json.loads((OUT / f'path_{mode}.json').read_text(encoding='utf8'))
        first = paths[mode]['first_hook_contact']
        stop = first['pull_mm'] if first else 140
        for pull, color in zip(np.linspace(0, stop, 4), ['k', 'tab:blue', 'tab:orange', 'tab:red']):
            r = path[int(round(pull))]
            uv = np.array(r['hook_uv_mm'])
            a.plot(uv[:, 0], uv[:, 1], 'o-', ms=3, color=color, label=f'pull {pull:.0f}mm, rot {r["rotation_deg"]:.0f}°')
            a.annotate('M0', uv[0], color=color, fontsize=7)
        a.set_aspect('equal')
        a.set_xlabel('u = X - 1100 (mm)')
        a.set_ylabel('v = Z - 1000 (mm)')
        a.set_title(f'{mode}: hook M0-M8 in target-pipe section')
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'offline_check.png', dpi=130)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--t0', type=float, default=T0)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    T0, OUT = args.t0, Path(args.out)
    main()
