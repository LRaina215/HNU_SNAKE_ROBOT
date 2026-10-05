"""弯曲轴顺应性（snake_control/curved_axis_compliance.py）的离线检查，不需要 EXE 或 MuJoCo 窗口。

1. 退化检查：把局部轴全部设为 EXE 的直轴、关闭螺距角调制，目标关节角与原方法 build_backbone 逐位一致；
2. 合成弯管：关节点取自绕“直管 → 90° 弯 → 直管”中心线的螺旋（机器人中点在弯头中点），比较只用局部轴（第 1、2 层）与加螺距角调制（第 3 层）
   重建的目标曲线离真实管轴的距离：调制后目标应沿弯曲管轴缠绕。
运行：python validation/curved_axis_unit.py
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from snake_control import curved_axis_compliance as cac
from snake_control.pole_compliance import ComplianceParameters, build_backbone

LINK = 64.5
JOINTS = 26


def straight_equivalence():
    geometry = json.loads((ROOT / 'validation/fixtures/geometry_example.json').read_text(encoding='utf-8'))
    params = ComplianceParameters((.04, .12), .25, False)
    reference = build_backbone(geometry, params)
    direction = np.asarray(geometry['projection_axis']['direction'], dtype=float)
    points = np.asarray(geometry['points_mm'], dtype=float)
    if (points[-1] - points[0]) @ direction < 0:
        direction = -direction
    original = cac.local_axes
    cac.local_axes = lambda p, w, s, f=None, used=None: (np.tile(direction, (len(p) - 1, 1)), np.zeros(len(p) - 1))
    try:
        backbone, _ = cac.build_curved_backbone(geometry, params, cac.CurvedAxisOptions(modulation=False))
    finally:
        cac.local_axes = original
    worst = max(float(np.abs(backbone.joint_angles(phase) - reference.joint_angles(phase)).max())
                for phase in (0., .7, 2.1))
    assert worst < 1e-9, worst
    return worst


def frenet_curve(backbone, step=.5):
    """按分段常 κ、τ 积分 Frenet 方程，返回沿弧长 step 的采样点与对应弧长。"""
    frame, point = np.eye(3), np.zeros(3)  # 列：T、N、B
    points, arcs, s = [point.copy()], [0.], 0.
    for segment in backbone.segments:
        n = max(1, math.ceil(segment.length / step))
        h = segment.length / n
        omega = np.array([segment.tau, 0., segment.kappa])  # 体坐标系下 Darboux 向量
        angle = np.linalg.norm(omega) * h
        if angle > 0:
            k = omega / np.linalg.norm(omega)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            rotation = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        else:
            rotation = np.eye(3)
        for _ in range(n):
            mid = frame @ (rotation if angle == 0 else np.eye(3) + .5 * (rotation - np.eye(3)))
            point = point + h * mid[:, 0]
            frame = frame @ rotation
            s += h
            points.append(point.copy())
            arcs.append(s)
    return np.array(points), np.array(arcs)


def kabsch(source, target):
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = (u @ flip @ vt).T
    return lambda x: (x - source.mean(0)) @ rotation.T + target.mean(0)


def bent_pipe_joints(bend_radius, rho=84., pitch=220., straight=700.):
    """中心线：直管 → 90° 圆弧弯（bend_radius，inf 为全直）→ 直管；螺旋按平行移动标架、每 pitch 一圈缠绕，
    机器人中点位于弯头中点。按 64.5 mm 弦长取 28 个关节点；返回 (关节点, 中心线采样)。"""
    ds = .05
    arc = bend_radius * math.pi / 2 if math.isfinite(bend_radius) else 0.
    sigma = np.arange(0, 2 * straight + arc, ds)
    center, tangent = np.zeros((len(sigma), 3)), np.zeros((len(sigma), 3))
    point, direction, hinge = np.zeros(3), np.array([0, 0, 1.]), np.array([0, -1., 0])  # 向 +x 弯
    for k, s in enumerate(sigma):
        center[k], tangent[k] = point, direction
        if straight <= s < straight + arc:
            a = ds / bend_radius
            direction = direction * math.cos(a) + np.cross(hinge, direction) * math.sin(a)
        point = point + direction * ds
    e1 = np.zeros_like(center)
    e1[0] = [1., 0, 0]
    for k in range(1, len(sigma)):
        v = e1[k - 1] - (e1[k - 1] @ tangent[k]) * tangent[k]
        e1[k] = v / np.linalg.norm(v)
    e2 = np.cross(tangent, e1)
    theta = 2 * np.pi * sigma / pitch
    curve = center + rho * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)
    k = int(np.argmin(np.abs(sigma - (straight + arc / 2 - 330))))  # 身体约覆盖 660 mm 中心线
    joints = [curve[k]]
    for _ in range(JOINTS + 1):
        distance = np.linalg.norm(curve[k:] - joints[-1], axis=1)
        k += int(np.argmax(distance >= LINK))
        joints.append(curve[k])
    return np.array(joints), center[::20]


def geometry_from_points(points):
    joints = [dict(joint_index=i + 1, axis='Yaw' if i % 2 == 0 else 'Pitch', link_length_mm=LINK,
                   minimum_deg=-90., maximum_deg=90.) for i in range(JOINTS)]
    return dict(points_mm=points.tolist(), config=dict(joint_count=JOINTS, head_length_mm=LINK, joints=joints),
                online=[True] * JOINTS, age_ms=0., sample_unix_ms=0., run_id='synthetic', config_id='synthetic',
                sequence=1)


def wrapping_error(bend_radius, modulation):
    points, center = bent_pipe_joints(bend_radius)
    params = ComplianceParameters(0., math.atan(220 / (2 * math.pi * 84)), False)  # K=0：只看形状重建
    backbone, info = cac.build_curved_backbone(geometry_from_points(points), params,
                                               cac.CurvedAxisOptions(modulation=modulation, axis_gain=1.))  # 几何检查：完全复现
    curve, arcs = frenet_curve(backbone)
    at_joints = np.array([curve[np.argmin(np.abs(arcs - s))] for s in np.r_[0., np.cumsum(backbone.links)]])
    placed = kabsch(at_joints, points)(curve)
    # 每个目标采样点到真实中心线的最近距离；理想情况下恒为 84 mm
    distance = np.array([np.min(np.linalg.norm(center - p, axis=1)) for p in placed[::10]])
    return dict(bend_mm=bend_radius, modulation=modulation, joint_rms_mm=float(np.sqrt(((kabsch(at_joints, points)(at_joints) - points) ** 2).sum(1).mean())),
                axis_distance_mean=float(distance.mean()), axis_distance_std=float(distance.std()),
                axis_bend_deg=info['axis_bend_deg'], beta_range_deg=(math.degrees(info['beta_min']), math.degrees(info['beta_max'])))


if __name__ == '__main__':
    print('退化检查（直轴、无调制，与 build_backbone 最大差 rad）:', straight_equivalence())
    for bend in (math.inf, 300., 165.):
        rows = {}
        for modulation in (False, True):
            r = rows[modulation] = wrapping_error(bend, modulation)
            print(f"弯曲半径 {bend:>5} 调制={'开' if modulation else '关'}：关节点 RMS {r['joint_rms_mm']:.1f} mm，"
                  f"目标到真实管轴 {r['axis_distance_mean']:.1f}±{r['axis_distance_std']:.1f} mm，"
                  f"估计轴弯曲 {r['axis_bend_deg']:.0f}°，β {r['beta_range_deg'][0]:.0f}–{r['beta_range_deg'][1]:.0f}°")
        # 直管上调制不起作用；弯管上调制后目标关节点误差至少降到 1/3
        if math.isinf(bend):
            assert abs(rows[True]['joint_rms_mm'] - rows[False]['joint_rms_mm']) < 1e-9
        else:
            assert rows[True]['joint_rms_mm'] < rows[False]['joint_rms_mm'] / 3, rows
    print('全部通过')
