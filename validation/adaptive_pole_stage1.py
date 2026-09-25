"""阶段 1 验证：独立旧步态对照、数值积分核对，以及真实 MuJoCo 短程执行。
运行：mujoco 环境 python validation/adaptive_pole_stage1.py
仅使用无界面 MuJoCo；关闭可视化网络，不访问串口或改变运行中的 EXE。
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from Adaptive_Pole_Climbing import HelixBackbone


def helix_from_pole_parameters(amplitude_deg, spatial_phase, link_mm):
    """仅用于旧 Pole 数值回归，不参与交互或主控制流程。

    对常曲率螺旋，2l 区间积分的幅值 B=2l*κ*sinc(v)，τ=v/l。
    因此 κ=|a/57.2958|/(2l*sinc(v))。负 a 用 π 相位差表示。
    当 κ>0 时对应 r=κ/(κ²+τ²)，b=τ/(κ²+τ²)。
    名义等间距近似与本地其他 Yamada 步态一致；实际模型不完全等长。
    限定 |v|<π，避开 sinc 零点导致的不可逆映射。
    """
    if not all(math.isfinite(x) for x in (amplitude_deg, spatial_phase, link_mm)):
        raise ValueError("a、v 和关节间距必须有限")
    if link_mm <= 0 or abs(spatial_phase) >= math.pi:
        raise ValueError("要求 link_mm>0 且 |v|<π")
    sinc = 1.0 if spatial_phase == 0 else math.sin(spatial_phase) / spatial_phase
    kappa = abs(amplitude_deg / 57.2958) / (2 * link_mm * sinc)
    tau = spatial_phase / link_mm
    offset = -spatial_phase + (math.pi if amplitude_deg >= 0 else 0.0)
    return HelixBackbone(kappa, tau, offset)


def numerical_checks():
    maximum = 0.0
    cases = 0
    # 标准螺旋 Yamada 区间积分 vs 原 Pole 直接公式：正反幅值/挠率/滚动。
    for count in (20, 28):
        for amplitude in (0, 30, -76):
            for spatial in (0, 1e-10, 0.05, -0.12, 0.8):
                for phase in (0, 0.7, -1.2, 2 * math.pi):
                    curve = helix_from_pole_parameters(amplitude, spatial, 68.5)
                    actual = curve.joint_angles(count, 68.5, phase)
                    idx = np.arange(count)
                    expected = amplitude / 57.2958 * np.sin(phase + spatial * idx + (idx % 2) * math.pi / 2)
                    maximum = max(maximum, float(np.max(np.abs(actual - expected))))
                    assert np.allclose(actual, expected, atol=2e-12, rtol=0)
                    cases += 1
    # 直接几何输入，不依赖 a/v 转换；以密集梯形积分独立核对 quad 和奇偶方向。
    curve = HelixBackbone.from_radius_pitch(150, 180, 0.3)
    assert curve.tau > 0
    assert math.isclose(curve.kappa / (curve.kappa**2 + curve.tau**2), 150)
    assert math.isclose(2 * math.pi * curve.tau / (curve.kappa**2 + curve.tau**2), 180)
    angles = curve.joint_angles(28, 68.5, 0.42)
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(28)])
    dense_error = 0.0
    for i in range(28):
        s = np.linspace(i * 68.5, (i + 2) * 68.5, 10001)
        psi = curve.tau * s + 0.42 + 0.3
        component = -curve.kappa * np.sin(psi) if i % 2 == 0 else curve.kappa * np.cos(psi)
        ds = s[1] - s[0]
        expected_paper = ds * (component[0] / 2 + component[1:-1].sum() + component[-1] / 2)
        dense_error = max(dense_error, abs(angles[i] * signs[i] - expected_paper))
    assert dense_error < 1e-9
    assert np.allclose(curve.joint_angles(28, 68.5, 0.42 + 2 * math.pi), angles, atol=2e-12)
    assert np.allclose(curve.joint_angles(28, 68.5, 0.42 + math.pi), -angles, atol=2e-12)
    straight = HelixBackbone(0, 0)
    assert np.count_nonzero(straight.joint_angles(20, 68.5, 2)) == 0
    for a, v, length in ((float('nan'), 0, 68), (30, math.pi, 68), (30, 0, 0)):
        try:
            helix_from_pole_parameters(a, v, length)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    return {"comparison_cases": cases, "pole_target_max_error_rad": maximum,
            "dense_integral_max_error_rad": dense_error}


def smoke(out, label, model=None, radius=260, pitch=0):
    env = os.environ.copy()
    path = out / (label + '.npz')
    env.update(SNAKE_MUJOCO_HEADLESS='1', SNAKE_VISUALIZER_ENABLED='0',
               SNAKE_CONTROL_VIA_VISUALIZER='0', SNAKE_SINGLE_STEP='0',
               SNAKE_MUJOCO_MAX_SECONDS='3', SNAKE_MUJOCO_TRACE=str(path),
               SNAKE_HELIX_RADIUS_MM=str(radius), SNAKE_HELIX_PITCH_MM=str(pitch))
    if model:
        env['SNAKE_MUJOCO_MODEL'] = str(model)
    else:
        env.pop('SNAKE_MUJOCO_MODEL', None)
    result = subprocess.run([sys.executable, str(ROOT / 'codes/Adaptive_Pole_Climbing.py')],
                            env=env, capture_output=True, text=True, errors='replace', timeout=45)
    (out / (label + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    data = np.load(path)
    assert len(data['ctrl']) >= 20
    assert np.all(np.isfinite(data['qpos'])) and np.all(np.isfinite(data['ctrl']))
    assert np.all(np.diff(data['sim_time']) >= 0)
    assert 'WARNING' not in result.stdout + result.stderr
    count = data['ctrl'].shape[1]
    # 独立解析积分，对照实际非等长连杆及有向 Frenet 挠率。
    bounds = np.r_[0., np.cumsum(data['link_lengths_mm'])]
    span, middle = bounds[2:] - bounds[:-2], (bounds[2:] + bounds[:-2]) / 2
    b = pitch / (2 * math.pi)
    kappa, tau = radius / (radius**2 + b**2), b / (radius**2 + b**2)
    amplitude = kappa * span * np.sinc(tau * span / (2 * math.pi))
    psi = data['roll_phase'][:, None] + tau * middle
    expected = -amplitude * np.where(np.arange(count) % 2, np.cos(psi), np.sin(psi))
    error = float(np.max(np.abs(data['ctrl'] - expected)))
    assert error < 2e-12, error
    # 以同样调度直接写原 Pole 公式；比较真实物理轨迹以排查循环中的角度/顺序错误。
    import mujoco
    resolved_model = model or ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml'
    m = mujoco.MjModel.from_xml_path(str(resolved_model))
    d = mujoco.MjData(m)
    qpos_error = 0.0
    for target, recorded, moving in zip(expected, data['qpos'], data['moving']):
        d.ctrl[:] = target
        if moving:
            for _ in range(10):
                mujoco.mj_step(m, d)
        qpos_error = max(qpos_error, float(np.max(np.abs(d.qpos - recorded))))
    assert qpos_error < 1e-7, qpos_error
    return {"joints": count, "samples": len(data['ctrl']), "sim_seconds": float(data['sim_time'][-1]),
            "nominal_link_mm": float(data['nominal_link_mm']),
            "target_max_error_rad": error, "same_schedule_qpos_max_error": qpos_error}


def main():
    out = ROOT / 'validation/results/adaptive-pole-stage1'
    out.mkdir(parents=True, exist_ok=True)
    pole = ROOT / 'codes/Pole_Climbing.py'
    before = hashlib.sha256(pole.read_bytes()).hexdigest()
    result = {"numerical": numerical_checks(), "runs": {}}
    result['runs']['20_planar'] = smoke(out, '20_planar')
    result['runs']['20_helix'] = smoke(out, '20_helix', radius=110, pitch=180)
    model28 = ROOT / 'validation/fixtures/pole28.xml'
    assert model28.is_file(), f'28-joint verification model missing: {model28}'
    result['runs']['28_helix'] = smoke(out, '28_helix', model28, 110, 180)
    assert hashlib.sha256(pole.read_bytes()).hexdigest() == before
    result['original_pole_sha256'] = before
    result['scope'] = 'Numerical and headless MuJoCo only; no stable climbing, interactive viewer or hardware acceptance.'
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
