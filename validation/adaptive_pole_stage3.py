"""顺应性数学核验、录制截面重建、隔离 EXE/MuJoCo 联调。无真实串口。"""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "visualizer"
OUT = ROOT / "validation/results/adaptive-pole-stage3"
sys.path[:0] = [str(ROOT / "codes"), str(VIS)]
from pole_compliance import ComplianceParameters, ComplianceController, build_backbone


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid parameters accepted")


def numerical():
    geometry = json.loads((ROOT / "validation/fixtures/geometry_example.json").read_text(encoding="utf-8"))
    geometry.update(age_ms=0, sample_unix_ms=1000, feedback_usable=True)
    count = geometry["config"]["joint_count"]
    parameters = ComplianceParameters(gain=[.04 if j < 4 or j >= count - 4 else .08 for j in range(count)])
    curve = build_backbone(geometry, parameters)
    neutral = build_backbone(geometry, ComplianceParameters(gain=0))
    inside = build_backbone(geometry, ComplianceParameters(gain=.04, inside=True))
    groups = {}
    for old, zero, new, inner in zip(geometry["curve"]["segments"], neutral.segments, curve.segments, inside.segments):
        groups.setdefault(new.joint_index, []).append(new)
        if old["kind"] == "arc":
            gain = parameters.gain[new.joint_index - 1]
            assert math.isclose(new.radius / zero.radius, 1 + gain)
            assert math.isclose(new.turn / zero.turn, 1 / (1 + gain))
            assert math.isclose(inner.radius / zero.radius, .96)
            assert math.isclose(inner.turn / zero.turn, 1 / .96)
            assert math.isclose(new.radius * abs(new.turn), zero.radius * abs(zero.turn))
        assert math.isclose(new.length, zero.length)
    for j in range(1, count + 1):
        assert math.isclose(sum(s.length for s in groups[j]), (curve.links[j-1] + curve.links[j]) / 2)
    assert math.isclose(curve.bounds[-1], sum(curve.links))
    assert math.isclose(curve.segments[0].length, curve.links[0] / 2)
    assert math.isclose(curve.segments[-1].length, curve.links[-1] / 2)
    # 独立 quad 数值积分核对解析积分，显式在 κ/τ 跳变处分区。
    phase = .73
    numerical_angles = []
    joint_s = np.r_[0., np.cumsum(curve.links)]
    for j in range(1, count + 1):
        value = 0.
        for k, segment in enumerate(curve.segments):
            lo, hi = max(joint_s[j - 1], curve.bounds[k]), min(joint_s[j + 1], curve.bounds[k + 1])
            if hi > lo:
                trig = math.sin if j % 2 else math.cos
                value += quad(lambda s: -segment.kappa * trig(phase + curve.twists[k] +
                              segment.tau * (s - curve.bounds[k])), lo, hi, epsabs=1e-12)[0]
        numerical_angles.append(value)
    angle_error = float(np.max(np.abs(curve.joint_angles(phase) - numerical_angles)))
    assert angle_error < 1e-11
    assert np.allclose(curve.joint_angles(phase + 2 * math.pi), numerical_angles, atol=1e-11)
    assert np.allclose(curve.joint_angles(phase + math.pi), -np.asarray(numerical_angles), atol=1e-11)
    flat = build_backbone(geometry, ComplianceParameters(gain=0, pitch_angle=0))
    assert all(segment.tau == 0 for segment in flat.segments)
    assert np.all(np.isfinite(flat.joint_angles(.3)))
    reverse_pitch = build_backbone(geometry, ComplianceParameters(gain=parameters.gain, pitch_angle=-.25))
    assert np.allclose([s.tau for s in reverse_pitch.segments], [-s.tau for s in curve.segments])
    reversed_winding = copy.deepcopy(geometry)
    for segment in reversed_winding["curve"]["segments"]:
        segment["turn_rad"] *= -1
    mirrored = build_backbone(reversed_winding, parameters)
    assert np.allclose([s.tau for s in mirrored.segments], [-s.tau for s in curve.segments])
    # 直接对三维曲线求导核对曲率和有向挠率，避开拼接点。
    sample_step = .03
    points = curve.sample(sample_step, world=False)
    offset, k_error, t_error = 1, 0., 0.
    for segment in curve.segments:
        steps = max(1, math.ceil(segment.length / sample_step))
        path = points[offset - 1:offset + steps]
        offset += steps
        if segment.radius == 0 or len(path) < 20:
            continue
        spacing = segment.length / steps
        first = np.gradient(path, spacing, axis=0, edge_order=2)
        second = np.gradient(first, spacing, axis=0, edge_order=2)
        third = np.gradient(second, spacing, axis=0, edge_order=2)
        index = len(path) // 2
        cross = np.cross(first[index], second[index])
        kappa = np.linalg.norm(cross) / np.linalg.norm(first[index]) ** 3
        tau = np.dot(cross, third[index]) / np.dot(cross, cross)
        k_error = max(k_error, abs(kappa - segment.kappa))
        t_error = max(t_error, abs(tau - segment.tau))
    assert k_error < 1e-6 and t_error < 1e-5, (k_error, t_error)
    world = curve.sample(3)
    local = curve.sample(3, world=False)
    assert np.all(np.isfinite(world))
    assert np.allclose(np.linalg.norm(np.diff(world, axis=0), axis=1), np.linalg.norm(np.diff(local, axis=0), axis=1))
    # 失效不改变上次有效骨架；同一截面不重复施加增益，但相位仍推进。
    controller = ComplianceController(parameters, curve.links)
    reference = np.zeros(count)
    targets = controller.compute(geometry, 0., reference, now_ms=1000)
    assert targets is not None, controller.status
    backbone = controller.backbone
    again = controller.compute(geometry, .2, targets, now_ms=1100)
    assert again is not None and controller.backbone is backbone and not np.allclose(again, targets)
    assert controller.compute(geometry, .2, targets, now_ms=1600) is None
    assert controller.backbone is backbone
    invalid = copy.deepcopy(geometry)
    invalid["feedback_usable"] = False
    assert controller.compute(invalid, .2, targets, now_ms=1000) is None
    invalid = copy.deepcopy(geometry)
    invalid["config"]["head_length_mm"] += 1
    assert controller.compute(invalid, .2, targets, now_ms=1000) is None
    invalid = copy.deepcopy(geometry)
    invalid["sequence"] -= 1
    assert controller.compute(invalid, .2, targets, now_ms=1000) is None
    invalid = copy.deepcopy(geometry)
    for joint in invalid["config"]["joints"]:
        joint.update(minimum_deg=0, maximum_deg=0)
    assert controller.compute(invalid, .2, targets, now_ms=1000) is None
    assert controller.backbone is backbone
    # 恢复反馈沿用滚动相位偏置，不重新初始化。
    phase_offset = controller.phase_offset
    fresh = copy.deepcopy(geometry)
    fresh["sequence"] += 1
    assert controller.compute(fresh, .3, targets, now_ms=1100) is not None
    assert controller.phase_offset == phase_offset
    # 回传预览必须引用实际参与计算的帧，不能偷换成网络线程更新后的帧。
    from python_client import GeometryExchange
    exchange = GeometryExchange.__new__(GeometryExchange)
    exchange._lock = threading.Lock()
    exchange._latest = copy.deepcopy(fresh)
    exchange._latest.update(mode="Mujoco", sample_unix_ms=time.time() * 1000)
    source = copy.deepcopy(exchange._latest)
    source["sequence"] -= 1
    exchange._received_at = time.monotonic()
    exchange.expected_lengths = curve.links
    exchange._pending = None
    exchange.dropped = 0
    exchange.execution = "preview"
    assert exchange.publish_targets([0.] * count, source_geometry=source)
    assert exchange._pending[0]["sequence"] == source["sequence"]
    source["sample_unix_ms"] -= 1000
    assert not exchange.publish_targets([0.] * count, source_geometry=source)
    exchange._pending = None
    reject(lambda: build_backbone(geometry, ComplianceParameters(gain=1., inside=True)))
    reject(lambda: build_backbone(geometry, ComplianceParameters(pitch_angle=math.pi / 2)))
    # 顺序反转、负弧长和非有限参数不可进入目标。
    invalid = copy.deepcopy(geometry)
    invalid["curve"]["segments"][1]["radius_mm"] = float("nan")
    reject(lambda: build_backbone(invalid, parameters))
    invalid = copy.deepcopy(geometry)
    invalid["config"]["joints"][0]["axis"] = "Pitch"
    reject(lambda: build_backbone(invalid, parameters))
    invalid = copy.deepcopy(geometry)
    invalid["projection_axis"]["horizontal"] = [0., 0., 0.]
    reject(lambda: build_backbone(invalid, parameters))
    np.savez(OUT / "reconstructed_curve.npz", points_mm=world,
             lengths_mm=[s.length for s in curve.segments], kappa=[s.kappa for s in curve.segments],
             tau=[s.tau for s in curve.segments], targets_rad=targets)
    from adaptive_pole_stage1 import numerical_checks
    baseline = numerical_checks()
    return dict(segments=len(curve.segments), joints=count, total_length_mm=curve.bounds[-1],
                stage1_regression=baseline,
                yamada_error_rad=angle_error, derivative_kappa_error=k_error, derivative_tau_error=t_error)


def run_gait():
    import Adaptive_Pole_Climbing as gait
    gait.DEBUG = False
    gait.CONTROL_VIA_VISUALIZER = False
    gait.COMPLIANCE_ENABLED = True
    gait.main()


def integration():
    from python_client import SnakeControlClient
    appdir = OUT / "app"
    appdir.mkdir(exist_ok=True)
    exe = appdir / "SnakeRobotVisualizer.exe"
    shutil.copy2(VIS / "windows_app/bin" / exe.name, exe)
    state = json.loads((ROOT / "validation/fixtures/app-state.json").read_text(encoding="utf-8"))
    state["Display"]["DisplayMode"] = 2
    (appdir / "app-state.json").write_text(json.dumps(state), encoding="utf-8")
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT="18785", SNAKE_MUJOCO_HEADLESS="1", SNAKE_VISUALIZER_ENABLED="1",
               SNAKE_GEOMETRY_EXCHANGE="1", SNAKE_MUJOCO_MAX_SECONDS="8", PYTHONIOENCODING="utf-8",
               SNAKE_MUJOCO_TRACE=str(OUT / "integration.npz"), SNAKE_MUJOCO_A="-76", SNAKE_MUJOCO_V=".05")
    env.pop("SNAKE_MUJOCO_MODEL", None)
    app = subprocess.Popen([str(exe)], cwd=appdir, env=env)
    client = SnakeControlClient(port=18785)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                assert client.get_state()["mode"] == "Idle"
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        run = subprocess.run([sys.executable, str(Path(__file__)), "--run"], env=env,
                             capture_output=True, encoding="utf-8", timeout=40)
        (OUT / "integration.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        assert run.returncode == 0, run.stdout + run.stderr
        traces = np.load(OUT / "integration.npz")
        assert np.all(np.isfinite(traces["ctrl"])) and np.all(np.isfinite(traces["qpos"]))
        active = traces["compliance_active"]
        assert np.count_nonzero(active) > 20, run.stdout
        assert np.max(np.abs(traces["ctrl"])) <= math.pi / 2
        assert np.ptp(traces["target_kappa"][active]) > 1e-5
        assert not client.get_state()["torque_active"]
        return dict(samples=len(active), adaptive_samples=int(np.count_nonzero(active)),
                    max_target_rad=float(np.max(np.abs(traces["ctrl"]))), sim_seconds=float(traces["sim_time"][-1]))
    finally:
        client.close()
        app.terminate()
        app.wait(timeout=5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(numerical=numerical(), integration=integration())
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_gait() if "--run" in sys.argv else main()
