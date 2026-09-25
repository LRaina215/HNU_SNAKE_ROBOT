"""完整 EXE 转发链验证；Windows 原生 MuJoCo，独立端口，无真实串口。"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "visualizer"
OUT = ROOT / "validation/results/adaptive-pole-stage4"
PORT = 18787
sys.path[:0] = [str(ROOT / "codes"), str(VIS)]


def run_gait():
    import Adaptive_Pole_Climbing as gait
    gait.DEBUG = False
    gait.CONTROL_VIA_VISUALIZER = True
    gait.main()


def verify_trace(label, require_relay=True):
    data = np.load(OUT / (label + ".npz"))
    received = json.loads((OUT / (label + ".npz.received.json")).read_text(encoding="utf-8"))
    records = {(c["run_id"], c["delivery_id"]): c for c in received}
    count = data["ctrl"].shape[1]
    signs = np.where(np.arange(count) % 2, -1, 1)
    keys = list(zip(data["applied_run_id"], data["applied_delivery_id"]))
    for actual, key in zip(data["ctrl"], keys):
        if key[1] == 0:
            assert np.count_nonzero(actual) == 0, "local algorithm bypassed EXE"
        else:
            packet = records[key]
            expected = np.deg2rad(np.array(packet["targets_deg"]) * signs)
            assert np.array_equal(actual, expected), "d.ctrl did not come from returned EXE frame"
    assert np.all(np.isfinite(data["qpos"])) and np.all(np.isfinite(data["ctrl"]))
    assert np.max(np.abs(data["requested_ctrl"])) > .1
    applied = {key for key in keys if key[1] != 0}
    if require_relay:
        assert len(applied) > 5
        assert np.max(np.abs(data["qpos"][:, -count:] - data["ctrl"])) > .001
        assert np.count_nonzero(data["compliance_active"]) > 10
    else:
        assert len(applied) == 0 and not received
        assert np.count_nonzero(data["ctrl"]) == 0
    return dict(joints=count, samples=len(keys), received=len(received), applied=len(applied),
                adaptive_samples=int(np.count_nonzero(data["compliance_active"])),
                run_ids=sorted({key[0] for key in applied}))


def verify_receiver_guards():
    import threading
    import mujoco
    from mujoco_execution_bridge import MujocoExecutionBridge
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco_model/Snake_Robot_o20_Pole_Climbing.xml"))
    data = mujoco.MjData(model)
    bridge = MujocoExecutionBridge.__new__(MujocoExecutionBridge)
    bridge.model, bridge.joint_count = model, model.nu
    bridge.signs = np.where(np.arange(model.nu) % 2, -1, 1)
    bridge._command_lock = threading.Lock()
    bridge._ready, bridge._run_id, bridge._response_at = True, "run", time.monotonic()
    bridge.applied_run_id, bridge.applied_delivery_id = "", 0
    bridge.applied = bridge.expired = bridge.rejected = 0
    command = dict(run_id="run", delivery_id=1, command_id=1,
                   targets_deg=[10.] * model.nu, expires_unix_ms=time.time() * 1000 - 1)
    bridge._command = dict(command)
    assert not bridge.apply_received(data) and np.count_nonzero(data.ctrl) == 0
    command["expires_unix_ms"] = time.time() * 1000 + 500
    bridge._command = dict(command)
    assert bridge.apply_received(data)
    last = data.ctrl.copy()
    bridge._command = dict(command)
    assert not bridge.apply_received(data)  # 同一投递不得重复应用。
    bridge._command = dict(command, delivery_id=2, run_id="old-run")
    assert not bridge.apply_received(data)
    bridge._command = dict(command, delivery_id=2, targets_deg=[999.] * model.nu)
    assert not bridge.apply_received(data) and np.array_equal(last, data.ctrl)
    bridge._command = dict(command, delivery_id=2)
    assert not bridge.apply_received(data, expected_targets=np.zeros(model.nu))
    assert np.array_equal(last, data.ctrl)  # 迟到的上一目标不能被当作本次单步回传。
    bridge._command = dict(command, delivery_id=2)
    assert bridge.apply_received(data, expected_targets=last)


def main():
    from python_client import SnakeControlClient
    OUT.mkdir(parents=True, exist_ok=True)
    # 确认本测试端口空闲；不会终止用户正在运行的 EXE。
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    verify_receiver_guards()
    appdir = OUT / "app"
    appdir.mkdir(exist_ok=True)
    exe = appdir / "SnakeRobotVisualizer.exe"
    source = Path(os.environ.get("SNAKE_STAGE4_EXE", str(VIS / "windows_app/bin/SnakeRobotVisualizer.exe")))
    shutil.copy2(source, exe)
    state = json.loads((ROOT / "validation/fixtures/app-state.json").read_text(encoding="utf-8"))
    state["Display"]["DisplayMode"] = 2
    (appdir / "app-state.json").write_text(json.dumps(state), encoding="utf-8")
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(PORT), SNAKE_MUJOCO_HEADLESS="1", SNAKE_SINGLE_STEP="0", SNAKE_VISUALIZER_ENABLED="1",
               SNAKE_GEOMETRY_EXCHANGE="1", SNAKE_CONTROL_VIA_VISUALIZER="1", PYTHONIOENCODING="utf-8",
               SNAKE_HELIX_RADIUS_MM="110", SNAKE_HELIX_PITCH_MM="180")
    env.pop("SNAKE_MUJOCO_MODEL", None)
    app, gait = None, None
    client = SnakeControlClient(port=PORT)
    report = {}

    def launch_app():
        nonlocal app
        app = subprocess.Popen([str(exe)], cwd=appdir, env=env)
        deadline = time.monotonic() + 10
        while True:
            try:
                assert client.get_state()["mode"] == "Idle"
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)

    def stop_app():
        nonlocal app
        client.close()
        if app is not None:
            app.terminate()
            app.wait(timeout=5)
            app = None

    def run(label, seconds):
        env.update(SNAKE_MUJOCO_MAX_SECONDS=str(seconds), SNAKE_MUJOCO_TRACE=str(OUT / (label + ".npz")))
        result = subprocess.run([sys.executable, str(Path(__file__)), "--run"], env=env,
                                capture_output=True, encoding="utf-8", timeout=45)
        (OUT / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARNING" not in result.stdout + result.stderr

    try:
        run("no_exe", 2)
        report["no_exe"] = verify_trace("no_exe", False)
        launch_app()
        run("relay20", 8)
        report["relay20"] = verify_trace("relay20")
        status = client.get_geometry()["target_state"]
        assert status["mujoco_applied_delivery_id"] > 5
        assert not client.get_state()["torque_active"]
        report["acknowledged20"] = status["mujoco_applied_delivery_id"]
        stop_app()
        launch_app()
        env["SNAKE_MUJOCO_MODEL"] = str(ROOT / "validation/fixtures/pole28.xml")
        run("relay28", 8)
        report["relay28"] = verify_trace("relay28")
        stop_app()
        launch_app()
        env.pop("SNAKE_MUJOCO_MODEL")
        env.update(SNAKE_MUJOCO_MAX_SECONDS="12", SNAKE_MUJOCO_TRACE=str(OUT / "restart.npz"))
        with (OUT / "restart.log").open("w", encoding="utf-8") as log:
            gait = subprocess.Popen([sys.executable, str(Path(__file__)), "--run"], env=env, stdout=log, stderr=log)
            deadline = time.monotonic() + 10
            while True:
                try:
                    if client.get_geometry()["target_state"]["mujoco_applied_delivery_id"] >= 5:
                        break
                except (OSError, RuntimeError):
                    pass
                assert time.monotonic() < deadline and gait.poll() is None
                time.sleep(.05)
            stop_app()
            offline_start = time.time() * 1000
            time.sleep(1.5)
            offline_end = time.time() * 1000
            launch_app()
            assert gait.wait(timeout=30) == 0
        report["restart"] = verify_trace("restart")
        assert len(report["restart"]["run_ids"]) == 2
        data = np.load(OUT / "restart.npz")
        mask = (data["wall_unix_ms"] > offline_start + 600) & (data["wall_unix_ms"] < offline_end)
        assert mask.sum() > 10 and np.all(data["ctrl"][mask] == data["ctrl"][mask][0])
        report["restart"]["held_samples_during_outage"] = int(mask.sum())
        assert not client.get_state()["torque_active"]
        report["scope"] = "No real servo bus; exact EXE packet to d.ctrl mapping; reconnect/hold; short MuJoCo run, not climbing acceptance"
        (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if gait is not None and gait.poll() is None:
            gait.terminate()
            gait.wait(timeout=5)
        stop_app()


if __name__ == "__main__":
    run_gait() if "--run" in sys.argv else main()
