"""Windows 原生调试窗口 + 隔离 EXE/MuJoCo 验证；不访问真实串口。"""
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VISUALIZER = ROOT.parents[1] / "two_joint_visualizer"
OUT = ROOT / "validation/results/geometry-debug"
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(VISUALIZER))


def run_gait(mode):
    import numpy as np
    import Adaptive_Pole_Climbing as gait
    from python_client import SnakeControlClient, ProtocolError
    startup_errors = []
    original_request = SnakeControlClient.request

    def request(self, payload):
        try:
            return original_request(self, payload)
        except ProtocolError as error:
            startup_errors.append(str(error))
            raise

    SnakeControlClient.request = request
    gait.DEBUG = mode != "off"
    gait.DEBUG_DRAW_CURVE = mode == "curve"
    timings, frames = [], []
    if gait.DEBUG:
        import geometry_debug_view as debug
        original_update = debug.GeometryDebugView.update
        original_offer = debug.GeometryDebugView._offer

        def update(self, exchange):
            start = time.perf_counter()
            original_update(self, exchange)
            timings.append((time.perf_counter() - start) * 1000)

        def offer(self, packet):
            if packet and packet.get("sequence") is not None:
                frames.append(packet["sequence"])
            original_offer(self, packet)

        debug.GeometryDebugView.update = update
        debug.GeometryDebugView._offer = offer
    gait.main()
    assert all(error.startswith("no_frame:") for error in startup_errors), startup_errors
    if not gait.DEBUG:
        assert "geometry_debug_view" not in sys.modules
    else:
        assert len(set(frames)) >= 5, frames
    print("DEBUG_VALIDATION " + json.dumps(dict(mode=mode, distinct_frames=len(set(frames)),
          startup_no_frame=len(startup_errors),
          update_p99_ms=float(np.percentile(timings, 99)) if timings else 0,
          update_max_ms=max(timings, default=0))))


def test_window():
    import tkinter as tk
    from geometry_debug_view import GeometryDebugWindow, segment_points
    for turn, endpoint in ((math.pi / 2, (10, 10)), (-math.pi / 2, (10, -10))):
        segment = dict(start_mm=[0, 0], tangent=[1, 0], kind="arc", radius_mm=10,
                       turn_rad=turn, length_mm=10 * abs(turn))
        assert math.dist(segment_points(segment)[-1], endpoint) < 1e-10
    geometry = json.loads((VISUALIZER / "validation/communication/results/geometry_example.json").read_text(encoding="utf-8"))
    for draw_curve in (True, False):
        root = tk.Tk()
        try:
            view = GeometryDebugWindow(root, queue.Queue(), draw_curve)
            root.update()
            packet = dict(sent_at=time.monotonic(), sequence=geometry["sequence"], age_ms=0,
                          valid=True, segments=geometry["curve"]["segments"])
            view.set_packet(packet)
            root.update()
            assert len(view.table.get_children()) == len(packet["segments"])
            assert "正在接收" in view.status.cget("text")
            iid = str(next(s["segment_index"] for s in packet["segments"] if s["kind"] == "arc"))
            view.table.selection_set(iid)
            root.update()
            if draw_curve:
                assert len(view.canvas.find_withtag("curve")) == len(packet["segments"])
                assert view.canvas.itemcget("segment-" + iid, "fill") == "#e62323"
            else:
                assert view.canvas is None and view.paths == []
            packet["sent_at"] -= 1
            view.update_status()
            assert "过期" in view.status.cget("text")
            packet.update(sent_at=time.monotonic(), valid=False)
            view.update_status()
            assert "曲线无效" in view.status.cget("text")
            view.set_packet({"sent_at": time.monotonic()})
            assert not view.table.get_children() and not view.paths
            assert "未收到" in view.status.cget("text")
        finally:
            root.destroy()


def main():
    import numpy as np
    from python_client import SnakeControlClient
    OUT.mkdir(parents=True, exist_ok=True)
    test_window()
    appdir = OUT / "app"
    appdir.mkdir(exist_ok=True)
    source = VISUALIZER / "windows_app/bin"
    exe = appdir / "SnakeRobotVisualizer.exe"
    shutil.copy2(source / exe.name, exe)
    state = json.loads((source / "app-state.json").read_text(encoding="utf-8"))
    state["Display"]["DisplayMode"] = 2
    (appdir / "app-state.json").write_text(json.dumps(state), encoding="utf-8")
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT="18783", SNAKE_MUJOCO_HEADLESS="1",
               SNAKE_MUJOCO_MAX_SECONDS="4", SNAKE_GEOMETRY_EXCHANGE="1",
               SNAKE_VISUALIZER_ENABLED="1", PYTHONIOENCODING="utf-8")
    for name in ("SNAKE_MUJOCO_MODEL", "SNAKE_MUJOCO_A", "SNAKE_MUJOCO_V"):
        env.pop(name, None)
    client = SnakeControlClient(port=18783)
    app = subprocess.Popen([str(exe)], cwd=appdir, env=env)
    report = {}
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                assert client.get_state()["mode"] == "Idle"
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        for mode in ("off", "table", "curve"):
            env["SNAKE_MUJOCO_TRACE"] = str(OUT / (mode + ".npz"))
            run = subprocess.run([sys.executable, str(Path(__file__)), "--run", mode], env=env,
                                 capture_output=True, encoding="utf-8", timeout=30)
            (OUT / (mode + ".log")).write_text(run.stdout + run.stderr, encoding="utf-8")
            assert run.returncode == 0, run.stdout + run.stderr
            assert "last_error=''" in run.stdout and "GEOMETRY_EXCHANGE received=" in run.stdout, run.stdout
            report[mode] = json.loads(next(line.split(" ", 1)[1] for line in run.stdout.splitlines()
                                           if line.startswith("DEBUG_VALIDATION ")))
            assert not client.get_state()["torque_active"]
        baseline = np.load(OUT / "off.npz")
        for mode in ("table", "curve"):
            actual = np.load(OUT / (mode + ".npz"))
            count = min(len(baseline["ctrl"]), len(actual["ctrl"]))
            assert count > 100
            for key in ("ctrl", "qpos"):
                difference = float(np.max(np.abs(baseline[key][:count] - actual[key][:count])))
                assert difference < 1e-12, (mode, key, difference)
                report[mode][key + "_max_difference"] = difference
        report["checks"] = "Tk drawing/table/selection/stale/invalid/disconnect; isolated EXE feedback; DEBUG off; trajectory equality"
        (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        client.close()
        app.terminate()
        app.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_gait(sys.argv[2])
    else:
        main()
