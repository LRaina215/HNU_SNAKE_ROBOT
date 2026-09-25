"""验证指定发布目录的 EXE 和客户端；隔离端口，不访问真实串口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from adaptive_pole_stage4 import ROOT, OUT, verify_trace
from python_client import SnakeControlClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    exe = release / "SnakeRobotVisualizer.exe"
    if not exe.is_file() or not (release / "python_client.py").is_file():
        parser.error("发布目录必须含 SnakeRobotVisualizer.exe 和 python_client.py")
    port = 18789
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(port), SNAKE_MUJOCO_HEADLESS="1", SNAKE_SINGLE_STEP="0",
               SNAKE_CONTROL_VIA_VISUALIZER="1", SNAKE_VISUALIZER_ENABLED="1",
               SNAKE_GEOMETRY_EXCHANGE="1", SNAKE_MUJOCO_MAX_SECONDS="4",
               SNAKE_HELIX_RADIUS_MM="110", SNAKE_HELIX_PITCH_MM="180",
               SNAKE_MUJOCO_TRACE=str(OUT / "release.npz"),
               SNAKE_CONTROL_CLIENT_DIR=str(release), PYTHONIOENCODING="utf-8")
    env.pop("SNAKE_MUJOCO_MODEL", None)
    app = subprocess.Popen([str(exe)], cwd=release, env=env)
    client = SnakeControlClient(port=port)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                assert client.get_state()["mode"] == "Idle"
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)
        run = subprocess.run([sys.executable, str(ROOT / "validation/adaptive_pole_stage4.py"), "--run"],
                             env=env, capture_output=True, encoding="utf-8", timeout=30)
        (OUT / "release.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        assert run.returncode == 0, run.stdout + run.stderr
        result = verify_trace("release")
        assert not client.get_state()["torque_active"]
        result["exe_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
        (OUT / "release-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        client.close()
        app.terminate()
        app.wait(timeout=5)


if __name__ == "__main__":
    main()
