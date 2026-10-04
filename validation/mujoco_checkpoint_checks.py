"""Checkpoint round trip and paused restore through an isolated EXE relay.

No serial access, global keyboard events, or climbing-success claims.
"""
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]
OUT = ROOT / 'validation/results/mujoco-checkpoint'


def run_child():
    import Adaptive_Pole_Climbing_Mujoco as gait
    import mujoco_checkpoint
    gait.DEBUG = False
    mujoco_checkpoint.DEFAULT_DIRECTORY = OUT / 'captures'
    threading.Thread(target=gait.uart_listener, daemon=True).start()
    gait.main()


def state(folder):
    with np.load(folder / 'state.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    metadata = json.loads((folder / 'checkpoint.json').read_text(encoding='utf-8'))
    return arrays, metadata


def main():
    from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
    from mujoco_checkpoint import load_checkpoint
    from python_client import SnakeControlClient
    OUT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    appdir = Path(tempfile.mkdtemp(prefix='checkpoint-exe-'))
    exe = appdir / 'SnakeRobotVisualizer.exe'
    shutil.copy2(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', exe)
    shutil.copy2(ROOT / 'validation/fixtures/app-state.json', appdir / 'app-state.json')
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(port), SNAKE_MUJOCO_HEADLESS='1', SNAKE_SINGLE_STEP='1',
               SNAKE_VISUALIZER_ENABLED='1', SNAKE_GEOMETRY_EXCHANGE='1', SNAKE_CONTROL_VIA_VISUALIZER='1',
               SNAKE_MUJOCO_MODEL=str(ROOT / 'mujoco_model/Snake_Robot_o20_Pipe_tee_vertical_down.xml'),
               PYTHONIOENCODING='utf-8')
    for name in ('SNAKE_MUJOCO_RESTORE', 'SNAKE_TRIAL_SEED', 'SNAKE_MUJOCO_MAX_SECONDS',
                 'SNAKE_MUJOCO_TRACE', 'SNAKE_HELIX_RADIUS_MM', 'SNAKE_HELIX_PITCH_MM'):
        env.pop(name, None)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    app = subprocess.Popen([str(exe)], cwd=appdir, env=env, startupinfo=startup)
    child = log = None
    client = SnakeControlClient(port=port)

    def wait(predicate, logpath=None):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if predicate():
                return
            if child is not None and child.poll() is not None:
                raise AssertionError(logpath.read_text(encoding='utf-8'))
            time.sleep(.05)
        raise AssertionError(logpath.read_text(encoding='utf-8')[-5000:] if logpath else 'EXE timeout')

    def ready():
        try:
            client.get_state()
            return True
        except OSError:
            return False

    def start(label, restore=None):
        nonlocal child, log
        runenv = env.copy()
        if restore:
            runenv['SNAKE_MUJOCO_RESTORE'] = str(restore)
        path = OUT / (label + '.log')
        log = path.open('w', encoding='utf-8')
        child = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--child'],
                                 cwd=ROOT, env=runenv, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        wait(lambda: 'MUJOCO_RELAY 已连接' in path.read_text(encoding='utf-8'), path)
        time.sleep(.3)
        return path

    def command(path, text, marker):
        prior = path.read_text(encoding='utf-8').count(marker)
        child.stdin.write(text + '\n')
        child.stdin.flush()
        wait(lambda: path.read_text(encoding='utf-8').count(marker) > prior, path)

    def capture(path):
        command(path, 'save', 'CHECKPOINT 已保存并暂停：')
        return Path(re.findall(r'CHECKPOINT 已保存并暂停：([^\r\n]+)', path.read_text(encoding='utf-8'))[-1])

    def stop():
        nonlocal child, log
        child.stdin.write('quit\n'); child.stdin.flush(); child.stdin.close()
        child.wait(timeout=8)
        assert child.returncode == 0
        child = None
        log.close(); log = None

    try:
        wait(ready)
        path = start('original')
        command(path, 'apply', '完成第')
        command(path, 'back', '完成第')
        saved = capture(path)
        a, meta = state(saved)
        assert meta['controller']['started'] and not meta['settings']['ignore_feedback_risk']
        assert meta['settings']['phase'] == -.15
        time.sleep(.4)
        idle = capture(path)
        assert np.array_equal(a['integration_state'], state(idle)[0]['integration_state'])
        command(path, 'hold', '完成第')
        continued = capture(path)
        stop()

        path = start('restored', saved)
        restored = capture(path)
        b, bmeta = state(restored)
        assert np.array_equal(a['integration_state'], b['integration_state'])
        assert meta['settings'] == bmeta['settings']
        assert meta['controller'] == bmeta['controller']
        command(path, 'hold', '完成第')
        resumed = capture(path)
        assert np.array_equal(state(continued)[0]['integration_state'], state(resumed)[0]['integration_state'])
        command(path, 'force', '已开启：')
        forced = capture(path)
        assert state(forced)[1]['settings']['ignore_feedback_risk']
        command(path, 'back', '完成第')
        next_step = capture(path)
        nmeta = state(next_step)[1]
        assert nmeta['settings']['phase'] == -.3
        assert nmeta['controller']['phase_offset'] == meta['controller']['phase_offset']
        stop()

        # Reconnect the simulation client and verify the saved force setting.
        path = start('force-restored', forced)
        forced_again = capture(path)
        assert state(forced_again)[1]['settings']['ignore_feedback_risk']
        stop()
        m, d, _, _ = load_checkpoint(saved)
        m2, d2, _, _ = load_checkpoint(saved)
        target = HelixBackbone.from_radius_pitch(81.5, 220).joint_angles(20, 64.5, -.3)
        for data in (d, d2):
            data.ctrl[:] = target
        for _ in range(200):
            mujoco.mj_step(m, d); mujoco.mj_step(m2, d2)
        assert np.array_equal(d.qpos, d2.qpos) and np.array_equal(d.qvel, d2.qvel)
        report = dict(mujoco=mujoco.__version__, saved=str(saved), physics_state_exact=True,
                      idle_frozen=True, relay_hold_continuation_exact=True,
                      next_phase=-.3, preserved_controller_phase_offset=True, restored_force=True)
        (OUT / f'summary-{mujoco.__version__}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        client.close()
        if child is not None and child.poll() is None:
            child.terminate(); child.wait(timeout=5)
        if log is not None:
            log.close()
        app.terminate(); app.wait(timeout=5)


if __name__ == '__main__':
    run_child() if '--child' in sys.argv else main()
