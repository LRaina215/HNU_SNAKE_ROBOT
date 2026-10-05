"""单步调度验收：隔离 EXE，真实 MuJoCo；只向测试子进程发送命令，不发送全局按键。"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/results/adaptive-pole-single-step'
PORT = 18791
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]


def run_gait():
    import Adaptive_Pole_Climbing_Mujoco as gait
    gait.DEBUG = False
    gait.COMPLIANCE_ENABLED = '--nominal' not in sys.argv
    gain = os.environ.get('SNAKE_COMPLIANCE_GAIN')  # 标量，或 "端部:中部"
    if gain:
        gait.COMPLIANCE_GAIN = tuple(map(float, gain.split(':'))) if ':' in gain else float(gain)
    # 无 viewer 的测试进程也用正式控制台命令入口；不伪造算法或网络。
    if os.environ.get('SNAKE_MUJOCO_HEADLESS') == '1':
        threading.Thread(target=gait.uart_listener, daemon=True).start()
    gait.main()


def math_checks():
    import copy
    from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
    from snake_control.pole_compliance import PiecewiseBackbone, TargetSegment, ComplianceController, ComplianceParameters, build_backbone
    # 同一常曲率/挠率输入，在非等长连杆上，两种 Yamada 实现应一致。
    links = np.array([35., 67.5, 69.5, 68.5, 34.])
    maximum = 0.
    for pitch in (-180., 0., 180.):
        helix = HelixBackbone.from_radius_pitch(110., pitch)
        segment = TargetSegment(None, sum(links), 110., 1., helix.kappa, helix.tau)
        curve = PiecewiseBackbone([segment], links, 0., [0., 0.], [1., 0.], {}, 0.)
        for phase in (-.15, 0., .15, 2 * math.pi):
            maximum = max(maximum, float(np.max(np.abs(
                curve.joint_angles(phase) - helix.joint_angles(4, links, phase)))))
    assert maximum < 1e-12
    geometry = json.loads((ROOT / 'validation/fixtures/geometry_example.json').read_text(encoding='utf-8'))
    geometry.update(age_ms=0, sample_unix_ms=1000, feedback_usable=True)
    params = ComplianceParameters()
    curve = build_backbone(geometry, params)
    reference = curve.joint_angles(.8)
    controller = ComplianceController(params, curve.links)
    target = controller.compute(copy.deepcopy(geometry), .45, reference, now_ms=1000, reference_phase=.3)
    assert np.allclose(target, curve.joint_angles(.95), atol=1e-12)
    assert not np.allclose(target, reference)
    return dict(nominal_piecewise_max_error_rad=maximum, first_adaptive_phase_increment_rad=.15)


def verify(path, executed=True):
    import mujoco
    data = np.load(path)
    records = json.loads(Path(str(path) + '.received.json').read_text(encoding='utf-8'))
    packets = {(x['run_id'], x['delivery_id']): x for x in records}
    count = data['ctrl'].shape[1]
    signs = np.where(np.arange(count) % 2, -1, 1)
    for ctrl, run, delivery in zip(data['ctrl'], data['applied_run_id'], data['applied_delivery_id']):
        if delivery:
            expected = np.deg2rad(np.array(packets[(run, int(delivery))]['targets_deg']) * signs)
            assert np.array_equal(ctrl, expected)
        else:
            assert not np.any(ctrl)
    # idle 的每次循环均不能推进任何物理量或相位，迟到网络目标也不得修改 ctrl。
    idle = ~data['moving'][1:]
    for key in ('qpos', 'ctrl', 'sim_time', 'roll_phase'):
        assert np.all(np.diff(data[key], axis=0)[idle] == 0), key
    assert np.count_nonzero(idle) > 80
    total = int(data['completed_steps'][-1])
    if executed:
        assert total == 5, total
        ends = np.flatnonzero(np.diff(np.r_[0, data['completed_steps']]) == 1)
        assert np.allclose(data['roll_phase'][ends], [0., .15, 0., 0., .15])
        moving = data['moving']
        assert np.count_nonzero(moving) == 5 * count
        # 每一步 N 组，每组 10 个 MuJoCo 物理步。
        positive = np.diff(np.r_[0., data['sim_time']])
        assert np.allclose(positive[positive > 0], positive.max())
        assert np.allclose(np.diff(np.r_[0., data['sim_time'][ends]]), count * positive.max())
        assert not np.allclose(data['ctrl'][ends[0]], data['ctrl'][ends[3]])  # R/P 应用确实生效
        assert np.allclose(data['kappa'][ends[3]], 130 / (130**2 + (180 / (2 * math.pi))**2))
    else:
        assert total == 0 and not np.any(data['sim_time']) and not records
    return dict(joints=count, completed_steps=total, idle_samples=int(idle.sum()),
                adaptive_samples=int(data['compliance_active'].sum()), final_sim_seconds=float(data['sim_time'][-1]))


def main():
    from python_client import SnakeControlClient
    OUT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    appdir = OUT / 'app'
    appdir.mkdir(exist_ok=True)
    exe = appdir / 'SnakeRobotVisualizer.exe'
    shutil.copy2(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', exe)
    shutil.copy2(ROOT / 'validation/fixtures/app-state.json', appdir / 'app-state.json')
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(PORT), SNAKE_MUJOCO_HEADLESS='1', SNAKE_SINGLE_STEP='1',
               SNAKE_VISUALIZER_ENABLED='1', SNAKE_GEOMETRY_EXCHANGE='1', SNAKE_CONTROL_VIA_VISUALIZER='1',
               SNAKE_HELIX_RADIUS_MM='110', SNAKE_HELIX_PITCH_MM='180', PYTHONIOENCODING='utf-8')
    env.pop('SNAKE_MUJOCO_MODEL', None)
    app = child = None
    client = SnakeControlClient(port=PORT)
    report = {'math': math_checks()}

    def run(label, nominal=False, offline=False, model=None, viewer=False):
        nonlocal child
        trace = OUT / (label + '.npz')
        env.update(SNAKE_MUJOCO_MAX_SECONDS='4' if offline else '18', SNAKE_MUJOCO_TRACE=str(trace),
                   SNAKE_MUJOCO_HEADLESS='0' if viewer else '1')
        if model:
            env['SNAKE_MUJOCO_MODEL'] = str(model)
        else:
            env.pop('SNAKE_MUJOCO_MODEL', None)
        with (OUT / (label + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__)), '--run'] +
                                     (['--nominal'] if nominal else []), env=env, stdin=subprocess.PIPE,
                                     stdout=log, stderr=log, text=True)
            time.sleep(1.5)
            def send(text):
                assert child.poll() is None, (OUT / (label + '.log')).read_text(encoding='utf-8')
                child.stdin.write(text + '\n')
                child.stdin.flush()
            if offline:
                send('apply')
            else:
                # 没有按键时通信仍有新帧，证明暂停没有堵塞主循环。
                deadline = time.monotonic() + 8
                while True:
                    try:
                        geometry = client.get_geometry()
                        if time.time() * 1000 - geometry['sample_unix_ms'] < 500:
                            before = geometry['sequence']
                            break
                        if time.monotonic() > deadline:
                            raise AssertionError('没有当前测试进程的新鲜反馈：' + (OUT / (label + '.log')).read_text(encoding='utf-8'))
                        time.sleep(.1)
                    except (OSError, RuntimeError):
                        assert time.monotonic() < deadline
                        time.sleep(.1)
                time.sleep(.4)
                after = client.get_geometry()['sequence']
                assert after > before
                for command in ('apply', 'step', 'back', 'r=130 p=180\napply', 'step'):
                    send(command)
                    time.sleep(1.5)
                    (OUT / (label + '-' + command.split()[0].replace('=', '') + '.json')).write_text(
                        json.dumps(client.get_geometry(), ensure_ascii=False, indent=2), encoding='utf-8')
            child.stdin.close()  # EOF 让测试输入线程退出，避免解释器关闭时仍锁住管道。
            child.wait(timeout=20)
            assert child.returncode == 0, (OUT / (label + '.log')).read_text(encoding='utf-8')
        report[label] = verify(trace, not offline)
        child = None

    try:
        run('no_exe', offline=True)
        app = subprocess.Popen([str(exe)], cwd=appdir, env=env)
        deadline = time.monotonic() + 10
        while True:
            try:
                client.get_state()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        if '--viewer-only' not in sys.argv:
            run('nominal20', nominal=True)
            run('adaptive20')
            run('adaptive28', model=ROOT / 'validation/fixtures/pole28.xml')
            assert report['adaptive28']['adaptive_samples'] == 84  # 正向、反向及 R/P 重设后的三步。
        if '--viewer' in sys.argv or '--viewer-only' in sys.argv:
            run('viewer20', viewer=True)
        assert not client.get_state()['torque_active']
        (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        client.close()
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        if app is not None:
            app.terminate()
            app.wait(timeout=5)


if __name__ == '__main__':
    run_gait() if '--run' in sys.argv else main()
