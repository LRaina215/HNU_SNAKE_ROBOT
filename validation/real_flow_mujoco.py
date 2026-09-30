"""实机版流程验证：Adaptive_Pole_Climbing_Real → EXE → MuJoCo 执行端（代替舵机）。

验证按键流程与目标生成，不验证真实舵机：缠绕（逐节、退回、单步上限）、Ctrl+U 不下发目标、
顺应性单步、Ctrl+E 切换到标准螺旋、Ctrl+R 回到缠绕。原 Pole 场景、零重力，机器人从直线姿态开始；
空中没有管壁，滚动时 K=0，只验证流程与相位对齐，不验证抱管效果。
运行：mujoco 环境 python validation/real_flow_mujoco.py
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/results/real-flow'
PORT = 18793
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]


def run_executor():
    """MuJoCo 执行端：持续物理步进，领取 EXE 转发的目标写入 ctrl，并上传实测帧。"""
    import mujoco
    from mujoco_execution_bridge import MujocoExecutionBridge
    m = mujoco.MjModel.from_xml_path(str(ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml'))
    # 只验证流程：关闭重力，让地面上的直线机器人能按目标弯成螺旋（原模型 kp=4 在重力下达不到）。
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bridge = MujocoExecutionBridge(m, d)
    applied = []
    stop = OUT / 'executor.stop'
    try:
        while not stop.exists():
            if bridge.apply_received(d):
                applied.append(d.ctrl.copy())
            for _ in range(10):
                mujoco.mj_step(m, d)
            bridge.publish(d)
            time.sleep(.005)
    finally:
        bridge.close()
        np.save(OUT / 'applied.npy', np.array(applied))


def run_controller():
    import Adaptive_Pole_Climbing_Real as real
    real.DEBUG = False
    real.STEP_SETTLE_S = .3
    # 空中没有管壁约束，K>0 会每步继续收紧；这里只验证流程，滚动时不做收紧。
    real.COMPLIANCE_GAIN = 0.
    # 下面的逐位比较与单步上限用例按 20 关节原模型的 R=110、P=180 设计，不随实机默认值变化。
    real.HELIX_RADIUS_MM, real.HELIX_PITCH_MM = 110., 180.
    real.main('mujoco', hotkeys=False, port=PORT)


def main():
    from python_client import SnakeControlClient
    from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
    from Adaptive_Pole_Climbing_Real import feedback_links, controller_angles, MAX_JOINT_STEP_DEG
    OUT.mkdir(parents=True, exist_ok=True)
    appdir = OUT / 'app'
    appdir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', appdir / 'SnakeRobotVisualizer.exe')
    shutil.copy2(ROOT / 'validation/fixtures/app-state.json', appdir / 'app-state.json')
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(PORT), PYTHONIOENCODING='utf-8')
    log = OUT / 'controller.log'
    (OUT / 'executor.stop').unlink(missing_ok=True)
    (OUT / 'applied.npy').unlink(missing_ok=True)
    app = executor = controller = None
    client = SnakeControlClient(port=PORT)
    try:
        app = subprocess.Popen([str(appdir / 'SnakeRobotVisualizer.exe')], cwd=appdir, env=env)
        time.sleep(1)
        executor = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--executor'], env=env)
        deadline = time.monotonic() + 15
        while True:
            try:
                geometry = client.get_geometry()
                if geometry.get('mode') == 'Mujoco' and geometry['target_state'].get('mujoco_executor_ready'):
                    break
            except (OSError, RuntimeError, KeyError):
                pass
            assert time.monotonic() < deadline, 'MuJoCo 执行端未就绪'
            time.sleep(.2)
        links = feedback_links(geometry)
        start = controller_angles(geometry['angles_deg'])
        with log.open('w', encoding='utf-8') as f:
            controller = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--controller'], env=env,
                                          stdin=subprocess.PIPE, stdout=f, stderr=f, text=True)

            def send(command, expect, timeout=8):
                before = log.read_text(encoding='utf-8').count(expect)
                controller.stdin.write(command + '\n')
                controller.stdin.flush()
                end = time.monotonic() + timeout
                while log.read_text(encoding='utf-8').count(expect) <= before:
                    assert controller.poll() is None and time.monotonic() < end, \
                        f'{command!r} 未得到 {expect!r}\n' + log.read_text(encoding='utf-8')[-1500:]
                    time.sleep(.05)

            time.sleep(1.5)
            send('status', 'STATUS')
            send('wind=2', '参数')
            send('step', '单步最大关节变化')  # 一次缠两节时第 2 关节需 -63.5°，超过 30° 上限，必须拒绝
            send('wind=0.25', '参数')
            for _ in range(80):
                send('step', '完成第')
            send('back', '完成第')
            send('step', '完成第')
            send('step', '已缠绕到尾部')
            send('apply', '已确认当前姿态')
            for _ in range(5):
                send('step', '顺应性 phase=')
            send('compliance', 'STATUS')
            send('step', '标准螺旋 phase=')
            send('compliance', 'STATUS')
            send('back', '顺应性 phase=')
            send('reset', '回到缠绕阶段')
            send('status', 'STATUS')
            controller.stdin.write('quit\n')
            controller.stdin.flush()
            controller.wait(timeout=15)
        text = log.read_text(encoding='utf-8')
        assert controller.returncode == 0, text[-1500:]
        completed = len(re.findall(r'完成第 \d+ 步', text))
        assert completed == 80 + 2 + 5 + 1 + 1, completed
        # MuJoCo 模式没有舵机状态块：遥测列为空，但记录文件照常生成，打印明确说明。
        assert '遥测记录' in text and '负载/温度=—' in text, text[-800:]
    finally:
        client.close()
        (OUT / 'executor.stop').touch()
        if executor is not None:
            executor.wait(timeout=10)
        for process in (controller, executor, app):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    applied = np.load(OUT / 'applied.npy')
    helix = HelixBackbone.from_radius_pitch(110., 180.).joint_angles(len(start), links, 0.)
    # 每次下发只执行一次；Ctrl+U 本身不下发目标。
    assert len(applied) == completed, (len(applied), completed)
    # 缠绕 1 节（第 4 步）：头部第一个关节精确到螺旋目标，其余仍为开始缠绕时的姿态。
    assert abs(applied[3][0] - helix[0]) < 1e-9
    assert np.allclose(applied[3][1:], start[1:], atol=1e-9)
    assert np.allclose(applied[79], helix, atol=1e-9)  # 缠满 20 节即标准螺旋
    jumps = np.degrees(np.abs(np.diff(np.vstack([start, applied]), axis=0))).max(axis=1)
    assert jumps.max() <= MAX_JOINT_STEP_DEG + 1e-9, jumps.max()
    summary = dict(completed_steps=completed, applied_frames=len(applied), max_step_change_deg=float(jumps.max()),
                   winding_step_change_deg=float(jumps[:80].max()), rolling_step_change_deg=float(jumps[82:].max()),
                   checks='wind limit/reject, per-joint winding, unwind, apply without dispatch, compliance, toggle, reset')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    if '--executor' in sys.argv:
        run_executor()
    elif '--controller' in sys.argv:
        run_controller()
    else:
        main()
