'''
说明：MuJoCo仿真实现从地面Rolling Arc横滚过渡到爬竖直杆子。本代码具有交互性。当a=-76，v=0.05时可以实现爬杆子。
注意，以上参数值仅供参考，并非标准。
参考文献：2013 Motion Estimation of Snake Robots in Straight Pipes
MuJoCo模型参考：https://github.com/VanguardDream/snake_RL
作者：肖剑0923
时间：20260122
'''
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import time
import queue
import mujoco
import threading
import numpy as np
import mujoco.viewer
from pynput import keyboard
import os
from contextlib import nullcontext
from snake_control.mujoco_visualizer_bridge import VisualizerBridge


NUM_JOINTS = 20
angle_rad = 57.2958


# 初始化RollingArc步态参数  全局变量
a = 30
v = 0.0
omega = 1.57
time_sleep = 0.01
time_step = 0.3
step = 1
y = 0.0
t = 0.0

'''
通过按键来调整参数值a和v，↑增大a，↓减小a，←减小v，→增大v，f5反向[单步调试]，f6正向[单步调试]
'''
def increase_a():
    global a
    a += 1
    print(f"a增加到: {a}")

def decrease_a():
    global a
    a -= 1
    print(f"a减少到: {a}")

def increase_v():
    global v
    v += 0.01
    print(f"v增加到: {v}")

def decrease_v():
    global v
    v -= 0.01
    print(f"v减少到: {v}")

def reverse_single_step():
    global step, time_step
    time_step = -abs(time_step)
    step += 1
    print(f"step = {step}, time_step < 0")

def single_step():
    global step, time_step
    time_step = abs(time_step)
    step += 1
    print(f"step = {step}, time_step > 0")

# 注册热键 ，，，
HOTKEYS = {
    keyboard.Key.up: increase_a,       # ↑增大a
    keyboard.Key.down: decrease_a,     # ↓减小a
    keyboard.Key.right: increase_v,    # ←减小v
    keyboard.Key.left: decrease_v,     # →增大v
}
CHAR_HOTKEYS = {
    '9': reverse_single_step,          # 反向[单步调试]
    '0': single_step,                  # 正向[单步调试]
}

def on_key_press(key):
    if key in HOTKEYS:
        HOTKEYS[key]()
    elif hasattr(key, 'char') and key.char in CHAR_HOTKEYS:
        CHAR_HOTKEYS[key.char]()

headless = os.environ.get('SNAKE_MUJOCO_HEADLESS') == '1'
if not headless:
    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.daemon = True
    key_listener.start()

'''
通过字符串快速设置参数
'''
# 创建命令队列
command_queue = queue.Queue()

def uart_listener():
    """模拟串口监听线程 - 相当于中断服务程序"""
    while True:
        try:
            # 这里模拟从串口接收数据
            # 实际使用时可以替换为 pyserial 的读取
            user_input = input("Enter command, please!")
            command_queue.put(user_input)
            time.sleep(0.1)  # 防止过度占用CPU
        except KeyboardInterrupt:
            break
        except EOFError:
            break

def command_processor():
    """命令处理线程 - 相当于中断处理"""
    global a, v, time_sleep, time_step, step

    while True:
        try:
            # 从队列获取命令（非阻塞）
            command = command_queue.get(timeout = 0.1)

            # 解析命令并修改变量
            if '=' in command:
                for subcommand in command.split():
                    key, value = subcommand.split('=')
                    key = key.strip().lower()

                    if key == 'a':
                        a = float(value)
                    elif key == 'v':
                        v = float(value)
                    elif key == 'time_sleep':
                        time_sleep = float(value)
                    elif key == 'time_step':
                        time_step = float(value)
                    elif key == 'step':
                        step = float(value)
                    else:
                        print(f"未知命令: {subcommand}")
                print(f"[参数更新] a = {a}, v = {v}, time_sleep = {time_sleep}, time_step = {time_step}, step = {step}")

            command_queue.task_done()

        except queue.Empty:
            continue
        except ValueError as e:
            print(f"命令格式错误: {e}")
        except KeyboardInterrupt:
            break

# 启动监听线程
if not headless:
    listener_thread = threading.Thread(target=uart_listener, daemon=True)
    processor_thread = threading.Thread(target=command_processor, daemon=True)
    listener_thread.start()
    processor_thread.start()

steps = 10  # step次数

model_path = os.environ.get('SNAKE_MUJOCO_MODEL') or str(
    Path(__file__).resolve().parent.parent / 'mujoco_model' / 'Snake_Robot_o20_Pole_Climbing.xml')
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
NUM_JOINTS = m.nu
data_ctrl = np.zeros(NUM_JOINTS)
bridge = VisualizerBridge(m, d)
max_seconds = float(os.environ.get('SNAKE_MUJOCO_MAX_SECONDS', '0'))
a = float(os.environ.get('SNAKE_MUJOCO_A', a))
v = float(os.environ.get('SNAKE_MUJOCO_V', v))
trace_path = os.environ.get('SNAKE_MUJOCO_TRACE')
trace_qpos, trace_ctrl, trace_time, trace_step_ms, trace_points = [], [], [], [], []


with (nullcontext(None) if headless else mujoco.viewer.launch_passive(m, d)) as viewer:
    start_wall = time.monotonic()
    while(step > 0):
        for i in range(NUM_JOINTS):
            if (i % 2 == 0):
                y = a / angle_rad * math.sin(omega * t + v * i)
            else:
                y = a / angle_rad * math.sin(omega * t + v * i + math.pi / 2)
            data_ctrl[i] = y
            step_started = time.perf_counter()
            with (nullcontext() if headless else viewer.lock()):
                d.ctrl = data_ctrl
                for k in range(steps):
                    mujoco.mj_step(m, d)
                step_elapsed_ms = (time.perf_counter() - step_started) * 1000
                if viewer is not None:
                    viewer.sync()
                bridge.publish(d)
                if trace_path:
                    trace_qpos.append(d.qpos.copy())
                    trace_ctrl.append(d.ctrl.copy())
                    trace_time.append(d.time)
                    trace_step_ms.append(step_elapsed_ms)
                    trace_points.append(bridge.world_points(d).copy())
                time.sleep(time_sleep)
            if max_seconds > 0 and time.monotonic() - start_wall >= max_seconds:
                break
        t = t + time_step
        if max_seconds > 0 and time.monotonic() - start_wall >= max_seconds:
            break
        # step = step - 1

bridge.close()
if trace_path:
    np.savez(trace_path, qpos=trace_qpos, ctrl=trace_ctrl, sim_time=trace_time,
             step_ms=trace_step_ms, points_world_m=trace_points)
print(f'VISUALIZER_BRIDGE sent={bridge.sent} rejected={bridge.rejected} joints={NUM_JOINTS} '
      f'enqueue_p99_ms={np.percentile(bridge.enqueue_ms, 99):.4f} '
      f'last_error={bridge.last_error!r}', flush=True)



