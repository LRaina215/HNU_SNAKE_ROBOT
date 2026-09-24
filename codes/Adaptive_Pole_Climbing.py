"""Adaptive Pole：以 Pole_Climbing.py 为模板的 Yamada 螺旋滚动与顺应性目标生成。

沿用 H-Crawler.py 的 K/Psi/Kp/Ky + scipy.integrate.quad 积分结构，
以及 Parallel_Crossing.py 的标准螺旋曲率/挠率公式。
已接入阶段 3：实测截面顺应性变形、三维重建及 Yamada 输出。实机下发仍留待阶段 4。

保留 a（有符号角度幅值）、v（相邻关节空间相位差）、omega、time_step
和原 Pole 的键盘/控制台操作。启动阶段 a/v 先转换为标准螺旋，再由 Yamada
积分生成目标；有效反馈到来后使用顺应性骨架。几何长度统一使用 mm。
默认 v=0 是平面圆弧退化情形，与原 Pole 从地面横滚开始的行为一致。

一轮统一提交全部目标，保持原 Pole 每轮 N*10 个仿真步和 N*time_sleep
的等待量；t 仍是原程序的相位时钟，不是 MuJoCo 的物理时间。
原脚本的爬杆参数不构成本脚本已通过稳定攀爬验证的证据。
"""
import math
import os
import queue
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import integrate

# True：打开接收曲线的调试窗口；False：不启动调试界面。
DEBUG = True
DEBUG_DRAW_CURVE = True  # False：调试窗口只显示各段参数表，不绘图。
COMPLIANCE_ENABLED = True
COMPLIANCE_GAIN = 0.04  # K_i；可改为按关节顺序排列的列表。
COMPLIANCE_PITCH_RAD = 0.25  # 论文的设计螺距角 alpha，单位 rad。
COMPLIANCE_INSIDE = False  # False 管外收紧；True 管内撑开（此时 K_i < 1）。


@dataclass(frozen=True)
class HelixBackbone:
    """标准螺旋的连续骨架；κ、τ 单位为 1/mm，相位单位为 rad。"""

    kappa: float
    tau: float
    phase_offset: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.kappa, self.tau, self.phase_offset)):
            raise ValueError("曲率、挠率和相位必须是有限数值")
        if self.kappa < 0:
            raise ValueError("曲率必须非负；弯曲方向由相位表示")

    @classmethod
    def from_radius_pitch(cls, radius_mm, pitch_mm, phase_offset=0.0):
        """复用 Parallel_Crossing.HelixSegment 的公式；pitch 为每圈轴向位移。"""
        if not math.isfinite(radius_mm) or radius_mm <= 0 or not math.isfinite(pitch_mm):
            raise ValueError("螺旋半径必须为有限正数，螺距必须有限")
        b = pitch_mm / (2 * math.pi)
        denominator = radius_mm ** 2 + b ** 2
        return cls(radius_mm / denominator, b / denominator, phase_offset)

    def K(self, s):
        return self.kappa

    def T(self, s):
        return self.tau

    def Psi(self, s, roll_phase):
        # 阶段 1 保留 Pole 空间相位的符号约定；这里的 tau 与阶段 3
        # 重建曲线使用的有向 Frenet 挠率不能直接混用。原论文 (33) 为 +∫τds。
        return -self.tau * s + roll_phase + self.phase_offset

    def Kp(self, s, roll_phase):
        # 沿用本地 Yamada 步态的函数名；实际奇偶轴映射见 joint_angles。
        return -self.K(s) * math.sin(self.Psi(s, roll_phase))

    def Ky(self, s, roll_phase):
        return self.K(s) * math.cos(self.Psi(s, roll_phase))

    def joint_angles(self, joint_count, link_mm, roll_phase):
        """Yamada 式 (32)，先算整组，再转换到现有 MuJoCo 的正负方向。

        j=1..N，积分区间 [(j-1)l,(j+1)l]，s 从头侧向尾侧递增。
        论文/可视化约定奇数关节 -κsinψ，偶数 +κcosψ。
        现有桥接的 MuJoCo→可视化符号为 [+1,-1,...]，反向映射相同。
        """
        if joint_count < 1 or not math.isfinite(link_mm) or link_mm <= 0:
            raise ValueError("关节数和名义关节间距必须为正")
        if not math.isfinite(roll_phase):
            raise ValueError("滚动相位必须有限")
        result = np.empty(joint_count)
        for j in range(1, joint_count + 1):
            component = self.Kp if j % 2 else self.Ky
            angle = integrate.quad(component, (j - 1) * link_mm, (j + 1) * link_mm,
                                   args=(roll_phase,), epsabs=1e-10, epsrel=1e-10)[0]
            result[j - 1] = angle if j % 2 else -angle
        return result


def helix_from_pole_parameters(amplitude_deg, spatial_phase, link_mm):
    """保留 Pole 的 a/v 操作语义，转换成 κ/τ 而非直接生成关节角。

    对常曲率螺旋，2l 区间积分的幅值 B=2l*κ*sinc(v)，τ=-v/l。
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
    tau = -spatial_phase / link_mm
    offset = -spatial_phase + (math.pi if amplitude_deg >= 0 else 0.0)
    return HelixBackbone(kappa, tau, offset)


# 与 Pole 相同的交互参数。模块导入不会启动仿真、键盘或网络。
a = 30.0
v = 0.0
omega = 1.57
time_sleep = 0.01
time_step = 0.3
step = 1
command_queue = queue.Queue()

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

def uart_listener():
    """保留 Pole 的函数名；实际读取控制台输入，不打开串口。"""
    while True:
        try:
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


def main():
    global a, v
    import mujoco
    import mujoco.viewer
    from mujoco_visualizer_bridge import VisualizerBridge

    headless = os.environ.get("SNAKE_MUJOCO_HEADLESS") == "1"
    key_listener = None
    if not headless:
        from pynput import keyboard
        hotkeys = {keyboard.Key.up: increase_a, keyboard.Key.down: decrease_a,
                   keyboard.Key.right: increase_v, keyboard.Key.left: decrease_v}
        char_hotkeys = {"9": reverse_single_step, "0": single_step}

        def on_key_press(key):
            if key in hotkeys:
                hotkeys[key]()
            elif getattr(key, "char", None) in char_hotkeys:
                char_hotkeys[key.char]()

    model_path = os.environ.get("SNAKE_MUJOCO_MODEL") or str(
        Path(__file__).resolve().parent.parent / "mujoco_model" / "Snake_Robot_o20_Pole_Climbing.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    joint_count = m.nu
    mujoco.mj_forward(m, d)
    # 采用模型内部相邻关节的平均间距作为名义 l，避免硬编码其他模型的 70 mm。
    anchors = np.array([d.joint(f"joint{j}").xanchor.copy() for j in range(1, joint_count + 1)])
    spacings = np.linalg.norm(np.diff(anchors, axis=0), axis=1) * 1000
    if joint_count < 2 or not np.all(np.isfinite(spacings)) or np.any(spacings <= 0):
        raise ValueError("模型需要至少两个具有有效间距的关节")
    link_mm = float(np.mean(spacings))
    a = float(os.environ.get("SNAKE_MUJOCO_A", a))
    v = float(os.environ.get("SNAKE_MUJOCO_V", v))
    max_seconds = float(os.environ.get("SNAKE_MUJOCO_MAX_SECONDS", "0"))
    trace_path = os.environ.get("SNAKE_MUJOCO_TRACE")
    bridge = None
    exchange = None
    debug_view = None
    compliance = None
    compliance_status = None
    traces = {name: [] for name in ("qpos", "ctrl", "sim_time", "step_ms", "roll_phase", "kappa", "tau",
                                   "compliance_active", "target_kappa", "target_tau")}
    last_error = None
    t = 0.0
    steps = 10
    print(f"YAMADA stage=3 joints={joint_count} nominal_link_mm={link_mm:.6f} "
          f"spacing_range_mm=[{min(spacings):.6f},{max(spacings):.6f}]", flush=True)
    if not headless:
        key_listener = keyboard.Listener(on_press=on_key_press)
        key_listener.daemon = True
        key_listener.start()
        threading.Thread(target=uart_listener, daemon=True).start()
        threading.Thread(target=command_processor, daemon=True).start()
    try:
        # 保留原有实测上传桥；另一个持久连接负责阶段 2 几何反馈/目标预览。
        if os.environ.get("SNAKE_VISUALIZER_ENABLED", "1") != "0":
            bridge = VisualizerBridge(m, d)
            if os.environ.get("SNAKE_GEOMETRY_EXCHANGE", "1") != "0":
                client_dir = Path(os.environ.get("SNAKE_CONTROL_CLIENT_DIR", str(
                    Path(__file__).resolve().parents[3] / "two_joint_visualizer")))
                sys.path.insert(0, str(client_dir))
                from python_client import GeometryExchange
                exchange = GeometryExchange(port=bridge.port, expected_lengths=bridge.lengths)
                if COMPLIANCE_ENABLED:
                    from pole_compliance import ComplianceController, ComplianceParameters
                    parameters = ComplianceParameters(COMPLIANCE_GAIN, COMPLIANCE_PITCH_RAD, COMPLIANCE_INSIDE)
                    parameters.gains(joint_count)
                    compliance = ComplianceController(parameters, bridge.lengths)
        if DEBUG:
            from geometry_debug_view import GeometryDebugView
            debug_view = GeometryDebugView(draw_curve=DEBUG_DRAW_CURVE)
        with (nullcontext(None) if headless else mujoco.viewer.launch_passive(m, d)) as viewer:
            start_wall = time.monotonic()
            while step > 0 and (viewer is None or viewer.is_running()):
                # 一帧内所有关节共享同一份参数及相位，后台交互只影响下一帧。
                frame_a, frame_v = a, v
                frame_sleep, frame_dt = time_sleep, time_step
                phase = omega * t
                feedback = None
                compliant_targets = None
                advance_phase = True
                try:
                    if not math.isfinite(frame_sleep) or frame_sleep < 0 or not math.isfinite(frame_dt):
                        raise ValueError("time_sleep 必须非负且有限，time_step 必须有限")
                    curve = helix_from_pole_parameters(frame_a, frame_v, link_mm)
                    targets = curve.joint_angles(joint_count, link_mm, phase)
                    if compliance is not None:
                        feedback = exchange.latest_geometry()
                        compliant_targets = compliance.compute(feedback, phase, d.ctrl.copy())
                        if compliant_targets is not None:
                            targets = compliant_targets
                        elif compliance.started:
                            targets = d.ctrl.copy()
                            advance_phase = False
                        if compliance.status != compliance_status:
                            print(f"COMPLIANCE {compliance.status}", flush=True)
                            compliance_status = compliance.status
                    joint_ids = m.actuator_trnid[:, 0]
                    limited = m.jnt_limited[joint_ids].astype(bool)
                    limits = m.jnt_range[joint_ids]
                    if (np.any(targets[limited] < limits[limited, 0]) or
                            np.any(targets[limited] > limits[limited, 1])):
                        raise ValueError("目标角超出模型关节范围，请减小 a")
                    last_error = None
                except ValueError as error:
                    if headless:
                        raise
                    if str(error) != last_error:
                        print(f"参数被拒绝：{error}", flush=True)
                        last_error = str(error)
                    # 不推进相位，不提交无效目标；等待交互修正或关闭。
                    if debug_view is not None:
                        debug_view.update(exchange)
                    time.sleep(0.05)
                    if max_seconds > 0 and time.monotonic() - start_wall >= max_seconds:
                        break
                    continue

                with (nullcontext() if viewer is None else viewer.lock()):
                    d.ctrl[:] = targets
                # 保持原 Pole 每轮的仿真步数/等待量，但整轮只提交这一组目标。
                # 这一同步化会改变瞬态动力学，因此不声称 qpos 与原脚本相同。
                for _ in range(joint_count):
                    step_started = time.perf_counter()
                    with (nullcontext() if viewer is None else viewer.lock()):
                        for _ in range(steps):
                            mujoco.mj_step(m, d)
                        if bridge is not None:
                            bridge.publish(d)
                        if exchange is not None and (compliance is None or not compliance.started or compliant_targets is not None):
                            # 仍只回传预览；真实串口下发留待阶段 4。
                            # bridge 的方向映射 [+1,-1,...]，发送可视化约定的角度（度）。
                            exchange.publish_targets(np.rad2deg(targets * bridge.signs).tolist(),
                                                     source_geometry=feedback if compliant_targets is not None else None)
                        if trace_path:
                            traces["qpos"].append(d.qpos.copy())
                            traces["ctrl"].append(d.ctrl.copy())
                            traces["sim_time"].append(d.time)
                            traces["step_ms"].append((time.perf_counter() - step_started) * 1000)
                            traces["roll_phase"].append(phase)
                            traces["kappa"].append(curve.kappa)
                            traces["tau"].append(curve.tau)
                            traces["compliance_active"].append(compliant_targets is not None)
                            if compliance is not None and compliance.started:
                                sample_s = np.cumsum(compliance.backbone.links)[:-1]
                                ks, ts = compliance.backbone.curvature_torsion(sample_s)
                            else:
                                ks, ts = np.full(joint_count, curve.kappa), np.full(joint_count, -curve.tau)
                            traces["target_kappa"].append(ks)
                            traces["target_tau"].append(ts)
                    if debug_view is not None:
                        debug_view.update(exchange)
                    if viewer is not None:
                        viewer.sync()
                        if not viewer.is_running():
                            break
                    time.sleep(frame_sleep)
                    if max_seconds > 0 and time.monotonic() - start_wall >= max_seconds:
                        break
                if advance_phase:
                    t += frame_dt
                if max_seconds > 0 and time.monotonic() - start_wall >= max_seconds:
                    break
    finally:
        if debug_view is not None:
            debug_view.close()
        if key_listener is not None:
            key_listener.stop()
        if exchange is not None:
            exchange.close()
            print(f"GEOMETRY_EXCHANGE received={exchange.received} sent={exchange.sent} "
                  f"replaced={exchange.dropped} failures={exchange.failures} "
                  f"last_error={exchange.last_error!r}", flush=True)
        if bridge is not None:
            bridge.close()
            p99 = np.percentile(bridge.enqueue_ms, 99) if bridge.enqueue_ms else 0.0
            print(f"VISUALIZER_BRIDGE sent={bridge.sent} rejected={bridge.rejected} "
                  f"joints={joint_count} enqueue_p99_ms={p99:.4f} last_error={bridge.last_error!r}", flush=True)
        if trace_path:
            np.savez(trace_path, **traces, nominal_link_mm=link_mm)


if __name__ == "__main__":
    main()
