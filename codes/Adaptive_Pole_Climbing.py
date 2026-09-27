"""Adaptive Pole：几何参数 → 曲率/挠率 → Yamada → EXE → MuJoCo。

默认单步：Ctrl+U 应用 R/P，Ctrl+J 正向一步，Ctrl+H 反向一步。
使用 --scene variable 加载连续变径管道，空格暂停，Ctrl+R 恢复实验初始条件。
等待按键时物理时间与滚动相位冻结，通信、反馈和窗口继续刷新。
R/P 定义初始螺旋及设计螺距角；顺应性运行中局部形状来自实测反馈。
长度 mm，相位 rad。模块导入不会启动仿真、键盘或网络。
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
COMPLIANCE_GAIN = (0.04, 0.08)  # K_i：标量，或论文式 (34) 的（首尾各 4 个关节, 中部）。
COMPLIANCE_INSIDE = False  # False 管外收紧；True 管内撑开（此时 K_i < 1）。
CONTROL_VIA_VISUALIZER = True  # 目标必须经 EXE 转发后才写入 MuJoCo。
SINGLE_STEP_MODE = True
HELIX_RADIUS_MM = 110.0  # 机器人中心线半径，不是管道表面半径。
HELIX_PITCH_MM = 180.0  # 每圈轴向位移；有符号，0 为平面圆弧。
ROLL_STEP_RAD = 0.15  # 每次 H/J 的相位增量，与形状参数分开。
FRAME_SLEEP = 0.01  # 每组 10 个物理步之后的刷新等待。


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
        # 与顺应性 PiecewiseBackbone 相同：沿头→尾弧长累计有向挠率。
        return self.tau * s + roll_phase + self.phase_offset

    def Kp(self, s, roll_phase):
        # 沿用本地 Yamada 步态的函数名；实际奇偶轴映射见 joint_angles。
        return -self.K(s) * math.sin(self.Psi(s, roll_phase))

    def Ky(self, s, roll_phase):
        return self.K(s) * math.cos(self.Psi(s, roll_phase))

    def joint_angles(self, joint_count, link_mm, roll_phase):
        """Yamada 式 (32)，先算整组，再转换到现有 MuJoCo 的正负方向。

        j=1..N，积分覆盖关节两侧连杆；link_mm 可为等长标量或 N+1 个实际长度。
        论文/可视化约定奇数关节 -κsinψ，偶数 +κcosψ。
        现有桥接的 MuJoCo→可视化符号为 [+1,-1,...]，反向映射相同。
        """
        links = np.broadcast_to(np.asarray(link_mm, dtype=float), (joint_count + 1,))
        if joint_count < 1 or not np.all(np.isfinite(links)) or np.any(links <= 0):
            raise ValueError("关节数和连杆长度必须为正")
        bounds = np.r_[0., np.cumsum(links)]
        if not math.isfinite(roll_phase):
            raise ValueError("滚动相位必须有限")
        result = np.empty(joint_count)
        for j in range(1, joint_count + 1):
            component = self.Kp if j % 2 else self.Ky
            angle = integrate.quad(component, bounds[j - 1], bounds[j + 1],
                                   args=(roll_phase,), epsabs=1e-10, epsrel=1e-10)[0]
            result[j - 1] = angle if j % 2 else -angle
        return result


# 键盘/控制台只入队；参数、相位与 mjData 均由主线程修改。
command_queue = queue.Queue()


def single_step():
    command_queue.put("step")


def reverse_single_step():
    command_queue.put("back")


def apply_helix():
    command_queue.put("apply")


def uart_listener():
    """沿用 Pole 的入口名，实际读取控制台，不打开串口。"""
    while True:
        try:
            command_queue.put(input())
        except (EOFError, KeyboardInterrupt):
            break


def main():
    import mujoco
    import mujoco.viewer
    from mujoco_visualizer_bridge import VisualizerBridge
    from pole_compliance import ComplianceController, ComplianceParameters
    from adaptive_debug import HELP, viewer_key_callback, measurements




    headless = os.environ.get("SNAKE_MUJOCO_HEADLESS") == "1"
    relay = os.environ.get("SNAKE_CONTROL_VIA_VISUALIZER", str(int(CONTROL_VIA_VISUALIZER))) != "0"
    manual = os.environ.get("SNAKE_SINGLE_STEP", str(int(SINGLE_STEP_MODE))) != "0"
    if relay and (os.environ.get("SNAKE_VISUALIZER_ENABLED", "1") == "0" or
                  os.environ.get("SNAKE_GEOMETRY_EXCHANGE", "1") == "0"):
        raise ValueError("转发执行模式要求可视化通信开启")
    radius = float(os.environ.get("SNAKE_HELIX_RADIUS_MM", HELIX_RADIUS_MM))
    pitch = float(os.environ.get("SNAKE_HELIX_PITCH_MM", HELIX_PITCH_MM))
    HelixBackbone.from_radius_pitch(radius, pitch)
    if not math.isfinite(ROLL_STEP_RAD) or ROLL_STEP_RAD <= 0:
        raise ValueError("ROLL_STEP_RAD 必须为有限正数")
    model_path = os.environ.get("SNAKE_MUJOCO_MODEL") or str(
        Path(__file__).resolve().parents[1] / "mujoco_model/Snake_Robot_o20_Pole_Climbing.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    start_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "experiment_start")
    if start_key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, start_key)
        preset = m.numeric("experiment_helix_mm").data
        radius = float(os.environ.get("SNAKE_HELIX_RADIUS_MM", preset[0]))
        pitch = float(os.environ.get("SNAKE_HELIX_PITCH_MM", preset[1]))
    mujoco.mj_forward(m, d)

    joint_count = m.nu
    points = [d.site("s_head").xpos.copy()]
    points.extend(d.joint(f"joint{j}").xanchor.copy() for j in range(1, joint_count + 1))
    points.append(d.site("s_tail").xpos.copy())
    links = np.linalg.norm(np.diff(points, axis=0), axis=1) * 1000
    link_mm = float(np.mean(links[1:-1]))
    joint_ids = m.actuator_trnid[:, 0]
    limited, limits = m.jnt_limited[joint_ids].astype(bool), m.jnt_range[joint_ids]



    max_seconds = float(os.environ.get("SNAKE_MUJOCO_MAX_SECONDS", "0"))
    trace_path = os.environ.get("SNAKE_MUJOCO_TRACE")
    bridge = exchange = debug_view = compliance = None
    traces = {name: [] for name in ("qpos", "ctrl", "sim_time", "step_ms", "roll_phase", "kappa", "tau",
                                   "compliance_active", "target_kappa", "target_tau", "requested_ctrl",
                                   "applied_delivery_id", "applied_command_id", "applied_run_id", "wall_unix_ms",
                                   "moving", "completed_steps", "reset_count", "robot_height_m", "pipe_contacts", "tracking_error_rad")}
    phase = 0.0
    phase_step = ROLL_STEP_RAD
    compliance_enabled = COMPLIANCE_ENABLED
    reset_count = 0
    feedback_after_ms = 0.
    gravity = m.opt.gravity.copy()
    initialized = False
    active_shape = None
    pending = None
    remaining = completed = 0
    curve = HelixBackbone.from_radius_pitch(radius, pitch)
    targets = d.ctrl.copy()
    compliant = False
    action_label = "标准螺旋"
    relay_status = None
    last_notice = None
    print(f"YAMADA execution={'mujoco_relay' if relay else 'direct_preview'} joints={joint_count} "
          f"mode={'单步' if manual else '连续'} R={radius:g}mm P={pitch:g}mm", flush=True)
    print(HELP, flush=True)
    print(f"场景：{Path(model_path).name}；重力={m.opt.gravity.tolist()}；{'启动暂停，先 Ctrl+U' if manual else '连续运行'}", flush=True)

    def show_status():
        height, contact_count, error = measurements(m, d)
        feedback = exchange.latest_geometry() if exchange is not None else None
        feedback_state = "有效" if feedback and feedback.get("feedback_usable") else "未就绪/无效"
        print(f"STATUS {'单步/暂停' if manual else '连续'} R={radius:g} P={pitch:g} "
              f"phase={phase:.4f} delta={phase_step:g} sim={d.time:.3f}s "
              f"顺应性={'开' if compliance_enabled else '关'} 反馈={feedback_state} "
              f"关节平均高度={height:.3f}m 管道接触={contact_count} "
              f"最大跟踪误差={np.degrees(error):.2f}deg 重力={'开' if np.any(m.opt.gravity) else '关'}", flush=True)

    try:
        if not headless:
            threading.Thread(target=uart_listener, daemon=True).start()
        if os.environ.get("SNAKE_VISUALIZER_ENABLED", "1") != "0":
            sys.path.insert(0, os.environ.get("SNAKE_CONTROL_CLIENT_DIR", str(
                Path(__file__).resolve().parents[1] / "visualizer")))
            if relay:
                from mujoco_execution_bridge import MujocoExecutionBridge
                bridge = MujocoExecutionBridge(m, d)
                bridge.record_commands = bool(trace_path)
            else:
                bridge = VisualizerBridge(m, d)
            if os.environ.get("SNAKE_GEOMETRY_EXCHANGE", "1") != "0":
                from python_client import GeometryExchange
                exchange = GeometryExchange(port=bridge.port, expected_lengths=bridge.lengths,
                                            execution="mujoco" if relay else "preview")
        if DEBUG:
            from geometry_debug_view import GeometryDebugView
            debug_view = GeometryDebugView(draw_curve=DEBUG_DRAW_CURVE)



        with (nullcontext(None) if headless else mujoco.viewer.launch_passive(m, d, key_callback=viewer_key_callback(command_queue))) as viewer:
            if viewer is not None and start_key >= 0:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0., 0., .85]
                    viewer.cam.distance = 3.4
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -12
            start_wall = time.monotonic()
            while viewer is None or viewer.is_running():
                now = time.monotonic()
                if max_seconds > 0 and now - start_wall >= max_seconds:
                    break
                if relay:
                    status = "已连接" if bridge.executor_ready else (bridge.last_error or "等待可视化执行端")
                    if status != relay_status:
                        print(f"MUJOCO_RELAY {status}", flush=True)
                        relay_status = status



                # 每个刷新周期都检查命令，暂停/重置不必等待当前动作结束。
                try:
                    command = command_queue.get_nowait().strip().lower()
                except queue.Empty:
                    command = ""
                if command in ("pause", "reset"):
                    manual, pending, remaining = True, None, 0
                    while not command_queue.empty():
                        command_queue.get_nowait()
                    if command == "reset":
                        with (nullcontext() if viewer is None else viewer.lock()):
                            if start_key >= 0:
                                mujoco.mj_resetDataKeyframe(m, d, start_key)
                            else:
                                mujoco.mj_resetData(m, d)
                            mujoco.mj_forward(m, d)
                        if relay:
                            bridge.reset_execution()
                        phase, completed = 0., 0
                        initialized, active_shape, compliance = False, None, None
                        compliant = False
                        targets = d.ctrl.copy()
                        reset_count += 1
                        feedback_after_ms = time.time() * 1000
                        print("实验已重置并暂停；保留当前 R/P、步长、顺应性和重力选项，Ctrl+U 重新应用", flush=True)
                    else:
                        print("已暂停；候选和剩余物理步已取消，未执行后续排队动作", flush=True)
                    show_status()
                    command = ""
                elif command in ("status", "help"):
                    print(HELP, flush=True) if command == "help" else show_status()
                    command = ""
                elif command == "run":
                    manual = not manual
                    if manual:
                        pending, remaining = None, 0
                    show_status()
                    command = ""
                if command and (pending is not None or remaining > 0):
                    print("当前一步尚未完成；该操作未排队，空格可立即暂停", flush=True)
                    command = ""
                if not command and not manual and pending is None and remaining == 0:
                    command = "step" if initialized else "apply"
                if command:
                    try:
                        if command == "compliance":
                            compliance_enabled = not compliance_enabled
                            compliance = None
                            if initialized and compliance_enabled and exchange is not None:
                                compliance = ComplianceController(ComplianceParameters(COMPLIANCE_GAIN,
                                    math.atan2(active_shape[1], 2*math.pi*active_shape[0]), COMPLIANCE_INSIDE), links)
                            show_status()
                        elif command in ("delta-", "delta+"):
                            phase_step = max(.01, phase_step/2) if command == "delta-" else min(.6, phase_step*2)
                            show_status()
                        elif command in ("r+", "r-", "p+", "p-") or "=" in command:
                            new_r, new_p = radius, pitch
                            if command in ("r+", "r-"):
                                new_r += 5 if command == "r+" else -5
                            elif command in ("p+", "p-"):
                                new_p += 10 if command == "p+" else -10
                            else:
                                for assignment in command.split():
                                    key, value = assignment.split("=")
                                    if key == "r":
                                        new_r = float(value)
                                    elif key == "p":
                                        new_p = float(value)
                                    elif key == "delta":
                                        value = float(value)
                                        if not math.isfinite(value) or value <= 0:
                                            raise ValueError("delta 必须为正数 rad")
                                        phase_step = value
                                    elif key == "gravity" and value in ("0", "1"):
                                        with (nullcontext() if viewer is None else viewer.lock()):
                                            m.opt.gravity[:] = gravity if value == "1" else 0.
                                    else:
                                        raise ValueError("参数：r=半径mm p=螺距mm delta=步长rad gravity=0或1")
                            HelixBackbone.from_radius_pitch(new_r, new_p)
                            radius, pitch = new_r, new_p
                            print(f"参数 R={radius:g}mm P={pitch:g}mm delta={phase_step:g}rad；"
                                  f"重力={'开' if np.any(m.opt.gravity) else '关'}。R/P 修改后 Ctrl+U 应用", flush=True)
                        elif command in ("apply", "step", "back", "hold"):
                            if command != "apply" and (not initialized or (command != "hold" and active_shape != (radius, pitch))):
                                raise ValueError("请先 Ctrl+U 应用当前 R/P，再单步滚动")
                            # 应用新 R/P 时沿用当前相位：清零会让已滚动的关节目标跳变上百度而脱管。
                            next_phase = phase if command in ("apply", "hold") else phase + (1 if command == "step" else -1) * phase_step
                            curve = HelixBackbone.from_radius_pitch(radius, pitch)
                            feedback = None
                            compliant = False
                            candidate = d.ctrl.copy() if command == "hold" else curve.joint_angles(joint_count, links, next_phase)
                            if command in ("step", "back") and compliance is not None:
                                feedback = exchange.latest_geometry()
                                if feedback and feedback["sample_unix_ms"] <= feedback_after_ms:
                                    raise ValueError("等待重置后的新反馈")
                                adaptive_target = compliance.compute(feedback, next_phase, d.ctrl.copy(), reference_phase=phase)
                                if adaptive_target is not None:
                                    candidate = adaptive_target
                                    compliant = True
                                elif compliance.started:
                                    raise ValueError(compliance.status + "；本步未执行")
                                else:
                                    # 直线启动尚无截面，必须允许手动形成初始螺旋；
                                    # 一旦顺应性接管，坏反馈不能退回名义形状。
                                    feedback = None
                                    print("标准螺旋初始化：尚无有效截面，本次仅推进一步", flush=True)
                            if np.any(candidate[limited] < limits[limited, 0]) or np.any(candidate[limited] > limits[limited, 1]):
                                raise ValueError("目标角超出模型范围，请增大半径或调整螺距")
                            targets = candidate
                            pending = dict(command=command, phase=next_phase, feedback=feedback, since=now)
                            last_notice = None
                        elif command:
                            raise ValueError("未知命令；输入 help 查看操作")
                    except ValueError as error:
                        if str(error) != last_notice:
                            print(f"未执行：{error}", flush=True)
                            last_notice = str(error)





                step_started = time.perf_counter()
                with (nullcontext() if viewer is None else viewer.lock()):
                    if pending is not None:
                        feedback = pending["feedback"]
                        stale = feedback is not None and time.time() * 1000 - feedback["sample_unix_ms"] > 500
                        if stale or now - pending["since"] > 2:
                            print("本步取消：反馈过期或未收到 EXE 转发；相位与物理时间未推进，请重新按键。", flush=True)
                            pending = None
                        else:
                            if exchange is not None:
                                exchange.publish_targets(np.rad2deg(targets * bridge.signs).tolist(), source_geometry=feedback)
                            applied = bridge.apply_received(d, expected_targets=targets) if relay else True
                            if applied:
                                if not relay:
                                    d.ctrl[:] = targets
                                phase = pending["phase"]
                                if pending["command"] == "apply":
                                    initialized = True
                                    active_shape = (radius, pitch)
                                    # R/P 只在明确应用时设定设计螺距角，不逐帧覆盖局部顺应性半径。
                                    if compliance_enabled and exchange is not None:
                                        alpha = math.atan2(pitch, 2 * math.pi * radius)
                                        compliance = ComplianceController(
                                            ComplianceParameters(COMPLIANCE_GAIN, alpha, COMPLIANCE_INSIDE), links)
                                action_label = "保持目标" if pending["command"] == "hold" else "顺应性" if compliant else "标准螺旋"
                                remaining = joint_count
                                pending = None
                    moving = remaining > 0
                    if moving:
                        # 一次操作保留 Pole 的 N*10 个物理步，下一次按键前完全暂停。
                        # 只在本步开始接收一次整组目标，迟到帧不能在空闲时改写 ctrl。
                        for _ in range(10):
                            mujoco.mj_step(m, d)
                        remaining -= 1
                        if remaining == 0:
                            completed += 1
                            print(f"完成第 {completed} 步 phase={phase:.6f}rad sim={d.time:.3f}s "
                                  f"{action_label}；等待下一步" if manual else
                                  f"完成第 {completed} 步 phase={phase:.6f}rad", flush=True)
                    if bridge is not None:
                        bridge.publish(d)
                    if trace_path:
                        traces["qpos"].append(d.qpos.copy())
                        traces["ctrl"].append(d.ctrl.copy())
                        traces["requested_ctrl"].append(targets.copy())
                        traces["applied_delivery_id"].append(bridge.applied_delivery_id if relay else 0)
                        traces["applied_command_id"].append(bridge.applied_command_id if relay else 0)
                        traces["applied_run_id"].append(bridge.applied_run_id if relay else "")
                        traces["wall_unix_ms"].append(time.time() * 1000)
                        traces["sim_time"].append(d.time)
                        traces["step_ms"].append((time.perf_counter() - step_started) * 1000)
                        traces["roll_phase"].append(phase)
                        traces["kappa"].append(curve.kappa)
                        traces["tau"].append(curve.tau)
                        traces["compliance_active"].append(compliant and moving)
                        traces["moving"].append(moving)
                        traces["completed_steps"].append(completed)
                        traces["reset_count"].append(reset_count)
                        height, contact_count, error = measurements(m, d)
                        traces["robot_height_m"].append(height)
                        traces["pipe_contacts"].append(contact_count)
                        traces["tracking_error_rad"].append(error)
                        if compliant and compliance is not None and compliance.started:
                            ks, ts = compliance.backbone.curvature_torsion(np.cumsum(links)[:-1])
                        else:
                            ks, ts = np.full(joint_count, curve.kappa), np.full(joint_count, curve.tau)
                        traces["target_kappa"].append(ks)
                        traces["target_tau"].append(ts)
                if debug_view is not None:
                    debug_view.update(exchange)
                if viewer is not None:
                    viewer.sync()
                time.sleep(FRAME_SLEEP)
    finally:
        if debug_view is not None:
            debug_view.close()
        if exchange is not None:
            exchange.close()
            print(f"GEOMETRY_EXCHANGE received={exchange.received} sent={exchange.sent} "
                  f"failures={exchange.failures} last_error={exchange.last_error!r}", flush=True)
        if bridge is not None:
            bridge.close()
            if relay:
                print(f"MUJOCO_RELAY applied={bridge.applied} expired={bridge.expired}", flush=True)
            print(f"VISUALIZER_BRIDGE sent={bridge.sent} rejected={bridge.rejected} "
                  f"last_error={bridge.last_error!r}", flush=True)
        if trace_path:
            np.savez(trace_path, **traces, nominal_link_mm=link_mm, link_lengths_mm=links)
            if relay and bridge is not None:
                import json
                Path(str(trace_path) + ".received.json").write_text(
                    json.dumps(bridge.received_commands), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Yamada 管道实验；默认保持原 Pole 场景")
    parser.add_argument("--scene", choices=("pole", "variable"), default=None)
    args = parser.parse_args()
    if args.scene is not None:
        filename = "Snake_Robot_o20_Variable_Pipe.xml" if args.scene == "variable" else "Snake_Robot_o20_Pole_Climbing.xml"
        os.environ["SNAKE_MUJOCO_MODEL"] = str(Path(__file__).resolve().parents[1] / "mujoco_model" / filename)
    main()
