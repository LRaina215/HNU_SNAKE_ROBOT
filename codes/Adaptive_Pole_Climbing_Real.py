"""Adaptive Pole 实机版：实测反馈 → 曲率/挠率 → Yamada → EXE（唯一串口所有者）→ 舵机。

与 Adaptive_Pole_Climbing_Mujoco.py 键位一致，但启动后不生成任何步态：
1. 缠绕阶段：Ctrl+J 从头部开始一节一节缠成标准螺旋（R/P），Ctrl+H 退回一节。
2. 位置合适后 Ctrl+U：以当前姿态为起点进入单步滚动，默认开顺应性（Ctrl+E 切换）。
3. 滚动阶段：Ctrl+J / Ctrl+H 相位 + / − 一步，与仿真相同。
扭矩只能在 EXE 界面手动解锁并使能；急停用 EXE 的紧急释放。
没有场景标记，管内/管外只由 COMPLIANCE_INSIDE 决定，启动时打印确认。
长度 mm，相位 rad。模块导入不会启动网络或键盘监听。
"""
import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from Adaptive_Pole_Climbing_Mujoco import HelixBackbone

# True：打开接收曲线的调试窗口；False：不启动调试界面。
DEBUG = True
DEBUG_DRAW_CURVE = True  # False：调试窗口只显示各段参数表，不绘图。
COMPLIANCE_ENABLED = True  # Ctrl+U 之后默认开顺应性；Ctrl+E 随时切换。
COMPLIANCE_GAIN = (0.04, 0.12)  # K_i：标量，或（首尾各 4 个关节, 中部）。
COMPLIANCE_INSIDE = False  # 手动选择：False 管外收紧；True 管内撑开。
HELIX_RADIUS_MM = 110.0  # 缠绕目标：机器人中心线螺旋半径，不是管道表面半径。
HELIX_PITCH_MM = 180.0  # 缠绕目标：每圈轴向位移。
ROLL_STEP_RAD = 0.15  # 滚动阶段每次 Ctrl+J/H 的相位增量。
WIND_STEP_JOINTS = 0.25  # 缠绕阶段每次 Ctrl+J/H 推进的关节数；0.25 表示一节分四次到位。
MAX_JOINT_STEP_DEG = 30.0  # 任一关节单步目标变化上限，超过则本步不执行；仿真中顺应性单步最大约 22°，缠绕步长 0.25 约 17°。
STEP_SETTLE_S = 1.0  # 目标下发后等待舵机运动的时间，之后才报告本步完成。
DISPATCH_TIMEOUT_S = 2.0  # 等待 EXE 确认已下发的时间。
PORT = 8765

HELP = """实机调试（全局热键，任何窗口都会响应；Ctrl+字母）：
  缠绕阶段：Ctrl+J 从头部多缠一节   Ctrl+H 退回一节   Ctrl+-/= 缠绕步长减半/加倍
  Ctrl+U 确认当前姿态，进入单步滚动（默认开顺应性）
  滚动阶段：Ctrl+J 相位+一步   Ctrl+H 相位-一步   Ctrl+-/= 相位步长减半/加倍
  Ctrl+N 重发当前目标   Ctrl+E 顺应性开/关   Ctrl+K 状态   Ctrl+R 回到缠绕阶段（不动机器人）
  Ctrl+上下 修改半径；Ctrl+左右 修改螺距（缠绕阶段下一步生效）   空格 暂停/中断等待
控制台：step / back / apply / hold / compliance / status / reset / pause / help / quit
参数：r=110 p=180 / delta=0.05 / wind=0.25
扭矩使能与紧急释放只在 EXE 界面操作。
"""

command_queue = queue.Queue()


def console_listener():
    """控制台命令入队；与全局热键共用同一队列。"""
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        command_queue.put(line)
        # 退出后不再阻塞读取，避免解释器关闭时读线程仍占用 stdin。
        if line.strip().lower() in ("quit", "exit"):
            break


def feedback_links(geometry):
    """EXE 反馈中的 N+1 段连杆长度（头段 + 各关节后连杆），mm。"""
    config = geometry["config"]
    return np.array([config["head_length_mm"]] + [j["link_length_mm"] for j in config["joints"]], dtype=float)


def controller_angles(degrees):
    """协议角（度，可视化方向）→ 控制器方向（rad）；奇偶关节方向相反。"""
    degrees = np.asarray(degrees, dtype=float)
    return np.deg2rad(degrees) * np.where(np.arange(len(degrees)) % 2, -1, 1)


def protocol_angles(radians):
    radians = np.asarray(radians, dtype=float)
    return np.rad2deg(radians * np.where(np.arange(len(radians)) % 2, -1, 1))


def winding_targets(start, helix, progress):
    """从头部逐节缠绕：关节 j（从 0 计）的权重为 clip(progress − j, 0, 1)，其余保持开始缠绕时的姿态。"""
    start, helix = np.asarray(start, dtype=float), np.asarray(helix, dtype=float)
    weights = np.clip(progress - np.arange(len(start)), 0., 1.)
    return start + weights * (helix - start)


def executed(state, command_id, execution):
    """EXE 是否已把该命令号交给执行端（实机：已写串口；MuJoCo：执行端已写入 ctrl）。"""
    if execution == "live":
        return state.get("dispatched_command_id", 0) >= command_id
    if execution == "mujoco":
        return (state.get("accepted_command_id", 0) >= command_id and not state.get("pending_command_id") and
                state.get("mujoco_delivery_id", 0) > 0 and
                state.get("mujoco_applied_delivery_id") == state.get("mujoco_delivery_id"))
    return state.get("accepted_command_id", 0) >= command_id


def main(execution="live", hotkeys=True, port=PORT):
    from pole_compliance import ComplianceController, ComplianceParameters
    from adaptive_debug import global_key_listener, circle_radius_mm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualizer"))
    from python_client import GeometryExchange

    radius, pitch = HELIX_RADIUS_MM, HELIX_PITCH_MM
    HelixBackbone.from_radius_pitch(radius, pitch)
    phase_step, wind_step = ROLL_STEP_RAD, WIND_STEP_JOINTS
    compliance_enabled = COMPLIANCE_ENABLED
    stage = "wind"  # wind：缠绕阶段；roll：Ctrl+U 之后的单步滚动
    progress = 0.  # 已缠绕的关节数，可为小数
    start = None  # 开始缠绕时的实测姿态（控制器方向，rad）
    targets = None  # 最近一次确认下发的整组目标
    phase = 0.
    compliance = None
    completed = 0
    last_notice = None
    exchange = GeometryExchange(port=port, execution=execution)
    debug_view = None

    def fresh_feedback(require_usable):
        feedback = exchange.latest_geometry()
        if feedback is None:
            raise ValueError("没有 EXE 反馈：确认 EXE 已启动、处于实机模式且串口已连接")
        if not all(feedback["online"]):
            raise ValueError("存在离线关节")
        if require_usable and not feedback.get("feedback_usable"):
            raise ValueError("反馈不可用：轴无效或关节离线")
        return feedback

    def wait_or_pause(seconds, condition=lambda: False):
        """等待期间只响应暂停；其他按键不排队，与仿真一致。"""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if condition():
                return True
            try:
                command = command_queue.get_nowait().strip().lower()
            except queue.Empty:
                command = ""
            if command == "pause":
                print("已暂停：停止等待，EXE 保持最后下发的目标", flush=True)
                return None
            if command:
                print("当前一步尚未完成；该操作未排队，空格可中断等待", flush=True)
            if debug_view is not None:
                debug_view.update(exchange)
            time.sleep(.02)
        return False

    def execute(candidate, feedback, label):
        """下发一整组目标并等待确认；返回 True 表示已下发（之后被暂停也算）。"""
        nonlocal targets, completed
        joints = feedback["config"]["joints"]
        degrees = protocol_angles(candidate)
        low = np.array([j["minimum_deg"] for j in joints])
        high = np.array([j["maximum_deg"] for j in joints])
        if not np.all(np.isfinite(degrees)) or np.any(degrees < low) or np.any(degrees > high):
            raise ValueError("目标超出关节范围")
        reference = targets if targets is not None else controller_angles(feedback["angles_deg"])
        jump = float(np.degrees(np.abs(candidate - reference)).max())
        if jump > MAX_JOINT_STEP_DEG:
            raise ValueError(f"单步最大关节变化 {jump:.1f}° 超过上限 {MAX_JOINT_STEP_DEG:g}°，请减小步长")
        before = exchange.last_command_id
        if not exchange.publish_targets(degrees.tolist(), source_geometry=feedback):
            raise ValueError("EXE 未接受目标：反馈过期、模式不符或未使能扭矩")
        done = wait_or_pause(DISPATCH_TIMEOUT_S, lambda: exchange.last_command_id > before and executed(
            (exchange.latest_geometry() or {}).get("target_state", {}), exchange.last_command_id, execution))
        if done is None:
            return False
        if not done:
            ack = exchange.last_ack or {}
            raise ValueError(f"等待 EXE 下发超时（{ack.get('error') or ack.get('status') or '无应答'}），目标可能未执行")
        targets = candidate
        if wait_or_pause(STEP_SETTLE_S) is None:
            return True
        completed += 1
        latest = exchange.latest_geometry()
        error = "—" if latest is None else f"{np.degrees(np.abs(controller_angles(latest['angles_deg']) - targets)).max():.1f}"
        print(f"完成第 {completed} 步 {label} 最大跟踪误差={error}deg；等待下一步", flush=True)
        return True

    def show_status():
        feedback = exchange.latest_geometry()
        text = lambda value: "—" if value is None else f"{value:.1f}"
        usable = bool(feedback and feedback.get("feedback_usable"))
        measured = circle_radius_mm(feedback["projected_mm"][1:-1]) if usable else None
        if compliance is not None and compliance.started:
            target = circle_radius_mm(compliance.backbone.sample(world=False)[:, :2])
        else:
            target = radius
        error = None
        if feedback is not None and targets is not None:
            error = float(np.degrees(np.abs(controller_angles(feedback["angles_deg"]) - targets)).max())
        count = feedback["config"]["joint_count"] if feedback else "?"
        online = "—" if feedback is None else f"{sum(feedback['online'])}/{len(feedback['online'])}"
        print(f"RADIUS 实测截面={text(measured)}mm 当前目标截面={text(target)}mm", flush=True)
        print(f"STATUS 阶段={'缠绕' if stage == 'wind' else '滚动'} 已缠绕={progress:g}/{count}节 "
              f"R={radius:g} P={pitch:g} phase={phase:.4f} delta={phase_step:g} wind={wind_step:g} "
              f"顺应性={'开' if compliance_enabled else '关'}（{'管内' if COMPLIANCE_INSIDE else '管外'}） "
              f"反馈={'有效' if usable else '未就绪/无效'} "
              f"在线={online} "
              f"最大跟踪误差={text(error)}deg", flush=True)

    print(f"YAMADA 实机版 execution={execution} R={radius:g}mm P={pitch:g}mm "
          f"顺应性={'开' if compliance_enabled else '关'} 模式={'管内' if COMPLIANCE_INSIDE else '管外'}"
          f"（COMPLIANCE_INSIDE={COMPLIANCE_INSIDE}，请确认）", flush=True)
    print(HELP, flush=True)
    print("启动后不下发任何目标：先在 EXE 手动使能扭矩，再按 Ctrl+J 从头部开始缠绕", flush=True)
    listener = None
    try:
        threading.Thread(target=console_listener, daemon=True).start()
        if hotkeys:
            listener = global_key_listener(command_queue)
        if DEBUG:
            from geometry_debug_view import GeometryDebugView
            debug_view = GeometryDebugView(draw_curve=DEBUG_DRAW_CURVE)
        while True:
            try:
                command = command_queue.get(timeout=.05).strip().lower()
            except queue.Empty:
                command = ""
            if debug_view is not None:
                debug_view.update(exchange)
            if not command:
                continue
            try:
                if command in ("quit", "exit"):
                    break
                if command in ("help", "status"):
                    print(HELP, flush=True) if command == "help" else show_status()
                elif command == "pause":
                    print("已暂停；EXE 保持最后下发的目标。紧急释放请用 EXE 界面", flush=True)
                elif command == "run":
                    print("实机版只支持单步，不提供连续运行", flush=True)
                elif command == "reset":
                    stage, progress, start, compliance, phase = "wind", 0., None, None, 0.
                    print("回到缠绕阶段：缠绕进度清零，下次 Ctrl+J 以当前实测姿态为起点；机器人不动", flush=True)
                elif command == "compliance":
                    controller = None
                    if stage == "roll" and not compliance_enabled:
                        controller = ComplianceController(ComplianceParameters(
                            COMPLIANCE_GAIN, math.atan2(pitch, 2 * math.pi * radius), COMPLIANCE_INSIDE),
                            feedback_links(fresh_feedback(False)))
                    compliance_enabled, compliance = not compliance_enabled, controller
                    show_status()
                elif command in ("delta-", "delta+"):
                    if stage == "wind":
                        wind_step = max(.125, wind_step / 2) if command == "delta-" else min(2., wind_step * 2)
                    else:
                        phase_step = max(.01, phase_step / 2) if command == "delta-" else min(.6, phase_step * 2)
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
                            value = float(value)
                            if key == "r":
                                new_r = value
                            elif key == "p":
                                new_p = value
                            elif key in ("delta", "wind"):
                                if not math.isfinite(value) or value <= 0:
                                    raise ValueError(f"{key} 必须为正数")
                                phase_step, wind_step = (value, wind_step) if key == "delta" else (phase_step, value)
                            else:
                                raise ValueError("参数：r=半径mm p=螺距mm delta=相位步长rad wind=缠绕步长(节)")
                    HelixBackbone.from_radius_pitch(new_r, new_p)
                    radius, pitch = new_r, new_p
                    print(f"参数 R={radius:g}mm P={pitch:g}mm delta={phase_step:g}rad wind={wind_step:g}节；"
                          f"{'下一步缠绕生效' if stage == 'wind' else '滚动阶段：标准 Yamada 下一步生效，顺应性螺距角需 Ctrl+R 后重新 Ctrl+U'}",
                          flush=True)
                elif command in ("step", "back") and stage == "wind":
                    feedback = fresh_feedback(False)  # 直线状态轴不可辨识，缠绕阶段不要求反馈可用
                    count = feedback["config"]["joint_count"]
                    if start is None:
                        start = controller_angles(feedback["angles_deg"])
                    new_progress = min(count, progress + wind_step) if command == "step" else max(0., progress - wind_step)
                    if new_progress == progress:
                        raise ValueError("已缠绕到尾部" if command == "step" else "已退回到起始姿态")
                    helix = HelixBackbone.from_radius_pitch(radius, pitch).joint_angles(count, feedback_links(feedback), 0.)
                    if execute(winding_targets(start, helix, new_progress), feedback, f"缠绕 {new_progress:g}/{count} 节"):
                        progress = new_progress
                elif command == "apply":
                    feedback = fresh_feedback(False)
                    count = feedback["config"]["joint_count"]
                    if targets is None:
                        targets = controller_angles(feedback["angles_deg"])
                    stage, phase = "roll", 0.
                    compliance = None
                    if compliance_enabled:
                        compliance = ComplianceController(ComplianceParameters(
                            COMPLIANCE_GAIN, math.atan2(pitch, 2 * math.pi * radius), COMPLIANCE_INSIDE),
                            feedback_links(feedback))
                    print(f"已确认当前姿态（已缠绕 {progress:g}/{count} 节），进入单步滚动；"
                          f"顺应性={'开' if compliance_enabled else '关'}。本操作不下发目标", flush=True)
                elif command in ("step", "back"):
                    next_phase = phase + (phase_step if command == "step" else -phase_step)
                    if compliance is not None:
                        feedback = fresh_feedback(True)
                        # 首次计算时把相位对齐到最近下发的目标，避免接管瞬间跳变。
                        candidate = compliance.compute(feedback, next_phase, targets, reference_phase=phase)
                        if candidate is None:
                            raise ValueError(compliance.status + "；本步未执行")
                        label = "顺应性"
                    else:
                        feedback = fresh_feedback(False)
                        candidate = HelixBackbone.from_radius_pitch(radius, pitch).joint_angles(
                            feedback["config"]["joint_count"], feedback_links(feedback), next_phase)
                        label = "标准螺旋"
                    if execute(candidate, feedback, f"{label} phase={next_phase:.4f}"):
                        phase = next_phase
                elif command == "hold":
                    if targets is None:
                        raise ValueError("还没有下发过目标")
                    execute(targets, fresh_feedback(False), "重发当前目标")
                else:
                    raise ValueError("未知命令；输入 help 查看操作")
                last_notice = None
            except ValueError as error:
                if str(error) != last_notice:
                    print(f"未执行：{error}", flush=True)
                    last_notice = str(error)
    except KeyboardInterrupt:
        pass
    finally:
        if listener is not None:
            listener.stop()
        if debug_view is not None:
            debug_view.close()
        exchange.close()
        print(f"GEOMETRY_EXCHANGE received={exchange.received} sent={exchange.sent} "
              f"failures={exchange.failures} last_error={exchange.last_error!r}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Pole 实机版：手动缠绕后单步顺应性攀爬")
    parser.add_argument("--execution", choices=("live", "mujoco", "preview"), default="live",
                        help="live 下发舵机（默认）；mujoco 用 MuJoCo 执行端验证本流程；preview 只在 EXE 中预览")
    parser.add_argument("--no-hotkeys", action="store_true", help="只用控制台命令，不启用全局热键")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    main(args.execution, hotkeys=not args.no_hotkeys, port=args.port)
