"""2023 Adaptive Helical Rolling，式 (23)–(33)。长度 mm，角度 rad。

只使用当前实测截面构造目标；不需要扭矩触发，不递归变形上一帧目标。
实际非等长连杆按相邻半连杆分组归一化，是对论文等长 l 的推广。
"""
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class ComplianceParameters:
    # 标量、（首尾各 4 个关节, 中部）二元组，或按关节顺序的 K_i；不按曲线段编号索引。
    gain: object = 0.04
    pitch_angle: float = 0.25
    inside: bool = False

    def gains(self, count):
        gain = self.gain
        if isinstance(gain, tuple) and len(gain) == 2 and count != 2:
            # 论文式 (34)：两端悬臂减小增益，避免端部离管或无约束收紧。
            gain = [gain[0] if i < 4 or i >= count - 4 else gain[1] for i in range(count)]
        values = np.broadcast_to(np.asarray(gain, dtype=float), (count,)).copy()
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("顺应性增益必须非负且有限")
        if self.inside and np.any(values >= 1):
            raise ValueError("管内模式要求 K_i < 1")
        if not math.isfinite(self.pitch_angle) or abs(self.pitch_angle) >= math.pi / 2:
            raise ValueError("螺距角必须满足 |alpha| < pi/2")
        return values


@dataclass(frozen=True)
class TargetSegment:
    """目标曲线的一段：length 为三维弧长 mm，radius 为截面半径 mm，turn 为有向 rad。"""
    joint_index: object
    length: float
    radius: float
    turn: float
    kappa: float
    tau: float


class PiecewiseBackbone:
    def __init__(self, segments, links, pitch, start_xy, tangent_xy, axis, start_z, axial_sign=-1.):
        self.segments = tuple(segments)
        self.links = np.asarray(links, dtype=float)
        self.bounds = np.r_[0., np.cumsum([p.length for p in segments])]
        self.twists = np.r_[0., np.cumsum([p.length * p.tau for p in segments])]
        self.pitch = pitch
        self.start_xy = np.asarray(start_xy, dtype=float)
        self.heading = math.atan2(tangent_xy[1], tangent_xy[0])
        self.axis = axis
        self.start_z = start_z
        self.axial_sign = axial_sign

    def curvature_torsion(self, s):
        s = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > self.bounds[-1] + 1e-8):
            raise ValueError("弧长不在目标曲线范围内")
        indices = np.clip(np.searchsorted(self.bounds, s, side="right") - 1, 0, len(self.segments) - 1)
        return (np.array([p.kappa for p in self.segments])[indices],
                np.array([p.tau for p in self.segments])[indices])

    def joint_angles(self, roll_phase):
        """分段解析积分式 (32)(33)，输出与现有 MuJoCo 同方向的 rad。

        ψ=∫τds+ψroll（原论文为正号）；先得到论文奇 -sin/偶 +cos，
        再做现有桥的奇 +1/偶 -1 映射。不对不连续的 κ/τ 作差分。
        """
        if not math.isfinite(roll_phase):
            raise ValueError("滚动相位必须有限")
        joints = np.r_[0., np.cumsum(self.links)]
        result = np.zeros(len(self.links) - 1)
        for j in range(1, len(self.links)):
            lo, hi = joints[j - 1], joints[j + 1]
            for k, segment in enumerate(self.segments):
                left, right = max(lo, self.bounds[k]), min(hi, self.bounds[k + 1])
                if right <= left or segment.kappa == 0:
                    continue
                span = right - left
                phase = roll_phase + self.twists[k] + segment.tau * ((left + right) / 2 - self.bounds[k])
                amplitude = segment.kappa * span * np.sinc(segment.tau * span / (2 * math.pi))
                result[j - 1] -= amplitude * (math.sin(phase) if j % 2 else math.cos(phase))
        return result

    def align_phase(self, reference, roll_phase):
        """首次接入/配置切换时拟合相位，减少目标跳变；以后不重置滚动相位。"""
        basis = np.column_stack((self.joint_angles(0), self.joint_angles(math.pi / 2)))
        weights = np.linalg.lstsq(basis, reference, rcond=None)[0]
        return math.atan2(weights[1], weights[0]) - roll_phase

    def sample(self, step_mm=5., world=True):
        """串接直线/圆螺旋，提供目标三维曲线；不参加控制主循环的积分。"""
        if not math.isfinite(step_mm) or step_mm <= 0:
            raise ValueError("采样间距必须为正")
        point = np.r_[self.start_xy, self.start_z]
        heading = self.heading
        points = [point.copy()]
        c, sine = math.cos(self.pitch), math.sin(self.pitch)
        for segment in self.segments:
            distances = np.linspace(0, segment.length, max(1, math.ceil(segment.length / step_mm)) + 1)[1:]
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            if segment.radius == 0:
                xy = point[:2] + distances[:, None] * c * tangent
            else:
                angles = segment.turn * distances / segment.length
                radius = math.copysign(segment.radius, segment.turn)
                xy = point[:2] + radius * (np.sin(angles)[:, None] * tangent +
                                          (2 * np.sin(angles / 2) ** 2)[:, None] * normal)
            # 曲线从头走向尾；axial_sign 为尾部相对头部在协议轴上的方向。
            path = np.column_stack((xy, point[2] + self.axial_sign * distances * sine))
            points.extend(path)
            point = path[-1]
            heading += segment.turn
        points = np.asarray(points)
        if world:
            basis = np.column_stack([self.axis[name] for name in ("horizontal", "vertical", "direction")])
            points = points @ basis.T + self.axis["point_mm"]
        return points


def build_backbone(geometry, parameters):
    """保留线/弧顺序，按 joint_index 将同一模块的线和弧共同缩放。"""
    config = geometry["config"]
    count = config["joint_count"]
    gains = parameters.gains(count)
    links = np.array([config["head_length_mm"]] + [j["link_length_mm"] for j in config["joints"]])
    if len(links) != count + 1 or not np.all(np.isfinite(links)) or np.any(links <= 0):
        raise ValueError("反馈连杆长度无效")
    if any(joint["joint_index"] != i + 1 or joint["axis"] != ("Yaw" if i % 2 == 0 else "Pitch")
           for i, joint in enumerate(config["joints"])):
        raise ValueError("关节顺序/轴类型不符合当前交替关节模型")
    source = geometry["curve"]["segments"]
    if not geometry["curve"]["valid"] or len(source) < count + 2:
        raise ValueError("缺少有效连续截面")
    if source[0]["joint_index"] is not None or source[-1]["joint_index"] is not None:
        raise ValueError("缺少首尾端部直线")
    groups = [[] for _ in range(count + 2)]
    previous = 0
    winding = set()
    for index, segment in enumerate(source):
        group = 0 if index == 0 else count + 1 if index == len(source) - 1 else segment["joint_index"]
        if not isinstance(group, int) or not previous <= group <= count + 1:
            raise ValueError("曲线模块顺序无效")
        previous = group
        length, radius, turn = (float(segment[key]) for key in ("length_mm", "radius_mm", "turn_rad"))
        if not all(math.isfinite(x) for x in (length, radius, turn)) or length <= 0 or radius < 0:
            raise ValueError("曲线参数无效")
        if segment["kind"] == "arc":
            if group in (0, count + 1) or radius <= 0 or turn == 0:
                raise ValueError("圆弧模块无效")
            if not math.isclose(radius * abs(turn), length, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("圆弧长度与半径/圆心角不一致")
            winding.add(math.copysign(1, turn))
        elif segment["kind"] != "line" or radius != 0 or turn != 0:
            raise ValueError("直线参数无效")
        groups[group].append((segment, length, radius, turn))
    if any(not group for group in groups) or len(winding) != 1:
        raise ValueError("模块缺失或截面弯曲内侧方向不一致")
    lengths = np.r_[links[0] / 2, (links[:-1] + links[1:]) / 2, links[-1] / 2]
    # 轴估计为保持连续不随姿态翻转，正向不一定指向头部（例如从直线逐节缠绕时）。
    # 挠率符号取决于身体沿轴前进的方向，按实测头尾位置判断，否则会重建出镜像螺旋。
    axis = geometry["projection_axis"]
    ends = np.asarray(geometry["points_mm"], dtype=float)
    axial_sign = math.copysign(1., float(np.dot(ends[-1] - ends[0], axis["direction"])))
    c, sine = math.cos(parameters.pitch_angle), math.sin(parameters.pitch_angle)
    result = []
    for group, target_length in zip(groups, lengths):
        projected_length = sum(row[1] for row in group)
        scale = target_length * c / projected_length  # 式 (25)(28)(29)(30)
        for segment, length, radius, turn in group:
            joint = segment["joint_index"]
            if radius:
                factor = 1 + (-1 if parameters.inside else 1) * gains[joint - 1]
                # 式 (23) 字面 (K̂r, φ/K̂) 在管外 K̂=1+K 时减小转角，闭环实测逐步张开，
                # 与论文正文“管外收紧、管内撑开”相反；按正文取 (r/K̂, K̂φ)，弧长不变。
                radius *= scale / factor
                turn *= factor
                kappa = c * c / radius
                tau = axial_sign * math.copysign(1, turn) * sine * c / radius
            else:
                kappa = tau = 0.
            result.append(TargetSegment(joint, scale * length / c, radius, turn, kappa, tau))
    basis = np.column_stack([axis[name] for name in ("horizontal", "vertical", "direction")])
    origin_data = np.r_[axis["point_mm"], geometry["points_mm"][0], source[0]["start_mm"], source[0]["tangent"]]
    if (basis.shape != (3, 3) or not np.all(np.isfinite(basis)) or
            not np.all(np.isfinite(origin_data)) or
            not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or np.linalg.det(basis) < 0 or
            not math.isclose(np.linalg.norm(source[0]["tangent"]), 1., abs_tol=1e-6)):
        raise ValueError("目标重建坐标基或初始切向无效")
    start_z = float(np.dot(np.asarray(geometry["points_mm"][0]) - axis["point_mm"], axis["direction"]))
    backbone = PiecewiseBackbone(result, links, parameters.pitch_angle,
                                 source[0]["start_mm"], source[0]["tangent"], axis, start_z, axial_sign)
    if not math.isclose(backbone.bounds[-1], sum(links), rel_tol=1e-12):
        raise ValueError("归一化后的目标长度不匹配")
    return backbone


class ComplianceController:
    """新鲜实测反馈驱动；无效帧返回 None，由调用者保持上次目标。

    ignore_feedback_risk 仅供纯仿真探测方法极限：跳过“轴无效或关节离线”这一检查，
    用当前能拿到的几何（即使轴置信度低、关节不全在线）强行继续计算目标。
    默认关闭；绝不应该在驱动真实舵机的入口打开——真实硬件上这意味着把目标发给
    可能已经断线或过期的关节，而且 Python 端无法撤回已经写出的指令。
    """

    def __init__(self, parameters, expected_lengths, ignore_feedback_risk=False):
        self.parameters = parameters
        self.expected_lengths = np.asarray(expected_lengths)
        self.backbone = None
        self.key = None
        self.phase_offset = 0.
        self.started = False
        self.status = "等待有效截面"
        self.ignore_feedback_risk = ignore_feedback_risk

    def compute(self, geometry, roll_phase, reference, now_ms=None, reference_phase=None):
        try:
            if geometry is None:
                raise ValueError("等待新鲜反馈")
            now_ms = time.time() * 1000 if now_ms is None else now_ms
            if not geometry.get("feedback_usable", False) and not self.ignore_feedback_risk:
                raise ValueError("反馈不可用：轴无效或关节离线")
            if geometry["age_ms"] > 500 or now_ms - geometry["sample_unix_ms"] > 500:
                raise ValueError(f"反馈过期：age={geometry['age_ms']:.0f}ms，"
                                 f"距采样 {now_ms - geometry['sample_unix_ms']:.0f}ms")
            config = geometry["config"]
            lengths = [config["head_length_mm"]] + [j["link_length_mm"] for j in config["joints"]]
            if (len(lengths) != len(self.expected_lengths) or
                    not np.allclose(lengths, self.expected_lengths, rtol=0, atol=1e-6)):
                raise ValueError("反馈机械配置不匹配")
            key = (geometry["run_id"], geometry["config_id"], geometry["sequence"])
            if self.key and key[:2] == self.key[:2] and key[2] < self.key[2]:
                raise ValueError("反馈帧倒序")
            if key != self.key:
                backbone = build_backbone(geometry, self.parameters)
                phase_offset = self.phase_offset
                if not self.started or key[:2] != self.key[:2]:
                    # 对齐上一已执行相位，再计算下一步；否则首次接入会抵消本步增量。
                    phase_offset = backbone.align_phase(reference, roll_phase if reference_phase is None else reference_phase)
            else:
                backbone, phase_offset = self.backbone, self.phase_offset
            targets = backbone.joint_angles(roll_phase + phase_offset)
            if not np.all(np.isfinite(targets)):
                raise ValueError("顺应性目标无效，保持上一组目标")
            # 个别无管壁支撑的关节会短暂收紧到极限；按关节限幅继续滚动，而不是整步拒绝（整步停住会失去夹紧）。
            # 可视化方向为奇 +1/偶 -1，范围关于 0 不一定对称，先换算回 MuJoCo 方向。
            signs = np.where(np.arange(len(targets)) % 2, -1, 1)
            bounds = np.radians([[j["minimum_deg"], j["maximum_deg"]] for j in config["joints"]]) + [.01, -.01]
            low, high = np.sort(bounds * signs[:, None], axis=1).T
            targets = np.clip(targets, low, high)
            # 全部角度校验通过才提交状态；坏反馈不污染下一轮骨架和相位。
            self.backbone, self.phase_offset, self.key = backbone, phase_offset, key
            self.started = True
            self.status = "顺应性控制运行中"
            return targets
        except (ValueError, KeyError, TypeError) as error:
            self.status = str(error)
            return None

# 别名：弯曲轴入口（Adaptive_Pole_Climbing_CurvedAxis_Mujoco）运行时会把本模块的
# ComplianceController 换成工厂函数；需要继承/实例化真类的地方（如 local_pitch_runtime）
# 使用这个别名，保证在任何入口下都拿到 2023 控制器类本身。
OriginalComplianceController = ComplianceController
