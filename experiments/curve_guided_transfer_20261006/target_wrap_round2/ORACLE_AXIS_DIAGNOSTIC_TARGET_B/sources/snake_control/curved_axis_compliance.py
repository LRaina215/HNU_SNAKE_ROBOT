"""弯曲轴顺应性：把 2023 论文的直轴顺应性扩展到弯曲管轴（论文第 V 节列为未来工作）。长度 mm，角度 rad。

与 pole_compliance.py（论文复现，直轴）并列，不修改它；只复用其中的 Yamada 积分（PiecewiseBackbone）与增益规则。
论文把螺旋分解为 轴 / 截面 / 螺距角：截面逐关节局部变形，轴为全身一根直线，螺距角为全身一个设计值。
本方法保留逐关节的截面变形，把另外两项改为沿身体变化：

1. 局部轴：每个连杆取约一圈的连杆窗口，最小化窗口内螺距角方差（切向协方差最小特征向量），得到轴方向场 d(s)。
2. 局部截面：每个关节只投影到自己的局部法平面，按论文式 (11)–(22) 的 A–D 情形构造直线/圆弧，
   逐关节按 K_i 变形（与 pole_compliance.build_backbone 相同的 (r/K̂, K̂φ)）。
3. 螺距角调制：任意曲线的 Darboux 轴方向 D = (τT+κB)/|·| 满足 D' = β'·u，其中 β = atan(τ/κ)，
   u = cosβ·T − sinβ·B 为截面切向。β 恒定时 D 不变，即论文的直轴螺旋（Lancret 定理）。
   按实测轴方向场的变化逐关节调制 β：Δβ = 2λ(d_{i+1} − d_i)·u_i（D 只能沿 u 转动，一圈内平均只得到一半，故乘 2；
   λ=axis_gain 为复现比例），再去掉均值使平均螺距角保持设计值。只在所有连杆都用一圈窗口辨识出局部轴时调制；
   身体被法兰、三通撑开而退到更大窗口时本步不调制（基线对比中，这类情况下调制会让身体松开而跌落）。目标仍为 κ = cos²β·k、τ = sinβcosβ·k（k 为局部截面曲率），对任意 β(s) 精确成立。

只使用 EXE 反馈中的关节点与机械配置，不使用 EXE 的直轴估计、截面曲线与 feedback_usable
（弯管上直轴置信度低于门限，原方法会拒绝执行）。局部轴全部取同一直轴且不调制时，与原方法目标逐位一致。
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from snake_control.pole_compliance import ComplianceParameters, PiecewiseBackbone, TargetSegment, build_backbone

DEFAULT_GAMMA_MIN = math.radians(1.8)  # 与 EXE 截面曲线默认圆弧阈值相同


@dataclass(frozen=True)
class CurvedAxisOptions:
    # 局部轴窗口（连杆数）：先用约一圈（26 关节、64.5 mm 绕 84 mm 半径约 9 节/圈），
    # 窗口内切向张不开（平整度超过门限，例如身体被法兰或弯头撑开）时逐级加大，最大到全身。
    window_links: tuple = (9, 13, 17, 27)
    smooth_links: int = 2  # 轴方向场再做 ±2 节平均
    max_window_flatness: float = .3  # 窗口内最小/中间特征值之比；全身窗口仍超过 .5 才拒绝执行
    modulation: bool = True  # 第 3 层：按实测轴弯曲调制螺距角；False 只做第 1、2 层
    # 复现实测轴弯曲的比例。1.0 时目标轴完全跟随实测，轴弯曲没有回复趋势，闭环中会越弯越大、身体被撑开后轴不可辨识
    # （bend90 / bend90_tight 6 次中 5 次停滞）；0.5 时目标只复现一半，保留回到直轴的趋势，6/6 通过。
    axis_gain: float = .5
    # 调制只用平滑后的轴方向场：高斯平滑 σ（连杆数），整条身体轴弯曲小于死区时不调制（直管上的估计噪声约 3–7°）
    modulation_sigma: float = 3.
    modulation_deadband_deg: float = 10.
    beta_swing: float = math.radians(15)  # 调制幅度上限 |β−β0|
    # 直管 / 弯管切换（默认关闭）：平滑轴方向场的头尾夹角超过 switch_on 进入弯曲轴模式，低于 switch_off 回到原方法。
    # 直管上该估计的中位数约 7–10°、90% 分位 13–16°（局部窗口噪声），弯管上 90% 分位 20–38°。
    # 对比实验中打开后直管负载降到与原方法相同，但短弯头 4/5、先弯后变径 3/5（关闭时均 5/5），总计 57/80 对 60/80。
    straight_switch: bool = False
    switch_on_deg: float = 20.
    switch_off_deg: float = 12.


def window_axis(block):
    """一段连杆切向的最小螺距角方差方向（协方差最小特征向量，朝前进方向）与平整度。"""
    values, vectors = np.linalg.eigh(np.cov(block.T, bias=True))
    axis = vectors[:, 0]
    return (axis if axis @ block.mean(axis=0) >= 0 else -axis), values[0] / max(values[1], 1e-12)


def local_axes(points, windows, smooth, max_flatness=.3, used=None):
    """各连杆的局部轴方向（单位向量，朝头→尾前进方向）与所用窗口的平整度（越小越可靠）。

    used 为列表时，按连杆顺序追加实际使用的窗口大小。
    """
    links = np.diff(points, axis=0)
    tangents = links / np.linalg.norm(links, axis=1)[:, None]
    count = len(tangents)
    windows = (windows,) if isinstance(windows, int) else tuple(windows)
    axes, flatness = np.empty((count, 3)), np.empty(count)
    for j in range(count):
        for window in windows:
            width = min(window, count)
            start = min(max(j - width // 2, 0), count - width)
            axes[j], flatness[j] = window_axis(tangents[start:start + width])
            if flatness[j] <= max_flatness:
                break
        if used is not None:
            used.append(width)
    if smooth > 0:
        kernel = np.array([axes[max(0, j - smooth):j + smooth + 1].sum(axis=0) for j in range(count)])
        axes = kernel / np.linalg.norm(kernel, axis=1)[:, None]
    return axes, flatness


def local_sections(points, axes, gamma_min, require_bisector=True):
    """逐关节的局部截面：在关节 i 的局部法平面内取前后连杆投影，返回 (a, b, γ, u, d)。

    a、b 为前后投影连杆的一半长度，γ 为绕局部轴 d 的有向转角，u 为截面切向（单位）。
    """
    count = len(points) - 2
    rows = []
    for i in range(1, count + 1):
        d = axes[i - 1] + axes[i]
        d /= np.linalg.norm(d)
        before, after = points[i] - points[i - 1], points[i + 1] - points[i]
        u0, v0 = before - (before @ d) * d, after - (after @ d) * d
        lu, lv = np.linalg.norm(u0), np.linalg.norm(v0)
        if lu < 1e-8 or lv < 1e-8:
            raise ValueError("连杆与局部轴平行，局部截面无效")
        gamma = math.atan2(float(np.cross(u0, v0) @ d), float(u0 @ v0))
        if require_bisector and abs(gamma) > math.pi - 1e-4:
            raise ValueError("局部截面接近折返")
        # Curved-axis pitch modulation needs the bisector; local A-D spans
        # only use lengths and the signed turn. Their normalized circular
        # segments can remain finite as the bisector becomes undefined.
        u = u0 / lu + v0 / lv if require_bisector else None
        rows.append((lu / 2, lv / 2, gamma, u / np.linalg.norm(u) if require_bisector else None, d))
    return rows


def joint_segments(a, b, gamma, winding, gamma_min):
    """论文式 (11)–(22) 的 A–D 情形，与 EXE CrossSectionCurve.Build 相同；返回 [(长度, 半径, 有向转角)]。"""
    g = gamma * winding
    if g <= gamma_min:  # D：外凸或几乎不转，用阈值圆弧近似
        return [(a + b, (a + b) / gamma_min, winding * gamma_min)]
    radius = min(a, b) / math.tan(g / 2)
    result = []
    if a - b > 1e-12:
        result.append((a - b, 0., 0.))
    result.append((radius * g, radius, winding * g))
    if b - a > 1e-12:
        result.append((b - a, 0., 0.))
    return result


def smooth_field(vectors, sigma):
    """沿身体的高斯平滑（端部按截断权重归一化），再归一化为单位向量。"""
    index = np.arange(len(vectors))
    weights = np.exp(-.5 * ((index[:, None] - index[None, :]) / sigma) ** 2)
    smoothed = weights @ vectors
    return smoothed / np.linalg.norm(smoothed, axis=1)[:, None]


def field_bend_deg(joint_axes):
    return math.degrees(math.acos(float(np.clip(joint_axes[0] @ joint_axes[-1], -1, 1))))


def pitch_modulation(sections, base, options, joint_axes):
    """逐关节螺距角：β_i = β0 + Σ 2λ(d_{h+1}−d_h)·ū_h，去均值后均值为 β0。

    d 为平滑后的关节轴方向；全身轴弯曲小于死区时返回常数 β0（直管上与论文一致）。
    """
    if field_bend_deg(joint_axes) < options.modulation_deadband_deg:
        return np.full(len(sections), base)
    deltas = [0.]
    for k in range(len(sections) - 1):
        d0, d1 = joint_axes[k], joint_axes[k + 1]
        mean_d = (d0 + d1) / np.linalg.norm(d0 + d1)
        u = sections[k][3] + sections[k + 1][3]
        u -= (u @ mean_d) * mean_d
        u /= np.linalg.norm(u)
        deltas.append(2 * options.axis_gain * float((d1 - d0) @ u))
    beta = np.cumsum(deltas)
    beta = np.clip(beta - beta.mean(), -options.beta_swing, options.beta_swing)
    return base + beta


def build_curved_backbone(geometry, parameters, options=CurvedAxisOptions()):
    """由实测关节点构造弯曲轴目标；返回 (PiecewiseBackbone, 诊断信息)。"""
    config = geometry["config"]
    count = config["joint_count"]
    gains = parameters.gains(count)
    links = np.array([config["head_length_mm"]] + [j["link_length_mm"] for j in config["joints"]], dtype=float)
    if len(links) != count + 1 or not np.all(np.isfinite(links)) or np.any(links <= 0):
        raise ValueError("反馈连杆长度无效")
    if any(joint["joint_index"] != i + 1 or joint["axis"] != ("Yaw" if i % 2 == 0 else "Pitch")
           for i, joint in enumerate(config["joints"])):
        raise ValueError("关节顺序/轴类型不符合当前交替关节模型")
    points = np.asarray(geometry["points_mm"], dtype=float)
    if points.shape != (count + 2, 3) or not np.all(np.isfinite(points)):
        raise ValueError("反馈关节点无效")
    gamma_min = float((geometry.get("curve") or {}).get("gamma_min_rad") or DEFAULT_GAMMA_MIN)
    used = []
    axes, flatness = local_axes(points, options.window_links, options.smooth_links, options.max_window_flatness, used)
    # 有连杆退到更大窗口（身体被法兰、三通等撑开，局部切向张不开）时轴方向场不可信，本步不调制螺距角
    reliable = all(width == min(options.window_links[0], len(points) - 1) for width in used)
    if flatness.max() > .5:
        raise ValueError(f"局部轴不可辨识（全身窗口平整度 {flatness.max():.2f}）")
    sections = local_sections(points, axes, gamma_min)
    total_turn = sum(row[2] for row in sections)
    if abs(total_turn) < 1e-5:
        raise ValueError("截面内侧方向不明确")
    winding = math.copysign(1., total_turn)
    joint_axes = smooth_field(np.array([row[4] for row in sections]), options.modulation_sigma)
    if options.modulation and reliable:
        beta = pitch_modulation(sections, parameters.pitch_angle, options, joint_axes)
    else:
        beta = np.full(count, parameters.pitch_angle)
    # 分组与 build_backbone 相同：头端半段、各关节、尾端半段；首尾端部沿用相邻关节的局部投影与螺距角。
    groups = [[(sections[0][0], 0., 0.)]]
    groups += [joint_segments(a, b, gamma, winding, gamma_min) for a, b, gamma, _, _ in sections]
    groups.append([(sections[-1][1], 0., 0.)])
    group_beta = np.r_[beta[0], beta, beta[-1]]
    lengths = np.r_[links[0] / 2, (links[:-1] + links[1:]) / 2, links[-1] / 2]
    result = []
    for joint, (group, target_length, b) in enumerate(zip(groups, lengths, group_beta)):
        c, sine = math.cos(b), math.sin(b)
        projected = sum(row[0] for row in group)
        scale = target_length * c / projected  # 式 (25)(28)–(30)，按本组螺距角
        index = joint if 0 < joint <= count else None
        for length, radius, turn in group:
            if radius:
                factor = 1 + (-1 if parameters.inside else 1) * gains[joint - 1]
                radius *= scale / factor  # 与 build_backbone 相同的 (r/K̂, K̂φ)：管外收紧、管内撑开
                turn *= factor
                kappa = c * c / radius
                tau = math.copysign(1, turn) * sine * c / radius  # 轴方向已取头→尾前进方向
            else:
                kappa = tau = 0.
            result.append(TargetSegment(index, scale * length / c, radius, turn, kappa, tau))
    # 采样（Ctrl+K 目标截面）只用平均螺距角近似；控制只用各段 κ、τ。
    axis = {"point_mm": np.zeros(3), "horizontal": np.array([1., 0, 0]), "vertical": np.array([0, 1., 0]),
            "direction": np.array([0, 0, 1.])}
    backbone = PiecewiseBackbone(result, links, parameters.pitch_angle, [0., 0.], [1., 0.], axis, 0., 1.)
    if not math.isclose(backbone.bounds[-1], sum(links), rel_tol=1e-9):
        raise ValueError("归一化后的目标长度不匹配")
    info = dict(axis_bend_deg=field_bend_deg(joint_axes), flatness_max=float(flatness.max()), beta_min=float(beta.min()),
                beta_max=float(beta.max()), axes=axes, points=points, reliable=reliable)
    return backbone, info


class CurvedAxisComplianceController:
    """接口与 pole_compliance.ComplianceController 相同，可直接替换；无效帧返回 None，由调用者保持上次目标。"""

    def __init__(self, parameters, expected_lengths, options=CurvedAxisOptions()):
        self.parameters = parameters if isinstance(parameters, ComplianceParameters) else ComplianceParameters(*parameters)
        self.options = options
        self.expected_lengths = np.asarray(expected_lengths)
        self.backbone = None
        self.info = {}
        self.key = None
        self.phase_offset = 0.
        self.started = False
        self.curved_mode = False  # False：直管，沿用原方法 build_backbone（论文直轴）
        self.status = "等待有效局部截面"

    def compute(self, geometry, roll_phase, reference, now_ms=None, reference_phase=None):
        try:
            if geometry is None:
                raise ValueError("等待新鲜反馈")
            now_ms = time.time() * 1000 if now_ms is None else now_ms
            if not all(geometry.get("online", [])):
                raise ValueError("反馈不可用：存在离线关节")
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
                backbone, info = build_curved_backbone(geometry, self.parameters, self.options)
                curved_mode = True
                if self.options.straight_switch:
                    bend = info['axis_bend_deg']
                    curved_mode = (bend > self.options.switch_on_deg or
                                   (self.curved_mode and bend >= self.options.switch_off_deg))
                    # 直管：与原方法完全相同（EXE 直轴与截面）；EXE 直轴不可用时仍用局部轴目标，不拒绝执行
                    if not curved_mode and geometry.get("feedback_usable", False):
                        backbone = build_backbone(geometry, self.parameters)
                info['curved_mode'] = curved_mode
                phase_offset = self.phase_offset
                if not self.started or key[:2] != self.key[:2]:
                    phase_offset = backbone.align_phase(reference, roll_phase if reference_phase is None else reference_phase)
            else:
                backbone, info, phase_offset = self.backbone, self.info, self.phase_offset
            targets = backbone.joint_angles(roll_phase + phase_offset)
            if not np.all(np.isfinite(targets)):
                raise ValueError("弯曲轴目标无效，保持上一组目标")
            signs = np.where(np.arange(len(targets)) % 2, -1, 1)
            bounds = np.radians([[j["minimum_deg"], j["maximum_deg"]] for j in config["joints"]]) + [.01, -.01]
            low, high = np.sort(bounds * signs[:, None], axis=1).T
            targets = np.clip(targets, low, high)
            self.backbone, self.info, self.phase_offset, self.key = backbone, info, phase_offset, key
            self.curved_mode = info.get('curved_mode', True)
            self.started = True
            self.status = (f"弯曲轴顺应性运行中（{'弯曲轴' if self.curved_mode else '直管·原方法'}）：轴弯曲 {info['axis_bend_deg']:.0f}°，"
                           f"β {math.degrees(info['beta_min']):.0f}–{math.degrees(info['beta_max']):.0f}°")
            return targets
        except (ValueError, KeyError, TypeError) as error:
            self.status = str(error)
            return None
