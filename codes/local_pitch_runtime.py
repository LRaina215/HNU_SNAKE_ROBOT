"""Optional local pitch for both interactive entries; no hardware or clock access.

The caller supplies a bounded command interval and commits a proposed controller
only after dispatch succeeds. Waiting, stale feedback and cancelled commands do
not advance the filter. The archived experiments remain unchanged.
"""
import copy
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pole_compliance import ComplianceController, ComplianceParameters, build_backbone
from local_pitch_feedback import estimate_module_radius
from local_pitch_geometry import VariablePitchBackbone

MAX_UPDATE_S = .2
PITCH_RATE_DEG_S = 1.
TRANSFER_S = 2.


def pitch_angles(radius, sign):
    """Archived reconfirmed P(R) rule, preserving the initial helix handedness."""
    p0 = 2 * np.pi * 85.5 * np.tan(np.arctan2(220., 2 * np.pi * 81.5))
    p1 = 2 * np.pi * 199. * np.tan(np.radians(8.))
    mix = np.clip((radius - 85.5) / (199. - 85.5), 0., 1.)
    return sign * np.clip(np.arctan2(p0 + mix * (p1 - p0), 2 * np.pi * radius),
                          np.radians(8.), np.radians(30.))


class InteractivePitchController(ComplianceController):
    def __init__(self, original):
        super().__init__(original.parameters, original.expected_lengths, original.ignore_feedback_risk)
        self._initial_controller = original
        self.backbone, self.key = original.backbone, original.key
        self.phase_offset, self.started = original.phase_offset, original.started
        self.requested = True
        self.standard_return = False
        self.alpha_state = np.full(len(self.expected_lengths) + 1, self.parameters.pitch_angle)
        self.radius_state = np.full(len(self.alpha_state), np.nan)
        self.radius_age = np.full(len(self.alpha_state), np.inf)
        self.transition_basis = None
        self.transition_elapsed = 0.
        self.standard_elapsed = 0.
        self.ready = False
        self.geometry = None
        self.last_sample_ms = None
        self.status = '局部螺距已开启，等待下一步'

    def capture_transition(self, reference, phase):
        if self.backbone is None:
            raise ValueError('请先完成一次有效的顺应性滚动，再开启局部螺距')
        basis = np.array([self.backbone.joint_angles(self.phase_offset),
                          self.backbone.joint_angles(self.phase_offset + np.pi / 2)])
        weights = np.array([math.cos(phase), math.sin(phase)])
        basis += weights[:, None] * (np.asarray(reference) - weights @ basis)
        self.transition_basis = basis
        self.transition_elapsed = 0.

    @property
    def mode_text(self):
        state = '开' if self.requested else '关（返回标准步态中）' if self.standard_return else '关（恢复设计角中）'
        lo, hi = np.degrees([self.alpha_state.min(), self.alpha_state.max()])
        return f'{state} α={lo:.1f}～{hi:.1f}deg'

    def compute(self, geometry, roll_phase, reference, now_ms=None, reference_phase=None,
                step_s=.2, standard_target=None):
        try:
            if not math.isfinite(step_s) or step_s <= 0:
                raise ValueError('局部螺距控制步长必须为正数')
            dt = min(step_s, MAX_UPDATE_S)
            validator = ComplianceController(self.parameters, self.expected_lengths, self.ignore_feedback_risk)
            if validator.compute(geometry, roll_phase, reference, now_ms, reference_phase) is None:
                raise ValueError(validator.status)
            key = (geometry['run_id'], geometry['config_id'], geometry['sequence'])
            if self.key and key[:2] != self.key[:2]:
                raise ValueError('反馈会话或机械配置已改变，请重置后重新接管')
            if self.key and key[2] < self.key[2]:
                raise ValueError('反馈帧倒序')
            sample_ms = float(geometry['sample_unix_ms'])
            feedback_dt = 0. if self.last_sample_ms is None else max(0., (sample_ms - self.last_sample_ms) / 1000.)
            radius, age = self.radius_state.copy(), self.radius_age + feedback_dt
            requested = np.full_like(self.alpha_state, self.parameters.pitch_angle)
            if self.requested:
                rows = estimate_module_radius(geometry['points_mm'], geometry['projection_axis'], 7)['nodes']
                if len(rows) != len(requested):
                    raise ValueError('局部半径节点与机械配置不匹配')
                valid = np.array([x['valid'] for x in rows])
                radius[valid] = [x['radius_mm'] for x in rows if x['valid']]
                age[valid] = 0.
                usable = np.isfinite(radius) & (age <= .5)
                requested[usable] = pitch_angles(radius[usable], math.copysign(1., self.parameters.pitch_angle))
                weights = gaussian_filter1d(usable.astype(float), 1., mode='nearest')
                values = gaussian_filter1d(np.where(usable, requested, 0.), 1., mode='nearest')
                filtered = np.divide(values, weights, out=requested.copy(), where=weights > 1e-12)
                requested = np.where(usable, filtered, self.parameters.pitch_angle)
            delta = (1. - .8 ** (dt / .2)) * (requested - self.alpha_state) if self.requested else requested - self.alpha_state
            limit = math.radians(PITCH_RATE_DEG_S) * dt
            alpha = self.alpha_state + np.clip(delta, -limit, limit)
            at_design = not self.requested and np.max(abs(alpha - self.parameters.pitch_angle)) < 1e-12
            if at_design:
                alpha[:] = self.parameters.pitch_angle
            bone = (build_backbone(geometry, ComplianceParameters(self.parameters.gain, float(alpha[0]), self.parameters.inside))
                    if np.ptp(alpha) < 1e-14 else
                    VariablePitchBackbone(geometry, self.parameters, alpha, np.r_[0., np.cumsum(self.expected_lengths)]))
            raw = bone.joint_angles(roll_phase + self.phase_offset)
            elapsed = min(TRANSFER_S, self.transition_elapsed + dt)
            if self.transition_basis is not None and elapsed < TRANSFER_S:
                moving = np.array([math.cos(roll_phase), math.sin(roll_phase)]) @ self.transition_basis
                raw = moving + (elapsed / TRANSFER_S) * (raw - moving)
            standard_elapsed = self.standard_elapsed
            if at_design and self.standard_return:
                if standard_target is None:
                    raise ValueError('关闭顺应性时缺少标准步态目标')
                standard_elapsed = min(TRANSFER_S, standard_elapsed + dt)
                raw = raw + (standard_elapsed / TRANSFER_S) * (np.asarray(standard_target) - raw)
            signs = np.where(np.arange(len(raw)) % 2, -1, 1)
            bounds = np.radians([[j['minimum_deg'], j['maximum_deg']] for j in geometry['config']['joints']]) + [.01, -.01]
            low, high = np.sort(bounds * signs[:, None], axis=1).T
            target = np.clip(raw, low, high)
            if not np.isfinite(target).all() or np.max(abs(target - reference)) > math.radians(30.):
                raise ValueError('局部螺距目标变化超过30°或无效；保持上一目标')
            self.backbone, self.key, self.geometry = bone, key, geometry
            self.last_sample_ms = sample_ms
            self.alpha_state, self.radius_state, self.radius_age = alpha, radius, age
            self.transition_elapsed, self.standard_elapsed = elapsed, standard_elapsed
            self.ready = at_design and elapsed >= TRANSFER_S and (not self.standard_return or standard_elapsed >= TRANSFER_S)
            self.started = True
            self.status = '局部螺距：' + self.mode_text
            return target
        except (ValueError, KeyError, TypeError) as error:
            self.status = str(error)
            return None

    def settled(self):
        if not self.ready:
            return self
        if self.standard_return:
            return None
        base = ComplianceController(self.parameters, self.expected_lengths, self.ignore_feedback_risk)
        for name in ('backbone', 'key', 'phase_offset', 'started'):
            setattr(base, name, getattr(self, name))
        base.status = '顺应性控制运行中；局部螺距已关闭'
        return base


def local_pitch_status(controller):
    return controller.mode_text if isinstance(controller, InteractivePitchController) else '关'


def toggle_local_pitch(controller, compliance_enabled, reference, phase):
    if not compliance_enabled or controller is None or not controller.started:
        raise ValueError('请先开启顺应性并完成一次有效滚动，再按 Ctrl+L')
    if controller.parameters.inside:
        raise ValueError('局部螺距当前仅支持管外模式')
    if abs(controller.parameters.pitch_angle) < 1e-8:
        raise ValueError('局部螺距需要非零设计Pitch角')
    if isinstance(controller, InteractivePitchController):
        changed = copy.copy(controller)
        changed.requested = not controller.requested
        changed.ready = False
        return changed
    changed = InteractivePitchController(controller)
    changed.capture_transition(reference, phase)
    return changed


def stop_local_pitch(controller):
    """Ctrl+E off: keep the current controller until a gradual standard return."""
    if not isinstance(controller, InteractivePitchController):
        return None
    changed = copy.copy(controller)
    changed.requested, changed.standard_return, changed.ready = False, True, False
    return changed


def resume_compliance(controller):
    changed = copy.copy(controller)
    changed.standard_return = False
    changed.standard_elapsed = 0.
    changed.ready = False
    return changed


def prepare_pitch_step(controller, geometry, phase, reference, reference_phase, step_s, standard_target):
    if not isinstance(controller, InteractivePitchController):
        return controller.compute(geometry, phase, reference, reference_phase=reference_phase), controller
    proposed = copy.copy(controller)
    target = proposed.compute(geometry, phase, reference, reference_phase=reference_phase,
                              step_s=step_s, standard_target=standard_target)
    if target is None:
        controller.status = proposed.status
    return target, proposed
