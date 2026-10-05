"""Shared controller: copied encoders + registered prior + commanded material feed.

The forward_points callback is kinematics only, in a fixed head frame. It must
not expose the simulator's free pose, contacts, velocities or hidden states.
"""
import math
import numpy as np

from snake_control.pole_compliance import ComplianceParameters
from snake_control.three_segment_compliance import LocalSection


class CurveTransferController:
    def __init__(self, curve, config, reference, forward_points, registered_rotation):
        self.curve, self.config = curve, config
        self.forward_points = forward_points
        self.reference = np.asarray(reference).copy()
        self.last = self.reference.copy()
        self.parameters = ComplianceParameters((.01, .02), curve.alpha, False)
        self.zero = ComplianceParameters(0., curve.alpha, False)
        _, nominal = curve.nominal_root(0.)
        self.registration = registered_rotation @ nominal.T

    def compute(self, measured, feed, elapsed, dt):
        measured = np.asarray(measured).copy()
        if measured.shape != (20,) or not np.all(np.isfinite(measured)):
            raise ValueError('need 20 finite fresh encoders')
        p = self.forward_points(measured)
        # Nominal head attitude from commanded feed; not an observed global attitude.
        # Its possible drift is an explicit limitation, not corrected by simulation truth.
        _, orientation = self.curve.nominal_root(feed)
        orientation = self.registration @ orientation
        head = self.curve.lambda0 + feed
        centers = head - (np.arange(20) + 1) * self.curve.p.link_mm
        regions = [(centers < -self.curve.p.link_mm, np.array([1., 0., 0.])),
                   (centers > self.curve.bounds[-1] + self.curve.p.link_mm, np.array([0., 1., 0.]))]
        delta = np.zeros(20)
        errors = []
        geometry = dict(config=self.config, points_mm=p,
                        curve=dict(gamma_min_rad=math.radians(1.)))
        for mask, world_axis in regions:
            ids = np.flatnonzero(mask)
            if len(ids) < 2:
                continue
            lo, hi = int(ids[0]), int(ids[-1] + 1)
            axis = orientation.T @ world_axis
            try:
                zero = LocalSection(geometry, self.zero, lo, hi, axis, np.zeros(3))
                shaped = LocalSection(geometry, self.parameters, lo, hi, axis, np.zeros(3))
                gauge = zero.align(measured[lo:hi], 0.)
                delta[lo:hi] = shaped.angles(gauge) - zero.angles(gauge)
            except ValueError as error:
                errors.append(str(error))
                # Explicitly omit invalid local correction, keep the motion reference.
        delta = np.clip(delta, -math.radians(1.), math.radians(1.))
        nominal = self.curve.joint_angles(feed)
        blend = min(1., max(0., elapsed / 5.))
        blend = blend * blend * (3 - 2 * blend)
        raw = (1 - blend) * self.reference + blend * (nominal + delta)
        if np.degrees(abs(raw)).max() > 90:
            raise ValueError('raw target exceeds joint limit; no hard angle clipping')
        maximum = math.radians(10.) * dt
        target = self.last + np.clip(raw - self.last, -maximum, maximum)
        limited = int(np.count_nonzero(abs(target - raw) > 1e-10))
        self.last = target.copy()
        return target, dict(feed_mm=feed, roll_rad=0., nominal_rad=nominal.tolist(),
            compliance_delta_rad=delta.tolist(), correction_errors=errors, rate_limited_joints=limited,
            feedback_fields=['encoder_angles', 'last_sent_targets'],
            limitation='axes are propagated by commanded curve attitude, not measured global motion')
