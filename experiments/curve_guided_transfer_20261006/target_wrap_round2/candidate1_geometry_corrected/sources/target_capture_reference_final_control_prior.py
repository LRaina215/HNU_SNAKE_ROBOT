"""Finite-module capture appended to a preserved body-material reference.

One feed clock; the saved start and entire first-half reference are unchanged.
Saved rigid transforms are offline diagnostics, never simulated root commands.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from snake_control.local_material_reference import BoundaryMaterialCurve


class TargetCaptureReference(BoundaryMaterialCurve):
    capture_start_mm = 80.625
    capture_end_mm = 161.25

    def __init__(self, path):
        super().__init__(path)
        with np.load(path) as z:
            self.capture_offset = z['capture_offset'].copy()
        if self.capture_offset.shape != (26,) or not np.isfinite(self.capture_offset).all():
            raise ValueError('26 finite capture coordinates required')

    @classmethod
    def capture_phase(cls, feed):
        u = float(np.clip((feed-cls.capture_start_mm)/(cls.capture_end_mm-cls.capture_start_mm), 0, 1))
        return u*u*(3-2*u)

    def state(self, feed):
        x = super().state(feed)
        if feed <= self.capture_start_mm:
            return x
        h = self.capture_phase(feed)
        x[:3] += h*self.capture_offset[:3]
        x[3:6] = (Rotation.from_rotvec(h*self.capture_offset[3:6]) * Rotation.from_rotvec(x[3:6])).as_rotvec()
        x[6:] += h*self.capture_offset[6:]
        return x

    def joint_derivative(self, feed):
        derivative = super().joint_derivative(feed)
        if feed <= self.capture_start_mm:
            return derivative
        span = self.capture_end_mm-self.capture_start_mm
        u = (feed-self.capture_start_mm)/span
        return derivative + 6*u*(1-u)/span*self.capture_offset[6:]

    def feed_tangent(self, feed):
        # Preserve the R2 material-direction prior in the compliance projector.
        # Capture has its own planned shape contribution at the SAME feed clock;
        # that contribution is not protected against compliance adjustment.
        # joint_derivative still includes capture and is used for real timing.
        # Replacing this with dq/dd immediately after a zero-velocity knot would
        # change the projector's limiting direction even at infinitesimal phase.
        return super().feed_tangent(feed)
