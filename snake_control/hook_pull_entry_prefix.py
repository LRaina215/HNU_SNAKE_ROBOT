"""One fixed-origin command-entry candidate; no feedback or physical stepping."""
import numpy as np


class FixedPreloadEntry:
    """q_hold + (q_shape(s) - q_shape(0)); no error-state accumulation.

    Exact zero input retains the executed target. Nonzero input does not imply
    preserved clamp force, feasible contact, or a compliance control law.
    """

    def __init__(self, held_target, measured_initial_shape):
        self.held = np.asarray(held_target, dtype=float).copy()
        self.initial_shape = np.asarray(measured_initial_shape, dtype=float).copy()
        if self.held.shape != (20,) or self.initial_shape.shape != (20,):
            raise ValueError('Twenty held targets and measured joints required')
        if not np.isfinite(self.held).all() or not np.isfinite(self.initial_shape).all():
            raise ValueError('Finite entry references required')
        self.held.setflags(write=False)
        self.initial_shape.setflags(write=False)

    def __call__(self, predicted_shape):
        shape = np.asarray(predicted_shape, dtype=float)
        if shape.shape != (20,) or not np.isfinite(shape).all():
            raise ValueError('Twenty finite predicted joint angles required')
        return self.held + (shape - self.initial_shape)
