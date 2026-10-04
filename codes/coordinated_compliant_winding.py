"""Measured-angle target lead and phase-linked formation for the crossing trial.

This changes position references, never the physical servo gain/torque limit.
The known-axis/contact scheduler is retained for this first, mechanical stage.
"""
from dataclasses import dataclass, asdict, replace
import math
import numpy as np

from traveling_compliant_winding import TravelingCompliantWinding


@dataclass(frozen=True)
class CoordinationOptions:
    target_lead_deg: float = 8.
    phase_feedback: bool = False


class CoordinatedCompliantWinding(TravelingCompliantWinding):
    def __init__(self, *args, coordination=CoordinationOptions(), **kwargs):
        super().__init__(*args, **kwargs)
        self.coordination = coordination
        self.requested_phase = self.phase
        self.progress_scale = 1.
        self.nominal_formation_speed = self.formation.joints_per_second

    def update_boundaries(self, points, contact_window, dt):
        # Formation follows the effective rolling rate. Contact/release timers,
        # middle yielding and membership smoothing still use physical dt.
        self.formation = replace(self.formation,
            joints_per_second=self.nominal_formation_speed*self.progress_scale)
        return super().update_boundaries(points, contact_window, dt)

    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        previous = self.last_target.copy()
        error = np.abs(previous-measured)
        requested_delta = phase-self.requested_phase
        self.requested_phase = phase
        if self.coordination.phase_feedback:
            # Slow continuous phase advancement when the active shape is not
            # following. A nonzero rate permits contact-guided rearrangement.
            active = (self.weights[:, 0]+self.weights[:, 2]) > .5
            lag = float(np.quantile(error[active], .8)) if np.any(active) else 0.
            budget = math.radians(self.coordination.target_lead_deg)
            self.progress_scale = float(np.clip(1-lag/budget, .15, 1.))
        effective_phase = self.phase+requested_delta*self.progress_scale
        candidate, info = super().compute(geometry, measured, effective_phase, dt,
                                          contact_window, migrate)
        budget = math.radians(self.coordination.target_lead_deg)
        conditioned = measured+np.clip(candidate-measured, -budget, budget)
        speed = self.options.maximum_speed_rad_s*dt
        output = previous+np.clip(conditioned-previous, -speed, speed)
        # The captured command can already exceed the error budget. Respect
        # the old rate limit while unloading it; do not reset the saved state.
        self.last_target = output.copy()
        info.update(unconditioned_target=candidate, conditioned_target=conditioned,
                    target_lead_limited=int(np.count_nonzero(abs(candidate-conditioned)>1e-9)),
                    progress_scale=self.progress_scale, effective_phase=effective_phase,
                    requested_phase=phase)
        return output, info

    def configuration(self):
        return dict(**super().configuration(), coordination=asdict(self.coordination))
