"""Position-reference velocity feedback using consecutive encoder readings.

The outer compliant controller still generates the shape. This short-period
correction opposes velocity tracking error during contact unloading. It changes
closed-loop damping, not MuJoCo's mechanical damping or actuator force limit.
"""
import numpy as np


class VelocityFeedbackDelivery:
    def __init__(self, measured, seconds, period, bounds, torque_error=None, filter_seconds=0.,
                 point_jacobians=None, body_damping=0., body_speed_limit=.2, position_gain=None):
        self.previous_measured = np.asarray(measured).copy()
        self.seconds = seconds
        self.period = period
        self.bounds = np.asarray(bounds)
        self.torque_error = torque_error
        self.filter_seconds = filter_seconds
        self.filtered_velocity = np.zeros_like(self.previous_measured)
        self.point_jacobians = point_jacobians
        self.body_damping = body_damping
        self.body_speed_limit = body_speed_limit
        self.position_gain = position_gain

    def compute(self, nominal, nominal_velocity, measured, previous_target, weight):
        measured = np.asarray(measured)
        velocity = (measured-self.previous_measured)/self.period
        prior_velocity = self.filtered_velocity.copy()
        # Backward-Euler low-pass: unfiltered sampled differentiation made
        # successive position corrections alternate at the delivery frequency.
        alpha = self.period/(self.period+self.filter_seconds)
        self.filtered_velocity += alpha*(velocity-self.filtered_velocity)
        correction = -self.seconds*weight*(self.filtered_velocity-nominal_velocity)
        reference = nominal.copy()
        if self.torque_error is not None:
            # Remove position error beyond the current servo's attainable
            # torque BEFORE velocity correction. Otherwise a large saturated
            # position lead can conceal the corrective torque for many samples.
            error = nominal-measured
            reference += weight*(np.clip(error, -self.torque_error, self.torque_error)-error)
        body_info = {}
        body_correction = np.zeros_like(measured)
        if self.point_jacobians is not None:
            jacobians = self.point_jacobians(measured)
            velocities = jacobians @ self.filtered_velocity
            speeds = np.linalg.norm(velocities, axis=1)
            module = int(np.argmax(speeds)); speed = float(speeds[module])
            excess = max(0., 1-self.body_speed_limit/max(speed, 1e-12))
            force = -self.body_damping*excess*velocities[module]
            torque = jacobians[module].T @ force
            body_correction = torque/self.position_gain
            body_info = dict(body_module=module, body_jacobian=jacobians[module],
                body_velocity=velocities[module], body_relative_speed=speed,
                body_damping_torque=torque, body_correction=body_correction,
                body_feedback_power=float(torque @ self.filtered_velocity))
        candidate = reference+correction+body_correction
        limit = np.radians(90)*self.period
        target = previous_target+np.clip(candidate-previous_target, -limit, limit)
        target = np.clip(target, self.bounds[:, 0]+.01, self.bounds[:, 1]-.01)
        info = dict(nominal=nominal.copy(), nominal_velocity=nominal_velocity.copy(),
                    measured=measured.copy(), previous_measured=self.previous_measured.copy(),
                    measured_velocity=velocity, filtered_velocity=self.filtered_velocity.copy(),
                    previous_filtered_velocity=prior_velocity, weight=weight.copy(), reference=reference,
                    correction=correction, target=target.copy(), **body_info)
        self.previous_measured = measured.copy()
        return target, info
