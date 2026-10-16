"""Servo telemetry and paired perturbation scores; no environment observations.

The signed load_fraction channel is normalized servo load, not contact force.
A simulator adapter may divide actuator torque by the known servo torque limit;
real hardware needs its own calibrated conversion. All angles are radians.
"""
from dataclasses import dataclass
import numpy as np


SIGNAL_KIND = 'normalized_servo_load'


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (20,) or not np.all(np.isfinite(array)):
        raise ValueError(name + ' must contain 20 finite values')
    # Immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(array.tobytes(), dtype=float)


def _rms(value):
    return float(np.sqrt(np.mean(np.asarray(value) ** 2)))


def _sample(value):
    if type(value) is not ServoObservation:
        raise TypeError('expected exactly ServoObservation, with no extra input fields')
    return value


@dataclass(frozen=True)
class ServoObservation:
    """Only actual angles, actually sent angles, time, and signed servo load."""
    __slots__ = ('t', 'q', 'sent', 'load_fraction')
    t: float
    q: np.ndarray
    sent: np.ndarray
    load_fraction: np.ndarray

    def __post_init__(self):
        if not np.isfinite(float(self.t)):
            raise ValueError('time must be finite')
        object.__setattr__(self, 't', float(self.t))
        for name in ('q', 'sent', 'load_fraction'):
            object.__setattr__(self, name, _vector(getattr(self, name), name))

    @property
    def signal_kind(self):
        return SIGNAL_KIND


class ServoObserver:
    def __init__(self, tracking_threshold_rad=np.deg2rad(2.), load_threshold=.10,
                 persistent_s=.20, baseline_tau_s=.4, calibration_stable_s=.5,
                 calibration_velocity_limit_rad_s=np.deg2rad(1.)):
        self.tracking_threshold_rad = float(tracking_threshold_rad)
        self.load_threshold = float(load_threshold)
        self.persistent_s = float(persistent_s)
        self.baseline_tau_s = float(baseline_tau_s)
        self.calibration_stable_s = float(calibration_stable_s)
        self.calibration_velocity_limit_rad_s = float(calibration_velocity_limit_rad_s)
        if min(self.tracking_threshold_rad, self.load_threshold, self.persistent_s,
               self.baseline_tau_s, self.calibration_stable_s,
               self.calibration_velocity_limit_rad_s) <= 0:
            raise ValueError('observer thresholds and durations must be positive')
        self._baseline_error = np.zeros(20)
        self._baseline_load = np.zeros(20)
        self.baseline_ready = False
        self._last = None
        self._calibration_last = None
        self._calibration_age = 0.
        self._calibration_initialized = False
        self._return_reference = None
        self._paired_response_support = False
        self._ages = dict(front_tracking=0., rear_tracking=0., front_load=0., rear_load=0., interaction=0.)

    @property
    def baseline(self):
        return dict(error_rad=_vector(self._baseline_error, 'baseline error'),
                    load_fraction=_vector(self._baseline_load, 'baseline load'))

    def set_baseline(self, error_rad, load_fraction):
        self._baseline_error = _vector(error_rad, 'baseline error').copy()
        self._baseline_load = _vector(load_fraction, 'baseline load').copy()
        self.baseline_ready = True
        self._calibration_initialized = True
        self._ages = dict.fromkeys(self._ages, 0.)

    def calibrate(self, sample):
        """Call only during a nominated baseline hold; update() never adapts it."""
        sample = _sample(sample)
        previous = self._calibration_last
        steady = False
        if previous is not None:
            dt = sample.t - previous.t
            if dt <= 0:
                raise ValueError('calibration time must strictly increase')
            speed = max(float(np.max(np.abs(sample.q - previous.q))),
                        float(np.max(np.abs(sample.sent - previous.sent)))) / dt
            steady = speed <= self.calibration_velocity_limit_rad_s
            if steady:
                if not self._calibration_initialized:
                    self._baseline_error = (sample.sent - sample.q).copy()
                    self._baseline_load = sample.load_fraction.copy()
                    self._calibration_initialized = True
                alpha = 1. - np.exp(-dt / self.baseline_tau_s)
                self._baseline_error += alpha * (sample.sent - sample.q - self._baseline_error)
                self._baseline_load += alpha * (sample.load_fraction - self._baseline_load)
                self._calibration_age += dt
                if self._calibration_age >= self.calibration_stable_s:
                    self.baseline_ready = True
            else:
                self._calibration_age = 0.
        self._calibration_last = sample
        return dict(baseline_ready=self.baseline_ready, calibration_steady=bool(steady),
                    calibration_age_s=float(self._calibration_age), **self.baseline)

    def set_return_reference(self, sample):
        """Record the pre-probe state; requires sent commands to return as well."""
        self._return_reference = _sample(sample)

    def set_paired_response_support(self, supported):
        """Caller clears this flag when its paired experiment is no longer current."""
        self._paired_response_support = bool(supported)

    def update(self, sample):
        sample = _sample(sample)
        dt = 0. if self._last is None else sample.t - self._last.t
        if self._last is not None and dt <= 0:
            raise ValueError('observation time must strictly increase')
        velocity = np.zeros(20) if self._last is None else (sample.q - self._last.q) / dt
        sent_velocity = np.zeros(20) if self._last is None else (sample.sent - self._last.sent) / dt
        error = sample.sent - sample.q
        residual = error - self._baseline_error
        load_change = sample.load_fraction - self._baseline_load
        metrics = dict(front_tracking=_rms(residual[:8]), rear_tracking=_rms(residual[-8:]),
                       front_load=_rms(load_change[:8]), rear_load=_rms(load_change[-8:]))
        flags = {key: self.baseline_ready and value > (self.load_threshold if 'load' in key else self.tracking_threshold_rad)
                 for key, value in metrics.items()}
        flags['interaction'] = bool(flags['front_load'] and (flags['front_tracking'] or self._paired_response_support))
        for key in self._ages:
            self._ages[key] = self._ages[key] + dt if flags[key] else 0.
        result = dict(t=sample.t, signal_kind=SIGNAL_KIND, baseline_ready=self.baseline_ready,
                      velocity_valid=self._last is not None, joint_velocity_rad_s=velocity,
                      sent_velocity_rad_s=sent_velocity, tracking_error_rad=error,
                      baseline_error_deviation_rad=residual, baseline_load_deviation=load_change,
                      front_error_deviation_rms_rad=metrics['front_tracking'],
                      rear_error_deviation_rms_rad=metrics['rear_tracking'],
                      front_load_deviation_rms=metrics['front_load'], rear_load_deviation_rms=metrics['rear_load'],
                      front_error_deviation_max_rad=float(np.max(np.abs(residual[:8]))),
                      rear_error_deviation_max_rad=float(np.max(np.abs(residual[-8:]))),
                      front_load_deviation_max=float(np.max(np.abs(load_change[:8]))),
                      rear_load_deviation_max=float(np.max(np.abs(load_change[-8:]))),
                      load_near_nominal_limit=bool(np.any(np.abs(sample.load_fraction) >= .95)),
                      front_interaction_candidate=flags['interaction'],
                      front_interaction_persistent=bool(self._ages['interaction'] >= self.persistent_s),
                      paired_response_support=self._paired_response_support,
                      interpretation='servo response change; does not identify contact, support, or pipe axis')
        for key, age in self._ages.items():
            result[key + '_anomaly_age_s'] = float(age)
            result[key + '_persistent'] = bool(age >= self.persistent_s)
        result.update(_return_metrics([sample], self._return_reference, self.baseline,
                                      np.ones(20, dtype=bool), self.tracking_threshold_rad, self.load_threshold))
        self._last = sample
        return result


def _read_baseline(baseline):
    if set(baseline) != {'error_rad', 'load_fraction'}:
        raise ValueError('baseline must contain exactly error_rad and load_fraction')
    return _vector(baseline['error_rad'], 'baseline error'), _vector(baseline['load_fraction'], 'baseline load')


def _return_metrics(samples, reference, baseline, affected, angle_tolerance, load_tolerance):
    if reference is None or not samples:
        return dict(return_available=False, return_recovered=False, return_cost=None)
    error0, load0 = _read_baseline(baseline)
    q = np.asarray([s.q for s in samples])
    sent = np.asarray([s.sent for s in samples])
    load = np.asarray([s.load_fraction for s in samples])
    tracking = _rms((sent - q - error0)[:, affected])
    load_change = _rms((load - load0)[:, affected])
    displacement = _rms((q - reference.q)[:, affected])
    sent_displacement = _rms((sent - reference.sent)[:, affected])
    cost = max(tracking / angle_tolerance, displacement / angle_tolerance,
               sent_displacement / angle_tolerance, load_change / load_tolerance)
    return dict(return_available=True, return_recovered=bool(cost <= 1.), return_cost=float(cost),
                return_tracking_residual_rms_rad=tracking, return_load_deviation_rms=load_change,
                return_q_displacement_rms_rad=displacement, return_sent_displacement_rms_rad=sent_displacement)


def score_probe(samples, direction, baseline, hold_start, hold_end,
                return_start=None, reference=None, recovery_tail_s=.15,
                angle_tolerance_rad=np.deg2rad(2.), load_tolerance=.10):
    """Score one signed joint-command direction using the explicitly held interval.

    direction is a 20-vector obtained from the curvature-to-joint mapping. It
    identifies a commanded perturbation, never a world direction or pipe axis.
    baseline is ServoObserver.baseline; sent must be the post-slew command.
    """
    samples = [_sample(s) for s in samples]
    if not samples or any(b.t <= a.t for a, b in zip(samples, samples[1:])):
        raise ValueError('probe samples must be nonempty and strictly time ordered')
    if not np.isfinite(hold_start) or not np.isfinite(hold_end) or hold_end <= hold_start:
        raise ValueError('hold_end must follow hold_start')
    if recovery_tail_s <= 0 or angle_tolerance_rad <= 0 or load_tolerance <= 0:
        raise ValueError('recovery duration and tolerances must be positive')
    reference = samples[0] if reference is None else _sample(reference)
    vector = _vector(direction, 'probe direction')
    if np.linalg.norm(vector) < 1e-10:
        raise ValueError('probe direction cannot be zero')
    vector = vector / np.linalg.norm(vector)
    affected = np.abs(vector) > 1e-8
    error0, load0 = _read_baseline(baseline)
    hold = [s for s in samples if hold_start <= s.t <= hold_end]
    result = dict(signal_kind=SIGNAL_KIND, direction=vector.tolist(), hold_sample_count=len(hold),
                  hold_duration_s=float(hold_end - hold_start), response_available=False,
                  interpretation='paired servo tracking preference; not pipe-axis identification')
    if hold:
        q = np.asarray([s.q for s in hold])
        sent = np.asarray([s.sent for s in hold])
        load = np.asarray([s.load_fraction for s in hold])
        command = float(np.median((sent - reference.sent) @ vector))
        response = float(np.median((q - reference.q) @ vector))
        valid = bool(len(hold) >= 2 and command > 1e-6)
        residual = sent - q - error0
        change = load - load0
        result.update(response_available=valid, commanded_response_rad=command,
                      observed_response_rad=response, tracking_response_ratio=response / command if valid else None,
                      tracking_residual_projection_rad=float(np.median(residual @ vector)),
                      tracking_residual_rms_rad=_rms(residual[:, affected]),
                      incremental_load_projection=float(np.median(change @ vector)),
                      incremental_load_rms=_rms(change[:, affected]),
                      commanded_hold_drift_rad=float(np.ptp((sent - reference.sent) @ vector)))
    returned = [] if return_start is None else [s for s in samples if s.t >= max(return_start, samples[-1].t - recovery_tail_s)]
    result.update(_return_metrics(returned, reference, baseline, affected, angle_tolerance_rad, load_tolerance))
    return result


class ProbeScoreAccumulator:
    """Convenience wrapper; add only ServoObservation, then call score()."""
    def __init__(self, direction, baseline, hold_start, hold_end, return_start=None, reference=None):
        error, load = _read_baseline(baseline)
        self.parameters = dict(direction=_vector(direction, 'probe direction'),
                               baseline=dict(error_rad=error, load_fraction=load),
                               hold_start=hold_start, hold_end=hold_end,
                               return_start=return_start, reference=reference)
        self.samples = []

    def add(self, sample):
        sample = _sample(sample)
        if self.samples and sample.t <= self.samples[-1].t:
            raise ValueError('probe time must strictly increase')
        self.samples.append(sample)

    def score(self):
        return score_probe(self.samples, **self.parameters)


def compare_probe_pair(positive, negative, response_difference=.20, load_difference=.08):
    """Compare opposite held probes; asymmetry is evidence, not a contact label."""
    valid = positive.get('response_available', False) and negative.get('response_available', False)
    reason = 'missing held response'
    if valid:
        a, b = positive['commanded_response_rad'], negative['commanded_response_rad']
        ta, tb = positive['hold_duration_s'], negative['hold_duration_s']
        opposite = np.dot(positive['direction'], negative['direction']) < -.95
        equal_amplitude = abs(a - b) <= .20 * max(a, b)
        equal_duration = abs(ta - tb) <= .20 * max(ta, tb)
        held = max(positive['commanded_hold_drift_rad'] / a, negative['commanded_hold_drift_rad'] / b) <= .15
        valid = bool(opposite and equal_amplitude and equal_duration and held)
        reason = 'comparable opposite holds' if valid else 'directions, amplitudes, durations, or held commands differ'
    result = dict(pair_comparable=bool(valid), reason=reason, paired_response_support=False,
                  preferred_probe=None, interpretation='relative servo response preference; no world geometry estimate')
    if not valid:
        return result
    rp, rn = positive['tracking_response_ratio'], negative['tracking_response_ratio']
    lp, ln = positive['incremental_load_projection'], negative['incremental_load_projection']
    weaker = positive if rp < rn else negative
    freer = negative if rp < rn else positive
    load_corrob = weaker['incremental_load_projection'] - freer['incremental_load_projection'] >= load_difference
    residual_corrob = weaker['tracking_residual_rms_rad'] - freer['tracking_residual_rms_rad'] >= np.deg2rad(.5)
    asymmetry = abs(rp - rn) >= response_difference
    support = bool(asymmetry and (load_corrob or residual_corrob))
    preferred = ('negative' if rp < rn else 'positive') if support and freer.get('return_recovered', False) else None
    result.update(paired_response_support=support, preferred_probe=preferred,
                  response_ratio_difference=float(rp - rn), incremental_load_difference=float(lp - ln),
                  both_return_recovered=bool(positive.get('return_recovered', False) and negative.get('return_recovered', False)))
    return result
