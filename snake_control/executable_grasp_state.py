"""R1: retained shape equals acknowledged feasible targets, with material gauge.

Encoder FK only. Cloud alignment transports the initially registered axis;
it is a shape hypothesis, not a measurement of world pose or pipe support.
"""
import math
import numpy as np
from snake_control.curve_grasp_revision import RetainedGraspController
from snake_control.three_segment_compliance import LocalSection


class ExecutableGraspController(RetainedGraspController):
    revision = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.last_feed = 0.
        self.pending = None

    def restoring_error(self, retained, dt):
        return np.zeros_like(retained)

    def compute(self, measured, feed, elapsed, dt):
        measured = np.asarray(measured, float)
        if measured.shape != (20,) or not np.isfinite(measured).all():
            raise ValueError('20 finite encoder angles required')
        nominal = self.curve.joint_angles(feed)
        old_nominal = self.curve.joint_angles(self.last_feed)
        retained = self.last - old_nominal
        p = self.forward_points(measured)
        pn = self.forward_points(nominal)
        _, orientation = self.curve.nominal_root(feed)
        orientation = self.registration @ orientation
        centers = self.curve.lambda0 + feed - (np.arange(20) + 1) * self.curve.p.link_mm
        l = self.curve.p.link_mm
        smooth = lambda x: np.clip(x, 0, 1)**2 * (3 - 2 * np.clip(x, 0, 1))
        weights = [smooth((-centers-l)/l), smooth((centers-self.curve.bounds[-1]-l)/l)]
        blend = smooth(elapsed / 5.)
        delta, update = np.zeros(20), np.zeros(20)
        axes = []
        geometry = dict(config=self.config, points_mm=p, curve=dict(gamma_min_rad=math.radians(1)))
        tangent = self.curve.feed_tangent(feed)
        for weight, world_axis in zip(weights, [np.array([1., 0, 0]), np.array([0., 1, 0])]):
            ids = np.flatnonzero(weight > 1e-8)
            if len(ids) < 2:
                continue
            lo, hi = int(ids[0]), int(ids[-1]+1)
            a, b = pn[lo:hi+2].copy(), p[lo:hi+2].copy()
            a -= a.mean(0); b -= b.mean(0)
            u, singular, v = np.linalg.svd(a.T @ b)
            transport = v.T @ np.diag([1, 1, np.linalg.det(v.T @ u.T)]) @ u.T
            rms = float(np.sqrt(np.mean(np.sum((a @ transport.T - b)**2, axis=1))))
            if rms > l/4 or singular[1] / singular[0] < .05:
                raise ValueError('material-cloud coordinate hypothesis invalid: residual or coverage')
            axis = transport @ (orientation.T @ world_axis)
            zero = LocalSection(geometry, self.zero, lo, hi, axis, np.zeros(3))
            shaped = LocalSection(geometry, self.parameters, lo, hi, axis, np.zeros(3))
            gauge = zero.align(measured[lo:hi], 0.)
            difference = shaped.angles(gauge) - zero.angles(gauge)
            delta[lo:hi] = difference
            error = nominal[lo:hi] + retained[lo:hi] - measured[lo:hi]
            error += self.restoring_error(retained[lo:hi], dt)
            change = blend * weight[lo:hi] * (difference - error)
            g = weight[lo:hi] * tangent[lo:hi]
            if float(g@g) > 1e-12:
                change -= g * float(g@change) / float(g@g)
            update[lo:hi] = change
            axes.append(dict(joints=[lo+1, hi], direction_head=axis.tolist(), cloud_rms_mm=rms))
        active = weights[0] + weights[1] > 1e-8
        retained[~active] = 0.
        proposed = retained + update
        raw = nominal + proposed
        limit = math.radians(10) * dt
        bounds = np.radians([[j['minimum_deg'], j['maximum_deg']] for j in self.config['joints']])
        signs = np.where(np.arange(20) % 2, -1., 1.)
        low, high = np.sort(bounds * signs[:, None], axis=1).T
        # Joint box and per-command step box constrain the STATE as well as output.
        target = np.clip(raw, np.maximum(low, self.last-limit), np.minimum(high, self.last+limit))
        feasible_retained = target - nominal
        self.pending = target.copy(), float(feed)
        return target, dict(feed_mm=float(feed), roll_rad=0., nominal_rad=nominal.tolist(),
            compliance_delta_rad=delta.tolist(), proposed_retained_rad=proposed.tolist(),
            retained_shape_rad=feasible_retained.tolist(), effective_residual_rad=feasible_retained.tolist(),
            shape_update_rad=(feasible_retained-retained).tolist(), raw_target_rad=raw.tolist(),
            execution_projection_rad=(target-raw).tolist(), grasp_weights=(weights[0]+weights[1]).tolist(),
            rate_limited_joints=int(np.count_nonzero(abs(target-raw)>1e-10)),
            angle_constrained_joints=int(np.count_nonzero((raw<low)|(raw>high))),
            correction_errors=[], material_axes=axes,
            restoring_time_constant_s=getattr(self, 'restoring_seconds', None),
            feedback_fields=['encoder_angles', 'acknowledged_targets', 'retained_executable_state'],
            limitation='Rigid world drift with unchanged joint angles is unobservable; cloud transport is conditional')

    def accept(self, target, feed):
        if self.pending is None or not np.allclose(target, self.pending[0], rtol=0, atol=1e-12):
            raise ValueError('Only the acknowledged pending target can commit grasp state')
        self.last = np.asarray(target).copy()
        self.last_feed = float(feed)
        self.shape = self.last - self.curve.joint_angles(feed)
        self.pending = None


class EquilibriumGraspController(ExecutableGraspController):
    """R2: finite reference-restoring equilibrium, no repeated free tightening.

    Away from execution boxes, steady updates obey
    P_g(delta - tracking_error - (dt/tau)*r)=0 in each weighted window;
    this is not equality of all components or a proof of whole-body stability.
    Nonzero deformation may remain supported by sustained A-D feedback. tau=.5 s
spans ten 50 ms commands and exceeds the recorded initial contact settling.
It is fixed before the run, not tuned after the result.
"""
    revision = 2
    restoring_seconds = .5

    def restoring_error(self, retained, dt):
        return (dt/self.restoring_seconds) * retained
