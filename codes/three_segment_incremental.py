"""Preserve measured spatial shape; apply A-D compliance as a DIFFERENCE.

This is an incremental adaptation of the existing section controller, not the
paper's unchanged absolute reconstruction and not the old K*gamma prototype.
Front/rear roll measured material bending; middle still follows joint angles.
Lengths mm, joint/phase angles rad. FK is supplied by the simulation caller.
"""
import math
import numpy as np

from pole_compliance import ComplianceParameters
from spatial_bishop_backbone import SpatialBishopBackbone
from three_segment_compliance import ThreeSegmentController, LocalSection


def integrated_field(bone, field):
    nodes = np.r_[0., np.cumsum(bone.links)]
    return np.array([bone.integral(lo, hi, field) for lo, hi in zip(nodes[:-2], nodes[2:])])


def rotate(field, phase):
    c, s = math.cos(phase), math.sin(phase)
    return field @ np.array([[c, s], [-s, c]])


def angles_from_integrals(field):
    return -np.where(np.arange(len(field)) % 2, field[:, 0], field[:, 1])


class IncrementalThreeSegmentController(ThreeSegmentController):
    def __init__(self, geometry, parameters, reference, phase, corner, source, target,
                 pipe_radius, *, measured, forward_points, **kwargs):
        super().__init__(geometry, parameters, reference, phase, corner, source, target,
                         pipe_radius, **kwargs)
        cfg = geometry['config']
        self.links = np.array([cfg['head_length_mm']]+[j['link_length_mm'] for j in cfg['joints']])
        self.forward_points = forward_points
        self.initial_phase = phase
        self.zero_parameters = ComplianceParameters(0., parameters.pitch_angle, parameters.inside)
        self.active = self.weights[:, 1] == 0
        observed = self.material_integrals(geometry['points_mm'], measured)
        nominal = self.material_integrals(forward_points(reference), reference)
        # Preserve captured servo preload as two bending components, so its
        # direction rotates with rolling rather than adding a fixed angle bias.
        self.preload = nominal-observed

    def material_integrals(self, points, angles):
        bone = SpatialBishopBackbone(points, self.links)
        field, _ = bone.material_field(angles)
        return integrated_field(bone, field)

    def rolling_observation(self, geometry, measured, observed):
        """Optional front-shape experiment hook; default preserves the baseline."""
        return observed, {}

    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        self.elapsed += dt
        status = self.update_boundaries(np.asarray(geometry['points_mm']), contact_window, dt) if migrate else {}
        desired = self.desired_weights()
        difference = desired-self.weights
        scale = np.minimum(1., (dt/self.options.switch_seconds)/np.maximum(abs(difference).max(axis=1), 1e-12))
        self.weights += difference*scale[:, None]
        middle = self.weights[:, 1] > 0
        self.middle_reference[middle] += -math.expm1(-dt/self.options.follow_seconds)*(
            measured[middle]-self.middle_reference[middle])

        observed = self.material_integrals(geometry['points_mm'], measured)
        delta, total = phase-self.phase, phase-self.initial_phase
        active = desired[:, 1] == 0
        entering = active & ~self.active
        if np.any(entering):
            # A joint admitted from the yielding middle must not regain its
            # obsolete captured preload. Start from its current executed target.
            nominal = self.material_integrals(self.forward_points(self.last_target), self.last_target)
            self.preload[entering] = rotate(nominal-observed, -(self.phase-self.initial_phase))[entering]
        self.active = active
        rolling_field, shape_info = self.rolling_observation(geometry, measured, observed)
        rolled = angles_from_integrals(rotate(rolling_field, delta)+rotate(self.preload, total))
        compliance = np.zeros(self.n)
        local_info = []
        for column, axis in ((0, self.target), (2, self.source)):
            ids = np.flatnonzero(self.weights[:, column] > 1e-12)
            if not len(ids):
                continue
            lo, hi = int(ids[0]), int(ids[-1])+1
            zero = LocalSection(geometry, self.zero_parameters, lo, hi, axis, self.corner)
            shaped = LocalSection(geometry, self.parameters, lo, hi, axis, self.corner)
            # Both maps use the SAME geometry, phase gauge and interval. Their
            # common absolute reconstruction bias cancels when K=0. Rolling is
            # already applied to the actual material field above, not reset by
            # this per-frame alignment of the small section correction.
            gauge = zero.align(measured[lo:hi], 0.)
            compliance[lo:hi] = shaped.angles(gauge+delta)-zero.angles(gauge+delta)
            local_info.append(dict(start=lo, stop=hi, gauge=gauge))
        raw = (self.weights[:, 0]+self.weights[:, 2])*(rolled+compliance)+self.weights[:, 1]*self.middle_reference
        speed = self.options.maximum_speed_rad_s*dt
        limited = self.last_target+np.clip(raw-self.last_target, -speed, speed)
        signs = np.where(np.arange(self.n) % 2, -1., 1.)
        bounds = np.radians([[j['minimum_deg'], j['maximum_deg']] for j in geometry['config']['joints']])
        low, high = np.sort(bounds*signs[:, None], axis=1).T
        target = np.clip(limited, low+.01, high-.01)
        self.last_target, self.phase = target.copy(), phase
        return target, dict(front=self.front, rear=self.rear, middle_max=self.middle_max,
            weights=self.weights.copy(), middle_reference=self.middle_reference.copy(),
            rolled=rolled, compliance_delta=compliance, preload=self.preload.copy(),
            raw=raw, rate_limited=int(np.count_nonzero(abs(raw-limited)>1e-9)),
            angle_limited=int(np.count_nonzero(abs(target-limited)>1e-9)), local_sections=local_info,
            **shape_info, **status)

    def configuration(self):
        return dict(**super().configuration(), method='measured Bishop/Yamada rolling + A-D shaped-minus-zero correction',
                    preload='material bending residual; refresh when entering from middle')
