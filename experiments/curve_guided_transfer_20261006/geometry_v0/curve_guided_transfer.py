"""One ipsilateral five-segment candidate. Lengths mm; angles rad.

Material feed and the spatial curve are distinct. No contact truth, body pose,
formation counters, or stopped-controller migration rules are accepted here.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CurveParameters:
    radius_mm: float = 95.
    pitch_mm: float = 220.
    arc_radius_mm: float = 100.
    bridge_mm: float = 96.75
    link_mm: float = 64.5
    joints: int = 20
    initial_target_links: float = 5.


class FiveSegmentCurve:
    def __init__(self, parameters=CurveParameters()):
        self.p = parameters
        self.alpha = math.atan(parameters.pitch_mm / (2 * math.pi * parameters.radius_mm))
        self.delta = math.pi / 4 - self.alpha
        if self.delta <= 0:
            raise ValueError('This ipsilateral construction needs 0 < alpha < pi/4')
        self.c, self.s = math.cos(self.alpha), math.sin(self.alpha)
        self.w = self.c / parameters.radius_mm
        self.k = self.c ** 2 / parameters.radius_mm
        self.tau = self.s * self.c / parameters.radius_mm
        self.arc = parameters.arc_radius_mm * self.delta
        self.bounds = np.array([0., self.arc, self.arc + parameters.bridge_mm,
                                2 * self.arc + parameters.bridge_mm])
        self.t1 = np.array([self.s, self.c, 0.])
        self.n1 = np.array([self.c, -self.s, 0.])
        self.t2 = np.array([1., 1., 0.]) / math.sqrt(2)
        self.n2 = np.array([1., -1., 0.]) / math.sqrt(2)
        offset = parameters.arc_radius_mm * (math.sin(self.delta) * self.t1 +
                  (1 - math.cos(self.delta)) * self.n1)
        offset4 = parameters.arc_radius_mm * (math.sin(self.delta) * self.t2 +
                  (1 - math.cos(self.delta)) * self.n2)
        a = (offset + parameters.bridge_mm * self.t2 + offset4)[0]
        self.A = np.array([-a, 0., parameters.radius_mm])
        self.B = self.A + offset
        self.C = self.B + parameters.bridge_mm * self.t2
        self.E = self.C + offset4
        self.lambda0 = self.bounds[-1] + parameters.initial_target_links * parameters.link_mm
        self.length = (parameters.joints + 1) * parameters.link_mm

    def sample(self, ell):
        x = np.asarray(ell, float)
        flat = x.reshape(-1)
        result = np.empty((len(flat), 3))
        for i, value in enumerate(flat):
            if value <= 0:
                a = self.w * value
                point = self.A + [self.s * value, self.p.radius_mm * math.sin(a),
                                  self.p.radius_mm * (math.cos(a) - 1)]
            elif value < self.bounds[1]:
                a = value / self.p.arc_radius_mm
                point = self.A + self.p.arc_radius_mm * (math.sin(a) * self.t1 + (1 - math.cos(a)) * self.n1)
            elif value < self.bounds[2]:
                point = self.B + (value - self.bounds[1]) * self.t2
            elif value < self.bounds[3]:
                a = (value - self.bounds[2]) / self.p.arc_radius_mm
                point = self.C + self.p.arc_radius_mm * (math.sin(a) * self.t2 + (1 - math.cos(a)) * self.n2)
            else:
                distance = value - self.bounds[3]
                a = self.w * distance
                point = self.E + [self.p.radius_mm * math.sin(a), self.s * distance,
                                  self.p.radius_mm * (math.cos(a) - 1)]
            result[i] = point
        return result.reshape(x.shape + (3,))

    def frame(self, ell):
        """Global parallel-transport frame; straight has inherited frame, not Frenet division."""
        if ell <= 0:
            a = self.w * ell
            tangent = np.array([self.s, self.c * math.cos(a), -self.c * math.sin(a)])
            normal = np.array([0., -math.sin(a), -math.cos(a)])
            psi = -self.tau * ell
        elif ell < self.bounds[1]:
            a = ell / self.p.arc_radius_mm
            tangent = math.cos(a) * self.t1 + math.sin(a) * self.n1
            normal = -math.sin(a) * self.t1 + math.cos(a) * self.n1
            psi = -math.pi / 2
        elif ell < self.bounds[2]:
            tangent, normal, psi = self.t2, self.n2, -math.pi / 2
        elif ell < self.bounds[3]:
            a = (ell - self.bounds[2]) / self.p.arc_radius_mm
            tangent = math.cos(a) * self.t2 + math.sin(a) * self.n2
            normal = -math.sin(a) * self.t2 + math.cos(a) * self.n2
            psi = -math.pi / 2
        else:
            a = self.w * (ell - self.bounds[3])
            tangent = np.array([self.c * math.cos(a), self.s, -self.c * math.sin(a)])
            normal = np.array([-math.sin(a), 0., -math.cos(a)])
            psi = self.tau * (ell - self.bounds[3])
        binormal = np.cross(tangent, normal)
        first = math.cos(psi) * normal - math.sin(psi) * binormal
        second = math.sin(psi) * normal + math.cos(psi) * binormal
        return np.column_stack((tangent, first, second))

    def integral(self, lo, hi):
        """Bending components for reversed, head-to-tail traversal."""
        result = np.zeros(2)
        for a, b, kind in [(-np.inf, 0., 'source'), (0., self.bounds[1], 'arc'),
                           (self.bounds[2], self.bounds[3], 'arc'),
                           (self.bounds[3], np.inf, 'target')]:
            left, right = max(lo, a), min(hi, b)
            if right <= left:
                continue
            if kind == 'arc':
                result[1] += (right - left) / self.p.arc_radius_mm
            else:
                rate = -self.tau if kind == 'source' else self.tau
                origin = 0. if kind == 'source' else self.bounds[3]
                psi = rate * ((left + right) / 2 - origin)
                amplitude = self.k * (right - left) * np.sinc(rate * (right - left) / (2 * math.pi))
                result += amplitude * np.array([math.cos(psi), -math.sin(psi)])
        return result

    def joint_angles(self, feed_mm, roll_rad=0.):
        # The initial normal gauge is fixed across the whole spatial curve.
        head = self.lambda0 + feed_mm
        nodes = np.arange(self.p.joints + 2) * self.p.link_mm
        b = np.array([self.integral(head - hi, head - lo) for lo, hi in zip(nodes[:-2], nodes[2:])])
        co, sn = math.cos(roll_rad), math.sin(roll_rad)
        b = b @ np.array([[co, -sn], [sn, co]])
        # Each mesh link has a fixed pi rotation around local Z in the XML.
        # Thus odd X axes are -root-X; even -Y axes retain -root-Y.
        return np.where(np.arange(self.p.joints) % 2, b[:, 1], -b[:, 0])

    def nominal_root(self, feed_mm):
        """Offline placement only. This must never update the physical free joint online."""
        head = self.lambda0 + feed_mm
        frame = self.frame(head - self.p.link_mm / 2)
        rotation = np.column_stack((-frame[:, 2], frame[:, 1], frame[:, 0]))
        position = self.sample(head) - self.p.link_mm * rotation[:, 2]
        return position, rotation

    def material_positions(self, feed_mm):
        return self.lambda0 + feed_mm - (np.arange(self.p.joints + 1) + .5) * self.p.link_mm


def feed_schedule(time_s, condition):
    """5 s entry + two 16.125 s feeds at 5 mm/s, 3 s/5 s prescribed holds."""
    if time_s < 5:
        return 0., 'entry'
    t = time_s - 5
    half = 80.625
    duration = half / 5.
    if t < duration:
        feed, state = 5 * t, 'feed_1'
    elif t < duration + 3:
        feed, state = half, 'hold_3s'
    elif t < 2 * duration + 3:
        feed, state = half + 5 * (t - duration - 3), 'feed_2'
    else:
        feed, state = 2 * half, 'hold_5s'
    return (feed if condition == 'B' else 0.), state
