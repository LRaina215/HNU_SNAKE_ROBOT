"""Offline variable-pitch prototype; no serial port or simulation entry point.

Lift the baseline's normalized planar turn field with a smooth alpha(s).
T = cos(alpha) t + a sin(alpha) z, n = planar normal, E2 = T x n.
In the oriented frame (w*n, w*E2), bending is
  b1 = w*cos(alpha)*theta', b2 = w*a*alpha',
and frame connection is omega = a*sin(alpha)*theta'.
Rotating by -integral(omega) gives a Bishop frame. This retains alpha'
and transports the frame through straight pieces without Frenet singularities.
All displays and joint integrals use the same ODE solution. Lengths are mm.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from pole_compliance import ComplianceParameters, build_backbone


def pitch_rule(radius_mm, k=0., p0=220., rref=85.5):
    radius = np.asarray(radius_mm, dtype=float)
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError('radius must be finite and positive')
    pitch = np.clip(p0 * (rref / radius)**k, 100., 300.)
    alpha = np.clip(np.arctan2(pitch, 2*np.pi*radius), np.radians(8.), np.radians(30.))
    return alpha, 2*np.pi*radius*np.tan(alpha)


def estimate_local_radius(points_mm, axis, window=9, min_coverage_deg=60., max_condition=100., alternating=False):
    """Circle fit in the feedback axis plane, with residual/coverage diagnostics.

    A fitted radius is a body wrapping radius, never a pipe radius. No model
    metadata, contact truth or true axis enters this function.
    """
    points = np.asarray(points_mm, dtype=float)
    basis = np.column_stack([axis['horizontal'], axis['vertical']])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid body points')
    if window < 5 or window % 2 != 1 or not np.allclose(basis.T @ basis, np.eye(2), atol=1e-6):
        raise ValueError('invalid window or feedback basis')
    xy = (points-np.asarray(axis['point_mm'])) @ basis
    rows = []
    for i in range(len(points)):
        lo = max(1 if alternating else 0, min(i-window//2, len(points)-window-(1 if alternating else 0)))
        hi = min(len(points)-(1 if alternating else 0), lo+window)
        p = xy[lo:hi]
        center = p.mean(0)
        scale = np.sqrt(np.mean(np.sum((p-center)**2, axis=1)))
        row = dict(index=i, lo=lo, hi=hi, radius_mm=None, residual_mm=None,
                   condition=None, coverage_deg=0., valid=False)
        if len(p) < 5 or scale < 1e-8:
            rows.append(row)
            continue
        q = (p-center)/scale
        parity=np.arange(lo,hi)%2
        design = (np.column_stack([2*q,parity,1-parity]) if alternating else
                  np.column_stack([2*q,np.ones(len(q))]))
        condition = float(np.linalg.cond(design))
        solution = np.linalg.lstsq(design, np.sum(q*q, axis=1), rcond=None)[0]
        radius_sq = solution[2:]+np.dot(solution[:2], solution[:2])
        row['condition'] = condition
        if np.any(radius_sq <= 0):
            rows.append(row)
            continue
        fitted_center = center+scale*solution[:2]
        distances = np.linalg.norm(p-fitted_center, axis=1)
        group_radii=scale*np.sqrt(radius_sq)
        radius = float(np.mean(group_radii))
        angles = np.unwrap(np.arctan2((p-fitted_center)[:, 1], (p-fitted_center)[:, 0]))
        coverage = float(np.degrees(np.ptp(angles)))
        expected=group_radii[1-parity] if alternating else radius
        residual = float(np.sqrt(np.mean((distances-expected)**2)))
        row.update(radius_mm=radius, residual_mm=residual, coverage_deg=coverage,alternating=alternating,
                   radial_parity_difference_mm=float(np.ptp(group_radii)),
                   center_mm=fitted_center.tolist(),
                   valid=condition <= max_condition and coverage >= min_coverage_deg and
                         residual <= max(2., .05*radius))
        rows.append(row)
    return rows


class VariablePitchBackbone:
    """Research geometry only. No target clipping hides a failed geometry gate."""
    def __init__(self, geometry, parameters, alpha_nodes, node_s=None):
        self.flat = build_backbone(geometry, ComplianceParameters(parameters.gain, 0., parameters.inside))
        self.links = self.flat.links
        self.bounds = self.flat.bounds
        self.axis = self.flat.axis
        self.axial_sign = self.flat.axial_sign
        alpha = np.asarray(alpha_nodes, dtype=float)
        if alpha.ndim != 1 or len(alpha) < 2 or not np.isfinite(alpha).all() or np.any(abs(alpha) >= np.pi/2):
            raise ValueError('invalid alpha field')
        self.node_s = np.linspace(0, self.bounds[-1], len(alpha)) if node_s is None else np.asarray(node_s)
        if len(self.node_s) != len(alpha) or self.node_s[0] != 0 or not np.isclose(self.node_s[-1], self.bounds[-1]) or np.any(np.diff(self.node_s) <= 0):
            raise ValueError('alpha nodes must span the whole body')
        self.alpha = PchipInterpolator(self.node_s, alpha)
        self.alpha_derivative = self.alpha.derivative()
        self.alpha_second = self.alpha.derivative(2)
        self.winding = math.copysign(1., next(s.turn for s in self.flat.segments if s.turn))
        self.constant = build_backbone(geometry, ComplianceParameters(parameters.gain, float(alpha[0]), parameters.inside)) if np.ptp(alpha) < 1e-14 else None
        self.turn_rates = np.array([p.turn/p.length for p in self.flat.segments])
        self.headings = self.flat.heading + np.r_[0., np.cumsum([p.turn for p in self.flat.segments])]
        # State: x,y,z, integrated frame connection, integral bending in Bishop axes.
        state = np.r_[self.flat.start_xy, self.flat.start_z, 0., 0., 0.]
        self.solutions = []
        self.end_states = [state.copy()]
        a, w = self.axial_sign, self.winding
        for index, rate in enumerate(self.turn_rates):
            left, right = self.bounds[index:index+2]
            heading = self.headings[index]
            def derivative(s, y, rate=rate, left=left, heading=heading):
                alpha = float(self.alpha(s))
                c, sn = math.cos(alpha), math.sin(alpha)
                theta = heading+rate*(s-left)
                b1, b2 = w*c*rate, w*a*float(self.alpha_derivative(s))
                co, si = math.cos(y[3]), math.sin(y[3])
                return [c*math.cos(theta), c*math.sin(theta), a*sn, a*sn*rate,
                        b1*co-b2*si, b1*si+b2*co]
            solution = solve_ivp(derivative, (left, right), state, rtol=2e-10, atol=2e-11,
                                 max_step=2., dense_output=True)
            if not solution.success:
                raise ValueError(solution.message)
            self.solutions.append(solution.sol)
            state = solution.y[:, -1]
            self.end_states.append(state.copy())
        self.end_states = np.array(self.end_states)

    def evaluate(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if not np.isfinite(s).all() or np.any(s < 0) or np.any(s > self.bounds[-1]+1e-8):
            raise ValueError('arc length outside backbone')
        indices = np.clip(np.searchsorted(self.bounds, s, side='right')-1, 0, len(self.solutions)-1)
        result = np.empty((len(s), 6))
        for i in np.unique(indices):
            mask = indices == i
            result[mask] = self.solutions[i](s[mask]).T
        return result

    def joint_angles(self, roll_phase):
        # Constant field retains the original analytical baseline exactly;
        # joint_angles_numerical separately checks the ODE's degeneration.
        if self.constant is not None:
            return self.constant.joint_angles(roll_phase)
        return self.joint_angles_numerical(roll_phase)

    def joint_angles_numerical(self, roll_phase):
        if not math.isfinite(roll_phase):
            raise ValueError('invalid phase')
        boundaries = np.r_[0., np.cumsum(self.links)]
        integrals = self.evaluate(boundaries)[:, 4:6]
        bending = integrals[2:]-integrals[:-2]
        co, si = math.cos(roll_phase), math.sin(roll_phase)
        b1 = bending[:, 0]*co-bending[:, 1]*si
        b2 = bending[:, 0]*si+bending[:, 1]*co
        return -np.where(np.arange(len(b1)) % 2 == 0, b2, b1)

    def sample_at(self, s, world=True):
        points = self.evaluate(s)[:, :3]
        if world:
            basis = np.column_stack([self.axis[name] for name in ('horizontal', 'vertical', 'direction')])
            points = points @ basis.T + self.axis['point_mm']
        return points

    def sample(self, step_mm=2., world=True):
        if not math.isfinite(step_mm) or step_mm <= 0:
            raise ValueError('invalid sampling interval')
        s = np.linspace(0, self.bounds[-1], math.ceil(self.bounds[-1]/step_mm)+1)
        return self.sample_at(s, world)

    def curvature_torsion(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        self.evaluate(s)  # bounds validation
        index = np.clip(np.searchsorted(self.bounds, s, side='right')-1, 0, len(self.turn_rates)-1)
        rate = self.turn_rates[index]
        alpha, ap, app = self.alpha(s), self.alpha_derivative(s), self.alpha_second(s)
        w, a = self.winding, self.axial_sign
        b1, b2 = w*np.cos(alpha)*rate, w*a*ap
        b1p, b2p = -w*np.sin(alpha)*ap*rate, w*a*app
        k2 = b1*b1+b2*b2
        gamma_prime = np.divide(b1*b2p-b2*b1p, k2, out=np.zeros_like(k2), where=k2>1e-20)
        return np.sqrt(k2), a*np.sin(alpha)*rate+gamma_prime
