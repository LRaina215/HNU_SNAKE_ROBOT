"""Spatially smooth front/rear phase scheduling for the simulation prototype.

The underlying measured Bishop field and section compliance are unchanged.
Rotate fields before Yamada integration; never splice actuator-angle arrays.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

from snake_control.segmented_pole_backbone import smoothstep


def schedule(mode, half_step, increment=.15):
    """One cycle = two equal physical intervals; both ends advance -increment.

    Returns cumulative and incremental phases [front, rear], relative to capture.
    The synchronised control advances half as much in each interval, so elapsed
    time and cumulative end phases match the alternating conditions exactly.
    """
    cycle, half = divmod(half_step, 2)
    if mode == 'synchronous':
        delta = np.full(2, -increment/2)
        total = delta*(half_step+1)
    else:
        first = 0 if mode == 'front_first' else 1
        delta = np.zeros(2);delta[first if half == 0 else 1-first] = -increment
        total = np.full(2, -cycle*increment)
        total[first] -= increment
        if half:total[1-first] -= increment
    return total, delta


class LocalPhaseCommand:
    def __init__(self, bounds, observed_with_compliance, preload, total, delta,
                 boundary_mm, width_mm):
        self.bounds = np.asarray(bounds)
        self.observed = np.asarray(observed_with_compliance)
        self.preload = np.asarray(preload)
        self.total, self.delta = np.asarray(total), np.asarray(delta)
        self.center, self.width = boundary_mm, width_mm

    def front_weight(self, s):
        return 1-smoothstep((np.asarray(s)-self.center)/self.width+.5)

    def evaluate(self, s):
        s = np.atleast_1d(s)
        index = np.clip(np.searchsorted(self.bounds, s, side='right')-1, 0, len(self.observed)-1)
        w = self.front_weight(s)
        def rotated(field, phase):
            angle = w*phase[0]+(1-w)*phase[1]
            co, sn = np.cos(angle), np.sin(angle)
            b = field[index]
            return np.column_stack([b[:,0]*co-b[:,1]*sn, b[:,0]*sn+b[:,1]*co])
        return rotated(self.observed, self.delta)+rotated(self.preload, self.total)

    def breaks(self, extra=()):
        lo, hi = self.center-self.width/2, self.center+self.width/2
        return np.unique(np.clip(np.r_[self.bounds, extra, lo, hi], self.bounds[0], self.bounds[-1]))

    def joint_angles(self, links, order=12):
        nodes = np.r_[0., np.cumsum(links)]
        breaks = self.breaks(nodes)
        x, weights = leggauss(order)
        integrated = np.zeros((len(links), 2))
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            s = (lo+hi)/2+(hi-lo)/2*x
            link = np.searchsorted(nodes, (lo+hi)/2)-1
            integrated[link] += weights @ self.evaluate(s)*(hi-lo)/2
        total = integrated[:-1]+integrated[1:]
        return -np.where(np.arange(len(total))%2, total[:,0], total[:,1])

    def sample(self, bone, material_rotation, step_mm=8.):
        """Reconstruct this same field; free placement uses the measured head.

        Phase gradients are represented inside the spatial bending field. A
        spatially varying phase generally changes the shape; it is not a rigid
        rotation of two independently movable robot halves.
        """
        frame = bone.start_frame.copy();frame[:,1:] = frame[:,1:] @ material_rotation
        state = np.r_[bone.start_point, frame.flatten()]
        points=[state[:3].copy()];arc=[0.]
        def derivative(s,y):
            b = self.evaluate([s])[0]
            T,N,E = y[3:].reshape(3,3).T
            return np.r_[T, np.column_stack([b[0]*N+b[1]*E, -b[0]*T, -b[1]*T]).flatten()]
        for lo,hi in zip(self.breaks()[:-1], self.breaks()[1:]):
            # Integrate one side of field discontinuities consistently.
            def rhs(s,y):return derivative(min(s,np.nextafter(hi,lo)),y)
            ss=np.linspace(lo,hi,max(2,math.ceil((hi-lo)/step_mm)+1))
            sol=solve_ivp(rhs,(lo,hi),state,t_eval=ss,rtol=2e-8,atol=2e-9,max_step=4.)
            if not sol.success:raise RuntimeError(sol.message)
            state=sol.y[:,-1];points.extend(sol.y[:3,1:].T);arc.extend(ss[1:])
        return dict(s_mm=arc,points_mm=points,frame_error=float(np.max(abs(
            state[3:].reshape(3,3).T@state[3:].reshape(3,3)-np.eye(3)))))
