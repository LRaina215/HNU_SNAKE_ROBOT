"""Simulation research: two known pipe axes, one transported bending field.

Units mm/rad. Reuse the 2023 per-joint section construction and gains. The
rotating axis frame contributes to bending AND twist; simply concatenating
constant-pitch kappa/tau would omit the bend between the two pipes.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from snake_control.curved_axis_compliance import local_sections, joint_segments, DEFAULT_GAMMA_MIN
from snake_control.pole_compliance import PiecewiseBackbone, TargetSegment


@dataclass(frozen=True)
class SegmentedOptions:
    initial_boundary_links: float = 6.
    transition_links: float = 4.
    migration_persistence: int = 3
    migration_max_links: float = .25
    switch_steps: int = 6


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10-15*x+6*x*x)


class AxisField:
    def __init__(self, target_direction, source_direction, first_link, center, width):
        # Directions follow the body head -> tail, opposite to pipe travel.
        self.target, self.source = -unit(target_direction), -unit(source_direction)
        cross = np.cross(self.target, self.source)
        self.angle = math.atan2(np.linalg.norm(cross), self.target @ self.source)
        self.rotation_axis = unit(cross) if np.linalg.norm(cross) > 1e-10 else np.array([1., 0., 0.])
        horizontal = first_link - (first_link @ self.target)*self.target
        if np.linalg.norm(horizontal) < 1e-8:
            raise ValueError('head tangent parallel to pipe axis')
        horizontal = unit(horizontal)
        self.initial = np.column_stack([horizontal, np.cross(self.target, horizontal), self.target])
        self.center, self.width = center, width

    def frames(self, s):
        x = (np.atleast_1d(s)-self.center)/self.width+.5
        angles = self.angle*smoothstep(x)
        rate = self.angle*30*np.clip(x, 0, 1)**2*(1-np.clip(x, 0, 1))**2/self.width
        frames = Rotation.from_rotvec(angles[:, None]*self.rotation_axis).as_matrix() @ self.initial
        return frames, rate


class MovingAxisBackbone:
    """Continuous tangent and Bishop components shared by shape and Yamada.

    In a moving pipe frame R, T=R(cos(alpha)*cos(theta),
    cos(alpha)*sin(theta), sin(alpha)). N follows the signed planar normal.
    b1=T' dot N, b2=T' dot (T cross N), omega=N' dot (T cross N).
    Rotate (b1,b2) by integral omega to obtain parallel-transported components.
    This keeps the axis-rotation terms and works through zero-curvature pieces.
    """
    def __init__(self, points, links, parameters, source_direction, target_direction,
                 boundary_mm, width_mm, gamma_min=DEFAULT_GAMMA_MIN):
        self.links = np.asarray(links, float)
        self.length = self.links.sum()
        self.axis_field = AxisField(target_direction, source_direction, points[1]-points[0],
                                    boundary_mm, width_mm)
        nodes = np.r_[0., np.cumsum(self.links)]
        axes = self.axis_field.frames((nodes[:-1]+nodes[1:])/2)[0][:, :, 2]
        rows = local_sections(points, axes, gamma_min)
        self.winding = math.copysign(1., sum(row[2] for row in rows))
        groups = [[(rows[0][0], 0., 0.)]]
        groups += [joint_segments(a, b, gamma, self.winding, gamma_min) for a,b,gamma,_,_ in rows]
        groups += [[(rows[-1][1], 0., 0.)]]
        lengths = np.r_[self.links[0]/2, (self.links[:-1]+self.links[1:])/2, self.links[-1]/2]
        self.alpha = parameters.pitch_angle
        gains = parameters.gains(len(links)-1)
        self.spans, self.rates = [], []
        segments = []
        for j, (group, length) in enumerate(zip(groups, lengths)):
            scale = length*math.cos(self.alpha)/sum(row[0] for row in group)
            for projected, radius, turn in group:
                span = projected*scale/math.cos(self.alpha)
                factor = 1+(-1 if parameters.inside else 1)*gains[j-1] if radius else 1.
                turn *= factor
                self.spans.append(span)
                self.rates.append(turn/span)
                r = radius*scale/factor if radius else 0.
                segments.append(TargetSegment(j if 0<j<len(links) else None, span, r, turn,
                    math.cos(self.alpha)**2/r if r else 0.,
                    math.copysign(1, turn)*math.sin(self.alpha)*math.cos(self.alpha)/r if r else 0.))
        self.bounds = np.r_[0., np.cumsum(self.spans)]
        self.headings = np.r_[0., np.cumsum(np.asarray(self.spans)*self.rates)]
        # Exact straight-axis degeneration uses the existing analytical integral.
        self.flat = PiecewiseBackbone(segments, links, self.alpha, [0,0], [1,0], {}, 0., 1.)
        self.solutions = []
        state = np.r_[np.asarray(points[0]), 0., 0., 0.]
        self.initial_point = state[:3].copy()
        for k in range(len(self.spans)):
            def derivative(s, y, k=k):
                tangent, b, omega = self.components(s, k)
                co, sn = np.cos(y[3]), np.sin(y[3])
                return np.r_[tangent, omega, b[0]*co-b[1]*sn, b[0]*sn+b[1]*co]
            solution = solve_ivp(derivative, self.bounds[k:k+2], state, rtol=2e-8, atol=2e-9,
                                 max_step=12., dense_output=True)
            if not solution.success:
                raise ValueError(solution.message)
            self.solutions.append(solution.sol)
            state = solution.y[:, -1]

    def components(self, s, k):
        frame, rate = self.axis_field.frames(s)
        R = frame[0]
        theta = self.headings[k]+self.rates[k]*(s-self.bounds[k])
        c, a = math.cos(self.alpha), math.sin(self.alpha)
        tangent = R @ [c*math.cos(theta), c*math.sin(theta), a]
        normal = self.winding * (R @ [-math.sin(theta), math.cos(theta), 0.])
        second = np.cross(tangent, normal)
        spin = self.axis_field.rotation_axis*rate[0]
        dt = np.cross(spin, tangent)+self.rates[k]*c*(R @ [-math.sin(theta), math.cos(theta), 0.])
        dn = np.cross(spin, normal)+self.winding*self.rates[k]*(R @ [-math.cos(theta), -math.sin(theta), 0.])
        return tangent, np.array([dt @ normal, dt @ second]), float(dn @ second)

    def evaluate(self, s):
        s = np.atleast_1d(s)
        index = np.clip(np.searchsorted(self.bounds, s, side='right')-1, 0, len(self.solutions)-1)
        return np.array([self.solutions[k](x) for x,k in zip(s,index)])

    def joint_angles(self, phase):
        nodes = np.r_[0., np.cumsum(self.links)]
        values = self.evaluate(nodes)[:, 4:6]
        b = values[2:]-values[:-2]
        co, sn = math.cos(phase), math.sin(phase)
        return -np.where(np.arange(len(b))%2, b[:,0]*co-b[:,1]*sn, b[:,0]*sn+b[:,1]*co)

    def align_phase(self, reference, phase):
        basis = np.column_stack([self.joint_angles(0.), self.joint_angles(math.pi/2)])
        weights = np.linalg.lstsq(basis, reference, rcond=None)[0]
        return math.atan2(weights[1], weights[0])-phase

    def bending_field(self, s, phase):
        s = np.atleast_1d(s)
        state = self.evaluate(s)
        index = np.clip(np.searchsorted(self.bounds,s,side='right')-1,0,len(self.spans)-1)
        b = np.array([self.components(x,k)[1] for x,k in zip(s,index)])
        angle = state[:,3]+phase
        co,sn = np.cos(angle),np.sin(angle)
        return np.column_stack([b[:,0]*co-b[:,1]*sn,b[:,0]*sn+b[:,1]*co])

    def sample_command(self, phase, switch=None, weight=1., step_mm=8.):
        """Reconstruct the exact commanded field, including switch correction.

        A free rigid placement is fixed at the measured head and phase-zero
        tangent/normal. This is a continuous backbone, not discrete robot FK.
        Clipped actuator angles (if any) must be displayed separately as FK.
        """
        T,b,omega = self.components(0.,0)
        N = self.winding*self.axis_field.initial[:,1]
        E = np.cross(T,N)
        state = np.r_[self.initial_point,T,N,E]
        links = np.r_[0.,np.cumsum(self.links)]
        breaks = np.unique(np.r_[self.bounds,links])
        points=[state[:3].copy()];arc=[0.]
        def derivative(s,y):
            bend = self.bending_field([s],phase)[0]
            if switch is not None:
                bend += (1-weight)*switch.correction_field([s],phase-switch.offset)[0]
            T,N,E=y[3:6],y[6:9],y[9:12]
            return np.r_[T,bend[0]*N+bend[1]*E,-bend[0]*T,-bend[1]*T]
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            if hi-lo<1e-8:continue
            ss=np.linspace(lo,hi,max(2,math.ceil((hi-lo)/step_mm)+1))
            sol=solve_ivp(derivative,(lo,hi),state,t_eval=ss,rtol=2e-8,atol=2e-9,max_step=4.)
            if not sol.success:raise ValueError(sol.message)
            points.extend(sol.y[:3,1:].T);arc.extend(ss[1:]);state=sol.y[:,-1]
        return dict(s_mm=arc,points_mm=np.asarray(points),frame_orthogonality_error=float(
            np.max(abs(state[3:].reshape(3,3)@state[3:].reshape(3,3).T-np.eye(3)))))


class BoundaryTracker:
    """Ordered prefix assignment with persistence and bounded movement.

    Geometry supplies a candidate cut; actual target contacts must corroborate
    it. Labels never form the success metric: the runner audits true positions.
    """
    def __init__(self, options, link_mm):
        self.options, self.link_mm = options, link_mm
        self.value = options.initial_boundary_links
        self.candidate = None
        self.streak = 0

    def update(self, centers, corner, source, target, contact_target):
        delta = centers-corner
        ds = np.linalg.norm(delta-(delta@source)[:,None]*source, axis=1)
        dt = np.linalg.norm(delta-(delta@target)[:,None]*target, axis=1)
        # Penalize assigning points behind the target branch to that branch.
        dt += np.maximum(0., -(delta@target))
        costs = [dt[:cut].sum()+ds[cut:].sum() for cut in range(2,len(centers)-2)]
        candidate = int(np.argmin(costs))+2
        corroborated = any(contact_target[max(0,candidate-3):candidate+1])
        if candidate == self.candidate and corroborated:
            self.streak += 1
        else:
            self.candidate, self.streak = candidate, int(corroborated)
        if self.streak >= self.options.migration_persistence:
            self.value += np.clip(candidate-self.value, -self.options.migration_max_links,
                                  self.options.migration_max_links)
        return self.value*self.link_mm


class SwitchContinuity:
    """One common phase; initial correction is a bending field, not angle splice.

    Each joint reads two adjacent link intervals. Alternating axes let a
    constant per-link correction reproduce the initial angle residual exactly.
    It rotates with the common phase and fades with the switch weight.
    """
    def __init__(self, backbone, reference, phase):
        self.phase = phase
        self.offset = backbone.align_phase(reference, phase)
        residual = reference-backbone.joint_angles(phase+self.offset)
        self.field = np.zeros((len(backbone.links),2))
        for i, delta in enumerate(residual):
            component = 0 if i%2 else 1
            self.field[i:i+2,component] = -delta/(backbone.links[i]+backbone.links[i+1])
        self.links = backbone.links

    def correction_angles(self, phase):
        co, sn = np.cos(phase-self.phase), np.sin(phase-self.phase)
        rot = np.array([[co,sn],[-sn,co]])
        integral = (self.field @ rot)*self.links[:,None]
        total = integral[:-1]+integral[1:]
        return -np.where(np.arange(len(total))%2, total[:,0], total[:,1])

    def correction_field(self,s,phase):
        bounds=np.r_[0.,np.cumsum(self.links)]
        indices=np.clip(np.searchsorted(bounds,s,side='right')-1,0,len(self.links)-1)
        co,sn=np.cos(phase-self.phase),np.sin(phase-self.phase)
        return self.field[indices]@np.array([[co,sn],[-sn,co]])

    def angles(self, backbone, phase, weight):
        return backbone.joint_angles(phase+self.offset)+(1-weight)*self.correction_angles(phase)
