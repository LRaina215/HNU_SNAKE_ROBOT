"""Known-axis three-region prototype, mm/rad/seconds; no MuJoCo dependency.

Front/rear use the existing A-D reconstruction and analytic Yamada integral.
Middle joints follow measured angles through a slow reference update. This is
position-feedback yielding, not zero torque or a change of actuator stiffness.
Pipe positions/contact flags are supplied by the simulation experiment only.
"""
from dataclasses import dataclass, asdict
import copy
import math

import numpy as np

from curved_axis_compliance import local_sections, joint_segments
from pole_compliance import ComplianceParameters, build_backbone


@dataclass(frozen=True)
class ThreeSegmentOptions:
    follow_seconds: float = .6
    switch_seconds: float = .8
    contact_seconds: float = .6
    boundary_seconds: float = .6
    extra_middle_joints: int = 2
    minimum_rear_joints: int = 6
    shell_radius_mm: float = 29.
    clearance_mm: float = 15.
    region_margin_mm: float = 10.
    maximum_speed_rad_s: float = math.radians(90)


class LocalSection:
    """Slice the original geometric chain; retain GLOBAL alternating axis parity.

    start/stop are zero-based joint indices, stop exclusive. A local span with
    an odd start begins with Pitch, whereas PiecewiseBackbone begins with Yaw.
    The pi/2 phase shift and alternate sign below implement that parity exactly.
    """
    def __init__(self, geometry, parameters, start, stop, direction, corner):
        self.start, self.stop = start, stop
        g = copy.deepcopy(geometry)
        points = np.asarray(g['points_mm'])[start:stop+2]
        cfg = g['config']
        links = [cfg['head_length_mm']] + [j['link_length_mm'] for j in cfg['joints']]
        count = stop-start
        cfg['joint_count'] = count
        cfg['head_length_mm'] = links[start]
        cfg['joints'] = cfg['joints'][start:stop]
        for i, joint in enumerate(cfg['joints']):
            joint.update(joint_index=i+1, axis='Yaw' if i % 2 == 0 else 'Pitch')
        d = np.asarray(direction, float)
        h = np.eye(3)[np.argmin(abs(d))]
        h -= h@d*d
        h /= np.linalg.norm(h)
        v = np.cross(d, h)
        xy = (points-corner) @ np.column_stack((h, v))
        gamma_min = geometry['curve']['gamma_min_rad']
        rows = local_sections(points, np.tile(d, (count+1, 1)), gamma_min)
        winding = math.copysign(1., sum(row[2] for row in rows))
        groups = [[(rows[0][0], 0., 0.)]]
        groups += [joint_segments(a, b, gamma, winding, gamma_min)
                   for a, b, gamma, _, _ in rows]
        groups += [[(rows[-1][1], 0., 0.)]]
        tangent = xy[1]-xy[0]
        tangent /= np.linalg.norm(tangent)
        segments = [dict(kind='arc' if radius else 'line', length_mm=length,
                         radius_mm=radius, turn_rad=turn,
                         joint_index=j if 0 < j <= count else None,
                         start_mm=xy[0], tangent=tangent)
                    for j, group in enumerate(groups) for length, radius, turn in group]
        g.update(points_mm=points, curve=dict(valid=True, segments=segments),
                 projection_axis=dict(point_mm=np.asarray(corner), direction=d,
                                      horizontal=h, vertical=v))
        gains = parameters.gains(len(links)-1)[start:stop]
        local_parameters = ComplianceParameters(gains, parameters.pitch_angle, parameters.inside)
        self.bone = build_backbone(g, local_parameters)

    def angles(self, phase):
        if self.start % 2:
            return self.bone.joint_angles(phase+math.pi/2) * np.where(
                np.arange(self.stop-self.start) % 2, -1., 1.)
        return self.bone.joint_angles(phase)

    def align(self, reference, phase):
        basis = np.column_stack((self.angles(0), self.angles(math.pi/2)))
        weights = np.linalg.lstsq(basis, reference, rcond=None)[0]
        return math.atan2(weights[1], weights[0])-phase


class ThreeSegmentController:
    def __init__(self, geometry, parameters, reference, phase, corner, source, target,
                 pipe_radius, options=ThreeSegmentOptions()):
        self.options = options
        self.parameters = parameters
        self.corner, self.source, self.target = map(np.asarray, (corner, source, target))
        self.radius = pipe_radius
        self.n = len(reference)
        self.link = float(np.mean([j['link_length_mm'] for j in geometry['config']['joints']]))
        points = np.asarray(geometry['points_mm'])
        self.initial_joints = points[1:-1].copy()
        ds, dt, ps, pt = self.distances(self.initial_joints)
        # One ordered split minimizing true-axis distances, rather than trying
        # to infer obscured joint IDs from a screenshot. The middle extends to
        # two link lengths before the intersection along the original pipe.
        cuts = range(2, self.n-options.minimum_rear_joints-1)
        self.front = min(cuts, key=lambda k: dt[:k].sum()+ds[k:].sum())
        rear_candidates = np.flatnonzero((np.arange(self.n) > self.front) & (ps < -2*self.link))
        self.rear = int(rear_candidates[0])
        self.initial_front, self.initial_rear = self.front, self.rear
        self.middle_max = self.rear-self.front+options.extra_middle_joints
        self.middle_reference = np.array(reference, float)
        self.last_target = np.array(reference, float)
        self.weights = self.desired_weights()
        self.offsets = [None, None]
        self.spans = [None, None]
        self.switch_residuals = [None, None]
        self.switch_elapsed = [0., 0.]
        self.front_ready = self.rear_ready = 0.
        self.elapsed = 0.
        self.events = []
        self.phase = phase

    def distances(self, positions):
        p = positions-self.corner
        return (np.linalg.norm(np.cross(p, self.source), axis=1),
                np.linalg.norm(np.cross(p, self.target), axis=1), p@self.source, p@self.target)

    def desired_weights(self):
        w = np.zeros((self.n, 3))
        w[:self.front, 0] = 1
        w[self.front:self.rear, 1] = 1
        w[self.rear:, 2] = 1
        return w

    def update_boundaries(self, points, contact_window, dt):
        """contact_window: 100 Hz pipe-normal-force samples, pipe x module.

        Membership can start before contact. Release permission additionally
        needs actual advance of previously middle joints and sustained target
        contact, plus remaining original-pipe support. Never a timer-only move.
        """
        o = self.options
        ds, distance, ps, progress = self.distances(points[1:-1])
        old_front, old_rear = self.front, self.rear
        near = (distance < self.radius+o.shell_radius_mm+self.link/2) & (
            distance < ds+o.region_margin_mm) & (progress > self.radius/2)
        self.front_ready = self.front_ready+dt if self.front < self.rear and near[self.front] else 0.
        if self.front_ready >= o.boundary_seconds-1e-9:
            self.front += 1
            self.front_ready = 0.
        # Actual transfer is stricter than pre-contact membership. It includes
        # progress relative to the captured pose; assigning a label earns none.
        moved = (points[1:-1]-self.initial_joints)@self.target >= self.link/4
        transferred = (distance+o.region_margin_mm < ds) & (progress > self.radius) & (
            distance < self.radius+o.shell_radius_mm+o.clearance_mm) & moved
        acquired = self.front > self.initial_front and bool(np.all(transferred[self.initial_front:self.front]))
        history = np.asarray(contact_window)
        needed = round(o.contact_seconds/.01)
        sustained = len(history) >= needed and bool(np.all(
            np.any(history[-needed:, 1, :self.front+1] > .1, axis=1)))
        source_supported = bool(len(history) and np.any(history[-1, 0, self.rear+1:] > .1))
        # Preserve six rear joints until all of those remaining modules have
        # themselves reached the target region with persistent target contact.
        final_transfer = bool(np.all(transferred[max(self.rear-1, 0):]) and sustained)
        room = self.rear-self.front < self.middle_max
        reserve = self.n-self.rear > o.minimum_rear_joints or final_transfer
        candidate_near_corner = self.rear < self.n and ps[self.rear] > -2*self.link
        allowed = room and acquired and sustained and (source_supported or final_transfer) and reserve
        self.rear_ready = self.rear_ready+dt if allowed and candidate_near_corner else 0.
        if self.rear_ready >= o.boundary_seconds-1e-9:
            self.middle_reference[self.rear] = self.last_target[self.rear]
            self.rear += 1
            self.rear_ready = 0.
        if (self.front, self.rear) != (old_front, old_rear):
            # time labels the upcoming command's END, matching control steps.
            self.events.append(dict(time=self.elapsed, old=[old_front, old_rear],
                                    new=[self.front, self.rear], acquired=acquired,
                                    target_contact=sustained, source_contact=source_supported))
        return dict(actual_transfer=transferred, acquired=acquired, target_contact=sustained,
                    source_contact=source_supported, release_allowed=allowed,
                    near_front=near, middle_room=room)

    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        self.elapsed += dt
        status = self.update_boundaries(np.asarray(geometry['points_mm']), contact_window, dt) if migrate else {}
        desired = self.desired_weights()
        # Linear cross-fade takes at least switch_seconds for a membership move.
        step = dt/self.options.switch_seconds
        difference = desired-self.weights
        magnitude = np.max(abs(difference), axis=1)
        scale = np.minimum(1., step/np.maximum(magnitude, 1e-12))
        self.weights += difference*scale[:, None]
        middle = self.weights[:, 1] > 0
        gain = -math.expm1(-dt/self.options.follow_seconds)
        self.middle_reference[middle] += gain*(measured[middle]-self.middle_reference[middle])
        targets = [self.last_target.copy(), self.middle_reference.copy(), self.last_target.copy()]
        # Include the fading boundary joints in the local reconstruction until
        # their weight is zero. Middle-only joints never enter either projection.
        for side, column, axis in ((0, 0, self.target), (1, 2, self.source)):
            active = np.flatnonzero(self.weights[:, column] > 1e-12)
            if not len(active):
                continue
            lo, hi = int(active[0]), int(active[-1])+1
            local = LocalSection(geometry, self.parameters, lo, hi, axis, self.corner)
            if self.spans[side] != (lo, hi):
                # Re-anchor only when the integration origin/span changes, not
                # every feedback frame (which would cancel ongoing rolling).
                self.offsets[side] = local.align(self.last_target[lo:hi], self.phase)
                self.switch_residuals[side] = self.last_target[lo:hi]-local.angles(self.phase+self.offsets[side])
                self.switch_elapsed[side] = 0.
                self.spans[side] = (lo, hi)
            self.switch_elapsed[side] += dt
            fade = max(0., 1-self.switch_elapsed[side]/self.options.switch_seconds)
            targets[column][lo:hi] = local.angles(phase+self.offsets[side])+fade*self.switch_residuals[side]
        raw = sum(self.weights[:, i]*targets[i] for i in range(3))
        limit = self.options.maximum_speed_rad_s*dt
        limited = self.last_target+np.clip(raw-self.last_target, -limit, limit)
        signs = np.where(np.arange(self.n) % 2, -1., 1.)
        limits = np.radians([[j['minimum_deg'], j['maximum_deg']] for j in geometry['config']['joints']])
        low, high = np.sort(limits*signs[:, None], axis=1).T
        output = np.clip(limited, low+.01, high-.01)
        self.last_target, self.phase = output.copy(), phase
        return output, dict(front=self.front, rear=self.rear, middle_max=self.middle_max,
            weights=self.weights.copy(), middle_reference=self.middle_reference.copy(),
            raw=raw, rate_limited=int(np.count_nonzero(abs(limited-raw)>1e-9)),
            angle_limited=int(np.count_nonzero(abs(output-limited)>1e-9)),
            offsets=self.offsets.copy(), **status)

    def configuration(self):
        return dict(options=asdict(self.options), initial_front=self.initial_front,
                    initial_rear=self.initial_rear, middle_max=self.middle_max,
                    index_convention='0-based joints; front [0,f), middle [f,r), rear [r,n)')
