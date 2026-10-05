"""Experimental travelling formation window with measured-shape compliance.

Formation and rolling are independent coordinates. Known pipe axes/contact
history are explicitly simulation inputs; this is not the unchanged 2023 law.
All commands still use position targets and the existing Yamada material field.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

from snake_control.three_segment_incremental import IncrementalThreeSegmentController, integrated_field
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone
from snake_control.segmented_pole_backbone import smoothstep
from snake_control.three_segment_compliance import LocalSection, ThreeSegmentOptions


@dataclass(frozen=True)
class FormationOptions:
    joints_per_second: float = .4
    maximum_lead: float = 4.
    recovery: float = .35
    release_distance_links: float = 3.
    end_recovery: float = .25
    complete_end_spans: bool = False


class TravelingCompliantWinding(IncrementalThreeSegmentController):
    def __init__(self, *args, pitch_mm=220., formation=FormationOptions(), options=None, **kwargs):
        if options is None:
            options = ThreeSegmentOptions(follow_seconds=.06)
        super().__init__(*args, options=options, **kwargs)
        self.formation = formation
        self.pitch_mm = pitch_mm
        self.progress = float(self.initial_front)
        self.handed_off = False
        self.finish_elapsed = None
        self.formation_weights = np.zeros(self.n)
        self.formation_weights[:self.initial_front] = 1.
        self.body_axis = -self.target
        p = np.asarray(args[0]['points_mm'])[:self.initial_front+2]-self.corner
        r = p-(p@self.body_axis)[:, None]*self.body_axis
        self.winding = float(np.sign(np.sum(np.cross(r[:-1], r[1:])@self.body_axis)))
        self.radius_reference = float(np.median(np.linalg.norm(r, axis=1)))
        # Select handedness from the actually wrapped front, not global phase 0.
        q = np.asarray(kwargs['measured'])
        candidates = []
        for sign in (-1., 1.):
            a, b = self.helix_basis(self.radius_reference, sign)
            fit = np.linalg.lstsq(np.column_stack([a, b])[:self.initial_front], q[:self.initial_front], rcond=None)[0]
            phase = math.atan2(fit[1], fit[0])
            error = np.linalg.norm((a*np.cos(phase)+b*np.sin(phase)-q)[:self.initial_front])
            candidates.append((error, sign, phase))
        _, self.pitch_sign, _ = min(candidates)

    def helix_basis(self, radius, sign):
        h = sign*self.pitch_mm/(2*np.pi)
        kappa, tau = radius/(radius*radius+h*h), h/(radius*radius+h*h)
        nodes = np.r_[0., np.cumsum(self.links)]
        length, mid = nodes[2:]-nodes[:-2], (nodes[2:]+nodes[:-2])/2
        amplitude = kappa*length*np.sinc(tau*length/(2*np.pi))
        parity = np.arange(self.n) % 2
        a = -amplitude*np.where(parity, np.cos(tau*mid), np.sin(tau*mid))
        b = -amplitude*np.where(parity, -np.sin(tau*mid), np.cos(tau*mid))
        return a, b

    def update_boundaries(self, points, contact_window, dt):
        ds, distance, ps, axial = self.distances(points[1:-1])
        history = np.asarray(contact_window)
        target_contact = bool(len(history) >= 40 and np.mean(np.any(history[-40:, 1, :self.front+1] > .1, axis=1)) >= .9)
        source_clear = bool(len(history) >= 60 and not np.any(history[-60:,0] > .1))
        # Once the entire chain has cleared the original pipe's envelope,
        # requiring its support to release the remaining source controller is
        # circular. This is a physical clearance/contact transition, not labels.
        cleared = bool(np.min((points-self.corner)@self.target) > self.radius+self.options.shell_radius_mm)
        if target_contact and source_clear and cleared:
            self.handed_off = True
        region = (distance+10 < ds) & (distance < self.radius+29+15) & (axial > self.radius/2)
        prefix = self.initial_front
        while prefix < self.n and region[prefix]:
            prefix += 1
        old = [self.front, self.rear]
        # Permit forming BEFORE contact, but do not call its progress transfer.
        if target_contact:
            limit = min(self.n, self.n if self.handed_off else prefix+self.formation.maximum_lead, self.rear)
            self.progress = max(self.progress, min(limit,
                                self.progress+dt*self.formation.joints_per_second))
        self.formation_weights = smoothstep(np.clip(self.progress-np.arange(self.n), 0., 1.))
        self.front = min(self.n, int(math.ceil(self.progress-1e-9)))
        room = self.rear-self.front < self.middle_max
        remaining_contact = bool(len(history) and np.any(history[-1, 0, self.rear+2:] > .1))
        # Rear reserve shrinks only after a physically transferred front grows.
        reserve = max(0, 6-max(0, prefix-8))
        near = self.rear < self.n and ps[self.rear] > -self.formation.release_distance_links*self.link
        if self.handed_off:
            reserve = 0
            near = self.rear < self.n
        release = (self.progress > self.initial_front+.5 and target_contact and room and near
                   and (remaining_contact or prefix >= 12 or self.handed_off) and self.n-self.rear > reserve)
        self.rear_ready = self.rear_ready+dt if release else 0.
        if self.rear_ready >= .6-1e-9:
            self.middle_reference[self.rear] = self.last_target[self.rear]
            self.rear += 1
            self.rear_ready = 0.
        if old != [self.front, self.rear]:
            self.events.append(dict(time=self.elapsed, old=old, new=[self.front,self.rear],
                actual_prefix=prefix, formation_progress=self.progress, release=release))
        return dict(actual_transfer=region, actual_prefix=prefix, formation_progress=self.progress,
                    target_contact=target_contact, release_allowed=release, rear_reserve=reserve,
                    release_near=near, remaining_source_contact=remaining_contact, middle_room=room,
                    handed_off=self.handed_off, original_pipe_clear=source_clear, junction_cleared=cleared)

    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        previous=self.last_target.copy()
        previous_phase=self.phase
        q,info=super().compute(geometry,measured,phase,dt,contact_window,migrate)
        if self.handed_off and self.progress >= self.n:
            local=LocalSection(geometry,self.parameters,0,self.n,self.target,self.corner)
            if self.finish_elapsed is None:
                self.finish_elapsed=0.
                self.finish_offset=local.align(previous,previous_phase)
                self.finish_residual=previous-local.angles(previous_phase+self.finish_offset)
            self.finish_elapsed+=dt
            fraction=float(smoothstep(self.finish_elapsed/2.))
            candidate=local.angles(phase+self.finish_offset)+(1-fraction)*self.finish_residual
            if self.formation.end_recovery:
                p=np.asarray(geometry['points_mm'])[1:-1]-self.corner
                radius=float(np.median(np.linalg.norm(np.cross(p[4:-4],self.target),axis=1)))
                a,b=self.helix_basis(radius,self.pitch_sign)
                fit=np.linalg.lstsq(np.column_stack([a,b])[4:-4],candidate[4:-4],rcond=None)[0]
                angle=math.atan2(fit[1],fit[0])
                nominal=a*np.cos(angle)+b*np.sin(angle)
                ends=np.r_[np.arange(4),np.arange(self.n-4,self.n)]
                candidate[ends]+=fraction*self.formation.end_recovery*(nominal[ends]-measured[ends])
            speed=self.options.maximum_speed_rad_s*dt
            limited=previous+np.clip(candidate-previous,-speed,speed)
            signs=np.where(np.arange(self.n)%2,-1.,1.)
            bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])
            low,high=np.sort(bounds*signs[:,None],axis=1).T
            q=np.clip(limited,low+.01,high-.01)
            self.last_target=q.copy()
            info.update(finish_local=True,finish_elapsed=self.finish_elapsed,finish_raw=candidate,
                        raw=candidate,rate_limited=int(np.count_nonzero(abs(candidate-limited)>1e-9)),
                        angle_limited=int(np.count_nonzero(abs(q-limited)>1e-9)))
        return q,info

    def formation_radius(self, distance):
        return float(np.median(distance[:self.initial_front]))

    def rolling_observation(self, geometry, measured, observed):
        points = np.asarray(geometry['points_mm'])
        bone = SpatialBishopBackbone(points, self.links)
        _, rotation = bone.material_field(measured)
        relative = points[1:-1]-self.corner
        radial = relative-(relative@self.body_axis)[:, None]*self.body_axis
        distance = np.linalg.norm(radial, axis=1)
        # Wrapped-front feedback updates the nominal radius. Incoming connector
        # distances must not enlarge the whole front's cross section.
        radius = self.formation_radius(distance)
        radius_dt=getattr(self,'current_dt',.2)
        radius_gain=.15 if radius_dt==.2 else 1-math.exp(math.log(.85)*radius_dt/.2)
        self.radius_reference += radius_gain*(radius-self.radius_reference)
        h = abs(self.pitch_mm)/(2*np.pi)
        alpha = math.atan2(h,self.radius_reference)
        kappa = self.radius_reference/(self.radius_reference**2+h*h)
        field = np.zeros_like(bone.field)
        for j, frame in enumerate(bone.mid_frames):
            normal = radial[j]/max(distance[j],1e-9)
            tangent = frame[:,0]
            desired_tangent = self.winding*np.cos(alpha)*np.cross(self.body_axis,normal)+np.sin(alpha)*self.body_axis
            derivative = -kappa*normal+(desired_tangent-tangent*(desired_tangent@tangent))/(2*self.link)
            field[j+1] = derivative @ (frame[:,1:]@rotation)
        if self.formation.complete_end_spans:
            # The measured discrete backbone has straight terminal half-links,
            # but the forming helix is a continuous target curve. Do not drop
            # its terminal half-span contribution from the Yamada integral.
            field[0]=field[1];field[-1]=field[-2]
        supported = integrated_field(bone,field)
        ramp = float(smoothstep(self.elapsed/1.6))
        # Existing support keeps the successful measured-axis recovery law.
        # New joints have an explicit travelling activation and a finite
        # recovery rate: actual deformation remains in every target update.
        gain = self.formation.recovery*self.formation_weights
        gain[:self.initial_front] = .1*ramp
        change = gain[:,None]*(supported-observed)
        return observed+change, dict(formation_weights=self.formation_weights.copy(),
            formation_radius_mm=self.radius_reference, formation_pitch_sign=self.pitch_sign,
            formation_nominal=supported,
            formation_observed=observed, formation_change=change, formation_gain=gain)

    def configuration(self):
        return dict(**super().configuration(), formation=asdict(self.formation),
                    pitch_mm=self.pitch_mm, pitch_sign=self.pitch_sign,
                    principle='measured bending + progressive nominal recovery + measured A-D difference')
