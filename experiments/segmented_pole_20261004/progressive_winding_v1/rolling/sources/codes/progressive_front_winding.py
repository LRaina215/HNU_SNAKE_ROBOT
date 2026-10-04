"""Experimental local helix-extension guidance on top of compliant rolling.

The added term is damped differential IK, not the unchanged paper controller.
Known pipe axes and a current-base FK callback are explicit simulation inputs.
Units: mm, rad, seconds. No actuator gains or physical constraints are changed.
"""
import numpy as np

from segmented_pole_backbone import smoothstep
from three_segment_front_support import FrontSupportedThreeSegmentController


class ProgressiveFrontWindingController(FrontSupportedThreeSegmentController):
    def __init__(self, *args, forming_gain=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.forming_gain = forming_gain
        self.forming_first = self.initial_front+1  # module 6 in this capture
        self.forming_ready = 0.
        self.forming_events = []
        self.last_reference = None

    def helix_extension(self, points):
        """An arc-length sampled continuation with measured position/tangent.

        The exponential tangent correction joins the measured incoming tangent
        to the target-axis helix. Samples are geometric objectives, not claimed
        to be an exactly reachable alternating-joint robot configuration.
        """
        anchor = self.forming_first-1
        p0 = points[anchor]
        radial = p0-self.corner
        radial -= (radial@self.body_axis)*self.body_axis
        radius = np.linalg.norm(radial)
        u = radial/radius
        v = np.cross(self.body_axis, u)
        h = abs(self.pitch_mm)/(2*np.pi)
        rate = 1/np.hypot(radius, h)
        length = self.links[anchor:anchor+3].sum()
        s = np.linspace(0., length+2*self.link, 401)
        theta = self.winding*s*rate
        ideal = (p0-radial + radius*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)
                 + (h*rate*s)[:, None]*self.body_axis)
        tangent = points[anchor]-points[anchor-1]
        tangent /= np.linalg.norm(tangent)
        helix_tangent = self.winding*radius*rate*v+h*rate*self.body_axis
        curve = ideal+(s*np.exp(-s/self.link))[:, None]*(tangent-helix_tangent)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        samples = np.cumsum(self.links[anchor:anchor+3])
        reference = np.column_stack([np.interp(samples, arc, curve[:, j]) for j in range(3)])
        return anchor, reference, curve[arc <= samples[-1]]

    def update_forming_window(self, points, contact_window, dt):
        """Advance only after BOTH new modules really transfer and contact.

        Labels alone cannot advance this window. Pre-contact guidance is active;
        the original rear-release gate still retains original-pipe support.
        """
        f = self.forming_first
        centers = (points[:-1]+points[1:])/2
        ds, distance, _, progress = self.distances(centers)
        region = ((distance+10 < ds) & (progress > self.radius)
                  & (distance < self.radius+self.options.shell_radius_mm+15))
        history = np.asarray(contact_window)
        ready = (len(history) >= 60 and np.all(region[f:f+2])
                 and np.all(history[-60:, 1, f:f+2] > .1))
        self.forming_ready = self.forming_ready+dt if ready else 0.
        if self.forming_ready >= .6-1e-9 and f+3 < self.rear:
            self.forming_events.append(dict(time=self.elapsed, old=f, new=f+1))
            self.forming_first += 1
            self.forming_ready = 0.

    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        previous = self.last_target.copy()
        base, info = super().compute(geometry, measured, phase, dt, contact_window, migrate)
        if self.forming_gain == 0:
            return base, info
        points = np.asarray(geometry['points_mm'], float)
        self.update_forming_window(points, contact_window, dt)
        anchor, reference, curve = self.helix_extension(points)
        endpoint_ids = np.arange(anchor+1, anchor+4)
        # Include the last established bend and the forming/connector bends.
        # No direct correction reaches rear joints, and front rolling continues.
        ids = np.arange(anchor-1, min(anchor+4, self.rear))
        q = np.asarray(measured, float)
        actual = self.forward_points(q)
        epsilon = 1e-4
        jacobian = []
        for j in ids:
            perturbed = q.copy()
            perturbed[j] += epsilon
            jacobian.append((self.forward_points(perturbed)-actual)/epsilon)
        jacobian = np.stack(jacobian, axis=-1)
        # Preserve connector exit position/direction softly, not by fixing a
        # body in MuJoCo. Infeasible tasks remain residuals instead of stretching
        # links or directly positioning bodies. Damping trades tracking for
        # modest joint corrections near kinematic singularities.
        exit_ids = np.array([self.rear+1, self.rear+2])
        endpoint_weights = np.array([.5, 1., 1.])[:, None, None]
        matrix = np.vstack([(jacobian[endpoint_ids]*endpoint_weights).reshape(-1, len(ids)),
                            .5*jacobian[exit_ids].reshape(-1, len(ids)),
                            20.*np.eye(len(ids))])
        ramp = float(smoothstep(self.elapsed/1.6))
        error = reference-actual[endpoint_ids]
        goal = np.r_[(self.forming_gain*ramp*error*endpoint_weights[:, :, 0]).ravel(),
                     np.zeros(6+len(ids))]
        correction = np.linalg.lstsq(matrix, goal, rcond=None)[0]
        correction = np.clip(correction, -np.radians(2.), np.radians(2.))
        raw = np.asarray(info['raw']).copy()
        raw[ids] += correction
        speed = self.options.maximum_speed_rad_s*dt
        limited = previous+np.clip(raw-previous, -speed, speed)
        signs = np.where(np.arange(self.n) % 2, -1., 1.)
        bounds = np.radians([[j['minimum_deg'], j['maximum_deg']] for j in geometry['config']['joints']])
        low, high = np.sort(bounds*signs[:, None], axis=1).T
        target = np.clip(limited, low+.01, high-.01)
        self.last_target = target.copy()
        self.last_reference = dict(anchor=anchor, endpoints=reference, curve=curve)
        info.update(raw=raw, forming_first_module=self.forming_first,
                    forming_joint_ids=ids, forming_delta_deg=np.degrees(correction),
                    forming_reference=self.last_reference, forming_error_mm=error,
                    forming_linear_error_after_mm=error-np.einsum('ijk,k->ij', jacobian[endpoint_ids], correction),
                    forming_connector_shift_mm=np.einsum('ijk,k->ij', jacobian[exit_ids], correction),
                    rate_limited=int(np.count_nonzero(abs(raw-limited)>1e-9)),
                    angle_limited=int(np.count_nonzero(abs(target-limited)>1e-9)))
        return target, info

    def configuration(self):
        return dict(**super().configuration(), progressive_winding=dict(
            gain=self.forming_gain, ramp_seconds=1.6, initial_first_module=self.forming_first,
            modules=2, tangent_transition_mm=self.link, damping_mm_per_rad=20.,
            connector_weight=.5, maximum_guidance_deg=2.,
            method='arc-length helix continuation + damped differential IK added to Yamada/compliance',
            window_gate='both forming modules transferred and contacted for 0.6 s, then 0.6 s confirmation'))
