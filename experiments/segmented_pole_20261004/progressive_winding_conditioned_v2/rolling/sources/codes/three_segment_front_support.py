"""Experimental front winding support during rolling, with unchanged rear/middle.

Known target pipe axis is simulation input. This is soft geometric feedback,
not a rigid anchor, not exact curve stitching, and not the unchanged 2023 law.
"""
import numpy as np

from segmented_pole_backbone import smoothstep
from spatial_bishop_backbone import SpatialBishopBackbone
from three_segment_incremental import IncrementalThreeSegmentController, integrated_field


class FrontSupportedThreeSegmentController(IncrementalThreeSegmentController):
    def __init__(self, *args, pitch_mm=220., shape_gain=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pitch_mm = pitch_mm
        self.shape_gain = shape_gain
        self.ramp_seconds = 1.6
        # Spatial tangents run head -> tail, opposite target-pipe travel.
        self.body_axis = -self.target
        p = np.asarray(args[0]['points_mm'])[:self.front+2]-self.corner
        radial = p-(p@self.body_axis)[:, None]*self.body_axis
        self.winding = float(np.sign(np.sum(np.cross(radial[:-1], radial[1:])@self.body_axis)))
        if self.winding == 0:
            raise ValueError('Captured front winding is undefined')

    def rolling_observation(self, geometry, measured, observed):
        p = np.asarray(geometry['points_mm'])
        bone = SpatialBishopBackbone(p, self.links)
        _, rotation = bone.material_field(measured)
        radial = p[1:-1]-self.corner
        radial -= (radial@self.body_axis)[:, None]*self.body_axis
        distances = np.linalg.norm(radial, axis=1)
        # Follow the observed radius, with no second K contraction: the existing
        # A-D shaped-minus-zero term still supplies section compliance.
        radius = float(np.median(distances[:self.front]))
        h = abs(self.pitch_mm)/(2*np.pi)
        alpha = np.arctan2(h, radius)
        kappa = radius/(radius*radius+h*h)
        desired = np.zeros_like(bone.field)
        errors = []
        for j, frame in enumerate(bone.mid_frames):
            tangent = frame[:, 0]
            inward = radial[j]/max(distances[j], 1e-9)
            reference_tangent = (self.winding*np.cos(alpha)*np.cross(self.body_axis, inward)
                                 + np.sin(alpha)*self.body_axis)
            tangent_error = reference_tangent-tangent*(reference_tangent@tangent)
            derivative = -kappa*inward+tangent_error/(2*self.link)
            desired[j+1] = derivative @ (frame[:, 1:]@rotation)
            errors.append(float(np.degrees(np.arccos(np.clip(tangent@reference_tangent, -1., 1.)))))
        reference = integrated_field(bone, desired)
        gain = self.shape_gain*float(smoothstep(self.elapsed/self.ramp_seconds))
        # Restrict the integrated correction to front membership. Middle-only
        # and rear targets receive exactly zero correction; existing 0.8 s
        # membership blending gradually admits a newly acquired front joint.
        correction = gain*self.weights[:, :1]*(reference-observed)
        return observed+correction, dict(front_shape_gain=gain,
            front_reference_radius_mm=radius, front_tangent_error_deg=errors[:self.front],
            front_shape_integrals_delta=correction, front_reference_integrals=reference)

    def configuration(self):
        return dict(**super().configuration(), front_support=dict(
            gain=self.shape_gain, ramp_seconds=self.ramp_seconds, pitch_mm=self.pitch_mm,
            winding=self.winding, tangent_relaxation_links=2.,
            radius='median measured front distance; A-D compliance unchanged'))
