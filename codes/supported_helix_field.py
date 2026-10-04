"""Known-axis helix feedback and a non-rolling axis-bend component.

Experimental geometric feedback, not an exact solution of two helices plus a
connector. All corrections are spatial bending fields before Yamada integration.
"""
import numpy as np
from alternating_pole_phase import LocalPhaseCommand
from segmented_pole_backbone import smoothstep


def reference_fields(bone, points, material_rotation, axis_field, corner, pitch,
                     winding, gains, front_end_links=6., taper_links=3.):
    """Helix reference: T = sign*cos(alpha)*e_theta + sin(alpha)*axis.

    Radius follows measured front distance, tightened by the existing mean K.
    The analytic helix curvature points inward. Tangent feedback also steers
    the local pitch; curvature magnitude alone would not enforce a helix.
    A two-link relaxation length sets tangent correction units (1/mm).
    """
    link=bone.links.mean();s=np.cumsum(bone.links)[:-1]
    _,rates=axis_field.frames(s)
    desired=np.zeros_like(bone.field);bend=np.zeros_like(bone.field)
    delta=np.asarray(points[1:7])-corner
    axis=axis_field.target
    radial=delta-(delta@axis)[:,None]*axis
    radius=float(np.median(np.linalg.norm(radial,axis=1)))
    target_radius=radius/(1+float(np.mean(gains[:6])))
    h=abs(pitch)/(2*np.pi);scale=np.hypot(target_radius,h)
    kappa=target_radius/scale**2;alpha=np.arctan2(h,target_radius)
    weights=1-smoothstep((s/link-front_end_links)/taper_links)
    tangent_errors=[]
    for j,frame in enumerate(bone.mid_frames):
        T=frame[:,0];normal=frame[:,1:]@material_rotation
        # R'(s)R(s)^T is the angular velocity of the smoothly turning pipe axis.
        spin=axis_field.rotation_axis*rates[j]
        bend[j+1]=np.cross(spin,T)@normal
        r=np.asarray(points[j+1])-corner;r-=axis*(r@axis)
        r/=max(np.linalg.norm(r),1e-9)
        circ=np.cross(axis,r)
        tangent=winding*np.cos(alpha)*circ+np.sin(alpha)*axis
        error=tangent-T*(tangent@T)
        derivative=-kappa*r+error/(2*link)
        desired[j+1]=derivative@normal
        tangent_errors.append(float(np.degrees(np.arccos(np.clip(T@tangent,-1,1)))))
    return desired,bend,np.r_[0.,weights,0.],dict(measured_radius_mm=radius,
        target_radius_mm=target_radius,pitch_mm=pitch,kappa_per_mm=kappa,
        front_tangent_error_deg=tangent_errors[:6])


class SupportedHelixCommand(LocalPhaseCommand):
    def __init__(self,bounds,observed,preload,total,delta,boundary_mm,width_mm,
                 helix_reference,axis_bend,front_weights,shape_gain=0.,separate=False):
        observed=np.asarray(observed);helix_reference=np.asarray(helix_reference)
        self.axis_bend=np.asarray(axis_bend) if separate else np.zeros_like(observed)
        shaped=observed+shape_gain*np.asarray(front_weights)[:,None]*(helix_reference-observed)
        super().__init__(bounds,shaped-self.axis_bend,preload,total,delta,boundary_mm,width_mm)

    def evaluate(self,s):
        s=np.atleast_1d(s)
        index=np.clip(np.searchsorted(self.bounds,s,side='right')-1,0,len(self.axis_bend)-1)
        # The measured/material projection is recomputed every command. Keeping
        # this world-axis contribution out of the phase rotation is not a
        # physical anchor and does not independently fix connector endpoints.
        return super().evaluate(s)+self.axis_bend[index]
