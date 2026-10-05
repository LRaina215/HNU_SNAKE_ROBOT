"""Apply the sectional A-D law only where its projection is well-defined.

Nearly opposite projected links do not define the circular fillet used by
the 2023 reconstruction. Split the correction into valid spans there; the
measured 3-D bending/rolling reference continues unchanged. This is a domain
restriction of that correction, not a clamp on physical joints or pipe axes.
"""
import numpy as np
from snake_control.three_segment_compliance import LocalSection
from snake_control.segmented_pole_backbone import smoothstep


def correction(geometry, zero_parameters, parameters, lo, hi, axis, corner, measured, phase_step):
    points=np.asarray(geometry['points_mm'])[lo:hi+2]
    links=np.diff(points,axis=0)
    projected=links-(links@axis)[:,None]*axis
    lengths=np.linalg.norm(projected,axis=1)
    gamma=np.arctan2(np.cross(projected[:-1],projected[1:])@axis,
                     (projected[:-1]*projected[1:]).sum(axis=1))
    minimum=geometry['curve']['gamma_min_rad']
    # Reuse the existing angular resolution instead of pretending a
    # sub-resolution fillet bisector is reliable. Fade over one resolution.
    angular_margin=np.pi-abs(gamma)
    ratio=lengths/np.linalg.norm(links,axis=1)
    projection_margin=np.minimum(ratio[:-1],ratio[1:])
    trust=smoothstep(np.clip((angular_margin-minimum)/minimum,0,1))
    trust*=smoothstep(np.clip((projection_margin-np.sin(minimum))/np.sin(minimum),0,1))
    valid=trust>0
    result=np.zeros(hi-lo);spans=[]
    start=0
    while start<hi-lo:
        if not valid[start]:start+=1;continue
        stop=start+1
        while stop<hi-lo and valid[stop]:stop+=1
        a,b=lo+start,lo+stop
        zero=LocalSection(geometry,zero_parameters,a,b,axis,corner)
        shaped=LocalSection(geometry,parameters,a,b,axis,corner)
        gauge=zero.align(measured[a:b],0.)
        result[start:stop]=(shaped.angles(gauge+phase_step)-zero.angles(gauge+phase_step))*trust[start:stop]
        spans.append([a,b]);start=stop
    return result,dict(joints=list(range(lo,hi)),projected_turn_rad=gamma,trust=trust,valid_spans=spans)
