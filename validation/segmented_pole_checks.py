"""Independent geometry/integral checks before paired physical trials."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import numpy as np
from numpy.polynomial.legendre import leggauss

from segmented_pole_experiment import (HORIZONTAL, BASE, load_checkpoint, pipe_frame,
    true_axis_geometry, build_backbone, MovingAxisBackbone, SwitchContinuity, write_json)
from snake_control.segmented_pole_backbone import BoundaryTracker,SegmentedOptions


def main():
    m,d,a,c=load_checkpoint(HORIZONTAL)
    corner,source,target,radius=pipe_frame(m)
    g=a['geometry_at_capture'];p=np.array(g['points_mm']);links=np.array(a['settings']['links'])
    phase=a['settings']['phase']
    bone=MovingAxisBackbone(p,links,c.parameters,source,target,6*64.5,4*64.5)
    straight=MovingAxisBackbone(p,links,c.parameters,source,source,6*64.5,4*64.5)
    flat_error=max(np.max(abs(straight.joint_angles(ph)-straight.flat.joint_angles(ph))) for ph in [0,.3,1.7])
    assert flat_error<1e-6,flat_error
    original=build_backbone(g,c.parameters)
    reprojected=build_backbone(true_axis_geometry(g,np.array(g['projection_axis']['direction']),
                                                np.array(g['projection_axis']['point_mm'])),c.parameters)
    section_error=np.max(abs(original.joint_angles(.3)-reprojected.joint_angles(.3)))
    assert section_error<1e-10,section_error
    switch=SwitchContinuity(bone,d.ctrl,phase)
    switch_error=np.max(abs(switch.angles(bone,phase,0)-d.ctrl))
    assert switch_error<1e-10,switch_error
    # Integrate sampled bending components independently of cumulative ODE states.
    nodes,weights=leggauss(20);linkbounds=np.r_[0.,np.cumsum(links)]
    breaks=np.unique(np.r_[bone.bounds,linkbounds]);integrals=np.zeros((21,2))
    ph=phase-.3;weight=.4
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        if hi-lo<1e-8:continue
        s=(lo+hi)/2+(hi-lo)/2*nodes
        field=bone.bending_field(s,ph+switch.offset)+(1-weight)*switch.correction_field(s,ph)
        j=np.clip(np.searchsorted(linkbounds,(lo+hi)/2,side='right')-1,0,20)
        integrals[j]+=np.sum(field*weights[:,None],axis=0)*(hi-lo)/2
    b=integrals[:-1]+integrals[1:]
    q=-np.where(np.arange(20)%2,b[:,0],b[:,1])
    quadrature_error=np.max(abs(q-switch.angles(bone,ph,weight)))
    assert quadrature_error<2e-6,quadrature_error
    # Independently differentiate world tangent; verify all axis-rotation terms.
    tangent_error=0.
    for k in range(len(bone.spans)):
        s=(bone.bounds[k]+bone.bounds[k+1])/2;eps=1e-3
        T,b,w=bone.components(s,k)
        derivative=(bone.components(s+eps,k)[0]-bone.components(s-eps,k)[0])/(2*eps)
        tangent_error=max(tangent_error,abs(np.linalg.norm(derivative)-np.linalg.norm(b)))
    assert tangent_error<1e-7,tangent_error
    tracker=BoundaryTracker(SegmentedOptions(),64.5)
    centers=np.vstack([np.column_stack([np.zeros(8),np.linspace(100,400,8),np.zeros(8)]),
                       np.column_stack([-np.linspace(100,700,13),np.zeros(13),np.zeros(13)])])
    contact=np.zeros(21,bool);contact[7]=True
    for _ in range(2):tracker.update(centers,np.zeros(3),np.array([1.,0,0]),np.array([0,1.,0]),contact)
    assert tracker.value==6.,'boundary changed before persistence gate'
    tracker.update(centers,np.zeros(3),np.array([1.,0,0]),np.array([0,1.,0]),contact)
    assert tracker.value==6.25,'boundary movement not bounded'
    before=tracker.value
    for _ in range(4):tracker.update(centers,np.zeros(3),np.array([1.,0,0]),np.array([0,1.,0]),np.zeros(21,bool))
    assert tracker.value==before,'unconfirmed target assignment moved boundary'
    sample=bone.sample_command(ph+switch.offset,switch,weight)
    assert sample['frame_orthogonality_error']<1e-5
    report=dict(straight_integral_error_rad=float(flat_error),same_axis_section_error_rad=float(section_error),
                zero_switch_error_rad=float(switch_error),independent_quadrature_error_rad=float(quadrature_error),
                tangent_derivative_error_per_mm=float(tangent_error),
                boundary_persistence_and_contact_gate=True,
                reconstruction_frame_error=sample['frame_orthogonality_error'])
    write_json(BASE/'geometry_checks.json',report)
    write_json(BASE/'geometry_sample.json',sample)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
