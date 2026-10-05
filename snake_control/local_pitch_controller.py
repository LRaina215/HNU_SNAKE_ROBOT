"""Experimental straight-axis controller for pitch comparisons, simulation only.

The isolated runner injects this class into the existing EXE relay entry point.
No model radius, contact, feature boundary, or true axis is available here.
"""
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from snake_control.pole_compliance import ComplianceController, ComplianceParameters, build_backbone
from snake_control.local_pitch_geometry import VariablePitchBackbone, estimate_local_radius, pitch_rule


class LocalPitchController(ComplianceController):
    method = 'D'
    exponent = 0.
    fixed_alpha = math.atan2(220.,2*np.pi*81.5)
    window = 9
    records = []

    def __init__(self, parameters, expected_lengths):
        self.initial_parameters = parameters
        super().__init__(ComplianceParameters(parameters.gain,self.fixed_alpha,parameters.inside),expected_lengths)
        self.alpha_state = None
        self.radius_state = None
        self.last_valid_ms = None
        self.radius_valid_ms = None
        self.frames = 0

    def compute(self, geometry, roll_phase, reference, now_ms=None, reference_phase=None):
        now_ms = time.time()*1000 if now_ms is None else now_ms
        # Reuse the production freshness/configuration checks without committing
        # a rejected candidate to the live controller state.
        validator = ComplianceController(self.parameters,self.expected_lengths)
        # A fresh validator deliberately rebuilds and validates the frame.
        checked = validator.compute(geometry,roll_phase,reference,now_ms,reference_phase)
        if checked is None:
            self.status = validator.status
            return None
        try:
            key=(geometry['run_id'],geometry['config_id'],geometry['sequence'])
            if self.key and key[:2]==self.key[:2] and key[2]<self.key[2]:
                raise ValueError('反馈帧倒序')
            estimates=estimate_local_radius(geometry['points_mm'],geometry['projection_axis'],self.window)
            count=len(geometry['points_mm'])
            valid=np.array([r['valid'] for r in estimates])
            radius=np.array([r['radius_mm'] if r['valid'] else np.nan for r in estimates])
            previous=np.full(count,85.5) if self.radius_state is None else self.radius_state
            radius=np.where(valid,radius,previous)
            valid_ms=np.full(count,-np.inf) if self.radius_valid_ms is None else self.radius_valid_ms.copy()
            valid_ms[valid]=now_ms
            expired=now_ms-valid_ms>500.
            timeout=bool(expired.all())
            degraded=not valid.all()
            if self.method in ('A','B'):
                requested=np.full(count,self.fixed_alpha)
            elif timeout:
                requested=np.full(count,self.fixed_alpha)
            else:
                used=radius if self.method=='D' else np.full(count,float(np.median(radius[valid])))
                requested,_=pitch_rule(used,self.exponent)
                requested=gaussian_filter1d(requested,1.,mode='nearest')
                requested=np.where(expired,self.fixed_alpha,requested)
            # One common spatial/temporal filter and one-degree step limit.
            previous_alpha=np.full(count,self.initial_parameters.pitch_angle) if self.alpha_state is None else self.alpha_state
            filtered=previous_alpha+.2*(requested-previous_alpha)
            alpha=previous_alpha+np.clip(filtered-previous_alpha,-np.radians(1.),np.radians(1.))
            uniform=bool(np.ptp(alpha)<1e-14)
            backbone=(build_backbone(geometry,ComplianceParameters(self.parameters.gain,float(alpha[0]),self.parameters.inside)) if uniform else
                      VariablePitchBackbone(geometry,self.parameters,alpha,np.r_[0.,np.cumsum(self.expected_lengths)]))
            phase_offset=self.phase_offset
            if not self.started or key[:2]!=self.key[:2]:
                basis=np.column_stack([backbone.joint_angles(0.),backbone.joint_angles(np.pi/2)])
                weights=np.linalg.lstsq(basis,reference,rcond=None)[0]
                phase_offset=math.atan2(weights[1],weights[0])-(roll_phase if reference_phase is None else reference_phase)
            raw=backbone.joint_angles(roll_phase+phase_offset)
            signs=np.where(np.arange(len(raw))%2,-1,1)
            bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])+[.01,-.01]
            low,high=np.sort(bounds*signs[:,None],axis=1).T
            clipped=np.clip(raw,low,high)
            jump=float(np.degrees(abs(clipped-reference)).max())
            # Freeze this rejection policy before training; all four methods share it.
            if jump>30.:
                raise ValueError(f'整组目标变化{jump:.3f}度超过30度')
            if not np.isfinite(clipped).all():
                raise ValueError('nonfinite pitch target')
            elapsed_ms=time.time()*1000-now_ms
            self.records.append(dict(sim_time=geometry.get('sim_time'),sequence=key[2],method=self.method,
                                     radius_mm=radius.tolist(),radius_valid=valid.tolist(),estimates=estimates,
                                     requested_alpha_rad=requested.tolist(),alpha_rad=alpha.tolist(),
                                     implied_pitch_mm=(2*np.pi*radius*np.tan(alpha)).tolist(),
                                     degraded=degraded,fallback_fixed=bool(expired.any()),
                                     alpha_rate_limited=int(np.count_nonzero(abs(filtered-previous_alpha)>np.radians(1.))),
                                     joint_clipped=int(np.count_nonzero(raw!=clipped)),max_jump_deg=jump,
                                     compute_ms=elapsed_ms,phase_offset=phase_offset,
                                     integrated_connection_rad=float(backbone.twists[-1] if uniform else backbone.end_states[-1,3]),
                                     target_mm=backbone.sample(10.).tolist()))
            self.backbone,self.phase_offset,self.key=backbone,phase_offset,key
            self.alpha_state,self.radius_state,self.radius_valid_ms=alpha,radius,valid_ms
            self.started=True;self.status='局部螺距实验控制运行中'
            return clipped
        except (ValueError,KeyError,TypeError) as error:
            self.status=str(error)
            return None
