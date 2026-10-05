"""Experimental pitch extension of the unchanged production compliance rule.

Version 4 adds common physical-time takeover to the version 3 feedback repair.
Only measured geometry enters targets; model metadata is never read here.
"""
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from snake_control.pole_compliance import ComplianceController, ComplianceParameters, build_backbone
from snake_control.local_pitch_geometry import VariablePitchBackbone, pitch_rule
from snake_control.local_pitch_feedback import estimate_module_radius


class LocalPitchController(ComplianceController):
    method='D'
    exponent=0.
    fixed_alpha=math.atan2(220.,2*np.pi*81.5)
    window=7
    records=[]
    control_clock=None
    transfer_s=2.

    def __init__(self,parameters,expected_lengths):
        self.initial_parameters=parameters
        super().__init__(ComplianceParameters(parameters.gain,self.fixed_alpha,parameters.inside),expected_lengths)
        self.alpha_state=None
        self.radius_state=None
        self.radius_valid_s=None
        self.global_radius=None
        self.global_valid_s=-np.inf
        self.last_control_s=None
        self.transfer_start_s=None
        self.transfer_reference=None
        # An experimental run must never silently roll with nominal targets
        # after common initialization if the compliance candidate is rejected.
        self.started=True

    def compute(self,geometry,roll_phase,reference,now_ms=None,reference_phase=None,control_time_s=None):
        started=time.perf_counter()
        wall_ms=time.time()*1000 if now_ms is None else now_ms
        validator=ComplianceController(self.parameters,self.expected_lengths)
        if validator.compute(geometry,roll_phase,reference,wall_ms,reference_phase) is None:
            self.status=validator.status
            return None
        try:
            clock=control_time_s if control_time_s is not None else self.control_clock() if self.control_clock is not None else geometry.get('sim_time')
            if clock is None or not math.isfinite(clock):
                raise ValueError('局部螺距实验缺少物理控制时间')
            key=(geometry['run_id'],geometry['config_id'],geometry['sequence'])
            reset=self.key is None or key[:2]!=self.key[:2]
            if not reset and (key[2]<self.key[2] or clock<self.last_control_s-1e-9):
                raise ValueError('反馈帧或物理时间倒序')
            estimate=estimate_module_radius(geometry['points_mm'],geometry['projection_axis'],self.window)
            rows=estimate['nodes'];count=len(rows)
            if count!=len(self.expected_lengths)+1:
                raise ValueError('身体点数与机械配置不匹配')
            valid=np.array([r['valid'] for r in rows])
            observed=np.array([r['radius_mm'] if r['valid'] else np.nan for r in rows])
            radius=np.full(count,np.nan) if reset else self.radius_state.copy()
            valid_s=np.full(count,-np.inf) if reset else self.radius_valid_s.copy()
            radius[valid]=observed[valid];valid_s[valid]=clock
            usable=np.isfinite(radius)&(clock-valid_s<=.5+1e-9)
            global_radius=None if reset else self.global_radius
            global_valid_s=-np.inf if reset else self.global_valid_s
            if valid.any():
                global_radius=float(np.median(observed[valid]));global_valid_s=clock
            requested=np.full(count,self.fixed_alpha)
            used=radius.copy();policy_valid=usable.copy()
            if self.method=='C':
                policy_valid[:]=global_radius is not None and clock-global_valid_s<=.5+1e-9
                used[:]=np.nan if global_radius is None else global_radius
            elif self.method not in ('A','B','D'):
                raise ValueError('unknown pitch method')
            if self.method in ('C','D') and policy_valid.any():
                requested[policy_valid]=pitch_rule(used[policy_valid],self.exponent)[0]
                # Confidence-normalized spatial filtering prevents unknown
                # placeholder radii from entering neighbouring valid targets.
                weights=gaussian_filter1d(policy_valid.astype(float),1.,mode='nearest')
                numerator=gaussian_filter1d(np.where(policy_valid,requested,0.),1.,mode='nearest')
                smoothed=np.divide(numerator,weights,out=requested.copy(),where=weights>1e-12)
                requested=np.where(policy_valid,smoothed,self.fixed_alpha)
            previous=np.full(count,self.initial_parameters.pitch_angle) if reset else self.alpha_state.copy()
            dt=.2 if reset else max(0.,clock-self.last_control_s)
            gain=1.-.8**(dt/.2)
            delta=gain*(requested-previous)
            step_limit=math.radians(1.)*dt/.2
            alpha=previous+np.clip(delta,-step_limit,step_limit)
            # Feasibility is evaluated AFTER filtering. Fixed-angle A/B and
            # explicit fallback are exempt from the candidate-P constraint;
            # they remain logged. For active C/D radii, enforce both bounds.
            p_active=policy_valid&(self.method in ('C','D'))
            low=np.full(count,math.radians(8.));high=np.full(count,math.radians(30.))
            low[p_active]=np.maximum(low[p_active],np.arctan2(100.,2*np.pi*used[p_active]))
            high[p_active]=np.minimum(high[p_active],np.arctan2(300.,2*np.pi*used[p_active]))
            if np.any(low>high):
                raise ValueError('反馈半径下P与alpha约束无交集')
            feasible=np.clip(alpha,low,high)
            if np.any(abs(feasible-previous)>step_limit+1e-12):
                raise ValueError('物理限速与P可行区无交集，保持上一组目标')
            alpha=feasible
            uniform=bool(np.ptp(alpha)<1e-14)
            backbone=(build_backbone(geometry,ComplianceParameters(self.parameters.gain,float(alpha[0]),self.parameters.inside)) if uniform else
                      VariablePitchBackbone(geometry,self.parameters,alpha,np.r_[0.,np.cumsum(self.expected_lengths)]))
            phase_offset=self.phase_offset
            if reset:
                basis=np.column_stack([backbone.joint_angles(0.),backbone.joint_angles(np.pi/2)])
                weights=np.linalg.lstsq(basis,reference,rcond=None)[0]
                phase_offset=math.atan2(weights[1],weights[0])-(roll_phase if reference_phase is None else reference_phase)
            raw=backbone.joint_angles(roll_phase+phase_offset)
            transfer_start=clock if reset else self.transfer_start_s
            transfer_reference=np.asarray(reference).copy() if reset else self.transfer_reference
            blend=1. if self.transfer_s==0 else min(1.,max(0.,(clock-transfer_start+.2)/self.transfer_s))
            unblended=raw.copy()
            raw=(1.-blend)*transfer_reference+blend*raw
            signs=np.where(np.arange(len(raw))%2,-1,1)
            bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])+[.01,-.01]
            joint_low,joint_high=np.sort(bounds*signs[:,None],axis=1).T
            target=np.clip(raw,joint_low,joint_high)
            jump=float(np.degrees(abs(target-reference)).max())
            if jump>30. or not np.isfinite(target).all():
                raise ValueError(f'整组目标变化{jump:.3f}度超限或非有限')
            implied=2*np.pi*used*np.tan(alpha)
            serial=lambda arr:[float(x) if np.isfinite(x) else None for x in arr]
            self.records.append(dict(version=4,sim_time=float(clock),feedback_sim_time=geometry.get('sim_time'),sequence=key[2],method=self.method,
                                     radius_mm=serial(radius),radius_used_mm=serial(used),radius_valid=valid.tolist(),radius_usable=policy_valid.tolist(),
                                     estimates=rows,module_estimates=estimate['modules'],requested_alpha_rad=requested.tolist(),alpha_rad=alpha.tolist(),
                                     implied_pitch_mm=serial(implied),pitch_constraint_active=p_active.tolist(),fallback_fixed=bool(not policy_valid.all()),
                                     degraded=bool(not valid.all()),dt_s=float(dt),alpha_rate_limited=int(np.count_nonzero(abs(delta)>step_limit)),
                                     joint_clipped=int(np.count_nonzero(raw!=target)),max_jump_deg=jump,compute_ms=(time.perf_counter()-started)*1000,
                                     phase_offset=phase_offset,takeover_blend=float(blend),unblended_target_rad=unblended.tolist(),integrated_connection_rad=float(backbone.twists[-1] if uniform else backbone.end_states[-1,3]),
                                     target_mm=backbone.sample(10.).tolist()))
            self.backbone,self.phase_offset,self.key=backbone,phase_offset,key
            self.alpha_state,self.radius_state,self.radius_valid_s=alpha,radius,valid_s
            self.global_radius,self.global_valid_s=global_radius,global_valid_s
            self.transfer_start_s,self.transfer_reference=transfer_start,transfer_reference
            self.last_control_s=float(clock);self.started=True
            self.status='原截面顺应性＋螺距反馈实验v4'
            return target
        except (ValueError,KeyError,TypeError) as error:
            self.status=str(error)
            return None
