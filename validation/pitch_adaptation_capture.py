"""Capture inputs/outputs around the unchanged v5 compute call, child-local.

No controller parameter, target, physical clock or rejection is changed.
Every attempt, including rejected input, is saved for independent limit replay.
"""
import copy
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(x) for x in value]
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def install_capture(controller,calls):
    original=controller.compute
    def wrapped(self,geometry,roll_phase,reference,*args,**kwargs):
        started=time.perf_counter();before_count=len(self.records)
        state={name:copy.deepcopy(getattr(self,name)) for name in ('key','alpha_state','radius_state','radius_valid_s','global_radius','global_valid_s','last_control_s')}
        clock=kwargs.get('control_time_s',args[2] if len(args)>2 else None)
        if clock is None:clock=self.control_clock() if self.control_clock is not None else geometry.get('sim_time')
        context=dict(method=self.method,k=self.exponent,fixed_alpha_rad=self.fixed_alpha,
                     initial_alpha_rad=self.initial_parameters.pitch_angle,gain=self.parameters.gain,inside=self.parameters.inside,
                     expected_lengths_mm=self.expected_lengths,physical_control_time_s=clock,before_state=state,
                     roll_phase_rad=roll_phase,reference_target_rad=np.asarray(reference).copy(),args=args,kwargs=kwargs)
        target=None;error=None
        try:
            target=original(self,geometry,roll_phase,reference,*args,**kwargs)
            return target
        except BaseException as exception:
            error=type(exception).__name__+': '+str(exception)
            raise
        finally:
            context.update(geometry=copy.deepcopy(geometry),accepted=target is not None,delivered_target_rad=None if target is None else np.asarray(target).copy(),
                           status=self.status,exception=error,original_control_record=copy.deepcopy(self.records[-1]) if len(self.records)>before_count else None)
            if target is not None:
                context['moving_initial_reference_rad']=(self.initial_backbone.joint_angles(len(self.expected_lengths)-1,self.expected_lengths,roll_phase)
                                                         if self.initial_backbone is not None else self.transfer_reference.copy())
            context['capture_total_call_ms']=(time.perf_counter()-started)*1000
            calls.append(serial(context))
    controller.compute=wrapped
    return original


def reconstruct_limits(call):
    if not call['accepted']:return dict(accepted=False,status=call['status'],exception=call['exception'])
    row=call['original_control_record'];used=np.asarray([np.nan if x is None else x for x in row['radius_used_mm']])
    active=np.asarray(row['pitch_constraint_active'],bool);count=len(used)
    candidate_p=np.full(count,np.nan);p_clipped=np.full(count,np.nan);before_bounds=np.full(count,call['fixed_alpha_rad'])
    requested=before_bounds.copy()
    if active.any():
        candidate_p[active]=220.*(85.5/used[active])**call['k'];p_clipped[active]=np.clip(candidate_p[active],100.,300.)
        before_bounds[active]=np.arctan2(p_clipped[active],2*np.pi*used[active])
        requested[active]=np.clip(before_bounds[active],math.radians(8.),math.radians(30.))
        weights=gaussian_filter1d(active.astype(float),1.,mode='nearest')
        numerator=gaussian_filter1d(np.where(active,requested,0.),1.,mode='nearest')
        spatial=np.divide(numerator,weights,out=requested.copy(),where=weights>1e-12)
        spatial=np.where(active,spatial,call['fixed_alpha_rad'])
    else:spatial=requested
    if not np.allclose(spatial,row['requested_alpha_rad'],atol=1e-12,rtol=0):raise ValueError('Captured requested alpha differs from limit replay')
    state=call['before_state'];key=state['key'];geometry=call['geometry']
    reset=key is None or list(key[:2])!=[geometry['run_id'],geometry['config_id']]
    previous=np.full(count,call['initial_alpha_rad']) if reset else np.asarray(state['alpha_state'])
    dt=row['dt_s'];delta=(1.-.8**(dt/.2))*(spatial-previous);step=math.radians(1.)*dt/.2
    after_time=previous+delta;after_rate=previous+np.clip(delta,-step,step)
    lo=np.full(count,math.radians(8.));hi=np.full(count,math.radians(30.))
    lo[active]=np.maximum(lo[active],np.arctan2(100.,2*np.pi*used[active]))
    hi[active]=np.minimum(hi[active],np.arctan2(300.,2*np.pi*used[active]))
    final=np.clip(after_rate,lo,hi)
    if not np.allclose(final,row['alpha_rad'],atol=1e-12,rtol=0):raise ValueError('Captured final alpha differs from limit replay')
    blend=row['takeover_blend'];raw=(1.-blend)*np.asarray(call['moving_initial_reference_rad'])+blend*np.asarray(row['unblended_target_rad'])
    bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])+[.01,-.01]
    signs=np.where(np.arange(len(raw))%2,-1,1);joint_lo,joint_hi=np.sort(bounds*signs[:,None],axis=1).T
    delivered=np.clip(raw,joint_lo,joint_hi);reference=np.asarray(call['reference_target_rad'])
    if not np.allclose(delivered,call['delivered_target_rad'],atol=1e-12,rtol=0):raise ValueError('Captured delivered target differs from limit replay')
    differences=delivered-reference
    return serial(dict(accepted=True,sim_time=row['sim_time'],policy_active=active,candidate_pitch_before_clip_mm=candidate_p,
                       pitch_after_clip_mm=p_clipped,alpha_before_angle_bounds_rad=before_bounds,
                       alpha_after_angle_bounds_rad=requested,alpha_after_spatial_filter_rad=spatial,
                       alpha_after_time_filter_before_rate_rad=after_time,alpha_after_rate_limit_rad=after_rate,
                       alpha_after_pitch_feasibility_rad=final,joint_before_limit_rad=raw,joint_after_limit_rad=delivered,
                       joint_change_rad=differences,candidate_pitch_clip_nodes=int(np.count_nonzero(abs(candidate_p-p_clipped)>1e-12)),
                       angle_bound_nodes=int(np.count_nonzero(abs(before_bounds-requested)>1e-12)),
                       physical_rate_limit_nodes=int(np.count_nonzero(abs(delta)>step)),
                       final_pitch_feasibility_nodes=int(np.count_nonzero(abs(after_rate-final)>1e-12)),
                       joint_limit_nodes=int(np.count_nonzero(raw!=delivered)),maximum_joint_change_deg=float(np.degrees(abs(differences)).max()),
                       integrated_connection_rad=row['integrated_connection_rad']))
