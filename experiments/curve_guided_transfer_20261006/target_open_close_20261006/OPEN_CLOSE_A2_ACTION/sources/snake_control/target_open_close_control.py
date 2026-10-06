"""Target-side opening coordinated with original R2 rear compliance.

Target A-D tightening smoothly pauses at full opening and recovers during
closing. Encoder tracking and existing residual restoration remain active;
rear A-D feedback is unchanged. Timing is prescribed, not contact switching.
"""
import math
import numpy as np
from snake_control.executable_grasp_state import EquilibriumGraspController
from snake_control.three_segment_compliance import LocalSection


class TargetOpenCloseController(EquilibriumGraspController):
    def compute(self, measured, action, elapsed, dt):
        if action == 0. and self.last_feed == 0.:
            # For every observation/ACK state the complete starting branch is R2.
            return super().compute(measured, action, elapsed, dt)
        measured = np.asarray(measured, float)
        if measured.shape != (20,) or not np.isfinite(measured).all():
            raise ValueError('20 finite encoders required')
        nominal = self.curve.joint_angles(action)
        retained = self.last-self.curve.joint_angles(self.last_feed)
        feed = self.curve.material_feed(action)
        opening = self.curve.opening(action)
        front_gain = 1.-opening/self.curve.opening_mm
        p = self.forward_points(measured)
        pn = self.forward_points(nominal)
        _, orientation = self.curve.nominal_root(action)
        orientation = self.registration @ orientation
        centers = self.curve.lambda0+feed-(np.arange(20)+1)*64.5
        smooth = lambda u:np.clip(u, 0, 1)**2*(3-2*np.clip(u, 0, 1))
        weights = [smooth((-centers-64.5)/64.5), smooth((centers-self.curve.bounds[-1]-64.5)/64.5)]
        blend = smooth(elapsed/5.)
        geometry = dict(config=self.config, points_mm=p, curve=dict(gamma_min_rad=math.radians(1)))
        delta, update = np.zeros(20), np.zeros(20)
        axes = []
        for side, (weight, world_axis) in enumerate(zip(weights, [np.array([1.,0,0]),np.array([0.,1.,0])])):
            ids = np.flatnonzero(weight>1e-8)
            if len(ids)<2:
                continue
            lo, hi = int(ids[0]), int(ids[-1]+1)
            a, b = pn[lo:hi+2].copy(), p[lo:hi+2].copy()
            a-=a.mean(0); b-=b.mean(0)
            u, singular, v = np.linalg.svd(a.T @ b)
            transport = v.T @ np.diag([1,1,np.linalg.det(v.T @ u.T)]) @ u.T
            rms = float(np.sqrt(np.mean(np.sum((a@transport.T-b)**2,axis=1))))
            valid = rms<=64.5/4 and singular[1]/singular[0]>=.05
            axis = transport @ (orientation.T @ world_axis)
            # Original validation remains mandatory whenever feedback is used.
            if not valid and (side == 0 or front_gain>1e-8):
                raise ValueError('material-cloud coordinate hypothesis invalid: residual or coverage')
            difference = np.zeros(hi-lo)
            if valid:
                zero = LocalSection(geometry,self.zero,lo,hi,axis,np.zeros(3))
                shaped = LocalSection(geometry,self.parameters,lo,hi,axis,np.zeros(3))
                gauge = zero.align(measured[lo:hi],0.)
                difference=shaped.angles(gauge)-zero.angles(gauge)
            delta[lo:hi]=difference
            error=self.last[lo:hi]-measured[lo:hi]+self.restoring_error(retained[lo:hi],dt)
            gain=1. if side==0 else front_gain
            change=blend*weight[lo:hi]*(gain*difference-error)
            # Retain original material projector on both sides. Switching to
            # a radial-action tangent at infinitesimal opening would create a
            # finite update discontinuity. AD gating coordinates opening.
            tangent=self.curve.original.feed_tangent(feed)
            g=weight[lo:hi]*tangent[lo:hi]
            if float(g@g)>1e-12:
                change-=g*float(g@change)/float(g@g)
            update[lo:hi]=change
            axes.append(dict(joints=[lo+1,hi],direction_head=axis.tolist(),cloud_rms_mm=rms,
                cloud_valid=bool(valid),feedback_gain=float(gain),side='source' if side==0 else 'target'))
        retained[weights[0]+weights[1]<=1e-8]=0.
        raw=nominal+retained+update
        bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in self.config['joints']])
        signs=np.where(np.arange(20)%2,-1.,1.)
        low,high=np.sort(bounds*signs[:,None],axis=1).T
        limit=math.radians(10)*dt
        target=np.clip(raw,np.maximum(low,self.last-limit),np.minimum(high,self.last+limit))
        feasible=target-nominal
        self.pending=target.copy(),float(action)
        return target,dict(action_mm=float(action),feed_mm=feed,opening_mm=opening,
            reference_plateau_radius_mm=95.+opening,target_AD_gain=float(front_gain),
            roll_rad=0.,nominal_rad=nominal.tolist(),compliance_delta_rad=delta.tolist(),
            retained_shape_rad=feasible.tolist(),effective_residual_rad=feasible.tolist(),
            proposed_retained_rad=(retained+update).tolist(),shape_update_rad=(feasible-retained).tolist(),
            raw_target_rad=raw.tolist(),execution_projection_rad=(target-raw).tolist(),
            grasp_weights=(weights[0]+weights[1]).tolist(),material_axes=axes,correction_errors=[],
            rate_limited_joints=int(np.count_nonzero(abs(raw-target)>1e-10)),
            angle_constrained_joints=int(np.count_nonzero((raw<low)|(raw>high))),
            restoring_time_constant_s=.5,feedback_fields=['encoder_angles','ACK_history','known_initial_geometry'],
            stage_switch='prescribed open-insert-close timing; no truth/contact switching',
            limitation='Conditional axis transport cannot observe arbitrary rigid world drift')
