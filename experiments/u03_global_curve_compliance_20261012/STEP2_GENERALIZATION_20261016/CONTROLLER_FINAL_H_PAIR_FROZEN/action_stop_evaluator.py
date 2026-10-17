"""Evaluation-only arbitrary true axis; orientation increments from actual pose."""
import numpy as np
from scipy.spatial.transform import Rotation
class ActionStopEvaluator:
    def __init__(self,truth,acceptance):
        self.truth=truth;self.reg=acceptance;self.previous_rotation=None;self.previous_tangent=None;self.cumulative=np.zeros(21);self.window=[];self.complete=None
    def update(self,row):
        c=row['centers'];p=row['points'];R=row['body_xmat'].reshape(21,3,3);tangent=p[:-1]-p[1:];tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        if self.previous_rotation is not None:
            delta=Rotation.from_matrix(R@np.transpose(self.previous_rotation,(0,2,1))).as_rotvec();direction=tangent+self.previous_tangent;direction/=np.linalg.norm(direction,axis=1)[:,None];self.cumulative+=np.sum(delta*direction,axis=1)
        self.previous_rotation=R.copy();self.previous_tangent=tangent.copy();row['actual_cumulative_orientation_axial_spin_rad']=self.cumulative.copy()
        axis=np.asarray(self.truth['target_axis']);origin=np.asarray(self.truth['target_straight_origin_m']);lo,hi=self.truth['target_effective_axial_bounds_m'];relative=c-origin;axial=relative@axis;radial=relative-axial[:,None]*axis;rho=np.linalg.norm(radial,axis=1);ends=(p-origin)@axis;e=np.array([1.,0.,0.]);e-=axis*(e@axis);e/=np.linalg.norm(e);b=np.cross(e,axis);span=np.ptp(np.unwrap(np.arctan2(radial@b,radial@e)));nf=row['eval_pipe_normal_N'];rlo,rhi=self.reg['center_radius_m']
        full=bool(np.all((rho>rlo)&(rho<rhi)&(axial>lo)&(axial<hi)) and np.all((ends>lo)&(ends<hi)) and nf[:2].sum()<.1 and nf[2].sum()>.2 and span>=2*np.pi)
        row['evaluation_full21']=full;row['evaluation_target_axial_m']=axial.copy()
        if not full:self.window=[];return False
        self.window.append((row['t'],axial.copy(),self.cumulative.copy()))
        start=self.window[0];duration=row['t']-start[0]
        if duration<self.reg['post_transfer_s']:return False
        spin=np.degrees(self.cumulative-start[2]);advance=axial-start[1];passed=bool((advance>self.reg['minimum_each_axial_advance_m']).all() and (abs(spin)>self.reg['minimum_each_same_direction_axial_spin_deg']).all() and ((spin>0).all() or (spin<0).all()))
        if passed:self.complete=dict(start_s=start[0],end_s=row['t'],duration_s=duration,each_advance_mm=(1000*advance).tolist(),each_orientation_spin_deg=spin.tolist())
        return passed
