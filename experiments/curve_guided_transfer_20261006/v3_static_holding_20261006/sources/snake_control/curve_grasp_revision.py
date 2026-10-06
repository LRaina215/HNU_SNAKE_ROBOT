"""V3: fixed material feed + fitted finite-shell reference + retained A-D shape state.

No world pose/contact inputs. Geometry fitting is offline and not a physical aid.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
from snake_control.curve_guided_transfer import FiveSegmentCurve, CurveParameters
from snake_control.pole_compliance import ComplianceParameters
from snake_control.three_segment_compliance import LocalSection


class FittedMaterialCurve:
    def __init__(self, path):
        self.prototype = FiveSegmentCurve(CurveParameters(radius_mm=95.,initial_target_links=7.))
        with np.load(path) as data:
            self.feeds = data['feed_mm'].copy()
            self.states = data['states'].copy()
        if len(self.feeds) < 2 or self.feeds[-1] < 161.25:
            raise ValueError('complete checked material reference required')
        self.interpolator = PchipInterpolator(self.feeds,self.states,axis=0)
        self.orientation = Slerp(self.feeds,Rotation.from_rotvec(self.states[:,3:6]))

    def __getattr__(self,name):
        return getattr(self.prototype,name)

    def state(self,feed):
        if not 0 <= feed <= self.feeds[-1]:
            raise ValueError('feed outside checked material reference')
        state=self.interpolator(feed)
        state[3:6]=self.orientation(feed).as_rotvec()
        return state

    def joint_angles(self,feed,roll_rad=0.):
        if roll_rad != 0:
            raise ValueError('extra rolling is not part of this candidate')
        return self.state(feed)[6:]

    def nominal_root(self,feed):
        state=self.state(feed)
        return state[:3]-np.array([1100.,0.,1000.]),Rotation.from_rotvec(state[3:6]).as_matrix()

    def feed_tangent(self,feed):
        return self.interpolator.derivative()(feed)[6:]


class RetainedGraspController:
    def __init__(self,curve,config,reference,forward_points,registered_rotation):
        self.curve,self.config,self.forward_points=curve,config,forward_points
        self.reference=np.asarray(reference).copy()
        self.last=self.reference.copy()
        self.shape=np.zeros(20)
        self.parameters=ComplianceParameters((.01,.02),curve.alpha,False)
        self.zero=ComplianceParameters(0.,curve.alpha,False)
        _,orientation=curve.nominal_root(0.)
        self.registration=registered_rotation @ orientation.T

    def compute(self,measured,feed,elapsed,dt):
        measured=np.asarray(measured).copy()
        if measured.shape != (20,) or not np.all(np.isfinite(measured)):
            raise ValueError('20 finite fresh encoders required')
        nominal=self.curve.joint_angles(feed)
        _,orientation=self.curve.nominal_root(feed)
        orientation=self.registration @ orientation
        points=self.forward_points(measured)
        centers=self.curve.lambda0+feed-(np.arange(20)+1)*self.curve.p.link_mm
        l=self.curve.p.link_mm
        smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
        weights=[smooth((-centers-l)/l),smooth((centers-self.curve.bounds[-1]-l)/l)]
        geometry=dict(config=self.config,points_mm=points,curve=dict(gamma_min_rad=math.radians(1.)))
        blend=np.clip(elapsed/5.,0,1);blend=blend*blend*(3-2*blend)
        delta=np.zeros(20);updates=np.zeros(20);errors=[]
        tangent=self.curve.feed_tangent(feed)
        for weight,axis in zip(weights,[np.array([1.,0.,0.]),np.array([0.,1.,0.])]):
            ids=np.flatnonzero(weight>0)
            if len(ids)<2:
                continue
            lo,hi=int(ids[0]),int(ids[-1]+1)
            try:
                zero=LocalSection(geometry,self.zero,lo,hi,orientation.T@axis,np.zeros(3))
                shaped=LocalSection(geometry,self.parameters,lo,hi,orientation.T@axis,np.zeros(3))
                gauge=zero.align(measured[lo:hi],0.)
                difference=shaped.angles(gauge)-zero.angles(gauge)
                delta[lo:hi]=difference
                # Retain successful deformation. Tracking error prevents target
                # integration through an immobile joint; no contact-force oracle.
                change=blend*weight[lo:hi]*(difference-(self.last[lo:hi]-measured[lo:hi]))
                g=weight[lo:hi]*tangent[lo:hi]
                norm=float(g@g)
                if norm>1e-12:
                    change-=g*float(g@change)/norm
                updates[lo:hi]=change
            except ValueError as error:
                errors.append(str(error))
        weight=weights[0]+weights[1]
        self.shape[weight==0]=0.
        self.shape+=updates
        residual=weight*self.shape
        raw=(1-blend)*self.reference+blend*(nominal+residual)
        if np.degrees(abs(raw)).max()>90:
            raise ValueError('retained grasp target exceeds existing 90 degree joint limit')
        maximum=math.radians(10.)*dt
        target=self.last+np.clip(raw-self.last,-maximum,maximum)
        self.last=target.copy()
        return target,dict(feed_mm=float(feed),roll_rad=0.,nominal_rad=nominal.tolist(),
            compliance_delta_rad=delta.tolist(),retained_shape_rad=self.shape.tolist(),effective_residual_rad=residual.tolist(),
            shape_update_rad=updates.tolist(),grasp_weights=weight.tolist(),correction_errors=errors,
            rate_limited_joints=int(np.count_nonzero(abs(raw-target)>1e-10)),
            feedback_fields=['encoder_angles','last_sent_targets','retained_shape_state'],
            limitation='axes are propagated by commanded offline reference; actual global drift remains unobserved')
