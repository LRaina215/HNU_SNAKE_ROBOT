"""One scalar segment parameter for translation, SO(3) and joint angles.

The segment image is the coupled straight/SO(3) connection. Cubic smoothstep
gives C1 zero-velocity joins; offline collision and sampled command checks are
required. This reference does not move the simulated base or read world pose.
"""
import numpy as np
from snake_control.curve_grasp_revision import FittedMaterialCurve


class CoupledMaterialCurve(FittedMaterialCurve):
    def coordinate(self,feed):
        if not 0<=feed<=self.feeds[-1]:
            raise ValueError('feed outside checked local material reference')
        i=min(max(np.searchsorted(self.feeds,feed,side='right')-1,0),len(self.feeds)-2)
        span=self.feeds[i+1]-self.feeds[i]
        u=(feed-self.feeds[i])/span
        return i,span,u,u*u*(3-2*u)

    def state(self,feed):
        i,span,u,h=self.coordinate(feed)
        state=self.states[i]+h*(self.states[i+1]-self.states[i])
        state[3:6]=self.orientation(self.feeds[i]+h*span).as_rotvec()
        return state

    def joint_derivative(self,feed):
        i,span,u,h=self.coordinate(feed)
        return 6*u*(1-u)*(self.states[i+1,6:]-self.states[i,6:])/span

    def feed_tangent(self,feed):
        # The projector needs a material direction, independent of scalar
        # parametrization speed. Use the segment's limiting tangent also at
        # its zero-speed join; otherwise a smoothstop erases this direction.
        i,span,u,h=self.coordinate(feed)
        return (self.states[i+1,6:]-self.states[i,6:])/span


class BoundaryMaterialCurve(CoupledMaterialCurve):
    """Preserve the old holding tangent at the first reference boundary.

    First joint segment uses Hermite interpolation with the saved V3 dq/dd
    at zero and zero derivative at its other end. Remaining connections keep
    the same coupled smoothstep. The saved zero pose is the original nominal
    pose, not measured held geometry. Initial state/prior are inherited by the
    experiment runner, without changing the frozen R2 update formula.
    """
    def __init__(self,path):
        super().__init__(path)
        with np.load(path) as data:
            self.start_tangent=data['start_tangent_rad_per_mm'].copy()

    def state(self,feed):
        state=super().state(feed)
        i,span,u,h=self.coordinate(feed)
        if i==0:
            state[6:]=(2*u**3-3*u*u+1)*self.states[0,6:]+(u**3-2*u*u+u)*span*self.start_tangent+(-2*u**3+3*u*u)*self.states[1,6:]
        return state

    def joint_derivative(self,feed):
        i,span,u,h=self.coordinate(feed)
        if i==0:
            return (6*u*u-6*u)/span*self.states[0,6:]+(3*u*u-4*u+1)*self.start_tangent+(-6*u*u+6*u)/span*self.states[1,6:]
        return super().joint_derivative(feed)

    def feed_tangent(self,feed):
        if feed<self.feeds[1]:
            return self.joint_derivative(feed)
        return super().feed_tangent(feed)
