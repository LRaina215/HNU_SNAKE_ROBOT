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
