"""Lower source outlet and lower target entry, from the same lower-hook seed."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from snake_control.hook_reform_route import ReformHookRoute
from snake_control.hook_start_transfer import MeasuredHookRoute
from snake_control.suspended_transfer_route import R,ALPHA,BETA
from snake_control.suspended_supply_transfer import LINK


class LowerBridgeReformHookRoute(ReformHookRoute):
    def __init__(self,initial_points,morph_length=.20,source_start=16):
        self.winding_sign=-1.;self.initial=np.asarray(initial_points).copy()
        self.seed=MeasuredHookRoute(initial_points,winding_sign=-1)
        self.phi0=-np.pi/2;self.target_y=.20;self.source_start=source_start
        self.origin=((self.seed.phi0-self.phi0)%(2*np.pi))*R/BETA
        self.source_cut=self.origin-source_start*LINK;self.target_join=0.
        self.total=21*LINK-self.origin;self.morph_length=morph_length
        # Proper 180-degree turn around the source X-axis defines a lower
        # nominal outlet. It is only a reference: physical body starts at the
        # identical lower-hook state and must perform the turn through ctrl.
        source=self.initial.copy();source[:,1]*=-1;source[:,2]=2-source[:,2]
        self.old=CubicSpline((self.origin-np.arange(22)*LINK)[::-1],source[::-1],axis=0)
        self.q=self.old(self.source_cut);self.e=self.target([0.])[0]
        ts=self.old(self.source_cut,1);ts/=np.linalg.norm(ts)
        te=np.array([BETA,ALPHA,0.]);self.entry_tangent=te
        t=np.linspace(0,1,2401)[:,None]
        def raw(h):return (2*t**3-3*t**2+1)*self.q+(t**3-2*t**2+t)*h*ts+(-2*t**3+3*t**2)*self.e+(t**3-t**2)*h*te
        def length(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        self.handle=brentq(lambda h:length(h)+self.source_cut,1e-6,2.)
        self.connector=raw(self.handle);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.connector,axis=0),axis=1))]
