"""Continue the measured CLOSE1B route into one forward target helix."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from snake_control.suspended_transfer_route import R,ALPHA,BETA
from snake_control.suspended_supply_transfer import LINK

class MeasuredHookRoute:
    def __init__(self,initial_points,extension_length=2.5*LINK):
        self.initial=np.asarray(initial_points).copy();self.knots=-np.arange(22)*LINK
        self.old=CubicSpline(self.knots[::-1],self.initial[::-1],axis=0)
        self.source_cut=-10*LINK;self.target_join=-8*LINK;self.total=21*LINK+extension_length
        self.origin=self.initial[0].copy();self.phi0=np.arctan2(self.origin[2]-1,self.origin[0]-1.1)
        self.extension_length=extension_length
        self.t0=self.old(0,1);self.t0/=np.linalg.norm(self.t0)
        end=self.helix(np.array([extension_length]))[0]
        phi=self.phi0-BETA*extension_length/R
        te=np.array([BETA*np.sin(phi),ALPHA,-BETA*np.cos(phi)])
        t=np.linspace(0,1,2401)[:,None]
        def raw(h):return (2*t**3-3*t**2+1)*self.origin+(t**3-2*t**2+t)*h*self.t0+(-2*t**3+3*t**2)*end+(t**3-t**2)*h*te
        def length(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        self.handle=brentq(lambda h:length(h)-extension_length,1e-6,2.)
        self.extension=raw(self.handle);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.extension,axis=0),axis=1))]

    def helix(self,v):
        v=np.atleast_1d(v);phi=self.phi0-BETA*v/R
        return np.column_stack([1.1+R*np.cos(phi),self.origin[1]+ALPHA*v,1+R*np.sin(phi)])

    def sample(self,v):
        v=np.atleast_1d(v);p=self.old(np.minimum(v,0));join=(v>=0)&(v<self.extension_length);high=v>=self.extension_length
        p[join]=np.column_stack([np.interp(v[join],self.arc,self.extension[:,j]) for j in range(3)])
        p[high]=self.helix(v[high]);return p

    def window(self,shift,coeff=None):
        v=shift-np.arange(22)*LINK;p=self.sample(v)
        if coeff is not None:
            w=np.clip((v-self.source_cut)/(self.target_join-self.source_cut),0,1);h=16*w*w*(1-w)*(1-w)
            p+=h[:,None]*(coeff[0]+(2*w-1)[:,None]*coeff[1])
        return p,v

    def supply_target(self,v):
        # Keep the existing actual hook until shared material motion replaces
        # it. Already released posterior nodes are measured with zero weight.
        return self.sample(v)
