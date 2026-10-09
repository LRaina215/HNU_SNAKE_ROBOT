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
        t=np.linspace(0,1,2401);h00=2*t**3-3*t**2+1;h10=t**3-2*t**2+t;h01=-2*t**3+3*t**2;h11=t**3-t**2
        r0=np.linalg.norm(self.origin[[0,2]]-[1.1,1.]);er=np.array([np.cos(self.phi0),0,np.sin(self.phi0)]);ep=np.array([-np.sin(self.phi0),0,np.cos(self.phi0)])
        dr=float(self.t0@er);dp=float(self.t0@ep)/r0;dy=float(self.t0[1])
        def raw(u):
            radius=h00*r0+h10*extension_length*dr+h01*R
            phi=h00*self.phi0+h10*extension_length*dp+h01*(self.phi0-BETA*u/R)+h11*extension_length*(-BETA/R)
            y=h00*self.origin[1]+h10*extension_length*dy+h01*(self.origin[1]+ALPHA*u)+h11*extension_length*ALPHA
            return np.column_stack([1.1+radius*np.cos(phi),y,1+radius*np.sin(phi)])
        def length(u):return np.linalg.norm(np.diff(raw(u),axis=0),axis=1).sum()
        self.helix_entry_parameter=brentq(lambda u:length(u)-extension_length,.02,.35)
        self.handle=extension_length;self.extension=raw(self.helix_entry_parameter);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.extension,axis=0),axis=1))]

    def helix(self,v):
        v=np.atleast_1d(v);phi=self.phi0-BETA*v/R
        return np.column_stack([1.1+R*np.cos(phi),self.origin[1]+ALPHA*v,1+R*np.sin(phi)])

    def sample(self,v):
        v=np.atleast_1d(v);p=self.old(np.minimum(v,0));join=(v>=0)&(v<self.extension_length);high=v>=self.extension_length
        p[join]=np.column_stack([np.interp(v[join],self.arc,self.extension[:,j]) for j in range(3)])
        p[high]=self.helix(v[high]-self.extension_length+self.helix_entry_parameter);return p

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
