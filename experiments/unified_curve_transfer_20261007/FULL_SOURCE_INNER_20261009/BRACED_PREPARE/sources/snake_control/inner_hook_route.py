"""Source +Y outlet, short inner bridge, and target -X upward hook."""
import numpy as np
from scipy.optimize import brentq
from snake_control.hook_reform_route import ReformHookRoute
from snake_control.hook_start_transfer import MeasuredHookRoute
from snake_control.suspended_supply_transfer import LINK
from snake_control.suspended_transfer_route import R,ALPHA,BETA


class InnerHookRoute(ReformHookRoute):
    def __init__(self,front=9,bridge=5,morph_length=.20):
        self.winding_sign=1.;self.front=front;self.bridge=bridge;self.origin=front*LINK
        self.source_start=front+bridge;self.source_cut=-bridge*LINK;self.target_join=0.
        self.total=(21-front)*LINK;self.morph_length=morph_length
        self.phi0=-3*np.pi/4;self.target_y=.20;self.source_phi=-np.pi/4
        self.q=np.array([.82,R*np.cos(self.source_phi),1+R*np.sin(self.source_phi)])
        self.e=self.target([0.])[0]
        ts=np.array([ALPHA,-BETA*np.sin(self.source_phi),BETA*np.cos(self.source_phi)])
        te=np.array([BETA*np.sin(self.phi0),ALPHA,-BETA*np.cos(self.phi0)])
        self.entry_tangent=te;t=np.linspace(0,1,2401)[:,None]
        def raw(h):return (2*t**3-3*t**2+1)*self.q+(t**3-2*t**2+t)*h*ts+(-2*t**3+3*t**2)*self.e+(t**3-t**2)*h*te
        def length(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        self.handle=brentq(lambda h:length(h)+self.source_cut,1e-6,2.)
        self.connector=raw(self.handle);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.connector,axis=0),axis=1))]
        self.seed=None

    def source(self,u):
        u=np.atleast_1d(u);phi=self.source_phi+BETA*u/R
        return np.column_stack([self.q[0]+ALPHA*u,R*np.cos(phi),1+R*np.sin(phi)])

    def nominal(self,v):
        v=np.atleast_1d(v);p=self.target(v);source=v<self.source_cut;bridge=(v>=self.source_cut)&(v<0)
        p[source]=self.source(v[source]-self.source_cut)
        p[bridge]=np.column_stack([np.interp(v[bridge]-self.source_cut,self.arc,self.connector[:,j]) for j in range(3)])
        return p

    def bind_initial(self,points):
        self.initial=np.asarray(points).copy();self.seed=MeasuredHookRoute(points,winding_sign=1)

    def window(self,shift,coeff=None):
        v=self.origin+shift-np.arange(22)*LINK;p=self.nominal(v);a=self.blend(shift)
        if self.seed is not None and a<1:p=(1-a)*self.seed.sample(v-self.origin)+a*p
        if coeff is not None:
            w=np.clip((v-self.source_cut)/(-self.source_cut),0,1);h=16*w*w*(1-w)*(1-w)
            p+=a*h[:,None]*(coeff[0]+(2*w-1)[:,None]*coeff[1])
        return p,v
