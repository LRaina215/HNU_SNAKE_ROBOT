"""Continuously unfold the real reverse hook into an extended L-transfer route."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation,Slerp
from snake_control.suspended_transfer_route import R,ALPHA,BETA
from snake_control.suspended_supply_transfer import LINK
from snake_control.hook_start_transfer import MeasuredHookRoute

class ReformHookRoute:
    def __init__(self,initial_points,morph_length=.20,source_start=16):
        self.initial=np.asarray(initial_points).copy();self.seed=MeasuredHookRoute(initial_points)
        self.origin=8*LINK;self.source_start=source_start;self.source_cut=(8-source_start)*LINK;self.target_join=0.;self.total=13*LINK;self.morph_length=morph_length
        self.old=CubicSpline((self.origin-np.arange(22)*LINK)[::-1],self.initial[::-1],axis=0)
        self.phi0=self.seed.phi0+BETA*self.origin/R;self.target_y=.20;self.q=self.initial[source_start].copy();self.e=self.target(np.array([0.]))[0]
        ts=self.old(self.source_cut,1);ts/=np.linalg.norm(ts);te=np.array([BETA*np.sin(self.phi0),ALPHA,-BETA*np.cos(self.phi0)])
        self.entry_tangent=te;t=np.linspace(0,1,2401)[:,None]
        def raw(h):return (2*t**3-3*t**2+1)*self.q+(t**3-2*t**2+t)*h*ts+(-2*t**3+3*t**2)*self.e+(t**3-t**2)*h*te
        def length(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        self.handle=brentq(lambda h:length(h)+self.source_cut,1e-6,2.);self.connector=raw(self.handle);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.connector,axis=0),axis=1))]

    def target(self,v):
        v=np.atleast_1d(v);phi=self.phi0-BETA*v/R
        return np.column_stack([1.1+R*np.cos(phi),self.target_y+ALPHA*v,1+R*np.sin(phi)])

    def nominal(self,v):
        v=np.atleast_1d(v);p=self.target(v);source=v<self.source_cut;bridge=(v>=self.source_cut)&(v<0)
        p[source]=self.old(v[source]);p[bridge]=np.column_stack([np.interp(v[bridge]-self.source_cut,self.arc,self.connector[:,j]) for j in range(3)])
        return p

    def blend(self,shift):
        x=float(np.clip(shift/self.morph_length,0,1));return x*x*(3-2*x)

    def sample(self,v):return self.nominal(v)

    def window(self,shift,coeff=None):
        v=self.origin+shift-np.arange(22)*LINK;p=self.nominal(v);a=self.blend(shift)
        if a<1:p=(1-a)*self.seed.sample(v-self.origin)+a*p
        if coeff is not None:
            w=np.clip((v-self.source_cut)/(-self.source_cut),0,1);h=16*w*w*(1-w)*(1-w)
            p+=a*h[:,None]*(coeff[0]+(2*w-1)[:,None]*coeff[1])
        return p,v

    def supply_target(self,v):
        v=np.atleast_1d(v);p=self.target(v);free=v<0;p[free]=self.e+v[free,None]*self.entry_tangent;return p

    def head_orientation(self,initial_rotation,shift,phase):
        if not hasattr(self,'rotation_blend'):
            p=self.target(np.array([self.origin,self.origin-LINK]));desired=p[0]-p[1];desired/=np.linalg.norm(desired);old=initial_rotation[:,2]
            cross=np.cross(old,desired);sine=np.linalg.norm(cross);angle=np.arctan2(sine,np.clip(old@desired,-1,1));align=Rotation.from_rotvec(cross*angle/max(sine,1e-12)).as_matrix()
            self.rotation_blend=Slerp([0,1],Rotation.from_matrix([initial_rotation,align@initial_rotation]))
        base=self.rotation_blend([self.blend(shift)]).as_matrix()[0]
        return Rotation.from_rotvec([0,BETA*shift/R,0]).as_matrix()@base@Rotation.from_rotvec([0,0,-phase]).as_matrix()
