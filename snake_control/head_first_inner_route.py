"""Head-first fixed spatial route, sampled by each physical 64.5mm rod."""
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from snake_control.suspended_supply_transfer import LINK
from snake_control.suspended_transfer_route import R,ALPHA,BETA


class HeadFirstInnerRoute:
    def __init__(self,points,left=.935,target_y=.20,smooth_source=False):
        self.initial=np.asarray(points).copy();self.origin=0.;self.source_cut=0.;self.winding_sign=1.;self.source_start=0
        self.knots=-np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        self.source_field=CubicSpline(self.knots[::-1],self.initial[::-1],axis=0)
        self.smooth_source=bool(smooth_source);self.source_phi=float(np.arctan2(points[0,2]-1,points[0,1]));self.source_node_radius=float(np.linalg.norm(points[0,[1,2]]-[0,1]))
        self.phi0=-3*np.pi/4;self.target_y=target_y;self.q=self.initial[0].copy();self.e=np.array([1.1+R*np.cos(self.phi0),target_y,1+R*np.sin(self.phi0)])
        ts=self.initial[0]-self.initial[1];ts/=np.linalg.norm(ts)
        self.entry_tangent=np.array([BETA*np.sin(self.phi0),ALPHA,-BETA*np.cos(self.phi0)])
        if self.q[2]<1 and self.q[1]>0:
            controls=np.array([self.q,self.q+.040*ts,[min(1.005,self.q[0]+.03),.12,.94],[1.025,.16,.845],self.e-.065*self.entry_tangent,self.e])
        else:controls=np.array([self.q,self.q+.040*ts,[left+.03,.070,1.135],[left,.12,.985],[left+.025,.17,.805],[1.075,.18,.835],self.e-.055*self.entry_tangent,self.e])
        t=np.linspace(0,1,8001)[:,None];n=len(controls)-1
        connector=sum(math.comb(n,j)*(1-t)**(n-j)*t**j*controls[j] for j in range(n+1))
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(connector,axis=0),axis=1))]
        self.controls=controls;self.connector=connector;self.arc=arc;self.bridge_length=float(arc[-1]);self.target_join=self.bridge_length
        self.bridge_field=CubicSpline(arc,connector,axis=0)
        self.total=self.bridge_length+21*LINK*1.04
        self.frame_s=np.linspace(-.04,self.total+.15,6001)
        pp=self.sample(self.frame_s)
        tt=np.array([point-self.sample([self.predecessor(s,point)])[0] for s,point in zip(self.frame_s,pp)]);tt/=np.linalg.norm(tt,axis=1)[:,None]
        normal=np.array([0.,self.q[1],self.q[2]-1]);normal-=normal@tt[0]*tt[0];normal/=np.linalg.norm(normal);normals=[normal.copy()]
        for a,b in zip(tt[:-1],tt[1:]):
            cross=np.cross(a,b);sine=np.linalg.norm(cross)
            if sine>1e-12:normal=Rotation.from_rotvec(cross*np.arctan2(sine,np.clip(a@b,-1,1))/sine).apply(normal)
            normal-=normal@b*b;normal/=np.linalg.norm(normal);normals.append(normal.copy())
        self.tangents=tt;self.normals=np.asarray(normals)

    def target(self,w):
        w=np.atleast_1d(w);phi=self.phi0-BETA*w/R
        return np.column_stack([1.1+R*np.cos(phi),self.target_y+ALPHA*w,1+R*np.sin(phi)])

    def sample(self,v):
        v=np.atleast_1d(v);p=self.target(v-self.bridge_length);source=v<0;bridge=(v>=0)&(v<self.bridge_length)
        if self.smooth_source:
            u=v[source];a=np.clip(1+u/(3*LINK),0,1);a=a*a*(3-2*a);radius=R+(self.source_node_radius-R)*a;phi=self.source_phi+BETA*u/R
            p[source]=np.column_stack([self.q[0]+ALPHA*u,radius*np.cos(phi),1+radius*np.sin(phi)])
        else:p[source]=self.source_field(v[source])
        p[bridge]=self.bridge_field(v[bridge]);return p

    def predecessor(self,v,point):
        def gap(u):return float(np.linalg.norm(self.sample([u])[0]-point)-LINK)
        grid=v-np.linspace(.001,.45,80);g=np.linalg.norm(self.sample(grid)-point,axis=1)-LINK
        for i in range(1,len(grid)):
            if g[i]>=0 and g[i-1]<0:return brentq(gap,grid[i],grid[i-1],xtol=1e-11)
        raise ValueError('Continuous curve has no preceding 64.5mm rod intersection within 450mm')

    def window(self,shift,coeff=None):
        v=[float(shift)];p=[self.sample(v)[0]]
        for _ in range(21):
            v.append(self.predecessor(v[-1],p[-1]));p.append(self.sample([v[-1]])[0])
        return np.asarray(p),np.asarray(v)

    def frame(self,s):
        t=np.array([np.interp(s,self.frame_s,self.tangents[:,j]) for j in range(3)]);t/=np.linalg.norm(t)
        n=np.array([np.interp(s,self.frame_s,self.normals[:,j]) for j in range(3)]);n-=n@t*t;n/=np.linalg.norm(n)
        return np.column_stack([n,np.cross(t,n),t])

    def head_orientation(self,initial_rotation,shift,phase):
        return self.frame(shift)@self.frame(0).T@initial_rotation@Rotation.from_rotvec([0,0,-phase]).as_matrix()

    def supply_target(self,v):return self.sample(v)
