"""Local C: actual known-source support boundary to prescribed G helix.

No actual-state G overwrite, target truth, integration or position optimizer.
Only C uses the current robot support pose. The same Bishop/native map executes.
"""
from math import comb
import numpy as np
from scipy.interpolate import CubicSpline
from native_mapping import LINK
from native_mapping_vectorized import FastNativeMapping

class FrontAnchoredCurve(FastNativeMapping):
    def __init__(self):
        self.supply_enabled=True;self.grow=1.;self.recover=0.;self.feed_per_cycle=7.9099945262477895
    def build(self,points,body_rotation,geometry,cut,pitch,phase,delta=0.,actual_angles=None):
        k=int(np.floor(cut));fraction=float(cut-k);self.cut=float(cut);self.geometry=geometry.copy();self.pitch=pitch;self.radius=geometry.radius+.039+delta
        # Source C boundary consists of current supported rods only. This never
        # enters global G and never writes to the actual physical data.
        boundary=(1-fraction)*points[k]+fraction*points[k+1];source=np.vstack([points[k+1:][::-1],boundary]);source=source[np.r_[True,np.linalg.norm(np.diff(source,axis=0),axis=1)>1e-8]];parameter=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(source,axis=0),axis=1))]
        ttail=source[1]-source[0];ttail/=np.linalg.norm(ttail);t0=source[-1]-source[-2];t0/=np.linalg.norm(t0)
        src=CubicSpline(parameter,source,axis=0,bc_type=((1,ttail),(1,t0)))
        us=np.unique(np.r_[np.linspace(0,parameter[-1],max(10,int(parameter[-1]/.003)+1)),parameter]);ps=src(us)
        start=source[-1];curvature=src(parameter[-1],2);curvature-=curvature@t0*t0
        beta=1/np.sqrt(self.radius**2+(pitch/(2*np.pi))**2);a=pitch*beta/(2*np.pi);e,b=geometry.frame();lead=.075
        phi=phase-beta*lead;radial=np.cos(phi)*e+np.sin(phi)*b;cross=-np.sin(phi)*e+np.cos(phi)*b
        finish=geometry.entry-a*lead*geometry.axis+self.radius*radial;t1=a*geometry.axis+self.radius*beta*cross;acc1=-self.radius*beta**2*radial
        scale=max(.20,1.5*np.linalg.norm(finish-start));cp=np.zeros((6,3));cp[0]=start;cp[1]=start+t0*scale/5;cp[2]=start+2*t0*scale/5+curvature*scale**2/20
        cp[5]=finish;cp[4]=finish-t1*scale/5;cp[3]=finish-2*t1*scale/5+acc1*scale**2/20
        u=np.linspace(0,1,241);connector=sum(comb(5,j)*(1-u[:,None])**(5-j)*u[:,None]**j*cp[j] for j in range(6))
        ut=np.arange(-lead+.002,3.502,.002);angles=phase+beta*ut;target=geometry.entry+(a*ut)[:,None]*geometry.axis+self.radius*(np.cos(angles)[:,None]*e+np.sin(angles)[:,None]*b)
        xyz=np.vstack([ps,connector[1:],target]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.tail_arc=0.;self.start_arc=float(arc[len(ps)-1]);self.target_arc=float(np.interp(0.,ut,arc[len(ps)+len(connector)-1:]));self.connector_arc=self.target_arc-self.start_arc
        self.anchor_arc=self.start_arc-(1.5-fraction)*LINK
        self.anchor_normal=(-1.)**(k+2)*body_rotation[:,0]
        tangent=-self.spline(self.anchor_arc,1);tangent/=np.linalg.norm(tangent);self.anchor_normal-=self.anchor_normal@tangent*tangent;self.anchor_normal/=np.linalg.norm(self.anchor_normal)
        self.current_dense=xyz
        self.gauge_angle=0.
        if actual_angles is not None:
            q,info=self.command(0.);bend=self.joint_bending(info['curvature_bounds_mm'],info['curvature_field'],info['material_arc_nodes_mm'])
            q90=-np.where(np.arange(20)%2,-bend[:,1],bend[:,0]);co=np.linalg.lstsq(np.column_stack([q[cut:],q90[cut:]]),actual_angles[cut:],rcond=None)[0]
            self.gauge_angle=float(np.arctan2(co[1],co[0]))
    def set_nominal_family(self,S):pass
    def material_anchor(self,S,locations):return self.anchor_arc,self.anchor_normal.copy(),self.gauge_angle,0.
    def body(self,S,return_locations=False):
        loc=[self.tail_arc];points=[self.spline(loc[0])]
        for _ in range(21):
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=loc[-1]+ds;distance=np.linalg.norm(self.spline(trial)-points[-1],axis=1);ids=np.flatnonzero(distance>=LINK)
            if not len(ids):raise ValueError('Local C has no forward physical chord')
            j=int(ids[0]);lo=loc[-1] if j==0 else trial[j-1];hi=trial[j]
            for _ in range(16):
                mid=.5*(hi+lo)
                if np.linalg.norm(self.spline(mid)-points[-1])>=LINK:hi=mid
                else:lo=mid
            loc.append(.5*(hi+lo));points.append(self.spline(loc[-1]))
        points=np.array(points)[::-1];loc=np.array(loc)[::-1]
        return (points,loc) if return_locations else points
