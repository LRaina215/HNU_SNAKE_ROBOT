"""A16 source-polar family + original native mapping; target prior only.

Source parameters are fitted once to the immutable supplied robot geometry.
Transport S is zero until actual bearing; shape progress has no source feed/roll.
"""
from math import comb
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,least_squares
from native_mapping import NativeMapping,LINK

class NativeStep1Reference(NativeMapping):
    def __init__(self,points,angles,geometry):
        self.geometry=geometry.copy();self.shape_progress=0.;self.release_materials=10.;self.delta=0.
        n=np.arange(22);phi=float(np.arctan2(points[0,2]-1,points[0,1]))
        def helix(v):
            R,P,ph,x=v;be=1/np.sqrt(R*R+(P/(2*np.pi))**2);a=P*be/(2*np.pi)
            h=brentq(lambda h:(a*h)**2+4*R*R*np.sin(be*h/2)**2-LINK**2,LINK,2*LINK)
            th=ph+be*h*n
            return np.column_stack([x-a*h*n,R*np.cos(th),1+R*np.sin(th)])
        fit=least_squares(lambda v:(helix(v)-points).ravel(),[.0968,.183,phi,points[0,0]],bounds=([.07,.14,phi-.7,points[0,0]-.04],[.12,.24,phi+.7,points[0,0]+.04]))
        self.radius,self.pitch,self.phi0,self.x0=fit.x
        self.beta=1/np.sqrt(self.radius**2+(self.pitch/(2*np.pi))**2);self.a=self.pitch*self.beta/(2*np.pi)
        self.feed_per_cycle=np.pi*.058/LINK*2.8;self.feed_early=.054;self.feed_late=.09
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.tail_u0=-21*h;self.anchor_u0=-4.;self.source_chart_shift=0.;self.spatial_phase=0.;self.supply_enabled=True
        self.gauge_angle=0.;self.set_nominal_family(0.);p,loc=self.body(0.,True)
        T=-self.spline(loc[0],1);T/=np.linalg.norm(T);N=np.eye(3)[np.argmin(abs(T))];N-=N@T*T;N/=np.linalg.norm(N)
        anchor=self.start_arc+self.anchor_u0-self.exit_u
        tt=-self.spline(np.linspace(loc[0],anchor,800),1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        self.supply_anchor_normal=self.transport_normal(N,tt)
        q0,info=NativeMapping.command(self,0.);b=self.joint_bending(info['curvature_bounds_mm'],info['curvature_field'],info['material_arc_nodes_mm'])
        q90=-np.where(np.arange(20)%2,-b[:,1],b[:,0]);co=np.linalg.lstsq(np.column_stack([q0,q90]),angles,rcond=None)[0]
        self.gauge_angle=float(np.arctan2(co[1],co[0]));self.initial_reference_error_m=float(np.linalg.norm(p-points,axis=1).max())
    def target_jet(self,u,entry_phi):
        phi=entry_phi+self.target_beta*u;e,b=self.geometry.frame();co=np.cos(phi);sn=np.sin(phi)
        radial=co*e+sn*b;tangent=-sn*e+co*b
        return (self.geometry.entry+self.target_a*u*self.geometry.axis+self.target_radius*radial,
                self.target_a*self.geometry.axis+self.target_radius*self.target_beta*tangent,
                -self.target_radius*self.target_beta**2*radial,
                -self.target_radius*self.target_beta**3*tangent)
    def target(self,u,phase):
        u=np.asarray(u);e,b=self.geometry.frame();phi=self.target_phi+self.target_beta*u+phase
        return self.geometry.entry+(self.target_a*u)[...,None]*self.geometry.axis+self.target_radius*(np.cos(phi)[...,None]*e+np.sin(phi)[...,None]*b)
    def nominal_command(self,S):
        key=(float(S),float(self.shape_progress),float(self.release_materials),float(self.delta),*self.geometry.entry,*self.geometry.axis,float(self.geometry.radius))
        cache=getattr(self,'_command_cache',{})
        if key in cache:
            self.set_nominal_family(S)
            return cache[key]
        result=NativeMapping.command(self,S)
        if len(cache)>=64:cache.pop(next(iter(cache)))
        cache[key]=result;self._command_cache=cache
        return result
    def receiving_phase(self,S):
        return -np.pi/4-np.pi/4*self.smooth((S-4.5)/1.5)-np.pi/3*self.smooth((S-7.5)/1.5)

    def set_nominal_family(self,S):
        transport=S;S=self.shape_progress
        self.target_radius=self.geometry.radius+.039+self.delta
        self.target_beta=1/np.sqrt(self.target_radius**2+(self.pitch/(2*np.pi))**2)
        self.target_a=self.pitch*self.target_beta/(2*np.pi)
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.exit_u=-self.release_materials*h;self.source_chart_shift=self.feed_early*self.phase_cells(transport);phase=self.beta*self.source_chart_shift;self.spatial_phase=phase
        ph0=self.phi0-self.beta*self.exit_u;p0=self.source(np.array([self.exit_u]),0.)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)]);a0=np.array([0.,-self.radius*self.beta**2*np.cos(ph0),-self.radius*self.beta**2*np.sin(ph0)])
        phi_initial=self.phi0;initial=self.source(np.array([0.]),0.)[0]
        ti=np.array([self.a,self.radius*self.beta*np.sin(phi_initial),-self.radius*self.beta*np.cos(phi_initial)]);ai=np.array([0.,-self.radius*self.beta**2*np.cos(phi_initial),-self.radius*self.beta**2*np.sin(phi_initial)])
        grow=self.smooth(S/6);lead=.075;entry_phi=self.receiving_phase(S);free_phi=entry_phi-self.target_beta*lead;free=self.target_jet(-lead,entry_phi)[0];theta_free=2*np.pi+np.arctan2(free[2]-1,free[1]);rfree=np.linalg.norm(free[1:]-[0,1])
        theta=(1-grow)*phi_initial+grow*theta_free;r=(1-grow)*self.radius+grow*rfree;x=(1-grow)*initial[0]+grow*free[0];end=np.array([x,r*np.cos(theta),1+r*np.sin(theta)])
        axial,radial,rr=self.geometry.coordinates(end);ee,bb=self.geometry.frame();at=np.arctan2(radial@bb,radial@ee);bet=1/np.sqrt(rr*rr+(self.pitch/(2*np.pi))**2);aa=self.pitch*bet/(2*np.pi)
        tv=aa*self.geometry.axis+rr*bet*(-np.sin(at)*ee+np.cos(at)*bb);av=-rr*bet**2*(np.cos(at)*ee+np.sin(at)*bb)
        t1=(1-grow)*ti+grow*tv;t1/=np.linalg.norm(t1);a1=(1-grow)*ai+grow*av;a1-=a1@t1*t1
        q0=self.coords(p0,ph0);q1=self.coords(end,theta);dt0,dd0=self.polar_derivatives(p0,t0,a0);dt1,dd1=self.polar_derivatives(end,t1,a1)
        scale=(self.release_materials*h)*(1-grow)+.20*grow;dt0*=scale;dt1*=scale;dd0*=scale**2;dd1*=scale**2
        cp=np.zeros((8,3));cp[0]=q0;cp[1]=q0+dt0/7;cp[2]=q0+2*dt0/7+dd0/42;cp[7]=q1;cp[6]=q1-dt1/7;cp[5]=q1-2*dt1/7+dd1/42
        for j in [3,4]:
            u=self.exit_u+(j/7)*(-self.exit_u);pp=self.source(np.array([u]),phase)[0];initialc=self.coords(pp,ph0+(phi_initial-ph0)*j/7)
            # Keep X before the target axis until the polar route is below it.
            # The receiving entry moved. Old fixed5.4/5.5 angles now overshoot
            # its endpoint and create a tight reverse turn. Preserve monotone
            # polar progression between the two fixed endpoint jet controls.
            middle_phi=cp[2,2]+(j-2)/3*(cp[5,2]-cp[2,2])
            desired=np.array([q0[0]+j/7*(q1[0]-q0[0]),.18 if j==3 else .23,middle_phi]);cp[j]=(1-grow)*initialc+grow*desired
        u=np.linspace(0,1,241);qq=sum(comb(7,j)*(1-u[:,None])**(7-j)*u[:,None]**j*cp[j] for j in range(8))
        # A C2 X support waits before the target axis while the polar path crosses
        # Z=1. It advances only in the lower sector; not a radial point push.
        plateau=min(p0[0]+.005,self.geometry.entry[0]-.105);X=qq[:,0].copy()
        def bezier5(values,v):return sum(comb(5,k)*(1-v)**(5-k)*v**k*values[k] for k in range(6))
        early=u<=.4;late=u>=.6;escape=np.full(len(u),plateau)
        e=.4;vals=[q0[0],q0[0]+dt0[0]*e/5,q0[0]+2*dt0[0]*e/5+dd0[0]*e**2/20,plateau,plateau,plateau];escape[early]=bezier5(vals,u[early]/e)
        vals=[plateau,plateau,plateau,q1[0]-2*dt1[0]*e/5+dd1[0]*e**2/20,q1[0]-dt1[0]*e/5,q1[0]];escape[late]=bezier5(vals,(u[late]-.6)/e)
        # The original wait plateau relied on approach source feed. With a held
        # source it compresses the entire X bridge into the final short sector.
        # Distribute X across the C2 Bezier bridge instead.
        qq[:,0]=X
        prefix=np.column_stack([qq[:,0],qq[:,1]*np.cos(qq[:,2]),1+qq[:,1]*np.sin(qq[:,2])])
        self.target_phi=entry_phi;entry,te,ae,je=self.target_jet(0.,entry_phi)
        scale=lead;dp0=t1*scale;dp1=te*scale;ddp0=a1*scale**2;ddp1=ae*scale**2
        ep=np.zeros((8,3));ep[0]=end;ep[1]=end+dp0/7;ep[2]=end+2*dp0/7+ddp0/42;ep[7]=entry;ep[6]=entry-dp1/7;ep[5]=entry-2*dp1/7+ddp1/42
        # A fixed75mm helical lead avoids collapsing a finite turn tozero.
        free,tf,af,jf=self.target_jet(-lead,entry_phi)
        final3=free+3*tf*lead/7+af*lead**2/14+jf*lead**3/210
        final4=entry-3*te*lead/7+ae*lead**2/14-je*lead**3/210
        ep[3]=(1-grow)*(ep[2]+(ep[5]-ep[2])/3)+grow*final3;ep[4]=(1-grow)*(ep[2]+2*(ep[5]-ep[2])/3)+grow*final4
        ext=sum(comb(7,j)*(1-u[:,None])**(7-j)*u[:,None]**j*ep[j] for j in range(8))
        us=np.arange(-5,self.exit_u,.002);ps=self.source(us,phase);ut=np.arange(.002,4,.002);pt=self.target(ut,0)
        extra=ext[1:]
        xyz=np.vstack([ps,prefix,extra,pt]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(arc[len(ps)]);self.prefix_end_arc=float(arc[len(ps)+len(prefix)-1]);self.target_arc=float(arc[len(ps)+len(prefix)+len(extra)-1]);self.connector_arc=self.target_arc-self.start_arc
        self.grow=grow;self.recover=self.smooth((S-6)/2);self.source_exit=p0;self.connector_controls=cp;self.continuation_controls=ep
        return np.array([phase,0.])

