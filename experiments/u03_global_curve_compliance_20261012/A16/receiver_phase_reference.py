"""A16: one receiving spatial-phase schedule, common A15 source/F/M/psi."""
from target_fit_reference import TargetFitReference
from continuous_window_reference import LINK
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from math import comb
import numpy as np

class ReceiverPhaseReference(TargetFitReference):
    def __init__(self,points,angles):
        self.receiver_phase_enabled=False
        super().__init__(points,angles)
        self.receiver_phase_enabled=True
        self.set_nominal_family(0)
        self.global_spline=self.spline;self.global_start_arc=self.start_arc;self.global_target_arc=self.target_arc
        self.spec.update(version=16,receiving_coordination='One spatial target entry-phase schedule: old entry minus pi/3*smooth((S-7.5)/1.5). Target radius/pitch and A15 source/F/M/common psi retained; no per-joint phase modification.',target_entry=dict(y_m=.20,polar_rad='old(S)-pi/3*smooth((S-7.5)/1.5)'),target_radius_m=self.target_radius,high_level_feedback=False)

    def receiving_phase(self,S):
        return -np.pi/4-np.pi/4*self.smooth((S-4.5)/1.5)-np.pi/3*self.smooth((S-7.5)/1.5)

    def set_nominal_family(self,S):
        if not self.receiver_phase_enabled:return super().set_nominal_family(S)
        if not self.target_fit_enabled:return super().set_nominal_family(S)
        if not self.supply_enabled:return super().set_nominal_family(S)
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.exit_u=-10*h;self.source_chart_shift=self.feed_early*self.phase_cells(S);phase=self.beta*self.source_chart_shift;self.spatial_phase=phase
        ph0=self.phi0-self.beta*self.exit_u;p0=self.source(np.array([self.exit_u]),0.)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)]);a0=np.array([0.,-self.radius*self.beta**2*np.cos(ph0),-self.radius*self.beta**2*np.sin(ph0)])
        phi_initial=self.phi0;initial=self.source(np.array([0.]),0.)[0]
        ti=np.array([self.a,self.radius*self.beta*np.sin(phi_initial),-self.radius*self.beta*np.cos(phi_initial)]);ai=np.array([0.,-self.radius*self.beta**2*np.cos(phi_initial),-self.radius*self.beta**2*np.sin(phi_initial)])
        grow=self.smooth(S/6);lead=.075;entry_phi=self.receiving_phase(S);free_phi=entry_phi-self.target_beta*lead;free=np.array([1.1+self.target_radius*np.cos(free_phi),.20-self.target_a*lead,1+self.target_radius*np.sin(free_phi)]);theta_free=2*np.pi+np.arctan2(free[2]-1,free[1]);rfree=np.linalg.norm(free[1:]-[0,1])
        theta=(1-grow)*phi_initial+grow*theta_free;r=(1-grow)*self.radius+grow*rfree;x=(1-grow)*initial[0]+grow*free[0];end=np.array([x,r*np.cos(theta),1+r*np.sin(theta)])
        rr=np.linalg.norm(end[[0,2]]-[1.1,1]);at=np.arctan2(end[2]-1,end[0]-1.1);bet=1/np.sqrt(rr*rr+(self.pitch/(2*np.pi))**2);aa=self.pitch*bet/(2*np.pi)
        tv=np.array([-rr*bet*np.sin(at),aa,rr*bet*np.cos(at)]);av=np.array([-rr*bet**2*np.cos(at),0.,-rr*bet**2*np.sin(at)])
        t1=(1-grow)*ti+grow*tv;t1/=np.linalg.norm(t1);a1=(1-grow)*ai+grow*av;a1-=a1@t1*t1
        q0=self.coords(p0,ph0);q1=self.coords(end,theta);dt0,dd0=self.polar_derivatives(p0,t0,a0);dt1,dd1=self.polar_derivatives(end,t1,a1)
        scale=(10*h)*(1-grow)+.20*grow;dt0*=scale;dt1*=scale;dd0*=scale**2;dd1*=scale**2
        cp=np.zeros((8,3));cp[0]=q0;cp[1]=q0+dt0/7;cp[2]=q0+2*dt0/7+dd0/42;cp[7]=q1;cp[6]=q1-dt1/7;cp[5]=q1-2*dt1/7+dd1/42
        for j in [3,4]:
            u=self.exit_u+(j/7)*(-self.exit_u);pp=self.source(np.array([u]),phase)[0];initialc=self.coords(pp,ph0+(phi_initial-ph0)*j/7)
            # Keep X before the target axis until the polar route is below it.
            # The receiving entry moved. Old fixed5.4/5.5 angles now overshoot
            # its endpoint and create a tight reverse turn. Preserve monotone
            # polar progression between the two fixed endpoint jet controls.
            middle_phi=cp[2,2]+(j-2)/3*(cp[5,2]-cp[2,2])
            desired=np.array([p0[0]+.005,.18,middle_phi]) if j==3 else np.array([p0[0]+.01,.23,middle_phi]);cp[j]=(1-grow)*initialc+grow*desired
        u=np.linspace(0,1,241);qq=sum(comb(7,j)*(1-u[:,None])**(7-j)*u[:,None]**j*cp[j] for j in range(8))
        # A C2 X support waits before the target axis while the polar path crosses
        # Z=1. It advances only in the lower sector; not a radial point push.
        plateau=min(p0[0]+.005,.995);X=qq[:,0].copy()
        def bezier5(values,v):return sum(comb(5,k)*(1-v)**(5-k)*v**k*values[k] for k in range(6))
        early=u<=.4;late=u>=.6;escape=np.full(len(u),plateau)
        e=.4;vals=[q0[0],q0[0]+dt0[0]*e/5,q0[0]+2*dt0[0]*e/5+dd0[0]*e**2/20,plateau,plateau,plateau];escape[early]=bezier5(vals,u[early]/e)
        vals=[plateau,plateau,plateau,q1[0]-2*dt1[0]*e/5+dd1[0]*e**2/20,q1[0]-dt1[0]*e/5,q1[0]];escape[late]=bezier5(vals,(u[late]-.6)/e)
        qq[:,0]=(1-grow)*X+grow*escape
        prefix=np.column_stack([qq[:,0],qq[:,1]*np.cos(qq[:,2]),1+qq[:,1]*np.sin(qq[:,2])])
        self.target_phi=entry_phi;self.target_y=.20;entry=self.target(np.array([0.]),0)[0];te=np.array([-self.target_radius*self.target_beta*np.sin(entry_phi),self.target_a,self.target_radius*self.target_beta*np.cos(entry_phi)]);ae=np.array([-self.target_radius*self.target_beta**2*np.cos(entry_phi),0.,-self.target_radius*self.target_beta**2*np.sin(entry_phi)])
        scale=lead;dp0=t1*scale;dp1=te*scale;ddp0=a1*scale**2;ddp1=ae*scale**2
        ep=np.zeros((8,3));ep[0]=end;ep[1]=end+dp0/7;ep[2]=end+2*dp0/7+ddp0/42;ep[7]=entry;ep[6]=entry-dp1/7;ep[5]=entry-2*dp1/7+ddp1/42
        # A fixed75mm helical lead avoids collapsing a finite turn tozero.
        tf=np.array([-self.target_radius*self.target_beta*np.sin(free_phi),self.target_a,self.target_radius*self.target_beta*np.cos(free_phi)])
        af=np.array([-self.target_radius*self.target_beta**2*np.cos(free_phi),0.,-self.target_radius*self.target_beta**2*np.sin(free_phi)])
        jf=np.array([self.target_radius*self.target_beta**3*np.sin(free_phi),0.,-self.target_radius*self.target_beta**3*np.cos(free_phi)])
        je=np.array([self.target_radius*self.target_beta**3*np.sin(entry_phi),0.,-self.target_radius*self.target_beta**3*np.cos(entry_phi)])
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

