"""Full source release -> rough receiver, using one material curve mapping."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from math import comb
from estimated_curve import EstimatedCurve
from native_mapping import LINK
from source_mode import reconstruct

class FromSourceCurve(EstimatedCurve):
    def __init__(self,constants,prior):
        super().__init__(constants,prior)
        self.theta_offset=0.;self.front_relief=np.zeros(2);self.initial_bias=None
    def receiving_phase(self,S):return super().receiving_phase(S)+self.theta_offset
    def set_nominal_family(self,S):
        if S<0:raise ValueError('negative release coordinate')
        self.target_radius=self.geometry.radius+.039
        self.target_beta=1/np.sqrt(self.target_radius**2+(self.pitch/(2*np.pi))**2)
        self.target_a=self.pitch*self.target_beta/(2*np.pi)
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.exit_u=-10*h;self.source_chart_shift=self.feed_early*self.phase_cells(S)
        phase=self.beta*self.source_chart_shift;self.spatial_phase=phase
        ph0=self.phi0-self.beta*self.exit_u;p0=self.source(np.array([self.exit_u]),0.)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)])
        a0=np.array([0.,-self.radius*self.beta**2*np.cos(ph0),-self.radius*self.beta**2*np.sin(ph0)])
        phi_initial=self.phi0;initial=self.source(np.array([0.]),0.)[0]
        ti=np.array([self.a,self.radius*self.beta*np.sin(phi_initial),-self.radius*self.beta*np.cos(phi_initial)])
        ai=np.array([0.,-self.radius*self.beta**2*np.cos(phi_initial),-self.radius*self.beta**2*np.sin(phi_initial)])
        grow=self.smooth(S/6);lead=.075;self.target_phi=self.receiving_phase(S)
        free=self.target_jet(-lead);entry=self.target_jet(0.)
        theta_free=2*np.pi+np.arctan2(free[2]-1,free[1]);rfree=np.linalg.norm(free[1:]-[0,1])
        theta=(1-grow)*phi_initial+grow*theta_free;r=(1-grow)*self.radius+grow*rfree
        x=(1-grow)*initial[0]+grow*free[0];end=np.array([x,r*np.cos(theta),1+r*np.sin(theta)])
        _,rad,rr=self.geometry.coordinates(end);e,b=self.geometry.frame();at=np.arctan2(rad@b,rad@e)
        bet=1/np.sqrt(rr*rr+(self.pitch/(2*np.pi))**2);aa=self.pitch*bet/(2*np.pi)
        tv=aa*self.geometry.axis+rr*bet*(-np.sin(at)*e+np.cos(at)*b)
        av=-rr*bet**2*(np.cos(at)*e+np.sin(at)*b)
        t1=(1-grow)*ti+grow*tv;t1/=np.linalg.norm(t1)
        a1=(1-grow)*ai+grow*av;a1-=a1@t1*t1
        q0=self.coords(p0,ph0);q1=self.coords(end,theta)
        dt0,dd0=self.polar_derivatives(p0,t0,a0);dt1,dd1=self.polar_derivatives(end,t1,a1)
        scale=10*h*(1-grow)+.20*grow;dt0*=scale;dt1*=scale;dd0*=scale**2;dd1*=scale**2
        cp=np.zeros((8,3));cp[0]=q0;cp[1]=q0+dt0/7;cp[2]=q0+2*dt0/7+dd0/42
        cp[7]=q1;cp[6]=q1-dt1/7;cp[5]=q1-2*dt1/7+dd1/42
        for j in [3,4]:
            u=self.exit_u+(j/7)*(-self.exit_u);pp=self.source(np.array([u]),phase)[0]
            initialc=self.coords(pp,ph0+(phi_initial-ph0)*j/7)
            middle_phi=cp[2,2]+(j-2)/3*(cp[5,2]-cp[2,2])
            desired=np.array([p0[0]+(.005 if j==3 else .01),.18 if j==3 else .23,middle_phi])
            cp[j]=(1-grow)*initialc+grow*desired
        u=np.linspace(0,1,241)
        def bezier(n,points,v):return sum(comb(n,j)*(1-v[:,None])**(n-j)*v[:,None]**j*points[j] for j in range(n+1))
        qq=bezier(7,cp,u);plateau=min(p0[0]+.005,self.source_endpoint_x_m+.045)
        escape=np.full(len(u),plateau);edge=.4
        vals=np.array([q0[0],q0[0]+dt0[0]*edge/5,q0[0]+2*dt0[0]*edge/5+dd0[0]*edge**2/20,plateau,plateau,plateau])[:,None]
        early=u<=.4;escape[early]=bezier(5,vals,u[early]/edge)[:,0]
        vals=np.array([plateau,plateau,plateau,q1[0]-2*dt1[0]*edge/5+dd1[0]*edge**2/20,q1[0]-dt1[0]*edge/5,q1[0]])[:,None]
        late=u>=.6;escape[late]=bezier(5,vals,(u[late]-.6)/edge)[:,0]
        qq[:,0]=(1-grow)*qq[:,0]+grow*escape
        prefix=np.column_stack([qq[:,0],qq[:,1]*np.cos(qq[:,2]),1+qq[:,1]*np.sin(qq[:,2])])
        te=self.target_jet(0.,1);ae=self.target_jet(0.,2)
        ep=np.zeros((8,3));ep[0]=end;ep[1]=end+t1*lead/7;ep[2]=end+2*t1*lead/7+a1*lead**2/42
        ep[7]=entry;ep[6]=entry-te*lead/7;ep[5]=entry-2*te*lead/7+ae*lead**2/42
        final3=free+3*self.target_jet(-lead,1)*lead/7+self.target_jet(-lead,2)*lead**2/14+self.target_jet(-lead,3)*lead**3/210
        final4=entry-3*te*lead/7+ae*lead**2/14-self.target_jet(0.,3)*lead**3/210
        ep[3]=(1-grow)*(ep[2]+(ep[5]-ep[2])/3)+grow*final3
        ep[4]=(1-grow)*(ep[2]+2*(ep[5]-ep[2])/3)+grow*final4
        ext=bezier(7,ep,u);us=np.arange(-5,self.exit_u,.002);ut=np.arange(.002,4,.002)
        xyz=np.vstack([self.source(us,phase),prefix,ext[1:],self.target_jet(ut)])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(arc[len(us)])
        self.prefix_end_arc=float(arc[len(us)+len(prefix)-1]);self.target_arc=float(arc[len(us)+len(prefix)+len(ext)-2])
        self.connector_arc=self.target_arc-self.start_arc;self.grow=grow;self.recover=self.smooth((S-6)/2);self.source_exit=p0
        self.nominal_spline=self.spline
        self.nominal_target_arc=self.target_arc;self.nominal_body_points,self.nominal_body_locations=self.body(S,True)
        self.current_global_dense=xyz[::8].copy()
        if abs(self.delta)>1e-15:
            old=self.spline.x;xyz=self.spline(old);station=old-self.target_arc
            w=self.smooth((station+.125)/.125)*self.smooth((.60-station)/.20)
            _,rad,rho=self.sensor_geometry.coordinates(xyz);xyz+=self.delta*w[:,None]*rad/np.maximum(rho[:,None],1e-9)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
            self.start_arc=float(np.interp(self.start_arc,old,arc));self.prefix_end_arc=float(np.interp(self.prefix_end_arc,old,arc));self.target_arc=float(np.interp(self.target_arc,old,arc))
            self.connector_arc=self.target_arc-self.start_arc;self.spline=CubicSpline(arc,xyz,axis=0)
        return np.array([phase,0.])
    def command(self,S):
        q,info=super().command(S);field=info['curvature_field'].copy();bounds=info['curvature_bounds_mm'];nodes=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:])
        v=np.clip(mid/nodes[8],0,1);w=np.sin(np.pi*v)**2;w[bounds[1:]>nodes[8]]=0.
        field+=w[:,None]*self.front_relief[None,:]/1000
        b=self.joint_bending(bounds,field,nodes);q=-np.where(np.arange(20)%2,b[:,0],b[:,1])
        dense,C=reconstruct(field,bounds,info['reference_frame'],info['reference_points'][0],nodes)
        info.update(curvature_field=field,nominal_curve_dense=dense,nominal_curve_points=C,reference_points=C,
            theta_offset_rad=self.theta_offset,front_relief_per_m=self.front_relief.copy(),initial_preload_fraction=0.)
        info['local_curve_chord_error_mm']=float(np.max(abs(np.linalg.norm(np.diff(C,axis=0),axis=1)-LINK)))*1000
        return q,info


    def nominal_command(self,S):
        delta=self.delta;gamma=self.source_gamma;probe=self.probe.copy();relief=self.front_relief.copy()
        self.delta=0.;self.source_gamma=0.;self.probe[:]=0.;self.front_relief[:]=0.
        try:return self.command(S)
        finally:self.delta=delta;self.source_gamma=gamma;self.probe=probe;self.front_relief=relief
