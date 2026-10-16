"""Shared curve jets/compliance/mapping helpers. The live controller uses
FromSourceCurve, which overrides the late-family method below at every S.
"""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from math import comb
from native_mapping import NativeMapping,LINK
from source_mode import SourceMode,reconstruct
from geometry import Geometry,normalize

class EstimatedCurve(NativeMapping,SourceMode):
    def __init__(self,constants,prior,initial_delta=0.):
        for key,value in constants.items():setattr(self,key,np.array(value) if isinstance(value,list) else value)
        self.geometry=prior.copy();self.sensor_geometry=prior.copy();self.supply_enabled=True
        self.delta=initial_delta;self.filtered_error=0.;self.filtered_load=0.;self.compliance_ready=True
        self.source_gamma=0.;self.source_direction=np.array([0.,1.]);self.source_gradient=0.;self.source_requested=0.
        self.probe=np.zeros(2)
        self.receiving_qualified_mask=np.zeros(21,bool)
        self.observation=dict(compliance_delta_m=self.delta,compliance_rate_m_s=0.,compliance_gap_m=0.,compliance_mean_load_N=0.,compliance_active_material=np.zeros(21,bool),compliance_amplitude_saturated=False,compliance_rate_saturated=False)
        self.spec=dict(representation='A16 source/prefix/75mm helix lead + estimated arbitrary target axis; original true-arc native mapping',radius_m=self.radius,pitch_m=self.pitch,helix_surface_offset_m=.039,high_level_feedback=True)
    def receiving_phase(self,S):return -np.pi/4-np.pi/4*self.smooth((S-4.5)/1.5)-np.pi/3*self.smooth((S-7.5)/1.5)
    def target_jet(self,u,order=0):
        g=self.geometry;e,b=g.frame();phi=self.target_phi+self.target_beta*np.asarray(u)
        radial=np.cos(phi)[...,None]*e+np.sin(phi)[...,None]*b
        tangent=-np.sin(phi)[...,None]*e+np.cos(phi)[...,None]*b
        if order==0:return g.entry+self.target_a*np.asarray(u)[...,None]*g.axis+self.target_radius*radial
        if order==1:return self.target_a*g.axis+self.target_radius*self.target_beta*tangent
        if order==2:return -self.target_radius*self.target_beta**2*radial
        return -self.target_radius*self.target_beta**3*tangent
    def set_nominal_family(self,S):
        if S<9.:raise ValueError('STEP2 given-contact family requires S>=9; no precontact search')
        self.target_radius=self.geometry.radius+.039
        self.target_beta=1/np.sqrt(self.target_radius**2+(self.pitch/(2*np.pi))**2)
        self.target_a=self.pitch*self.target_beta/(2*np.pi)
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.exit_u=-10*h;self.source_chart_shift=self.feed_early*self.phase_cells(S);phase=self.beta*self.source_chart_shift;self.spatial_phase=phase
        ph0=self.phi0-self.beta*self.exit_u;p0=self.source(np.array([self.exit_u]),0.)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)])
        a0=np.array([0.,-self.radius*self.beta**2*np.cos(ph0),-self.radius*self.beta**2*np.sin(ph0)])
        lead=.075;self.target_phi=self.receiving_phase(S)
        free=self.target_jet(-lead);entry=self.target_jet(0.);t1=self.target_jet(-lead,1);a1=self.target_jet(-lead,2)
        theta=2*np.pi+np.arctan2(free[2]-1,free[1]);q0=self.coords(p0,ph0);q1=self.coords(free,theta)
        dt0,dd0=self.polar_derivatives(p0,t0,a0);dt1,dd1=self.polar_derivatives(free,t1,a1)
        scale=.20;dt0*=scale;dt1*=scale;dd0*=scale**2;dd1*=scale**2
        cp=np.zeros((8,3));cp[0]=q0;cp[1]=q0+dt0/7;cp[2]=q0+2*dt0/7+dd0/42;cp[7]=q1;cp[6]=q1-dt1/7;cp[5]=q1-2*dt1/7+dd1/42
        for j in [3,4]:
            middle_phi=cp[2,2]+(j-2)/3*(cp[5,2]-cp[2,2]);cp[j]=[p0[0]+(.005 if j==3 else .01),(.18 if j==3 else .23),middle_phi]
        u=np.linspace(0,1,241);qq=sum(comb(7,j)*(1-u[:,None])**(7-j)*u[:,None]**j*cp[j] for j in range(8))
        # X wait bound derives only from the known source endpoint/clearance.
        plateau=min(p0[0]+.005,self.source_endpoint_x_m+.045)
        def bezier5(values,v):return sum(comb(5,k)*(1-v)**(5-k)*v**k*values[k] for k in range(6))
        early=u<=.4;late=u>=.6;escape=np.full(len(u),plateau);e=.4
        vals=[q0[0],q0[0]+dt0[0]*e/5,q0[0]+2*dt0[0]*e/5+dd0[0]*e**2/20,plateau,plateau,plateau];escape[early]=bezier5(vals,u[early]/e)
        vals=[plateau,plateau,plateau,q1[0]-2*dt1[0]*e/5+dd1[0]*e**2/20,q1[0]-dt1[0]*e/5,q1[0]];escape[late]=bezier5(vals,(u[late]-.6)/e)
        qq[:,0]=escape;prefix=np.column_stack([qq[:,0],qq[:,1]*np.cos(qq[:,2]),1+qq[:,1]*np.sin(qq[:,2])])
        te=self.target_jet(0.,1);ae=self.target_jet(0.,2)
        ep=np.zeros((8,3));ep[0]=free;ep[1]=free+t1*lead/7;ep[2]=free+2*t1*lead/7+a1*lead**2/42
        ep[7]=entry;ep[6]=entry-te*lead/7;ep[5]=entry-2*te*lead/7+ae*lead**2/42
        ep[3]=free+3*t1*lead/7+a1*lead**2/14+self.target_jet(-lead,3)*lead**3/210
        ep[4]=entry-3*te*lead/7+ae*lead**2/14-self.target_jet(0.,3)*lead**3/210
        ext=sum(comb(7,j)*(1-u[:,None])**(7-j)*u[:,None]**j*ep[j] for j in range(8))
        us=np.arange(-5,self.exit_u,.002);ps=self.source(us,phase);ut=np.arange(.002,4,.002);pt=self.target_jet(ut);extra=ext[1:]
        xyz=np.vstack([ps,prefix,extra,pt]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(arc[len(ps)]);self.prefix_end_arc=float(arc[len(ps)+len(prefix)-1]);self.target_arc=float(arc[len(ps)+len(prefix)+len(extra)-1]);self.connector_arc=self.target_arc-self.start_arc
        self.grow=1.;self.recover=1.;self.source_exit=p0
        # G is captured before local compliance; it never copies actual body.
        self.nominal_target_arc=self.target_arc;self.nominal_body_points,self.nominal_body_locations=self.body(S,True)
        self.current_global_dense=xyz[::8].copy()
        if abs(self.delta)>1e-15:
            old=self.spline.x;xyz=self.spline(old);station=old-self.target_arc
            weight=self.smooth((station+.125)/.125)*self.smooth((.60-station)/.20)
            _,rad,rho=self.sensor_geometry.coordinates(xyz);xyz+=self.delta*weight[:,None]*rad/np.maximum(rho[:,None],1e-9)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
            self.start_arc=float(np.interp(self.start_arc,old,arc));self.prefix_end_arc=float(np.interp(self.prefix_end_arc,old,arc));self.target_arc=float(np.interp(self.target_arc,old,arc));self.connector_arc=self.target_arc-self.start_arc
            self.spline=CubicSpline(arc,xyz,axis=0)
        return np.array([phase,0.])
    def observe(self,centers,target_normal,previous_info,dt):
        loc=previous_info.get('global_reference_locations',previous_info['reference_locations']);station=.5*(loc[:-1]+loc[1:])-previous_info.get('global_target_arc_m',self.target_arc)
        axial,_,rho=self.sensor_geometry.coordinates(centers)
        mask=(station>=-.075)&(station<=.40)&(axial>-.05)&(axial<1.)&self.receiving_qualified_mask
        err=float(np.mean(rho[mask])-(self.sensor_geometry.radius+.036)) if mask.any() else 0.
        load=float(np.mean(target_normal[mask])) if mask.any() else 0.
        alpha=1-np.exp(-dt/.4);self.filtered_error+=alpha*(err-self.filtered_error);self.filtered_load+=alpha*(load-self.filtered_load)
        gap=np.sign(self.filtered_error)*max(0.,abs(self.filtered_error)-.004)
        desired=-.06*gap+.00005*max(0.,self.filtered_load-12.)-.1*self.delta
        if not mask.any():desired=-.1*self.delta
        rate=float(np.clip(desired,-.00025,.00025));before=self.delta;self.delta=float(np.clip(before+dt*rate,-.002,.002))
        self.observation=dict(compliance_delta_m=self.delta,compliance_rate_m_s=(self.delta-before)/dt,compliance_gap_m=self.filtered_error,compliance_mean_load_N=self.filtered_load,compliance_active_material=mask,compliance_amplitude_saturated=abs(self.delta)>=.002-1e-12,compliance_rate_saturated=abs(desired)>.00025)
    def command(self,S):
        q,info=NativeMapping.command(self,S)
        info.update(geometric_curvature_field=info['curvature_field'].copy(),geometric_reference_points=info['reference_points'].copy(),geometric_native_command=q.copy())
        info.update(global_reference_points=self.nominal_body_points.copy(),global_reference_locations=self.nominal_body_locations.copy(),global_target_arc_m=self.nominal_target_arc,**self.observation,
            source_gamma_per_m=self.source_gamma,source_curve_direction=self.source_direction.copy(),probe_curvature_per_m=self.probe.copy())
        field=info['curvature_field'].copy();bounds=info['curvature_bounds_mm'];nodes=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:])
        if self.source_gamma>0:
            v=np.clip((mid-nodes[15])/(nodes[-1]-nodes[15]),0,1);weight=np.sin(np.pi*v)**2;weight[(bounds[:-1]<nodes[15])|(bounds[1:]>nodes[-1])]=0.
            field+=self.source_gamma*weight[:,None]*self.source_direction[None,:]/1000
        probe_field=np.zeros_like(field)
        if np.any(self.probe):
            v=np.clip(mid/nodes[8],0,1);weight=np.sin(np.pi*v)**2;weight[bounds[1:]>nodes[8]]=0.
            probe_field=weight[:,None]*self.probe[None,:]/1000;field+=probe_field
        probe_bending=self.joint_bending(bounds,probe_field,nodes)
        info['probe_native_delta_rad']=-np.where(np.arange(20)%2,probe_bending[:,0],probe_bending[:,1])
        if self.source_gamma>0 or np.any(self.probe):
            b=self.joint_bending(bounds,field,nodes);q=-np.where(np.arange(20)%2,b[:,0],b[:,1])
            dense,C=reconstruct(field,bounds,info['reference_frame'],info['reference_points'][0],nodes)
            info.update(curvature_field=field,nominal_curve_dense=dense,nominal_curve_points=C,reference_points=C)
        info['local_curve_chord_error_mm']=float(np.max(abs(np.linalg.norm(np.diff(info['nominal_curve_points'],axis=0),axis=1)-LINK)))*1000
        info['target_helix_radius_m']=self.target_radius;info['target_helix_pitch_m']=self.pitch
        return q,info

