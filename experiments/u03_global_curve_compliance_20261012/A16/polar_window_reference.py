"""C2 source-polar connector; initial uncoiling and receiving-radius recovery."""
from two_window_reference import TwoWindowReference,LINK
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from math import comb
import numpy as np

class PolarWindowReference(TwoWindowReference):
    def __init__(self,points,angles):
        self.cut_cells=8;self.initial_station=None;self.last_compensation=0.
        super().__init__(points,angles)
        self.release_curve=False;self.torsion_compensation=0.
        _,loc=self.body(0,True);self.initial_station=.5*(loc[-1]+loc[-2])-self.start_arc
        self.set_nominal_family(8.)
        self.global_spline=self.spline;self.global_start_arc=self.start_arc;self.global_target_arc=self.target_arc
        self.set_nominal_family(0.)
        self.spec.update(version=10,representation='two world-pipe helices and C2 degree7 source-cylinder-coordinate connector',connector='control coefficients in (X, radius about X-pipe, continuously unwrapped source polar angle); no Cartesian projection',source_interface='eight initial rods posterior to initial head; within original source straight',S_units='nominal bend-cell count, not actual displacement',window_mapping='initial8-cell connector uncoils byS=6; receiving window advancesLM*(S-6) while its released radius recovers to fixed target radius byS=8',roll_mapping='psi=2pi*chi; source material station torsion transported from once-initialized gauge',release='geometric uncoiling to head at[1.24,.20,.80], then radius recovery into target helix; no post-extraction curvature scaling',target_entry=dict(y_m=.20,polar_rad=-np.pi/2),high_level_feedback=False)

    @staticmethod
    def smooth(x):
        x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))

    def coords(self,p,theta):
        return np.array([p[0],np.linalg.norm(p[1:]-[0,1]),theta])

    @staticmethod
    def polar_derivatives(p,t,acc):
        rad=p[1:]-[0,1];r=np.linalg.norm(rad);er=rad/r;ep=np.array([-er[1],er[0]])
        rd=er@t[1:];td=ep@t[1:]/r
        return np.array([t[0],rd,td]),np.array([acc[0],er@acc[1:]+r*td*td,(ep@acc[1:]-2*rd*td)/r])

    def set_nominal_family(self,S):
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        exit_u=-self.cut_cells*h;source_phi=self.phi0-self.beta*exit_u
        p0=self.source(np.array([exit_u]),0)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(source_phi),-self.radius*self.beta*np.cos(source_phi)])
        a0=np.array([0.,-self.radius*self.beta**2*np.cos(source_phi),-self.radius*self.beta**2*np.sin(source_phi)])
        initial_head=self.source(np.array([0.]),0)[0];initial_t=np.array([self.a,self.radius*self.beta*np.sin(self.phi0),-self.radius*self.beta*np.cos(self.phi0)])
        initial_acc=np.array([0.,-self.radius*self.beta**2*np.cos(self.phi0),-self.radius*self.beta**2*np.sin(self.phi0)])
        self.target_phi=-np.pi/2
        entry=self.target(np.array([0.]),0)[0];free=np.array([1.24,.20,.80])
        grow=self.smooth(S/6);recover=self.smooth((S-6)/2)
        end_theta=self.phi0+grow*(2*np.pi+np.arctan2(free[2]-1,free[1])-self.phi0)+recover*(np.arctan2(entry[2]-1,entry[1])-np.arctan2(free[2]-1,free[1]))
        # Interpolate in this same cylindrical chart. Cartesian blending of
        # endpoint positions would shrink through the source pipe during unwind.
        end_x=initial_head[0]+grow*(free[0]-initial_head[0])+recover*(entry[0]-free[0])
        free_radius=np.linalg.norm(free[1:]-[0,1]);entry_radius=np.linalg.norm(entry[1:]-[0,1])
        end_radius=self.radius+grow*(free_radius-self.radius)+recover*(entry_radius-free_radius)
        end=np.array([end_x,end_radius*np.cos(end_theta),1+end_radius*np.sin(end_theta)])
        rr=np.linalg.norm(end[[0,2]]-[1.1,1]);theta_t=np.arctan2(end[2]-1,end[0]-1.1)
        b=self.pitch/(2*np.pi);bet=1/np.sqrt(rr*rr+b*b);aa=b*bet
        target_t=np.array([-rr*bet*np.sin(theta_t),aa,rr*bet*np.cos(theta_t)])
        target_acc=np.array([-rr*bet*bet*np.cos(theta_t),0.,-rr*bet*bet*np.sin(theta_t)])
        t1=(1-grow)*initial_t+grow*target_t;t1/=np.linalg.norm(t1)
        acc1=(1-grow)*initial_acc+grow*target_acc;acc1-=acc1@t1*t1
        q0=self.coords(p0,source_phi);q1=self.coords(end,end_theta)
        dt0,dd0=self.polar_derivatives(p0,t0,a0);dt1,dd1=self.polar_derivatives(end,t1,acc1)
        scale=(self.cut_cells*h)*(1-grow)+.20*grow
        dt0*=scale;dt1*=scale;dd0*=scale*scale;dd1*=scale*scale
        cp=np.zeros((8,3));cp[0]=q0;cp[1]=q0+dt0/7;cp[2]=q0+2*dt0/7+dd0/42
        cp[7]=q1;cp[6]=q1-dt1/7;cp[5]=q1-2*dt1/7+dd1/42
        initial3=self.coords(self.source(np.array([exit_u+(3/7)*(-exit_u)]),0)[0],source_phi+(self.phi0-source_phi)*3/7)
        initial4=self.coords(self.source(np.array([exit_u+(4/7)*(-exit_u)]),0)[0],source_phi+(self.phi0-source_phi)*4/7)
        cp[3]=(1-grow)*initial3+grow*np.array([.98,.18,5.4])
        cp[4]=(1-grow)*initial4+grow*np.array([1.04,.23,5.5])
        uu=np.linspace(0,1,241);qq=sum(comb(7,j)*(1-uu[:,None])**(7-j)*uu[:,None]**j*cp[j] for j in range(8))
        bridge=np.column_stack([qq[:,0],qq[:,1]*np.cos(qq[:,2]),1+qq[:,1]*np.sin(qq[:,2])])
        us=np.arange(-3,exit_u,.002);ps=self.source(us,0)
        if S>=6:
            ut=np.arange(.002,4,.002)
            ph=theta_t+bet*ut
            pt=np.column_stack([1.1+rr*np.cos(ph),end[1]+aa*ut,1+rr*np.sin(ph)])
        else:
            # Only a short C2 continuation beyond the free head. A premature
            # jump to the distant receiving helix would pollute the spline.
            ut=np.linspace(.0005,.035,70);pt=end+ut[:,None]*t1+.5*ut[:,None]**2*acc1
        xyz=np.vstack([ps,bridge,pt]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(arc[len(ps)]);self.target_arc=float(arc[len(ps)+len(bridge)-1]);self.connector_arc=self.target_arc-self.start_arc
        self.connector_controls=cp;self.connector_raw=bridge;self.grow=grow;self.recover=recover;self.source_exit=p0;self.head_parameter=self.target_arc+LINK*max(0,S-6)
        return np.zeros(2)

    def body(self,S,return_locations=False):
        # Parent's chord sampler starts from start_arc + supplied distance.
        from curve_reference import GlobalCurveReference
        return GlobalCurveReference.body(self,self.head_parameter-self.start_arc,return_locations)

    def discrete_head_normal(self,points,locations,S,phases,angle):
        station=.5*(locations[-1]+locations[-2])-self.start_arc
        tau=self.pitch/(2*np.pi)/(self.radius**2+(self.pitch/(2*np.pi))**2)
        compensation=0. if self.initial_station is None else tau*(station-self.initial_station)
        self.last_compensation=compensation
        return super().discrete_head_normal(points,locations,S,phases,angle-compensation)

    def command(self,S):
        q,i=super().command(S)
        # Whole geometric C; post-field release from old candidates is off.
        if 'nominal_curve_points' not in i:
            from geometry_backbone import SpatialBishopBackbone
            bone=SpatialBishopBackbone(i['reference_points']*1000,np.full(21,LINK*1000))
            sample=bone.sample(field=i['curvature_field'],material_rotation=i['material_rotation'],step_mm=8)
            cp=np.asarray(sample['points_mm']);i['nominal_curve_points']=np.column_stack([np.interp(np.arange(22)*LINK*1000,sample['s_mm'],cp[:,j]) for j in range(3)])/1000
        i.update(source_station_transport_rad=self.last_compensation,uncoil_fraction=self.grow,receiving_recovery_fraction=self.recover)
        return q,i
