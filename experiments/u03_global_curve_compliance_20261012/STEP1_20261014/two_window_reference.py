"""Two pipe helices and a true geometric connector; S is nominal cell count."""
from curve_reference import GlobalCurveReference, LINK
from geometry_backbone import SpatialBishopBackbone
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.optimize import least_squares,brentq
import numpy as np

class TwoWindowReference(GlobalCurveReference):
    def __init__(self,points,angles,clearance_connector=False):
        self.clearance_connector=clearance_connector
        p=np.asarray(points);n=np.arange(22);phi=float(np.arctan2(p[0,2]-1,p[0,1]))
        def helix_nodes(params):
            R,P,phi0,x0=params;beta=1/np.sqrt(R*R+(P/(2*np.pi))**2);a=P*beta/(2*np.pi)
            h=brentq(lambda h:(a*h)**2+4*R*R*np.sin(beta*h/2)**2-LINK**2,LINK,2*LINK)
            th=phi0+beta*h*n
            return np.column_stack([x0-a*h*n,R*np.cos(th),1+R*np.sin(th)])
        fit=least_squares(lambda v:(helix_nodes(v)-p).ravel(),[.0968,.183,phi,p[0,0]],bounds=([.07,.14,phi-.7,p[0,0]-.04],[.12,.24,phi+.7,p[0,0]+.04]))
        self.radius,self.pitch,self.phi0,self.x0=fit.x
        self.beta=1/np.sqrt(self.radius**2+(self.pitch/(2*np.pi))**2);self.a=self.pitch*self.beta/(2*np.pi)
        self.target_phi=-np.pi/4;self.target_y=.20;self.connector_parameter=.36
        self.feed_per_cycle=np.pi*.058/LINK*2.8
        self.roll_sign=1.;self.torsion_compensation=self.pitch/(2*np.pi)/(self.radius**2+(self.pitch/(2*np.pi))**2)*LINK
        self.moving_family=True;self.anchor_frame=True;self.bezier_bridge=False;self.local_material=False;self.discrete_transport=True;self.axial_family=False;self.release_curve=True
        self.set_nominal_family(0)
        self.global_spline=self.spline;self.global_start_arc=self.start_arc;self.global_target_arc=self.target_arc
        pp=self.body(0);bone=SpatialBishopBackbone(pp*1000,np.full(21,LINK*1000));_,rot=bone.material_field(angles)
        normal=(bone.start_frame[:,1:]@rot)[:,0]
        self.anchor_arc=self.start_arc-1.48
        tt=-self.spline(np.linspace(self.start_arc,self.anchor_arc,500),1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        normal-=normal@tt[0]*tt[0];normal/=np.linalg.norm(normal);self.anchor_normal=self.transport_normal(normal,tt)
        self.initial_error_m=float(np.max(np.linalg.norm(pp-p,axis=1)))
        self.spec=dict(version=8,representation='source X-axis helix, target Y-axis helix, C1 Hermite geometric connector; curvature-release C integrated from template',high_level_feedback=False,initial_fit_only=True,radius_m=float(self.radius),pitch_m=float(self.pitch),source_initial_phi_rad=float(self.phi0),source_initial_head_x_m=float(self.x0),initial_helix_fit_rms_mm=float(np.sqrt(np.mean(np.sum((helix_nodes(fit.x)-p)**2,axis=1)))*1000),initial_reference_max_error_m=self.initial_error_m,S_units='nominal bend-cell count, not actual displacement',S_mapping='S=pi*DM/LM*2.8*chi; chi=0.02*max(0,t-0.5)',window_mapping='template head arclength=start_arc+LM*S; connector arclength computed from geometry each S, source/target roles follow same window',roll_mapping='psi=2*pi*chi-tau*LM*S, common material frame from registered source',target_entry=dict(y_m=self.target_y,polar_rad=self.target_phi),connector='C1 cubic Hermite, endpoint derivatives6*LM; two-link handles; no fixed lower waypoint; actual arc length recorded',release='first6 curvature cells progressive0.4 then sigmoid recovery atS=6; integrated C holds posterior template frame and point')
        self.spec.update(clearance_connector=clearance_connector)
        if clearance_connector:self.spec.update(version=9,connector_clearance='fixed known target-cylinder radial corridor radius0.110m for Y>=0.12, tapers to original helix radius at entry; geometric projection before curvature, never uses actual body/contact')

    def set_nominal_family(self,S):
        v=LINK*S;phases=np.array([self.beta*v,-self.beta*v])
        us=np.arange(-3,0,.002);ut=np.arange(0,4,.002)
        ps=self.source(us,phases[0]);pt=self.target(ut,phases[1]);p0=self.source(np.array([0.]),phases[0])[0];p1=pt[0]
        ph0=self.phi0+phases[0];ph1=self.target_phi+phases[1]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)])
        t1=np.array([-self.radius*self.beta*np.sin(ph1),self.a,self.radius*self.beta*np.cos(ph1)])
        # Handles two rod lengths; endpoint directions are analytic derivatives.
        u=np.linspace(0,1,181);bridge=CubicHermiteSpline([0.,1.],np.vstack([p0,p1]),(6*LINK)*np.vstack([t0,t1]))(u)
        if self.clearance_connector:
            # Fixed world geometry. A smooth radial corridor preserves the
            # target endpoint/tangent since its derivative is zero there.
            gate=np.clip((bridge[:,1]-.08)/.04,0,1);gate=gate*gate*(3-2*gate)
            end=np.clip((1-u)/.25,0,1);end=end*end*(3-2*end)
            start=np.clip(u/.25,0,1);start=start*start*(3-2*start)
            minimum=(self.radius+(.110-self.radius)*end)*gate*start
            radial=bridge[:,[0,2]]-[1.1,1.];r=np.linalg.norm(radial,axis=1)
            proposed=r+.5*(minimum-r+np.sqrt((minimum-r)**2+(.001*gate*start*end)**2))
            bridge[:,[0,2]]=[1.1,1.]+radial*(proposed/np.maximum(r,1e-8))[:,None]
        xyz=np.vstack([ps,bridge,pt[1:]]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(arc[len(ps)]);self.target_arc=float(arc[len(ps)+len(bridge)-1]);self.connector_arc=self.target_arc-self.start_arc
        return phases

    def source(self,u,phase):
        ph=self.phi0-self.beta*u+phase
        return np.column_stack([self.x0+self.a*u,self.radius*np.cos(ph),1+self.radius*np.sin(ph)])

    def target(self,u,phase):
        ph=self.target_phi+self.beta*u+phase
        return np.column_stack([1.1+self.radius*np.cos(ph),self.target_y+self.a*u,1+self.radius*np.sin(ph)])

    def body(self,S,return_locations=False):
        return super().body(S*LINK,return_locations)

    def command(self,S):
        q,i=super().command(S);i.update(connector_arc_m=self.connector_arc,nominal_shift_cells=S)
        return q,i
