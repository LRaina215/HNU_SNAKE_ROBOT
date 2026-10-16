"""Prepare a separate native Step1 candidate; no simulation or state mutation."""
from pathlib import Path
import json, shutil

P=Path(__file__).resolve().parent
B=P/'CONTROLLER_V8_NATIVE_STEP1'
OLD=P/'CONTROLLER_V7_FINAL_CONTRACT_R1'
FORMAL=P.parent/'STEP1_20261014/R1_READY_FROZEN'
B.mkdir(exist_ok=False)
for f in OLD.glob('*.py'):
    shutil.copy2(f,B/f.name)
for name in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']:
    shutil.copy2(OLD/name,B/name)
r=json.loads((OLD/'REGISTERED.json').read_text(encoding='utf-8'))
r.update(version='STEP2_V8_NATIVE_STEP1',current_code_self_checked=False,new_dynamics_starts=0,
         prior_version='V7_FINAL_CONTRACT_R1',execution_backend='Step1 continuous curve/Bishop/integrals before and after bearing',
         revision_reason='Separate candidate for the observed nominal-wrap/actual-shape gap. Preserve given physical t0. Decouple nominal shape progression from source feed and material roll; no world-position optimizer.',
         native_shape=dict(source_release_materials=10.,maximum_release_materials=14.,shape_rate_per_s=.10,shape_max=9.,
             partial_bearing_fraction=.10,full_bearing_fraction=.40,partial_bearing_persistence_s=.5,release_rate_materials_s=.5,
             source_grip_fade_after_capture_s=15.,receiving_virtual_offset_m=.039,
             source_polar_endpoint_branch='Original A16 continuous unwrapped 2pi+atan2 branch; checked on current given without replacing its state'))
r['direct_extension']['capture_timeout_after_release_s']=120.
(B/'REGISTERED.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')

# Reuse the A16 family body literally, replacing target jets with the supplied
# rough/estimated cylinder and replacing shared approach feed by a separate S.
s=(FORMAL/'receiver_phase_reference.py').read_text(encoding='utf-8')
s=s[s.index('    def receiving_phase'):]
s=s.replace("        if not self.receiver_phase_enabled:return super().set_nominal_family(S)\n        if not self.target_fit_enabled:return super().set_nominal_family(S)\n        if not self.supply_enabled:return super().set_nominal_family(S)\n", "        transport=S;S=self.shape_progress\n        self.target_radius=self.geometry.radius+.039+self.delta\n        self.target_beta=1/np.sqrt(self.target_radius**2+(self.pitch/(2*np.pi))**2)\n        self.target_a=self.pitch*self.target_beta/(2*np.pi)\n")
s=s.replace('self.exit_u=-10*h;self.source_chart_shift=self.feed_early*self.phase_cells(S)', 'self.exit_u=-self.release_materials*h;self.source_chart_shift=self.feed_early*self.phase_cells(transport)')
s=s.replace('free=np.array([1.1+self.target_radius*np.cos(free_phi),.20-self.target_a*lead,1+self.target_radius*np.sin(free_phi)])', 'free=self.target_jet(-lead,entry_phi)[0]')
s=s.replace('rr=np.linalg.norm(end[[0,2]]-[1.1,1]);at=np.arctan2(end[2]-1,end[0]-1.1)', 'axial,radial,rr=self.geometry.coordinates(end);ee,bb=self.geometry.frame();at=np.arctan2(radial@bb,radial@ee)')
s=s.replace('tv=np.array([-rr*bet*np.sin(at),aa,rr*bet*np.cos(at)]);av=np.array([-rr*bet**2*np.cos(at),0.,-rr*bet**2*np.sin(at)])', 'tv=aa*self.geometry.axis+rr*bet*(-np.sin(at)*ee+np.cos(at)*bb);av=-rr*bet**2*(np.cos(at)*ee+np.sin(at)*bb)')
s=s.replace('(10*h)*(1-grow)', '(self.release_materials*h)*(1-grow)')
s=s.replace('plateau=min(p0[0]+.005,.995)', 'plateau=min(p0[0]+.005,self.geometry.entry[0]-.105)')
start='        self.target_phi=entry_phi;self.target_y=.20;entry=self.target(np.array([0.]),0)[0];te='
i=s.index(start);j=s.index('\n        scale=lead;',i)
s=s[:i]+"        self.target_phi=entry_phi;entry,te,ae,je=self.target_jet(0.,entry_phi)"+s[j:]
i=s.index('        tf=np.array(');j=s.index('\n        final3=',i)
s=s[:i]+"        free,tf,af,jf=self.target_jet(-lead,entry_phi)"+s[j:]
header='''"""A16 source-polar family + original native mapping; target prior only.

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
        return NativeMapping.command(self,S)
'''
(B/'native_step1_reference.py').write_text(header+s,encoding='utf-8')
print(str(B))
