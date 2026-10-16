"""Convected tail window, early spatial-phase cancellation, covariant gauge.

Only fixed nominal geometry and shared S are inputs. No recorded poses replayed.
"""
from continuous_window_reference import ContinuousWindowReference,LINK
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from math import comb
import numpy as np

class OverlapWindowReference(ContinuousWindowReference):
    def __init__(self,points,angles):
        self.supply_enabled=False;self.source_chart_shift=0.;self.feed_early=.054;self.feed_late=.09
        super().__init__(points,angles)
        self.set_nominal_family(0);oldp,oldloc=self.body(0,True)
        self.tail_u0=float((oldp[-1,0]-self.x0)/self.a)
        self.anchor_u0=-4.
        self.supply_enabled=True
        self.set_nominal_family(0)
        p,loc=self.body(0,True)
        headT=-self.spline(loc[0],1);headT/=np.linalg.norm(headT)
        n=np.eye(3)[np.argmin(abs(headT))];n-=n@headT*headT;n/=np.linalg.norm(n)
        anchor=self.start_arc+self.anchor_u0-self.exit_u
        tt=-self.spline(np.linspace(loc[0],anchor,800),1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        self.supply_anchor_normal=self.transport_normal(n,tt)
        self.gauge_angle=0.;q0,i=self.command(0);b=self.joint_bending(i['curvature_bounds_mm'],i['curvature_field'],i['material_arc_nodes_mm'])
        q90=-np.where(np.arange(20)%2,-b[:,1],b[:,0]);coef=np.linalg.lstsq(np.column_stack([q0,q90]),angles,rcond=None)[0];self.gauge_angle=float(np.arctan2(coef[1],coef[0]))
        self.initial_error_m=float(np.max(np.linalg.norm(p-points,axis=1)))
        self.global_spline=self.spline;self.global_start_arc=self.start_arc;self.global_target_arc=self.target_arc
        self.spec.update(version=11,representation='moving source interface, polar uncoiling prefix and C2 world under-to-up receiving connector',window_mapping='tail arc=start_arc+tail_u0-exit_u+LM*(S-mu); first forward64.5mm chords, reverse to head->tail',source_spatial_phase='world equivalent Phi=beta*LM*mu(S); local source chart phase0, x0 advancesa*LM*mu; mu=S until5, C2 slope decay5..7, mu=6 after7',source_material_polar='theta_i=theta_i0-beta*LM*(S-mu), source material X advancesa*LM*S',roll_mapping='material anchor convectsLM*(S-mu) in moving chart; anchor normal rotates aboutX by-beta*LM*(S-mu); common psi=gauge+2pi*chi; no duplicate torsion subtraction',target_entry=dict(y_m=.27,polar_rad=0.),release='existing quintic uncoiling0..6; entire future receiving connector exists fromS0; material head is not pinned at prefix end',tail_initial_source_u_m=self.tail_u0,anchor_initial_source_u_m=self.anchor_u0,initial_reference_max_error_m=self.initial_error_m,initial_material_gauge_rad=self.gauge_angle,material_sampling='first forward64.5mm chords from prescribed tail station; true geometric arc joint supports',source_interface='ten initial rods posterior to head; material polar kept at initial value while interface advances',high_level_feedback=False)
        self.spec.update(window_mapping='tail arc=start_arc+tail_u0-exit_u+F-M; first forward64.5mm chords',source_spatial_phase='world Phi=beta*M, moving-source-chart x0 advancesa*M; M=.054*mu(S), mu slope decays5..7 to6',source_material_polar='source X_i=X_i0+a*F; theta_i=theta_i0-beta*(F-M)',roll_mapping='material anchor convectsF-M in moving chart, normal rotatesX by-beta*(F-M), psi=gauge+2pi*chi; no duplicate torsion subtraction',feed_early_m_per_S=self.feed_early,feed_late_m_per_S=self.feed_late,feed_mapping='F=.054*S+.036*integral_6_to_S(smoothstep((s-6)/2))',feed_evidence='rounded aggregate U03 rear0..40 axial speed and full-transfer post133.2..234.9 each-module Y advance; no per-frame fitting',connector='A10 source-polar prefix with C2 X wait support before target axis, then fixed C2 lower-to-right-up receiver; no radial push')
        self.spec.update(version=13,representation='A11 source support/feed and common material frame, polar uncoiling into short lower target receiving lead',target_entry=dict(y_m=.20,polar_rad='-pi/4-pi/4*smooth((S-4.5)/1.5); -pi/2 afterS6'),overlap_change='prefix final endpoint is target helix75mm upstream of lower entry; fixed C2 receiving lead instead of long freepoint/right-side detour',release='same quintic S0..6 uncoiling; entry moves from rising sector tolower tangent during final uncoilingS4.5..6 toavoid forced tight reversal',connector='source-polar C2 X wait support preserved, middle polar progression interpolates endpoint jet angles; fixed75mm C2 Cartesian continuation converges to helix through endpoint third derivative jets',native_mapping='original continuous curvature integrals; A12 finite-hinge corrections disabled due anchor-switch/slew cost and no acquisition benefit',handover_evidence='U03 first6 stable53.8s with9 source straight-geometry rods; A11/A12 head has not entered nominal target helix60s, posterior supports leave source earlier',high_level_feedback=False)

    @staticmethod
    def phase_cells(S):
        x=np.clip((S-5)/2,0,1)
        return min(S,5)+2*(x-2.5*x**4+3*x**5-x**6)

    def feed_distance(self,S):
        x=np.clip((S-6)/2,0,1);integral=2*(2.5*x**4-3*x**5+x**6)+max(0,S-8)
        return self.feed_early*S+(self.feed_late-self.feed_early)*integral

    def source(self,u,phase):
        if not self.supply_enabled:return super().source(u,phase)
        phi=self.phi0-self.beta*u
        return np.column_stack([self.x0+self.a*(u+self.source_chart_shift),self.radius*np.cos(phi),1+self.radius*np.sin(phi)])

    def set_nominal_family(self,S):
        if not self.supply_enabled:return super().set_nominal_family(S)
        h=brentq(lambda h:(self.a*h)**2+4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2,LINK,2*LINK)
        self.exit_u=-10*h;self.source_chart_shift=self.feed_early*self.phase_cells(S);phase=self.beta*self.source_chart_shift;self.spatial_phase=phase
        ph0=self.phi0-self.beta*self.exit_u;p0=self.source(np.array([self.exit_u]),0.)[0]
        t0=np.array([self.a,self.radius*self.beta*np.sin(ph0),-self.radius*self.beta*np.cos(ph0)]);a0=np.array([0.,-self.radius*self.beta**2*np.cos(ph0),-self.radius*self.beta**2*np.sin(ph0)])
        phi_initial=self.phi0;initial=self.source(np.array([0.]),0.)[0]
        ti=np.array([self.a,self.radius*self.beta*np.sin(phi_initial),-self.radius*self.beta*np.cos(phi_initial)]);ai=np.array([0.,-self.radius*self.beta**2*np.cos(phi_initial),-self.radius*self.beta**2*np.sin(phi_initial)])
        grow=self.smooth(S/6);lead=.075;entry_phi=-np.pi/4-np.pi/4*self.smooth((S-4.5)/1.5);free_phi=entry_phi-self.beta*lead;free=np.array([1.1+self.radius*np.cos(free_phi),.20-self.a*lead,1+self.radius*np.sin(free_phi)]);theta_free=2*np.pi+np.arctan2(free[2]-1,free[1]);rfree=np.linalg.norm(free[1:]-[0,1])
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
        self.target_phi=entry_phi;self.target_y=.20;entry=self.target(np.array([0.]),0)[0];te=np.array([-self.radius*self.beta*np.sin(entry_phi),self.a,self.radius*self.beta*np.cos(entry_phi)]);ae=np.array([-self.radius*self.beta**2*np.cos(entry_phi),0.,-self.radius*self.beta**2*np.sin(entry_phi)])
        scale=lead;dp0=t1*scale;dp1=te*scale;ddp0=a1*scale**2;ddp1=ae*scale**2
        ep=np.zeros((8,3));ep[0]=end;ep[1]=end+dp0/7;ep[2]=end+2*dp0/7+ddp0/42;ep[7]=entry;ep[6]=entry-dp1/7;ep[5]=entry-2*dp1/7+ddp1/42
        # A fixed75mm helical lead avoids collapsing a finite turn tozero.
        tf=np.array([-self.radius*self.beta*np.sin(free_phi),self.a,self.radius*self.beta*np.cos(free_phi)])
        af=np.array([-self.radius*self.beta**2*np.cos(free_phi),0.,-self.radius*self.beta**2*np.sin(free_phi)])
        jf=np.array([self.radius*self.beta**3*np.sin(free_phi),0.,-self.radius*self.beta**3*np.cos(free_phi)])
        je=np.array([self.radius*self.beta**3*np.sin(entry_phi),0.,-self.radius*self.beta**3*np.cos(entry_phi)])
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

    def body(self,S,return_locations=False):
        if not self.supply_enabled:return super().body(S,return_locations)
        tail=self.start_arc+self.tail_u0-self.exit_u+self.feed_distance(S)-self.source_chart_shift;locations=[tail];points=[self.spline(tail)]
        for _ in range(21):
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=locations[-1]+ds;dist=np.linalg.norm(self.spline(trial)-points[-1],axis=1);ids=np.flatnonzero(dist>=LINK)
            if not len(ids):raise ValueError('no forward chord-length material intersection')
            j=int(ids[0]);hi=trial[j];lo=locations[-1] if j==0 else trial[j-1]
            for _ in range(12):
                mid=(hi+lo)/2
                if np.linalg.norm(self.spline(mid)-points[-1])>=LINK:hi=mid
                else:lo=mid
            locations.append((hi+lo)/2);points.append(self.spline(locations[-1]))
        p=np.asarray(points)[::-1];loc=np.asarray(locations)[::-1];return (p,loc) if return_locations else p

    def material_anchor(self,S,locations):
        if not self.supply_enabled:return super().material_anchor(S,locations)
        anchor=self.start_arc+self.anchor_u0+self.feed_distance(S)-self.source_chart_shift-self.exit_u;delta=self.spatial_phase-self.beta*self.feed_distance(S);n=self.supply_anchor_normal.copy();co,sn=np.cos(delta),np.sin(delta);n[1:]=[co*n[1]-sn*n[2],sn*n[1]+co*n[2]]
        return anchor,n,self.gauge_angle+2*np.pi*S/self.feed_per_cycle,0.

    def command(self,S):
        if not self.supply_enabled:return super().command(S)
        q,i=super().command(S);i.update(nominal_pipe_phase=np.array([self.spatial_phase,self.target_phi+np.pi/2]),target_entry_polar_rad=float(self.target_phi),receiving_lead_arc_m=.075,supply_tail_arc_m=float(self.start_arc+self.tail_u0-self.exit_u+self.feed_distance(S)-self.source_chart_shift),supply_distance_m=self.feed_distance(S),prefix_end_arc_m=self.prefix_end_arc,source_interface_x_m=float(self.source_exit[0]),source_chart_shift_m=self.source_chart_shift)
        return q,i
