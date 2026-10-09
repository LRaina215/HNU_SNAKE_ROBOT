"""Brace nine source rods, unfold twelve front rods, then admit actual grip."""
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from snake_control.inner_hook_route import InnerHookRoute
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,nodes,WholeCurveMapper
from snake_control.suspended_transfer_route import R,ALPHA,BETA


class BracedInnerRoute(InnerHookRoute):
    def __init__(self):
        super().__init__(front=6,bridge=6)
        self.target_y=.17;self.q[0]=.70;self.e=self.target([0])[0]
        self.initial=None;self.source_field=None;self.residual=None
        self.build_connector(np.array([ALPHA,-BETA*np.sin(self.source_phi),BETA*np.cos(self.source_phi)]))

    def initial_window(self):return self.source((12-np.arange(22))*LINK)

    def build_connector(self,ts):
        ts=ts/np.linalg.norm(ts);te=self.entry_tangent;t=np.linspace(0,1,2401)[:,None]
        def raw(h):return (2*t**3-3*t**2+1)*self.q+(t**3-2*t**2+t)*h*ts+(-2*t**3+3*t**2)*self.e+(t**3-t**2)*h*te
        def length(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        h=brentq(lambda h:length(h)-6*LINK,1e-6,2.)
        self.connector=raw(h);self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.connector,axis=0),axis=1))]
        def probe(y):
            control=np.array([self.q,self.q+.08*ts,[.86,y,.78],[1.07,y,.78],self.e-.08*te,self.e])
            return sum(math.comb(5,j)*(1-t)**(5-j)*t**j*control[j] for j in range(6))
        def plength(y):return np.linalg.norm(np.diff(probe(y),axis=0),axis=1).sum()
        y=brentq(lambda y:plength(y)-12*LINK,.20,1.5)
        self.probe=probe(y);self.probe_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.probe,axis=0),axis=1))]
        v=np.arange(13)*LINK
        self.probe_nodes=np.column_stack([np.interp(12*LINK-v,self.probe_arc,self.probe[:,j]) for j in range(3)])

    def bind_source(self,points):
        self.initial=np.asarray(points).copy();v=self.origin-np.arange(22)*LINK
        self.source_field=CubicSpline(v[::-1],self.initial[::-1],axis=0)
        self.q=self.initial[12].copy();self.build_connector(self.initial[12]-self.initial[13])

    def nominal(self,v):
        p=super().nominal(v);v=np.atleast_1d(v)
        if self.source_field is not None:p[v<self.source_cut]=self.source_field(v[v<self.source_cut])
        return p

    def probe_window(self,stage,a):
        a=float(np.clip(a,0,1));a=a*a*(3-2*a);p=self.initial.copy()
        final=self.nominal(self.origin-np.arange(22)*LINK)
        if stage=='unfold':
            p[:13]=(1-a)*self.initial[:13]+a*self.probe_nodes
            weight=np.clip((12-np.arange(13))/3,0,1)
            p[:13,1]+=.22*np.sin(np.pi*a)*weight
            p[:13,2]-=.10*np.sin(np.pi*a)*weight
        else:p[:13]=(1-a)*self.probe_nodes+a*final[:13]
        return p,self.origin-np.arange(22)*LINK

    def bind_acquired(self,points):
        p=np.asarray(points);v=self.origin-np.arange(22)*LINK
        phi=np.unwrap(np.arctan2(p[:7,2]-1,p[:7,0]-1.1))
        self.phi0=float(np.mean(phi+BETA*v[:7]/R));self.target_y=float(np.mean(p[:7,1]-ALPHA*v[:7]));self.e=self.target([0])[0]
        self.build_connector(p[12]-p[13]);self.residual=CubicSpline(v[::-1],(p-self.nominal(v))[::-1],axis=0)

    def window(self,shift,coeff=None):
        v=self.origin+shift-np.arange(22)*LINK;p=self.nominal(v);a=self.blend(shift)
        if self.residual is not None and a<1:p+=(1-a)*self.residual(np.clip(v,self.origin-21*LINK,self.origin))
        if coeff is not None:
            w=np.clip((v-self.source_cut)/(-self.source_cut),0,1);h=16*w*w*(1-w)*(1-w);p+=a*h[:,None]*(coeff[0]+(2*w-1)[:,None]*coeff[1])
        return p,v


class BracedProbe:
    def __init__(self,model,data,route,unfold_seconds=70,coil_seconds=80):
        self.m=model;self.route=route;self.mapper=WholeCurveMapper(model);self.mapper.bearing_module=14
        self.qa=self.mapper.qa;self.last=data.ctrl.copy();self.rear_hold=data.ctrl.copy();self.kp=np.full(20,127.)
        self.unfold_seconds=unfold_seconds;self.coil_seconds=coil_seconds;self.previous=nodes(data).copy();self.received=set();self.exited=set();self.receive_age=np.zeros(21);self.shift=0.;self.phase=0.;self.failure=None
        self.acquire_since=None;self.ready=False;self.best_distance=float(np.linalg.norm(self.previous[0]-route.e));self.last_progress=0.

    def command(self,data,t,forces,source_any):
        p=nodes(data);c=(p[:-1]+p[1:])/2
        stage='unfold' if t<self.unfold_seconds else 'coil';a=t/self.unfold_seconds if stage=='unfold' else (t-self.unfold_seconds)/self.coil_seconds
        guide,v=self.route.probe_window(stage,a)
        desired=np.full(20,63.5);desired[12:]=127.;self.kp+=np.clip(desired-self.kp,-127*DT/1.5,127*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        anchor_forces=forces.copy();anchor_forces[1]=0.;anchor_forces[0,:12]=0.
        weights=np.full(22,2.5);active=np.arange(20)<12
        _,q,projection=self.mapper.solve(guide,self.previous,data,anchor_forces,self.last,active_mask=active,node_weights=weights)
        q[12:]=self.rear_hold[12:];target=self.last+np.clip(q-self.last,-np.radians(20)*DT,np.radians(20)*DT);self.last=target.copy();self.previous=guide.copy()
        rr=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1]);near=(rr>.065)&(rr<.125)&(c[:,1]>.15)
        connected=near.copy();connected[1:]=near[1:]&near[:-1]&(da>.3)&(da<1.1)&(dy<-.002)&(dy>-.045)&(abs(dy+.180/(2*np.pi)*da)<.025)
        local=np.array([forces[1,max(0,j-1):min(21,j+2)].sum()>.2 for j in range(21)]);connected&=local
        self.receive_age=np.where(connected,self.receive_age+DT,0);self.received=set(np.flatnonzero(self.receive_age>=1.))
        sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);self.exited=set(np.flatnonzero((sr>.130)&(~source_any)&(forces[0]<.1)))
        wrap=float(np.degrees(np.ptp(phi[:6])));acquired=all(j in self.received for j in range(6)) and wrap>180 and forces[1,:6].sum()>1.
        if acquired and self.acquire_since is None:self.acquire_since=t
        if not acquired:self.acquire_since=None
        self.ready=self.acquire_since is not None and t-self.acquire_since>=1.
        distance=float(np.linalg.norm(p[0]-guide[0]))
        endpoint_distance=float(np.linalg.norm(p[0]-self.route.e))
        if endpoint_distance<self.best_distance-.010:self.best_distance=endpoint_distance;self.last_progress=t
        if t>self.unfold_seconds+self.coil_seconds+15:self.failure='braced_probe_without_six_rod_actual_acquisition'
        if forces[0,12:].sum()<1. and t>5:self.failure='posterior_source_brace_lost'
        projection['points_mm']=nodes(self.mapper.d)*1000
        info=dict(stage='rear_braced_front_'+stage,shape_parameter=a,shift_mm=0.,tracking_deg=float(np.degrees(abs(target-data.qpos[self.qa]).max())),curve_radius_delta_mm=0.,reference_points_mm=guide*1000,material_coordinate_mm=v*1000,body_route_regions=np.r_[np.ones(12,int),np.zeros(9,int)],kp_nm_per_rad=self.kp.copy(),actual_received_modules=sorted(self.received),actual_source_exited_modules=sorted(self.exited),actual_new_retained_modules=sorted(self.received),actual_front6_wrap_deg=wrap,projection=projection,roll_factor=0.,roll_phase_rad=0.,last_actual_progress_s=self.last_progress,probe_head_reference_error_mm=distance*1000,maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),yamada_compliance_enabled=False,posterior_held_hinges=list(range(13,21)),physical_root_assigned=False)
        return target,info
