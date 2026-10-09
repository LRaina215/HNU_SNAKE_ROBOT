"""One continuous source-bridge-target curve with transported material frames."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from snake_control.suspended_transfer_route import receiving,R,ALPHA,BETA,INTAKE,A
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,nodes
from snake_control.horizontal_pipe_roll import HingePhaseMapper

FRONT,BRIDGE,SOURCE=8,7,6
CUT=-BRIDGE*LINK;TOTAL=(BRIDGE+SOURCE)*LINK
Q=np.array([.82,0.,1.-R]);PHI=-np.pi/2

def source_curve(u):
    u=np.atleast_1d(u);phi=PHI+BETA*u/R
    return np.column_stack([Q[0]+ALPHA*u,R*np.cos(phi),1.+R*np.sin(phi)])

class ThreeSegmentRoute:
    def __init__(self):
        grid=np.linspace(0.,1.,2401);t=grid[:,None];start=Q;end=A
        tangent=np.array([ALPHA,BETA,0.]);up=np.array([0.,0.,1.]);length=BRIDGE*LINK-INTAKE
        def raw(h):return (2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*h*tangent+(-2*t**3+3*t**2)*end+(t**3-t**2)*h*up
        def arc(h):return np.linalg.norm(np.diff(raw(h),axis=0),axis=1).sum()
        h=brentq(lambda x:arc(x)-length,1e-6,3.)
        self.connector=raw(h);self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.connector,axis=0),axis=1))]
        self.handle=h;self.length=length
        # One source radial normal is transported from a fixed curve location.
        # No per-frame world-axis normal selection or all-body world-Y rotation.
        self.frame_s=np.linspace(-TOTAL-.1,21*LINK+.1,4201);p=self.sample(self.frame_s)
        tangent=np.gradient(p,self.frame_s,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        phi=PHI+BETA*(self.frame_s[0]-CUT)/R
        normal=np.array([0.,np.cos(phi),np.sin(phi)]);normal-=normal@tangent[0]*tangent[0];normal/=np.linalg.norm(normal)
        normals=[normal.copy()]
        for before,after in zip(tangent[:-1],tangent[1:]):
            cross=np.cross(before,after);sine=np.linalg.norm(cross);angle=np.arctan2(sine,np.clip(before@after,-1,1))
            if sine>1e-12:normal=Rotation.from_rotvec(cross*angle/sine).apply(normal)
            normal-=normal@after*after;normal/=np.linalg.norm(normal);normals.append(normal.copy())
        self.tangent=tangent;self.normal=np.array(normals)

    def sample(self,v):
        v=np.atleast_1d(v);p=receiving(v);source=v<CUT;bridge=(v>=CUT)&(v<-INTAKE)
        p[source]=source_curve(v[source]-CUT)
        p[bridge]=np.column_stack([np.interp(v[bridge]-CUT,self.arc,self.connector[:,k]) for k in range(3)])
        return p

    def frame(self,s):
        t=np.array([np.interp(s,self.frame_s,self.tangent[:,j]) for j in range(3)]);t/=np.linalg.norm(t)
        n=np.array([np.interp(s,self.frame_s,self.normal[:,j]) for j in range(3)]);n-=n@t*t;n/=np.linalg.norm(n)
        return np.column_stack([n,np.cross(t,n),t])

    def window(self,shift,coeff=None):
        v=FRONT*LINK+shift-np.arange(22)*LINK;p=self.sample(v)
        if coeff is not None:
            w=np.clip((v-CUT)/(-CUT),0,1);h=16*w*w*(1-w)*(1-w)
            p+=h[:,None]*(coeff[0]+(2*w-1)[:,None]*coeff[1])
        return p,v

class ThreeSegmentTransfer:
    def __init__(self,model,data,speed=.005,rate_deg_s=20.,roll_factor=None,bridge_gain=.25):
        self.m=model;self.mapper=HingePhaseMapper(model,data);self.qa=self.mapper.qa;self.va=self.mapper.va;self.route=ThreeSegmentRoute()
        self.last=data.ctrl.copy();self.kp=np.full(20,127.);self.shift=0.;self.phase=0.;self.speed=speed;self.step=np.radians(rate_deg_s)*DT
        self.roll_factor=ALPHA/BETA**2 if roll_factor is None else roll_factor;self.bridge_gain=bridge_gain
        self.frame_offset=self.route.frame(FRONT*LINK).T@self.mapper.rotation
        self.coeff=np.zeros((2,3));self.receive_age=np.zeros(21);self.received=set(range(FRONT));self.engaged=set(range(FRONT))
        self.exit_age=np.zeros(21);self.exited=set();self.best_counts=0;self.last_progress=0.;self.failure=None
        self.initial_centers=.5*(nodes(data)[:-1]+nodes(data)[1:]);self.initial_head_axis=float(self.initial_centers[0,1])

    def command(self,data,t,forces,source_any):
        p=nodes(data);c=.5*(p[:-1]+p[1:]);rt=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1);rs=np.linalg.norm(c[:,[1,2]]-[0.,1.],axis=1)
        near=(rt>.065)&(rt<.125)&(c[:,1]>.15);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1])
        connected=np.zeros(21,bool);connected[:FRONT]=near[:FRONT]
        connected[1:]=near[1:]&near[:-1]&(da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)&(np.abs(dy+.180/(2*np.pi)*da)<.025)
        self.receive_age=np.where(connected,self.receive_age+DT,0.);self.received={j for j in range(21) if self.receive_age[j]>=1.};self.engaged.update(self.received)
        clear=(rs>.130)&(~source_any)&(forces[0]<.1)
        self.exit_age=np.where(clear,self.exit_age+DT,0.);self.exited={j for j in range(15,21) if self.exit_age[j]>=.5}
        count=len(self.received)+len(self.exited)
        if count>self.best_counts:self.best_counts=count;self.last_progress=t
        tracking=float(np.degrees(abs(self.last-data.qpos[self.qa])).max())
        scale=float(np.clip((25.-tracking)/20.,.2,1.));increment=self.speed*DT*scale;self.shift=min(TOTAL,self.shift+increment)
        # Match roll feeding to the helix axial fraction; J03's hanging supply
        # could drift along the pipe, whereas the L bridge needs both anchors.
        axial_error=ALPHA*self.shift-(c[0,1]-self.initial_head_axis)
        factor=float(np.clip(self.roll_factor+1.5*axial_error,.06,.65))
        self.phase+=increment*BETA/.029*factor
        guide,v=self.route.window(self.shift,self.coeff)
        middle=(v>CUT)&(v<0.)
        if np.count_nonzero(middle)>=2:
            w=np.clip((v[middle]-CUT)/(-CUT),0,1);h=16*w*w*(1-w)*(1-w);basis=np.column_stack([h,h*(2*w-1)])
            raw=self.route.sample(v[middle]);fit=np.linalg.lstsq(basis,p[middle]-raw,rcond=None)[0]
            desired=np.clip(fit,-.10,.10);self.coeff+=np.clip(.35*desired-self.coeff,-.002*DT,.002*DT)
            guide,v=self.route.window(self.shift,self.coeff)
        carried_source=(rs>.065)&(rs<.130)&(c[:,0]<.98)&(~clear)
        roles=np.ones(21,int);roles[(v[:-1]+v[1:])*.5<CUT]=0
        roles[carried_source]=0;roles[(v[:-1]+v[1:])*.5>=0.]=2
        for j in self.engaged:roles[j]=2
        desired_kp=np.full(20,127.*self.bridge_gain)
        weights=np.full(22,.35)
        for j in range(21):
            if roles[j]==0:weights[j:j+2]=np.maximum(weights[j:j+2],2.5)
            elif roles[j]==2:weights[j:j+2]=np.maximum(weights[j:j+2],3.)
            if j==0:continue
            if roles[j]==0 or j in self.engaged:desired_kp[j-1]=127.
            elif v[j]>=-2*LINK:desired_kp[j-1]=63.5
        self.kp+=np.clip(desired_kp-self.kp,-127*DT/1.5,127*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        tau=ALPHA*BETA/R
        orientation=self.route.frame(FRONT*LINK+self.shift)@Rotation.from_rotvec([0.,0.,tau*self.shift-self.phase]).as_matrix()@self.frame_offset
        q,projection=self.mapper.solve(guide,self.shift,self.phase,self.last,node_weights=weights,orientation_override=orientation,command_step=self.step)
        target=self.last+np.clip(q-self.last,-self.step,self.step);self.last=target.copy()
        if self.shift>=2*LINK and t-self.last_progress>45.:self.failure='no_actual_source_exit_or_target_intake_45s'
        if self.shift>=TOTAL and t-self.last_progress>25.:self.failure='whole_reference_complete_without_whole_actual_transfer'
        region_codes=np.where(self.kp>126,0,np.where(self.kp<.5,2,1))
        info=dict(stage='simultaneous_source_bridge_target',shape_parameter=self.phase,shift_mm=self.shift*1000,tracking_deg=tracking,
            curve_radius_delta_mm=0.,reference_points_mm=guide*1000,material_coordinate_mm=v*1000,body_route_regions=roles,
            kp_nm_per_rad=self.kp.copy(),active_joint_mask=self.kp>.5,region_codes=region_codes,
            actual_new_retained_modules=sorted(j for j in self.received if j>=FRONT),actual_received_modules=sorted(self.received),actual_source_exited_modules=sorted(self.exited),
            actual_new_winding_rad=np.zeros(21),current_spatial_connection_mask=connected,
            roll_phase_rad=self.phase,roll_factor=factor,actual_head_axis_advance_mm=(c[0,1]-self.initial_head_axis)*1000,
            head_axis_reference_error_mm=axial_error*1000,bridge_coefficients_mm=self.coeff*1000,projection=projection,
            material_frame_head=orientation,material_frame_method='fixed source normal parallel-transported through source/bridge/target, common roll gauge; exact physical hinge FK',
            control_layers=['common material shift','continuous transported roll frame','true20-hinge bounded numerical projection','state-based moving stiffness','smooth bridge-shape feedback'],
            yamada_compliance_enabled=False,maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),failure=self.failure)
        return target,info
