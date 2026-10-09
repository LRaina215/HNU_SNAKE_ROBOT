"""One continuous source-bridge-target curve with transported material frames."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
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
    def __init__(self,model,data,speed=.005,rate_deg_s=20.,roll_factor=None,bridge_gain=.25,adaptive_supply=False,route=None,source_start=15,intake_y=.15,preserve_preload=False,handoff_on_posterior_exit=False):
        self.m=model;self.mapper=HingePhaseMapper(model,data);self.qa=self.mapper.qa;self.va=self.mapper.va;self.route=ThreeSegmentRoute() if route is None else route
        self.custom_route=route is not None
        self.winding_sign=float(getattr(self.route,'winding_sign',1.))
        self.front=FRONT;self.source_start=source_start;self.source_count=21-source_start;self.intake_y=intake_y
        self.total=getattr(self.route,'total',TOTAL);self.cut=getattr(self.route,'source_cut',CUT);self.join=getattr(self.route,'target_join',0.)
        self.preload=data.ctrl-data.qpos[self.qa] if preserve_preload else np.zeros(20)
        self.handoff_on_posterior_exit=handoff_on_posterior_exit
        self.last=data.ctrl.copy();self.kp=np.full(20,127.);self.shift=0.;self.phase=0.;self.speed=speed;self.step=np.radians(rate_deg_s)*DT
        self.roll_factor=ALPHA/BETA**2 if roll_factor is None else roll_factor;self.bridge_gain=bridge_gain
        self.adaptive_supply=adaptive_supply;self.source_release_latch=set();self.free_mode_age=0.;self.free_supply_latch=False
        self.frame_offset=np.eye(3) if route is not None else self.route.frame(FRONT*LINK).T@self.mapper.rotation
        self.coeff=np.zeros((2,3));self.receive_age=np.zeros(21);self.received=set(range(FRONT));self.engaged=set(range(FRONT))
        self.exit_age=np.zeros(21);self.exited=set();self.best_counts=0;self.last_progress=0.;self.failure=None
        self.initial_centers=.5*(nodes(data)[:-1]+nodes(data)[1:]);self.initial_head_axis=float(self.initial_centers[0,1])

    def command(self,data,t,forces,source_any):
        p=nodes(data);c=.5*(p[:-1]+p[1:]);rt=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1);rs=np.linalg.norm(c[:,[1,2]]-[0.,1.],axis=1)
        near=(rt>.065)&(rt<.125)&(c[:,1]>self.intake_y);phi=self.winding_sign*np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1])
        connected=np.zeros(21,bool);connected[:FRONT]=near[:FRONT]
        connected[1:]=near[1:]&near[:-1]&(da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)&(np.abs(dy+.180/(2*np.pi)*da)<.025)
        self.receive_age=np.where(connected,self.receive_age+DT,0.);self.received={j for j in range(21) if self.receive_age[j]>=1.};self.engaged.update(self.received)
        local_source=np.array([source_any[max(0,j-1):min(21,j+2)].any() or forces[0,max(0,j-1):min(21,j+2)].sum()>.1 for j in range(21)])
        clear=(~local_source) if self.adaptive_supply else (rs>.130)&(~source_any)&(forces[0]<.1)
        self.exit_age=np.where(clear,self.exit_age+DT,0.);self.exited={j for j in range(self.source_start,21) if self.exit_age[j]>=.5}
        self.source_release_latch.update(self.exited)
        count=len(self.received)+len(self.exited)
        if count>self.best_counts:self.best_counts=count;self.last_progress=t
        controlled=self.kp>.5 if self.custom_route else np.ones(20,dtype=bool)
        tracking=float(np.degrees(abs(self.last-data.qpos[self.qa])[controlled]).max()) if controlled.any() else 0.
        scale=float(np.clip((25.-tracking)/20.,.2,1.));increment=self.speed*DT*scale;self.shift=min(self.total,self.shift+increment)
        # Match roll feeding to the helix axial fraction; J03's hanging supply
        # could drift along the pipe, whereas the L bridge needs both anchors.
        axial_error=ALPHA*self.shift-(c[0,1]-self.initial_head_axis)
        fully_free=self.adaptive_supply and len(self.exited)==self.source_count and not source_any.any() and forces[0].sum()<.1
        posterior_released=self.adaptive_supply and len(self.exited)==self.source_count
        self.free_supply_latch=self.free_supply_latch or fully_free or (self.handoff_on_posterior_exit and posterior_released)
        self.free_mode_age=self.free_mode_age+DT if self.free_supply_latch else 0.
        handoff=float(np.clip(self.free_mode_age/1.5,0.,1.))
        factor=float((1-handoff)*np.clip(self.roll_factor+1.5*axial_error,.06,.65)+handoff*1.)
        self.phase+=increment*BETA/.029*factor
        guide,v=self.route.window(self.shift,self.coeff)
        middle=(v>self.cut)&(v<self.join)
        if np.count_nonzero(middle)>=2:
            w=np.clip((v[middle]-self.cut)/(self.join-self.cut),0,1);h=16*w*w*(1-w)*(1-w);basis=np.column_stack([h,h*(2*w-1)])
            raw=self.route.sample(v[middle]);fit=np.linalg.lstsq(basis,p[middle]-raw,rcond=None)[0]
            desired=np.clip(fit,-.10,.10);self.coeff+=np.clip(.35*desired-self.coeff,-.002*DT,.002*DT)
            guide,v=self.route.window(self.shift,self.coeff)
        carried_source=(rs>.065)&(rs<.130)&(~clear)
        if not self.custom_route:carried_source &= c[:,0]<.98
        roles=np.ones(21,int);roles[(v[:-1]+v[1:])*.5<self.cut]=0
        roles[carried_source]=0;roles[(v[:-1]+v[1:])*.5>=0.]=2
        for j in self.engaged:roles[j]=2
        if self.adaptive_supply:
            for j in self.source_release_latch:
                if j not in self.engaged and not carried_source[j]:roles[j]=1
        desired_kp=np.full(20,127.*self.bridge_gain)
        weights=np.full(22,.35)
        for j in range(21):
            if roles[j]==0:weights[j:j+2]=np.maximum(weights[j:j+2],2.5)
            elif roles[j]==2:weights[j:j+2]=np.maximum(weights[j:j+2],3.)
            if j==0:continue
            if roles[j]==0 or j in self.engaged:desired_kp[j-1]=127.
            elif v[j]>=-2*LINK:desired_kp[j-1]=63.5
        prefix=FRONT
        while prefix in self.engaged:prefix+=1
        if self.adaptive_supply:
            active_end=min(20,prefix+3)
            # A measured free supply curve replaces the exited source helix.
            # Register only after both pipe supports are no longer required.
            supply=p.copy()
            if self.free_supply_latch:
                nominal,_=self.route.window(self.shift)
                n=min(prefix+1,22);x=p[:n]-p[:n].mean(0);y=nominal[:n]-nominal[:n].mean(0)
                u,_,vh=np.linalg.svd(x.T@y);rotation=u@vh
                if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@vh
                supply=(p-p[:n].mean(0))@rotation+nominal[:n].mean(0)
                guide=self.route.supply_target(v) if hasattr(self.route,'supply_target') else receiving(v)
                weights[:]=0.;weights[:prefix+1]=3.;weights[prefix+1:min(22,prefix+5)]=.5
                desired_kp[:]=0.;desired_kp[:min(20,prefix-1)]=127.;desired_kp[max(0,prefix-1):active_end]=63.5
                guide[min(22,prefix+5):]=supply[min(22,prefix+5):]
                if self.handoff_on_posterior_exit:
                    for j in np.flatnonzero(carried_source):
                        if j>0:desired_kp[j-1]=127.
                        if j>=prefix:
                            guide[j:j+2]=supply[j:j+2];weights[j:j+2]=np.maximum(weights[j:j+2],.75)
            else:
                for j in self.source_release_latch:
                    if j>=active_end+1 and not carried_source[j] and j not in self.engaged:
                        desired_kp[j-1]=0.;weights[j+1]=0.;guide[j+1]=supply[j+1]
            # This C2 spatial curve is sampled at the same material window.
            # Released posterior positions have zero fit weight and zero Kp;
            # they are never pulled toward a prescribed hanging straight line.
            continuous=CubicSpline(v[::-1],guide[::-1],axis=0)
            guide=continuous(v)
        self.kp+=np.clip(desired_kp-self.kp,-127*DT/1.5,127*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        # Head remains on the target Y-pipe. Use the proven J03 head formula:
        # the roll is about the actual head's local Z, after its initial tilt.
        # Source/bridge/target guide points are not rotated about world Y.
        orientation=Rotation.from_rotvec([0.,BETA*self.shift/R,0.]).as_matrix()@self.mapper.rotation@Rotation.from_rotvec([0.,0.,-self.phase]).as_matrix()
        if hasattr(self.route,'head_orientation'):orientation=self.route.head_orientation(self.mapper.rotation,self.shift,self.phase)
        active_mask=desired_kp>.5 if self.adaptive_supply else None
        fade=float(np.clip(1-self.shift/.20,0,1));preload=self.preload*(fade*fade*(3-2*fade))
        q,projection=self.mapper.solve(guide,self.shift,self.phase,self.last-preload,node_weights=weights,orientation_override=orientation,command_step=self.step,active_mask=active_mask,observed_q=data.qpos[self.qa])
        q=np.clip(q+preload,-LIMIT,LIMIT)
        if self.adaptive_supply:q[~active_mask]=np.clip(data.qpos[self.qa][~active_mask],-LIMIT,LIMIT)
        target=self.last+np.clip(q-self.last,-self.step,self.step);self.last=target.copy()
        if self.shift>=2*LINK and t-self.last_progress>45.:self.failure='no_actual_source_exit_or_target_intake_45s'
        # Once all registered modules have been connected, the runner still
        # verifies the whole reference window and actual tail geometry. A wide
        # connection count must not stop that final physical winding action.
        if self.shift>=self.total and len(self.received)<21 and t-self.last_progress>25.:self.failure='whole_reference_complete_without_whole_actual_transfer'
        region_codes=np.where(self.kp>126,0,np.where(self.kp<.5,2,1))
        info=dict(stage='simultaneous_source_bridge_target',shape_parameter=self.phase,shift_mm=self.shift*1000,tracking_deg=tracking,
            curve_radius_delta_mm=0.,reference_points_mm=guide*1000,material_coordinate_mm=v*1000,body_route_regions=roles,
            kp_nm_per_rad=self.kp.copy(),active_joint_mask=self.kp>.5,region_codes=region_codes,
            actual_new_retained_modules=sorted(j for j in self.received if j>=FRONT),actual_received_modules=sorted(self.received),actual_source_exited_modules=sorted(self.exited),
            actual_new_winding_rad=np.zeros(21),current_spatial_connection_mask=connected,
            roll_phase_rad=self.phase,roll_factor=factor,actual_head_axis_advance_mm=(c[0,1]-self.initial_head_axis)*1000,
            head_axis_reference_error_mm=axial_error*1000,bridge_coefficients_mm=self.coeff*1000,projection=projection,
            adaptive_released_supply=self.adaptive_supply,fully_released_supply=fully_free,free_handoff_fraction=handoff,
            posterior_original_source_released=posterior_released,handoff_on_posterior_exit=self.handoff_on_posterior_exit,
            source_release_latch=sorted(self.source_release_latch),free_supply_mode_latched=self.free_supply_latch,
            saved_clamp_preload_remaining_rad=preload,
            material_frame_head=orientation,material_frame_method='J03 target-pipe head screw followed by head-local roll; full unrotated source/bridge/target guide; exact physical hinge FK',
            control_layers=['common material shift','continuous transported roll frame','true20-hinge bounded numerical projection','state-based moving stiffness','smooth bridge-shape feedback'],
            yamada_compliance_enabled=False,maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),failure=self.failure)
        info['actual_winding_sign']=self.winding_sign
        return target,info
