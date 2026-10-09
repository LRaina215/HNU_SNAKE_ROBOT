"""Actual source approach, a pitch-change repositioning loop, head-first feed."""
import numpy as np,mujoco
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from snake_control.braced_inner_probe import BracedInnerRoute
from snake_control.head_first_inner_route import HeadFirstInnerRoute
from snake_control.horizontal_pipe_roll import HingePhaseMapper
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,nodes,WholeCurveMapper
from snake_control.suspended_transfer_route import R,BETA


class HeadFirstInnerControl:
    def __init__(self,model,data,actual_exit_gate=False,anchored_feed=False):
        self.m=model;self.mapper=HingePhaseMapper(model,data);self.qa=self.mapper.qa
        self.actual_exit_gate=actual_exit_gate;self.anchored_feed=anchored_feed
        self.last=data.ctrl.copy();self.kp=np.full(20,127.);self.shift=0.;self.phase=0.;self.failure=None;self.received=set();self.exited=set();self.receive_age=np.zeros(21)
        self.stage='source_approach';self.stage_start=0.;self.pose0=nodes(data);self.approach=BracedInnerRoute();self.residual=self.pose0-self.approach.initial_window();self.route=None;self.events={};self.source_s=0.;self.source_phase=0.;self.pitch=.18;self.support_age=0.;self.material_shift=0.;self.material_offset=0.;self.observed_received=set()

    def enter(self,stage,data,t):
        self.stage=stage;self.stage_start=t;self.mapper=HingePhaseMapper(self.m,data)
        p=nodes(data);self.source_phi=float(np.arctan2(p[0,2]-1,p[0,1]));self.source_s=0.;self.source_phase=0.
        self.source_frame_offset=None;self.phase_base=self.phase
        self.events[stage]=dict(t_s=t,head_mm=p[0]*1000,shared_material_shift_mm=self.material_shift*1000,physical_reset=False)

    def begin_feed(self,p,data,t,smooth_source):
        route=HeadFirstInnerRoute(p,smooth_source=smooth_source);checks=[]
        for s in [0,.025,.07,.15,.25,.45,.70,route.total]:
            g,_=route.window(s);private=WholeCurveMapper(self.m);_,fit=private.initialize(g)
            checks.append(dict(shift_mm=s*1000,rod_error_mm=float(abs(np.linalg.norm(np.diff(g,axis=0),axis=1)-LINK).max()*1000),fit_rms_mm=fit['guide_rms_mm'],pipe_depth_mm=fit['pipe_depth_mm'],self_depth_mm=fit['self_depth_mm'],maximum_joint_deg=fit['maximum_joint_deg']))
        ready=all(x['rod_error_mm']<.001 and x['fit_rms_mm']<35 and x['pipe_depth_mm']<2 and x['self_depth_mm']<.5 and x['maximum_joint_deg']<=88.00001 for x in checks)
        self.events['actual_exit_route_preflight']=dict(t_s=t,physical_steps=0,seed='current actual state, private fits only; physical qpos/qvel untouched',checks=checks,ready=ready)
        if not ready:self.failure='actual_exit_curve_not_executable';return
        self.route=route;self.enter('head_first_inner_feed',data,t);self.shift=0.;self.material_offset=self.material_shift;self.feed_phase=0.;self.feed_phase_base=self.phase;self.progress_head=p[0].copy();self.last_progress=t;self.before_feed_guide=route.window(0)[0];self.feed_mapper=WholeCurveMapper(self.m)
        self.events['actual_inner_exit']=dict(t_s=t,head_mm=p[0]*1000,entry_vector_mm=(p[0]-p[1])*1000,bridge_length_mm=route.bridge_length*1000,physical_reset=False)

    def source_command(self,data,t):
        age=t-self.stage_start;direction=0.
        if self.stage=='source_pitch_low':self.pitch=.18+(.08-.18)*min(age/10,1)
        elif self.stage=='source_retreat':self.pitch=.08;direction=-1.
        elif self.stage=='source_pitch_high':self.pitch=.08+(.30-.08)*min(age/10,1)
        elif self.stage=='source_reapproach':self.pitch=.30;direction=1.
        alpha=self.pitch/np.sqrt((2*np.pi*R)**2+self.pitch**2);beta=np.sqrt(1-alpha*alpha)
        self.source_s+=direction*.005*DT;self.material_shift+=direction*.005*DT;self.source_phase+=direction*.005*DT*beta/.029;self.phase=self.phase_base+self.source_phase
        delta=brentq(lambda h:(alpha*h)**2+(2*R*np.sin(beta*h/(2*R)))**2-LINK**2,LINK,LINK*1.2)
        p=nodes(data);phi=self.source_phi+beta*self.source_s/R
        u=-np.arange(22)*delta;guide=np.column_stack([p[0,0]+alpha*u,R*np.cos(phi+beta*u/R),1+R*np.sin(phi+beta*u/R)])
        z=guide[0]-guide[1];z/=np.linalg.norm(z);n=np.array([0,np.cos(phi),np.sin(phi)]);n-=n@z*z;n/=np.linalg.norm(n);frame=np.column_stack([n,np.cross(z,n),z])
        if self.source_frame_offset is None:self.source_frame_offset=frame.T@self.mapper.rotation
        orientation=frame@self.source_frame_offset@Rotation.from_rotvec([0,0,-self.source_phase]).as_matrix()
        q,projection=self.mapper.solve(guide,0,self.source_phase,self.last,node_weights=np.full(22,3.),orientation_override=orientation,command_step=np.radians(20)*DT)
        if self.actual_exit_gate and self.stage=='source_retreat':
            c=(p[:-1]+p[1:])/2;rr=np.linalg.norm(c[3:,[1,2]]-[0,1],axis=1);head_r=np.linalg.norm(p[0,[1,2]]-[0,1]);entry=p[0]-p[1]
            if .85<p[0,0]<1.025 and p[0,1]>.04 and p[0,2]<.97 and entry[1]>.007 and entry[2]>.01 and .065<head_r<.125 and np.count_nonzero((rr>.075)&(rr<.115))>=16:
                self.begin_feed(p,data,t,smooth_source=False)
                return q,guide,projection
        if self.stage=='source_pitch_low' and age>=10:self.enter('source_retreat',data,t)
        elif self.stage=='source_retreat' and p[0,0]<=.75:self.enter('source_pitch_high',data,t)
        elif self.stage=='source_pitch_high' and age>=10:self.enter('source_reapproach',data,t)
        elif self.stage=='source_reapproach':
            entry=p[0]-p[1]
            if .94<=p[0,0]<=1.025 and p[0,1]>.035 and p[0,2]<.97 and entry[1]>0 and entry[2]>0:
                self.begin_feed(p,data,t,smooth_source=True)
            elif p[0,0]>1.025:self.failure='source_reposition_returned_without_inner_lower_exit'
        if t>220 and self.route is None:self.failure='source_reposition_not_completed_220s'
        return q,guide,projection

    def command(self,data,t,forces,source_any):
        p=nodes(data);c=(p[:-1]+p[1:])/2;v=None;projection=None
        stage_before=self.stage
        if self.stage=='source_approach':
            self.shift+=.005*DT;self.material_shift+=.005*DT;self.phase+=.005*DT*BETA/.029
            guide=self.approach.source((13-np.arange(22))*LINK+self.shift);fade=np.clip(1-self.shift/.20,0,1);guide+=fade*fade*(3-2*fade)*self.residual
            orientation=Rotation.from_rotvec([BETA*self.shift/R,0,0]).as_matrix()@self.mapper.rotation@Rotation.from_rotvec([0,0,-self.phase]).as_matrix()
            q,projection=self.mapper.solve(guide,self.shift,self.phase,self.last,node_weights=np.full(22,3.),orientation_override=orientation,command_step=np.radians(20)*DT)
            if p[0,0]>=1.025:self.events['actual_junction_arrival']=dict(t_s=t,head_mm=p[0]*1000,blocked=False);self.enter('source_pitch_low',data,t)
            if t>150:self.failure='source_approach_without_junction_arrival_150s'
        elif self.stage!='head_first_inner_feed':q,guide,projection=self.source_command(data,t)
        else:
            tracking=float(np.degrees(abs(self.last-data.qpos[self.qa])).max());inc=min(self.route.total-self.shift,.005*DT*np.clip((25-tracking)/20,.2,1));self.shift+=inc;self.material_shift+=inc
            rr=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1]);near=(rr>.065)&(rr<.125)&(c[:,1]>.15)
            connected=near.copy();connected[1:]=near[1:]&near[:-1]&(da>.3)&(da<1.1)&(dy<-.002)&(dy>-.045)&(abs(dy+.180/(2*np.pi)*da)<.025)
            local=np.array([forces[1,max(0,j-1):min(21,j+2)].sum()>.2 for j in range(21)]);connected&=local;self.receive_age=np.where(connected,self.receive_age+DT,0);self.received=set(np.flatnonzero(self.receive_age>=1))
            sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);self.exited=set(np.flatnonzero((sr>.130)&(~source_any)&(forces[0]<.1)))
            prefix=0
            while prefix in self.received:prefix+=1
            free=prefix>=8 and not source_any.any() and forces[0].sum()<.1
            factor=1. if free else .328;self.feed_phase+=inc*BETA/.029*factor;self.phase=self.feed_phase_base+self.feed_phase
            guide,v=self.route.window(self.shift);roles=np.where((v[:-1]+v[1:])/2<0,0,np.where((v[:-1]+v[1:])/2>=self.route.target_join,2,1))
            desired=np.full(20,63.5);weights=np.full(22,1.)
            for j in range(21):
                if roles[j]==0:weights[j:j+2]=3.
                if roles[j]==2 or j in self.received:weights[j:j+2]=3.
                if j and (roles[j]==0 or j in self.received):desired[j-1]=127.
            if free:
                weights[:]=0.;weights[:prefix+1]=3.;weights[prefix+1:min(22,prefix+5)]=.5;desired[:]=0.;desired[:max(0,prefix-1)]=127.;desired[max(0,prefix-1):min(20,prefix+3)]=63.5
                nominal=guide.copy();n=min(prefix+1,22);x=p[:n]-p[:n].mean(0);y=nominal[:n]-nominal[:n].mean(0);u,_,vh=np.linalg.svd(x.T@y);rot=u@vh
                if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vh
                supply=(p-p[:n].mean(0))@rot+nominal[:n].mean(0);guide[min(22,prefix+5):]=supply[min(22,prefix+5):]
            self.kp+=np.clip(desired-self.kp,-127*DT/1.5,127*DT/1.5)
            if self.anchored_feed and not free:
                _,q,projection=self.feed_mapper.solve(guide,self.before_feed_guide,data,forces,self.last,active_mask=desired>.5,node_weights=weights)
                self.before_feed_guide=guide.copy()
            else:
                q,projection=self.mapper.solve(guide,self.shift,self.feed_phase,self.last,node_weights=weights,orientation_override=self.route.head_orientation(self.mapper.rotation,self.shift,self.feed_phase),command_step=np.radians(20)*DT,active_mask=desired>.5,observed_q=data.qpos[self.qa])
            if np.linalg.norm(p[0]-self.progress_head)>.01 or self.received-self.observed_received:self.last_progress=t;self.progress_head=p[0].copy()
            self.observed_received|=self.received
            if t-self.last_progress>45:self.failure='no_actual_head_or_intake_progress_45s'
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        target=self.last+np.clip(q-self.last,-np.radians(20)*DT,np.radians(20)*DT);self.last=target.copy()
        if len(self.received)<8 and forces[0].sum()<5:self.support_age+=DT
        else:self.support_age=0.
        if self.support_age>1:self.failure='source_support_lost_before_target_acquisition'
        info=dict(stage=stage_before,shape_parameter=self.pitch,shift_mm=self.material_shift*1000,local_feed_shift_mm=self.shift*1000,roll_phase_rad=self.phase,roll_factor=1.,curve_radius_delta_mm=0.,tracking_deg=float(np.degrees(abs(target-data.qpos[self.qa])).max()),reference_points_mm=guide*1000,material_coordinate_mm=((v+self.material_offset)*1000 if v is not None else self.material_shift*1000-np.arange(22)*LINK*1000),kp_nm_per_rad=self.kp.copy(),actual_received_modules=sorted(self.received),actual_source_exited_modules=sorted(self.exited),actual_new_retained_modules=sorted(self.received),projection=projection,maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),yamada_compliance_enabled=False,physical_root_assigned=False)
        return target,info
