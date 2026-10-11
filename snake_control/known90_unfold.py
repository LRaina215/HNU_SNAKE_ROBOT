"""Third candidate: unwind a specified front length, open approach, then intake.

The source-side cut is a calibrated material location, NOT an outlet rotated
by a sliding source-window coordinate. A polar homotopy removes the front's
original winding counterclockwise before an open under-entry bow goes upward.
The same shared coordinate subsequently grows a single receiving helix.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from snake_control.known90_global_local import ActualBearingMapper
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,nodes


def hermite(start,end,ts,te,length):
    t=np.linspace(0,1,1501)[:,None]
    def raw(h):return (2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*h*ts+(-2*t**3+3*t**2)*end+(t**3-t**2)*h*te
    chord=np.linalg.norm(end-start)
    if chord>=length:raise ValueError(f'Open bow chord {chord:.4f} exceeds available front length {length:.4f}')
    h=brentq(lambda v:np.linalg.norm(np.diff(raw(v),axis=0),axis=1).sum()-length,1e-8,4.)
    p=raw(h);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    return CubicSpline(arc,p,axis=0),h


class FixedDepartureRoute:
    def __init__(self,seed,cut=10,receiving=5):
        self.seed=np.asarray(seed).copy();self.cut=cut;self.front=receiving;self.radius=.0915;self.pitch=.180
        self.alpha=self.pitch/np.hypot(self.pitch,2*np.pi*self.radius);self.beta=np.sqrt(1-self.alpha**2)
        self.target_phi=-5*np.pi/6;self.target_y=.18;self.q=seed[cut].copy()
        self.e=np.array([1.1+self.radius*np.cos(self.target_phi),self.target_y,1+self.radius*np.sin(self.target_phi)])
        self.ts=seed[cut]-seed[cut+1];self.ts/=np.linalg.norm(self.ts)
        self.te=np.array([self.beta*np.sin(self.target_phi),self.alpha,-self.beta*np.cos(self.target_phi)])
        self.bridge_length=(cut-receiving)*LINK;self.origin=receiving*LINK
        self.field,self.handle=hermite(self.q,self.e,self.ts,self.te,self.bridge_length)
        self.source=CubicSpline((-np.arange(22-cut)*LINK)[::-1],seed[cut:][::-1],axis=0)
        self.total=1.15

    def target(self,v):
        v=np.atleast_1d(v);phi=self.target_phi-self.beta*v/self.radius
        return np.column_stack([1.1+self.radius*np.cos(phi),self.target_y+self.alpha*v,1+self.radius*np.sin(phi)])

    def sample(self,v):
        v=np.atleast_1d(v);p=self.target(v);src=v<-self.bridge_length;mid=(v>=-self.bridge_length)&(v<0)
        p[src]=self.source(v[src]+self.bridge_length);p[mid]=self.field(v[mid]+self.bridge_length);return p

    def window(self,extra):
        v=self.origin+extra-np.arange(22)*LINK;return self.sample(v),v

    def description(self):
        return dict(source_exit_m=self.q,target_entry_m=self.e,source_spatial_hand=1,source_phi_rad=float(np.arctan2(self.q[2]-1,self.q[1])),
                    target_phi_rad=self.target_phi,bridge_length_mm=self.bridge_length*1000,node_radius_mm=self.radius*1000,pitch_mm=180,
                    initial_cut_module=self.cut,initial_front_release_mm=self.cut*LINK*1000,first_target_modules=self.front,remaining_source_modules=21-self.cut)


class Known90Unfold:
    def __init__(self,model,data,feedback=True,speed=.005):
        self.m=model;self.mapper=ActualBearingMapper(model);self.qa=self.mapper.qa;self.va=self.mapper.va
        self.seed=nodes(data).copy();self.route=FixedDepartureRoute(self.seed);self.cut=self.route.cut
        self.last=data.ctrl.copy();self.kp=np.full(20,127.);self.feedback=feedback;self.speed=speed
        self.shift=0.;self.phase=0.;self.coeff=np.zeros(2);self.target_delta=0.;self.angle=0.;self.axial=0.;self.target_angle=0.
        self.unfold_end=.50;self.approach_end=.60;self.hook_end=.60+self.route.origin;self.total=self.hook_end+self.route.total
        self.open_head=np.array([1.015,.17,.88]);self.open_field,self.open_handle=hermite(self.route.q,self.open_head,self.route.ts,self.route.te,self.cut*LINK)
        self.open_prefix=self.open_field(np.linspace(self.cut*LINK,0,self.cut+1))
        self.seed_phi=np.unwrap(np.arctan2(self.seed[:self.cut+1,2]-1,self.seed[:self.cut+1,1])[::-1])[::-1]
        self.open_phi=np.unwrap(np.arctan2(self.open_prefix[:,2]-1,self.open_prefix[:,1])[::-1])[::-1]
        self.open_phi+=2*np.pi*round((self.seed_phi[-1]-self.open_phi[-1])/(2*np.pi))
        self.capture_age=0.;self.acquired=False;self.events=[];self.gated=False;self.last_execution_stations=np.zeros(22)

    def nominal(self,shared):
        r=self.route;p=self.seed.copy();n=self.cut
        if shared<self.unfold_end:
            a=np.clip(shared/self.unfold_end,0,1);a=a*a*(3-2*a)
            rad0=np.linalg.norm(self.seed[:n+1,[1,2]]-[0,1],axis=1);rad1=np.linalg.norm(self.open_prefix[:,[1,2]]-[0,1],axis=1)
            radius=(1-a)*rad0+a*rad1;phi=(1-a)*self.seed_phi+a*self.open_phi
            p[:n+1,0]=(1-a)*self.seed[:n+1,0]+a*self.open_prefix[:,0]
            p[:n+1,1]=radius*np.cos(phi);p[:n+1,2]=1+radius*np.sin(phi)
            return p,np.r_[np.linspace(n*LINK,0,n+1),-np.arange(1,22-n)*LINK],'front_polar_unwinding'
        if shared<self.approach_end:
            a=(shared-self.unfold_end)/(self.approach_end-self.unfold_end);a=a*a*(3-2*a)
            head=(1-a)*self.open_head+a*r.e
            field,_=hermite(r.q,head,r.ts,r.te,n*LINK)
            p[:n+1]=field(np.linspace(n*LINK,0,n+1))
            return p,np.r_[np.linspace(n*LINK,0,n+1),-np.arange(1,22-n)*LINK],'open_lower_upward_approach'
        if shared<self.hook_end:
            v=shared-self.approach_end;length=n*LINK-v;field,_=hermite(r.q,r.e,r.ts,r.te,length)
            u=n*LINK-np.arange(n+1)*LINK;target=u>=length
            p[:n+1][target]=r.target(u[target]-length)
            p[:n+1][~target]=field(u[~target])
            return p,np.r_[u,-np.arange(1,22-n)*LINK],'growing_first_target_wrap'
        p,s=r.window(shared-self.hook_end)
        return p,s,'continuous_single_helix_intake'

    def progress_path(self):
        s=np.linspace(0,self.total,4001);p=np.array([self.nominal(v)[0][0] for v in s]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        return s,p,arc

    def locate(self,p,nominal,stage):
        c=(p[:-1]+p[1:])/2;nc=(nominal[:-1]+nominal[1:])/2
        sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);rt=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1)
        source=(sr>.074)&(sr<.120)&(c[:,0]<1.035);target=(rt>.069)&(rt<.120)&(c[:,1]>.15)
        carrying=source.copy()
        if stage!='continuous_single_helix_intake':carrying[:self.cut]=False
        if carrying.sum()>=4:
            a=np.arctan2(c[carrying,2]-1,c[carrying,1])-np.arctan2(nc[carrying,2]-1,nc[carrying,1])
            wanted=np.arctan2(np.sin(a).mean(),np.cos(a).mean());self.angle+=np.clip(np.angle(np.exp(1j*(wanted-self.angle))),-.025,.025)
            self.axial+=np.clip(np.median(c[carrying,0]-nc[carrying,0])-self.axial,-.001,.001)
        if self.acquired and target.sum()>=5:
            a=np.arctan2(c[target,2]-1,c[target,0]-1.1)-np.arctan2(nc[target,2]-1,nc[target,0]-1.1)
            wanted=np.arctan2(np.sin(a).mean(),np.cos(a).mean());self.target_angle+=np.clip(np.angle(np.exp(1j*(wanted-self.target_angle))),-.025,.025)
        return c,sr,rt,source,target

    def execution(self,nominal,s,stage):
        p=nominal.copy();r=self.route
        sx=Rotation.from_rotvec([self.angle,0,0]).as_matrix()
        transformed=(p-[0,0,1])@sx.T+[self.axial,0,1]
        if stage!='continuous_single_helix_intake':
            # The observed source departure changes the cut endpoint only;
            # the prescribed global head trajectory remains fixed.
            delta=transformed[self.cut]-p[self.cut];w=np.linspace(0,1,self.cut+1);smooth=w*w*(3-2*w)
            p[:self.cut+1]+=smooth[:,None]*delta;p[self.cut:]=transformed[self.cut:]
            h=np.sin(np.pi*w)**2;p[:self.cut+1]+=h[:,None]*[0,self.coeff[0],self.coeff[1]]
        else:
            src=s<-r.bridge_length;mid=(s>=-r.bridge_length)&(s<0);tgt=s>=0
            p[src]=transformed[src]
            ty=Rotation.from_rotvec([0,-self.target_angle,0]).as_matrix();target=(p-[1.1,0,1])@ty.T+[1.1,0,1]
            p[tgt]=target[tgt]
            q=(r.q-[0,0,1])@sx.T+[self.axial,0,1];e=(r.e-[1.1,0,1])@ty.T+[1.1,0,1]
            ts=sx@r.ts;te=ty@r.te
            t=np.clip((s+r.bridge_length)/r.bridge_length,0,1)[:,None]
            # Endpoint relocation only; use the registered source/receiving poses
            # and retain this bridge's shape rather than fitting measured sag.
            p[mid]=(2*t[mid]**3-3*t[mid]**2+1)*q+(t[mid]**3-2*t[mid]**2+t[mid])*r.handle*ts+(-2*t[mid]**3+3*t[mid]**2)*e+(t[mid]**3-t[mid]**2)*r.handle*te
            h=(16*t*t*(1-t)*(1-t)).ravel();p+=h[:,None]*[0,self.coeff[0],self.coeff[1]]
        return p

    def command(self,data,t):
        p=nodes(data);q=data.qpos[self.qa];load=data.qfrc_actuator[self.va];nominal,s,stage=self.nominal(self.shift)
        c,sr,rt,source,target=self.locate(p,nominal,stage);execution=self.execution(nominal,s,stage)
        prefix=0
        while prefix<21 and target[prefix]:prefix+=1
        wrap=np.ptp(np.unwrap(np.arctan2(c[:prefix,2]-1,c[:prefix,0]-1.1))) if prefix>=2 else 0.
        track=float(np.degrees(abs(self.last-q)).max());head_error=float(np.linalg.norm(p[0]-execution[0]))
        condition=prefix>=5 and wrap>2.3 and track<25
        self.capture_age=self.capture_age+DT if condition else 0
        if not self.acquired and self.capture_age>=1:
            self.acquired=True;self.events.append(dict(t_s=t,event='active_geometric_wrap',prefix=prefix,wrap_rad=wrap))
        self.gated=head_error>.07 or track>32 or (self.shift>=self.hook_end and not self.acquired)
        scale=0. if self.gated else float(np.clip((30-track)/22,.15,1))
        if head_error>.035:scale*=.4
        inc=self.speed*DT*scale;self.shift=min(self.total,self.shift+inc)
        phase_inc=(inc if inc else self.speed*DT*.08)*r_beta(self.route)/.029*(self.route.alpha/self.route.beta**2);self.phase+=phase_inc
        nominal,s,stage=self.nominal(self.shift);execution=self.execution(nominal,s,stage)
        bridge=np.zeros(21,bool)
        if stage=='continuous_single_helix_intake':bridge=((s[:-1]+s[1:])/2<0)&((s[:-1]+s[1:])/2>=-self.route.bridge_length)&(~source)&(~target)
        else:bridge[:self.cut]=~target[:self.cut]
        if self.feedback and bridge.any():
            cc=(execution[:-1]+execution[1:])/2;err=cc-c;ids=np.flatnonzero(bridge);resistance=abs(load[np.minimum(ids,19)]).max()
            desired=np.array([.25*np.mean(err[ids,1]),.25*np.mean(err[ids,2])])
            if resistance>1.8:desired[1]=min(desired[1],-.005)
            self.coeff+=np.clip(np.clip(desired,-.020,.020)-self.coeff,-.00025,.00025)
            execution=self.execution(nominal,s,stage)
        if self.feedback and self.acquired and prefix>=5:
            self.target_delta+=np.clip(np.clip(.25*(.087-np.median(rt[:prefix])),-.003,.003)-self.target_delta,-.0001,.0001)
            radial=execution[:,[0,2]]-[1.1,1];radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
            node_target=np.zeros(22);node_target[:-1]=np.maximum(node_target[:-1],target);node_target[1:]=np.maximum(node_target[1:],target)
            # Blend across the last receiving rod rather than mixing 21 rod
            # masks with 22 backbone nodes.
            if prefix<21:node_target[prefix]*=.5
            execution[:,[0,2]]+=node_target[:,None]*self.target_delta*radial
        carrying=np.flatnonzero(source)
        anchor=3 if self.acquired and (len(carrying)<4 or np.count_nonzero(source[14:])<3) else int(carrying[-3]) if len(carrying)>=3 else 18
        weights=np.ones(22);weights[:self.cut+1]=2.
        if self.acquired:weights[:min(22,prefix+2)]=3.
        desired,report=self.mapper.solve(execution,data,self.last,anchor,phase_inc,weights)
        kp=np.full(20,127.)
        if track>18 and abs(load).max()>1.8:kp[bridge[:20]]=95.25
        self.kp+=np.clip(kp-self.kp,-127*DT/1.5,127*DT/1.5);self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        step=float(np.degrees(abs(desired-self.last)).max());self.last=desired.copy()
        return desired,dict(stage=stage,shared_shift_mm=self.shift*1000,roll_phase_rad=self.phase,tracking_deg=track,
            head_global_error_mm=float(np.linalg.norm(p[0]-nominal[0]))*1000,head_execution_error_mm=head_error*1000,
            estimated_prefix=prefix,estimated_wrap_rad=float(wrap),estimated_capture=self.acquired,advancement_gate=self.gated,speed_scale=scale,
            global_points_mm=nominal*1000,execution_points_mm=execution*1000,estimated_points_mm=p*1000,material_stations_mm=s*1000,
            bridge_coeff_mm=np.array([[0,self.coeff[0],self.coeff[1]],[0,0,0]])*1000,radius_delta_mm=np.array([0,self.target_delta])*1000,
            source_pose_azimuth_rad=self.angle,source_pose_axial_mm=self.axial*1000,receiving_pose_phase_rad=self.target_angle,kp_nm_per_rad=self.kp.copy(),
            maximum_dispatch_step_deg=step,controller_source_geometry_mask=source,controller_target_geometry_mask=target,
            physical_root_assigned=False,exact_contact_used_for_local_feedback=False,**report)


def r_beta(route):return route.beta
