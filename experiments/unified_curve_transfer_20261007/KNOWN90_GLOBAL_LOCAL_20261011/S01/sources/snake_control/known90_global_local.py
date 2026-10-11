"""Known 90-degree pipe: read-only actual pose, one local curve, one hinge fit.

Source support keeps its calibrated cylinder. Its observed axial/azimuth pose
locates the departure, rather than treating the body as frozen in world space.
Contacts are NOT inputs to this controller. Actuation is joint targets only.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq,least_squares
from snake_control.inner_ccw_compliance import InnerCCWRoute,initial_pose
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,nodes,skew


class LocatedCurve:
    def __init__(self,global_route,angle,axial,coeff,target_delta=0.):
        self.global_route=global_route;self.angle=angle;self.axial=axial;self.coeff=np.asarray(coeff)
        self.target_delta=target_delta;self.join=global_route.bridge_length
        self.rot=Rotation.from_rotvec([angle,0,0]).as_matrix()
        self.q=self.source([0.])[0];self.e=global_route.e.copy()
        tangent=global_route.initial_field(0,1) if global_route.initial_field is not None else global_route.frame(0)[:,2]
        self.ts=self.rot@tangent;self.ts/=np.linalg.norm(self.ts)
        phi=global_route.target_phi
        self.te=np.array([global_route.beta*np.sin(phi),global_route.alpha,-global_route.beta*np.cos(phi)])

    def source(self,s):
        p=self.global_route.sample(np.minimum(np.atleast_1d(s),0))
        p=(p-[0,0,1])@self.rot.T+[self.axial,0,1]
        return p

    def sample(self,s):
        s=np.atleast_1d(s);r=self.global_route;p=r.sample(s);src=s<0;mid=(s>=0)&(s<self.join);tgt=s>=self.join
        p[src]=self.source(s[src]);w=(s[mid]/self.join)[:,None]
        # Only the source-side departure position/orientation is located anew.
        # The target end is fixed; the feedback field has zero value and slope
        # at both endpoints. This does not copy a measured hanging bridge.
        p[mid]=(2*w**3-3*w**2+1)*self.q+(w**3-2*w**2+w)*r.handles[0]*self.ts+(-2*w**3+3*w**2)*self.e+(w**3-w**2)*r.handles[1]*self.te
        h=16*w*w*(1-w)*(1-w);p[mid]+=h*np.array([0,self.coeff[0],self.coeff[1]])
        if self.target_delta:
            radial=p[:,[0,2]]-[1.1,1];radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
            blend=np.clip((s-self.join)/(2*LINK),0,1);p[:,[0,2]]+=blend[:,None]*self.target_delta*radial
        return p

    def predecessor(self,s,p):
        grid=s-np.linspace(.001,.30,50);g=np.linalg.norm(self.sample(grid)-p,axis=1)-LINK
        for i in range(1,len(grid)):
            if g[i]>=0 and g[i-1]<0:return brentq(lambda v:np.linalg.norm(self.sample([v])[0]-p)-LINK,grid[i],grid[i-1],xtol=1e-10)
        raise ValueError('Execution curve has no continuous previous rod')

    def window(self,head_station):
        s=[head_station];p=[self.sample(s)[0]]
        for _ in range(21):s.append(self.predecessor(s[-1],p[-1]));p.append(self.sample([s[-1]])[0])
        return np.asarray(p),np.asarray(s)


class ActualBearingMapper:
    def __init__(self,model):
        self.m=model;self.d=mujoco.MjData(model);self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.va=model.jnt_dofadr[model.actuator_trnid[:,0]];self.head=model.body('head').id
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv))

    def solve(self,guide,actual,previous,anchor,roll_increment,weights):
        body=self.head+anchor
        rotation=actual.xmat[body].reshape(3,3)@Rotation.from_rotvec([0,0,-roll_increment]).as_matrix()
        origin=actual.xpos[body].copy();cache={}
        def evaluate(q):
            if 'q' in cache and np.array_equal(q,cache['q']):return cache
            d=self.d;d.qpos[:3]=0;d.qpos[3:7]=[1,0,0,0];d.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,d)
            rr=rotation@d.xmat[body].reshape(3,3).T;tt=origin-rr@d.xpos[body]
            d.qpos[:3]=tt;d.qpos[3:7]=Rotation.from_matrix(rr).as_quat()[[3,0,1,2]];mujoco.mj_fwdPosition(self.m,d)
            mujoco.mj_jac(self.m,d,self.jp,self.jr,origin,body);ap=self.jp[:,self.va].copy();ar=self.jr[:,self.va].copy()
            def jac(point,b):
                mujoco.mj_jac(self.m,d,self.jp,self.jr,point,b)
                return self.jp[:,self.va]-ap+skew(point-origin)@ar
            p=nodes(d);jj=np.asarray([jac(point,self.head+min(i,20)) for i,point in enumerate(p)])
            penalty=np.zeros(42);pj=np.zeros((42,20));depth=0.
            for con in d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]]);one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                if one!=two:
                    pipe=con.geom2 if one else con.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                    b=b1 if one else b2;idx=b-self.head;v=max(0.,-con.dist-.0008);depth=max(depth,-con.dist)
                    if v>penalty[idx]:penalty[idx]=v;pj[idx]=(1 if one else -1)*con.frame[:3]@jac(con.pos,b)
                elif one and two and abs(b1-b2)>1:
                    idx=21+min(b1,b2)-self.head;v=max(0.,-con.dist-.0002)
                    if v>penalty[idx]:penalty[idx]=v;pj[idx]=con.frame[:3]@(jac(con.pos,b1)-jac(con.pos,b2))
            cache.clear();cache.update(q=q.copy(),residual=np.r_[(weights[:,None]*(p-guide)/.012).ravel(),penalty/.0005,.02*(q-previous)],
                jac=np.vstack([(weights[:,None,None]*jj/.012).reshape(66,20),pj/.0005,.02*np.eye(20)]),points=p.copy(),depth=depth)
            return cache
        lo=np.maximum(-LIMIT,previous-np.radians(20)*DT);hi=np.minimum(LIMIT,previous+np.radians(20)*DT)
        guess=np.clip(actual.qpos[self.qa],lo+1e-9,hi-1e-9)
        result=least_squares(lambda q:evaluate(q)['residual'],guess,jac=lambda q:evaluate(q)['jac'],bounds=(lo,hi),max_nfev=20,ftol=1e-6,xtol=1e-6,gtol=1e-5)
        a=evaluate(result.x)
        return result.x.copy(),dict(projection_rms_mm=float(np.sqrt(np.mean(np.sum((a['points']-guide)**2,axis=1)))*1000),
                                    predicted_pipe_depth_mm=a['depth']*1000,predicted_points_mm=a['points']*1000,anchor_module=anchor)


class Known90GlobalLocal:
    def __init__(self,model,data,route,feedback=True,speed=.0035):
        self.m=model;self.route=route;self.mapper=ActualBearingMapper(model);self.qa=self.mapper.qa;self.va=self.mapper.va
        self.last=data.ctrl.copy();self.shift=0.;self.phase=0.;self.feedback=feedback;self.speed=speed
        self.kp=np.full(20,127.);self.coeff=np.zeros(2);self.target_delta=0.;self.angle=0.;self.axial=0.
        self.capture_age=0.;self.acquired=False;self.events=[];self.captured_y=None
        self.best_progress=0.;self.last_progress_time=0.;self.gated=False

    def command(self,data,t):
        p=nodes(data);c=(p[:-1]+p[1:])/2;q=data.qpos[self.qa].copy();load=data.qfrc_actuator[self.va].copy()
        sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);rt=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1)
        source=(sr>.074)&(sr<.120)&(c[:,0]<1.035)
        target=(rt>.069)&(rt<.120)&(c[:,1]>.15)
        prefix=0
        while prefix<21 and target[prefix]:prefix+=1
        wrap=float(np.ptp(np.unwrap(np.arctan2(c[:prefix,2]-1,c[:prefix,0]-1.1)))) if prefix>=2 else 0.
        tracking=float(np.degrees(abs(self.last-q)).max())
        if prefix>=5 and wrap>2.3 and tracking<25:
            if self.captured_y is None:self.captured_y=float(c[0,1])
            self.capture_age+=DT
            if not self.acquired and self.capture_age>=1. and c[0,1]>=self.captured_y-.003:
                self.acquired=True;self.events.append(dict(t_s=t,event='located_geometry_capture',prefix=prefix,wrap_rad=wrap))
        else:self.capture_age=0.;self.captured_y=None
        # Progress gates use true poses, not planned module counts alone.
        head_error=float(np.linalg.norm(p[0]-self.route.sample([self.shift])[0]))
        self.gated=(not self.acquired and self.shift>self.route.bridge_length+.33) or head_error>.07 or tracking>32
        scale=0. if self.gated else float(np.clip((28-tracking)/20,.12,1))
        if not self.acquired and head_error>.03:scale*=.4
        inc=self.speed*DT*scale;self.shift=min(self.route.total,self.shift+inc)
        phase_inc=(inc if inc>0 else self.speed*DT*.08)*self.route.beta/.029*(self.route.alpha/self.route.beta**2)
        self.phase+=phase_inc
        nominal,ns=self.route.window(self.shift);nc=(nominal[:-1]+nominal[1:])/2
        carrying=source & ((ns[:-1]+ns[1:])/2<-.04)
        if np.count_nonzero(carrying)>=4:
            observed=np.arctan2(c[carrying,2]-1,c[carrying,1]);desired=np.arctan2(nc[carrying,2]-1,nc[carrying,1])
            delta=np.angle(np.exp(1j*(observed-desired)))
            wanted=np.arctan2(np.sin(delta).mean(),np.cos(delta).mean())
            angle_step=np.angle(np.exp(1j*(wanted-self.angle)));self.angle+=np.clip(angle_step,-.025,.025)
            wanted_x=float(np.median(c[carrying,0]-nc[carrying,0]));self.axial+=np.clip(wanted_x-self.axial,-.001,.001)
        located=LocatedCurve(self.route,self.angle,self.axial,self.coeff,self.target_delta)
        execution,es=located.window(self.shift)
        em=(es[:-1]+es[1:])/2;bridge=(em>=0)&(em<self.route.bridge_length)&(~source)&(~target)
        if self.feedback and np.count_nonzero(bridge)>=1:
            i=np.flatnonzero(bridge);error=(execution[:-1]+execution[1:])/2-c
            resistance=float(np.max(abs(load[np.minimum(i,19)])))
            wanted=np.array([.3*np.mean(error[i,1]),.3*np.mean(error[i,2])])
            if resistance>1.8:wanted[1]=min(wanted[1],-.005)
            wanted=np.clip(wanted,-.020,.020);self.coeff+=np.clip(wanted-self.coeff,-.00025,.00025)
        if self.feedback and self.acquired and prefix>=5:
            desired=np.clip(.25*(.087-np.median(rt[:prefix])),-.003,.003)
            self.target_delta+=np.clip(desired-self.target_delta,-.0001,.0001)
        located=LocatedCurve(self.route,self.angle,self.axial,self.coeff,self.target_delta);execution,es=located.window(self.shift)
        carrying_ids=np.flatnonzero(source)
        if self.acquired and (len(carrying_ids)<4 or np.count_nonzero(source[14:])<3):anchor=min(3,prefix-1)
        elif len(carrying_ids):anchor=int(carrying_ids[min(len(carrying_ids)-1,max(0,len(carrying_ids)-3))])
        else:anchor=18
        weights=np.ones(22);weights[:min(10,max(3,prefix+3))]=2.
        # The source suffix remains a supported canonical helix, not a measured
        # sagging supply. Its physical material frame is read explicitly.
        desired,report=self.mapper.solve(execution,data,self.last,anchor,phase_inc,weights)
        target_q=np.clip(desired,np.maximum(-LIMIT,self.last-np.radians(20)*DT),np.minimum(LIMIT,self.last+np.radians(20)*DT))
        kp=np.full(20,127.)
        if tracking>18 and abs(load).max()>1.8:
            joint_bridge=((es[1:-1]>=0)&(es[1:-1]<self.route.bridge_length))&(~source[:20])
            kp[joint_bridge]=95.25
        self.kp+=np.clip(kp-self.kp,-127*DT/1.5,127*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        step=float(np.degrees(abs(target_q-self.last)).max());self.last=target_q.copy()
        return target_q,dict(stage='continuous_intake' if self.acquired else 'located_open_upward_entry',shared_shift_mm=self.shift*1000,
            roll_phase_rad=self.phase,tracking_deg=tracking,head_global_error_mm=head_error*1000,estimated_prefix=prefix,estimated_wrap_rad=wrap,
            estimated_capture=self.acquired,advancement_gate=self.gated,speed_scale=scale,global_points_mm=nominal*1000,
            execution_points_mm=execution*1000,estimated_points_mm=p*1000,material_stations_mm=es*1000,
            bridge_coeff_mm=np.array([[0,self.coeff[0],self.coeff[1]],[0,0,0]])*1000,radius_delta_mm=np.array([0,self.target_delta])*1000,
            source_pose_azimuth_rad=self.angle,source_pose_axial_mm=self.axial*1000,kp_nm_per_rad=self.kp.copy(),
            controller_source_geometry_mask=source,controller_target_geometry_mask=target,maximum_dispatch_step_deg=step,
            localization='read-only exact MuJoCo body pose; explicitly not a hardware localization result',
            exact_contact_used_for_local_feedback=False,physical_root_assigned=False,**report)
