"""Independent baseline: source helix, inner upward entry, shared material feed.

Only joint feedback and actuator load enter the controller. Support-registered
FK is a MODEL ESTIMATE, not an observed world pose. Exact simulation contacts
and poses are reserved for the experiment evaluator.
"""
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,least_squares
from scipy.spatial.transform import Rotation
from snake_control.suspended_supply_transfer import LINK, DT, LIMIT, nodes, WholeCurveMapper,skew
from snake_control.horizontal_pipe_roll import HingePhaseMapper


class InnerCCWRoute:
    def __init__(self, source_x=.95, source_phi=-np.pi/2+.12,
                 target_phi=-5*np.pi/6, target_radius=.0915, source_hand=1., initial_points=None):
        self.radius=.0915; self.pitch=.180; self.source_hand=source_hand
        self.alpha=self.pitch/np.hypot(self.pitch,2*np.pi*self.radius)
        self.beta=np.sqrt(1-self.alpha**2)
        self.source_phi=source_phi; self.target_phi=target_phi
        self.target_radius=target_radius; self.target_y=.20
        self.q=np.array([source_x,self.radius*np.cos(source_phi),1+self.radius*np.sin(source_phi)])
        self.initial_field=None
        if initial_points is not None:
            initial_points=np.asarray(initial_points);self.q=initial_points[0].copy()
            # Once-only setup calibration of the artificially placed chain.
            # Frozen before dynamics; never replaced by measured runtime sag.
            ds=brentq(lambda u:np.hypot(self.alpha*u,2*self.radius*np.sin(self.beta*u/(2*self.radius)))-LINK,.05,.08)
            self.initial_field=CubicSpline((-np.arange(22)*ds)[::-1],initial_points[::-1],axis=0)
        self.e=self.target([0.])[0]
        ts=np.array([self.alpha,-source_hand*self.beta*np.sin(source_phi),source_hand*self.beta*np.cos(source_phi)])
        if initial_points is not None:
            ts=initial_points[0]-initial_points[1];ts/=np.linalg.norm(ts)
        te=np.array([self.beta*np.sin(target_phi),self.alpha,-self.beta*np.cos(target_phi)])
        t=np.linspace(0,1,3001)[:,None]
        # Asymmetric Hermite handles limit the lower entry bend without the
        # deep below-pipe detour of the previous head-first route.
        self.handles=np.array([.09,.11])
        connector=(2*t**3-3*t**2+1)*self.q+(t**3-2*t**2+t)*self.handles[0]*ts+(-2*t**3+3*t**2)*self.e+(t**3-t**2)*self.handles[1]*te
        self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(connector,axis=0),axis=1))]
        self.bridge_length=float(self.arc[-1]); self.bridge=CubicSpline(self.arc,connector,axis=0)
        self.total=self.bridge_length+1.5
        self.frame_s=np.linspace(-1.7,self.total+.10,2401)
        p=self.sample(self.frame_s); tang=np.gradient(p,self.frame_s,axis=0);tang/=np.linalg.norm(tang,axis=1)[:,None]
        phi=source_phi+source_hand*self.beta*self.frame_s[0]/self.radius
        n=np.array([0,np.cos(phi),np.sin(phi)]);n-=n@tang[0]*tang[0];n/=np.linalg.norm(n);normal=[n.copy()]
        for a,b in zip(tang[:-1],tang[1:]):
            cross=np.cross(a,b);sine=np.linalg.norm(cross)
            if sine>1e-12:n=Rotation.from_rotvec(cross*np.arctan2(sine,np.clip(a@b,-1,1))/sine).apply(n)
            n-=n@b*b;n/=np.linalg.norm(n);normal.append(n.copy())
        self.tangents=tang;self.normals=np.asarray(normal)

    def target(self,w):
        w=np.atleast_1d(w);phi=self.target_phi-self.beta*w/self.target_radius
        return np.column_stack([1.1+self.target_radius*np.cos(phi),self.target_y+self.alpha*w,1+self.target_radius*np.sin(phi)])

    def sample(self,v):
        v=np.atleast_1d(v);p=self.target(v-self.bridge_length)
        src=v<0;mid=(v>=0)&(v<self.bridge_length)
        phi=self.source_phi+self.source_hand*self.beta*v[src]/self.radius
        p[src]=np.column_stack([self.q[0]+self.alpha*v[src],self.radius*np.cos(phi),1+self.radius*np.sin(phi)])
        if self.initial_field is not None:p[src]=self.initial_field(v[src])
        p[mid]=self.bridge(v[mid]);return p

    def predecessor(self,s,p):
        grid=s-np.linspace(.001,.20,35);g=np.linalg.norm(self.sample(grid)-p,axis=1)-LINK
        for i in range(1,len(grid)):
            if g[i]>=0 and g[i-1]<0:return brentq(lambda u:np.linalg.norm(self.sample([u])[0]-p)-LINK,grid[i],grid[i-1],xtol=1e-10)
        raise ValueError('No preceding physical rod chord on registered route')

    def window(self,shift):
        s=[shift];p=[self.sample(s)[0]]
        for _ in range(21):
            s.append(self.predecessor(s[-1],p[-1]));p.append(self.sample([s[-1]])[0])
        return np.asarray(p),np.asarray(s)

    def frame(self,s):
        t=np.array([np.interp(s,self.frame_s,self.tangents[:,j]) for j in range(3)]);t/=np.linalg.norm(t)
        n=np.array([np.interp(s,self.frame_s,self.normals[:,j]) for j in range(3)]);n-=n@t*t;n/=np.linalg.norm(n)
        return np.column_stack([n,np.cross(t,n),t])

    def head_orientation(self,rotation,shift,phase):
        return self.frame(shift)@self.frame(0).T@rotation@Rotation.from_rotvec([0,0,-phase]).as_matrix()

    def description(self):
        return dict(source_exit_m=self.q,target_entry_m=self.e,source_spatial_hand=self.source_hand,
                    source_phi_rad=self.source_phi,target_phi_rad=self.target_phi,bridge_length_mm=self.bridge_length*1000,
                    node_radius_mm=self.radius*1000,pitch_mm=self.pitch*1000,
                    estimated_front_release_mm=(self.bridge_length+np.pi*self.target_radius/self.beta)*1000,
                    chord_mm=LINK*1000,starting_target_modules=0)


def initial_pose(model,route):
    guide,stations=route.window(0);mapper=WholeCurveMapper(model);pose,report=mapper.initialize(guide)
    p=nodes(mapper.d);c=(p[:-1]+p[1:])/2
    sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
    target_any=[]
    for con in mapper.d.contact:
        n=[model.geom(int(g)).name or '' for g in [con.geom1,con.geom2]]
        if any(x in ('pipe_straight_04','pipe_sharp_out_03') for x in n):target_any.append(float(con.dist))
    report.update(source_radii_mm=sr*1000,reference_stations_mm=stations*1000,
                  target_initial_contact_count=len(target_any),source_fitting_rms_mm=report['guide_rms_mm'])
    calibrated=InnerCCWRoute(source_x=route.q[0],source_phi=route.source_phi,target_phi=route.target_phi,
                            target_radius=route.target_radius,source_hand=route.source_hand,initial_points=p)
    route.__dict__.update(calibrated.__dict__);guide,stations=route.window(0)
    report['setup_calibrated_reference_rms_mm']=float(np.sqrt(np.mean(np.sum((p-guide)**2,axis=1)))*1000)
    report['calibration_scope']='fixed artificial initial source curve only; before first physical step'
    report['ready']=bool(report['pipe_depth_mm']<=.9 and report['self_depth_mm']<=.5 and report['guide_rms_mm']<35
                         and np.all((sr>.078)&(sr<.105)) and not target_any and report['maximum_joint_deg']<88.01)
    return pose,guide,report


def register(a,b):
    ac=a.mean(0);bc=b.mean(0);u,_,vh=np.linalg.svd((a-ac).T@(b-bc));rot=vh.T@u.T
    if np.linalg.det(rot)<0:vh[-1]*=-1;rot=vh.T@u.T
    return rot,bc-rot@ac


class InnerCCWCompliance:
    def __init__(self,model,data,route,feedback=True,speed=.005,roll_sign=1.,roll_factor=None,rolling_support=False,anchor_mapper=False):
        self.m=model;self.route=route;self.feedback=feedback;self.mapper=HingePhaseMapper(model,data)
        self.qa=self.mapper.qa;self.va=self.mapper.va;self.fk=mujoco.MjData(model);self.fk.qpos[:]=data.qpos
        self.last=data.ctrl.copy();self.shift=0.;self.phase=0.;self.speed=speed;self.roll_sign=roll_sign
        self.roll_factor=route.alpha/route.beta**2 if roll_factor is None else roll_factor
        self.coeff=np.zeros((2,3));self.rdelta=np.zeros(2);self.kp=np.full(20,127.);self.age=0.;self.acquired=False
        self.events=[];self.failure=None;self.last_ref=route.window(0)[0];self.initial_rotation=self.mapper.rotation.copy()
        self.rolling_support=rolling_support
        self.anchor_mapper=anchor_mapper;self.afk=mujoco.MjData(model);self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv))

    def support_motion(self,p,s):
        p=p.copy()
        if not self.rolling_support:return p
        # B01 diagnosed that axial rolling is accompanied by source orbit.
        # A frozen no-slip/calibrated odometry model changes this LOCAL support
        # pose only. No runtime world pose or contact is read here.
        angle=-self.route.source_hand*self.route.beta*self.shift/self.route.radius-.14*self.phase
        dx=-self.route.alpha*self.shift+.045*self.phase
        moved=p.copy();moved[:,[1,2]]=(p[:,[1,2]]-[0,1])@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])+[0,1]
        moved[:,0]+=dx
        w=np.clip(s/self.route.bridge_length,0,1);blend=1-w*w*(3-2*w)
        return p+blend[:,None]*(moved-p)

    def estimate(self,q,reference,stations):
        self.fk.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,self.fk);p=nodes(self.fk)
        # Assumed support geometry registers FK; no exact actual root/contact.
        support=np.flatnonzero(stations<-.03)
        mode='source_support_registered_joint_FK'
        if self.acquired and np.count_nonzero(stations>self.route.bridge_length+.05)>=7:
            support=np.flatnonzero(stations>self.route.bridge_length+.05);mode='target_support_registered_joint_FK'
        if len(support)<4:support=np.arange(22);mode='whole_route_registered_joint_FK_low_confidence'
        rot,trans=register(p[support],reference[support]);estimated=p@rot.T+trans
        self.estimate_rotation=rot;self.estimate_translation=trans
        return estimated,mode,float(np.sqrt(np.mean(np.sum((estimated[support]-reference[support])**2,axis=1)))*1000)

    def anchored_solve(self,guide,observed,phase_increment,stations):
        # Register the assumed bearing module's complete material frame, rather
        # than allowing an independently imposed head frame to float the chain.
        anchor=3 if self.acquired and stations[4]>self.route.bridge_length else 18
        body=self.m.body('head').id+anchor
        rotation=self.estimate_rotation@self.fk.xmat[body].reshape(3,3)@Rotation.from_rotvec([0,0,-phase_increment]).as_matrix()
        origin=guide[anchor+1];head=self.m.body('head').id;cache={}
        def evaluate(q):
            if 'q' in cache and np.array_equal(q,cache['q']):return cache
            d=self.afk;d.qpos[:3]=0;d.qpos[3:7]=[1,0,0,0];d.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,d)
            rr=rotation@d.xmat[body].reshape(3,3).T;tt=origin-rr@d.xpos[body]
            d.qpos[:3]=tt;d.qpos[3:7]=Rotation.from_matrix(rr).as_quat()[[3,0,1,2]];mujoco.mj_fwdPosition(self.m,d)
            mujoco.mj_jac(self.m,d,self.jp,self.jr,origin,body);ap=self.jp[:,self.va].copy();ar=self.jr[:,self.va].copy()
            def jac(point,b):
                mujoco.mj_jac(self.m,d,self.jp,self.jr,point,b)
                return self.jp[:,self.va]-ap+skew(point-origin)@ar
            p=nodes(d);jj=np.asarray([jac(point,head+min(i,20)) for i,point in enumerate(p)])
            penalty=np.zeros(42);pj=np.zeros((42,20));depth=0.
            for con in d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]]);one=head<=b1<head+21;two=head<=b2<head+21
                if one!=two:
                    pipe=con.geom2 if one else con.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                    b=b1 if one else b2;idx=b-head;v=max(0.,-con.dist-.0008);depth=max(depth,-con.dist)
                    if v>penalty[idx]:penalty[idx]=v;pj[idx]=(1 if one else -1)*con.frame[:3]@jac(con.pos,b)
                elif one and two and abs(b1-b2)>1:
                    idx=21+min(b1,b2)-head;v=max(0.,-con.dist-.0002)
                    if v>penalty[idx]:penalty[idx]=v;pj[idx]=con.frame[:3]@(jac(con.pos,b1)-jac(con.pos,b2))
            weights=np.ones(22);weights[:8]=2.
            cache.clear();cache.update(q=q.copy(),residual=np.r_[(weights[:,None]*(p-guide)/.012).ravel(),penalty/.0005,.01*(q-observed)],
                jac=np.vstack([(weights[:,None,None]*jj/.012).reshape(66,20),pj/.0005,.01*np.eye(20)]),points=p.copy(),depth=depth)
            return cache
        lo=np.maximum(-LIMIT,self.last-np.radians(20)*DT);hi=np.minimum(LIMIT,self.last+np.radians(20)*DT)
        result=least_squares(lambda q:evaluate(q)['residual'],np.clip(observed,lo+1e-9,hi-1e-9),jac=lambda q:evaluate(q)['jac'],bounds=(lo,hi),
                             max_nfev=18,ftol=1e-6,xtol=1e-6,gtol=1e-5)
        a=evaluate(result.x)
        return result.x,dict(whole_fit_rms_mm=float(np.sqrt(np.mean(np.sum((a['points']-guide)**2,axis=1)))*1000),predicted_pipe_depth_mm=a['depth']*1000,
                             anchor_module=anchor,mapper_kind='encoder_estimated_material_frame_anchor')

    def command(self,q,load,t):
        nominal,s=self.route.window(self.shift);estimated,mode,esterr=self.estimate(q,self.last_ref,s)
        c=(estimated[:-1]+estimated[1:])/2;sm=(s[:-1]+s[1:])/2
        rt=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);near=(rt>.065)&(rt<.125)&(c[:,1]>.15)
        prefix=0
        while prefix<21 and near[prefix]:prefix+=1
        span=float(np.ptp(np.unwrap(np.arctan2(c[:prefix,2]-1,c[:prefix,0]-1.1)))) if prefix>=2 else 0.
        err=np.degrees(abs(self.last-q));tracking=float(err.max());loadmag=float(abs(load).max())
        # Model-estimated geometry AND a sustained drive response, never load alone.
        condition=prefix>=5 and span>2.3 and tracking<30 and np.mean(abs(load[:max(1,prefix-1)]))>.05
        self.age=self.age+DT if condition else 0
        if not self.acquired and self.age>=1:
            self.acquired=True;self.events.append(dict(t_s=t,event='controller_estimated_capture',prefix=prefix,span_rad=span))
        nominal_target=int(np.count_nonzero(sm>self.route.bridge_length))
        scale=float(np.clip((32-tracking)/24,.12,1)) if self.feedback else 1.
        gate=(nominal_target>=8 and not self.acquired)
        if gate:scale=0.
        if self.feedback:
            src=sm<-.06;rec=(sm>self.route.bridge_length+.04)&near
            rs=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
            desired=np.zeros(2)
            if np.count_nonzero(src)>=4:desired[0]=np.clip(.35*(.087-np.median(rs[src])),-.004,.004)
            if self.acquired and np.count_nonzero(rec)>=3:desired[1]=np.clip(.35*(.087-np.median(rt[rec])),-.004,.004)
            self.rdelta+=np.clip(desired-self.rdelta,-.00015,.00015)
            w=np.clip(s/self.route.bridge_length,0,1);h=16*w*w*(1-w)*(1-w)
            valid=h>.12
            if np.count_nonzero(valid)>=2:
                basis=np.column_stack([h[valid],h[valid]*(2*w[valid]-1)])
                fitted=np.linalg.lstsq(basis,(estimated-nominal)[valid],rcond=None)[0]
                fitted=np.clip(.25*fitted,-.015,.015)
                # Front resistance is accommodated here before receiving radius.
                if tracking>18:fitted[0,2]-=.003
                self.coeff+=np.clip(fitted-self.coeff,-.0003,.0003)
        increment=self.speed*DT*scale;self.shift=min(self.route.total,self.shift+increment)
        phase_inc=increment*self.route.beta/.029*self.roll_factor
        if gate:phase_inc=self.speed*DT*self.route.beta/.029*self.roll_factor*.25
        self.phase+=self.roll_sign*phase_inc
        nominal,s=self.route.window(self.shift);execution=nominal.copy();w=np.clip(s/self.route.bridge_length,0,1)
        h=16*w*w*(1-w)*(1-w);execution+=h[:,None]*(self.coeff[0]+(2*w-1)[:,None]*self.coeff[1])
        for side,(mask,cols,origin) in enumerate([(s<0,[1,2],[0,1]),(s>self.route.bridge_length,[0,2],[1.1,1])]):
            radial=execution[:,cols]-origin;radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
            blend=np.clip(-s/(2*LINK),0,1) if side==0 else np.clip((s-self.route.bridge_length)/(2*LINK),0,1)
            execution[:,cols]+=blend[:,None]*self.rdelta[side]*radial
        execution=self.support_motion(execution,s)
        orientation=self.route.head_orientation(self.initial_rotation,self.shift,self.phase)
        if self.anchor_mapper:desired,report=self.anchored_solve(execution,q,self.roll_sign*phase_inc,s)
        else:desired,report=self.mapper.solve(execution,self.shift,self.phase,self.last,orientation_override=orientation,
                                            command_step=np.radians(20)*DT,node_weights=np.ones(22))
        target=np.clip(desired,np.maximum(-LIMIT,self.last-np.radians(20)*DT),np.minimum(LIMIT,self.last+np.radians(20)*DT))
        self.last=target.copy();self.last_ref=execution.copy()
        mid=(s[1:-1]>0)&(s[1:-1]<self.route.bridge_length)
        kp=np.full(20,127.);kp[mid]=63.5
        self.kp+=np.clip(kp-self.kp,-127*DT/1.5,127*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        return target,dict(stage='transfer' if self.acquired else 'active_inner_entry',shared_shift_mm=self.shift*1000,
            roll_phase_rad=self.phase,tracking_deg=tracking,load_peak_nm=loadmag,estimated_prefix=prefix,estimated_wrap_rad=span,
            estimated_capture=self.acquired,advancement_gate=gate,speed_scale=scale,global_points_mm=nominal*1000,
            execution_points_mm=execution*1000,estimated_points_mm=estimated*1000,material_stations_mm=s*1000,
            bridge_coeff_mm=self.coeff*1000,radius_delta_mm=self.rdelta*1000,kp_nm_per_rad=self.kp.copy(),
            estimator_mode=mode,estimator_support_fit_mm=esterr,projection_rms_mm=report['whole_fit_rms_mm'],
            predicted_pipe_depth_mm=report['predicted_pipe_depth_mm'],physical_root_assigned=False)
