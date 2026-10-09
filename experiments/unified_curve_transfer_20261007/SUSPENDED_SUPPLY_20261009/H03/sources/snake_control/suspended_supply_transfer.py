"""Whole-curve source peeling, hanging supply and shared-window hauling.

The private FK mapper reconstructs all twenty real hinges. Its root is a
mathematical registration only. Actual dynamics receive joint ctrl only.
"""
import math
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from snake_control import suspended_transfer_route as route
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone

DT, LINK = .05, .0645
LIMIT, STEP = np.radians(88.), np.radians(10.)*DT

def nodes(d):
    return np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])

def skew(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

def desired_curve(stage, parameter, shift=0., radius_delta=0.,initial_field=None):
    sample = route.make_peel(parameter)[0] if stage=='peel' else route.make_unfold(parameter)[0] if stage=='unfold' else route.receiving
    v=9*LINK+shift-np.arange(22)*LINK
    p=sample(v)
    if initial_field is not None:
        captured=initial_field(np.clip(v,-12*LINK,9*LINK))
        beyond=np.clip((v-9*LINK)/(3*LINK),0.,1.);captured*= (1-beyond*beyond*(3-2*beyond))[:,None]
        if stage=='unfold':
            prefix=np.clip((v+route.INTAKE)/route.INTAKE,0.,1.);prefix=prefix*prefix*(3-2*prefix)
            fade=parameter*parameter*(3-2*parameter)
            captured*=(1-fade*(1-prefix))[:,None]
        elif stage=='haul':
            prefix=np.clip((v+route.INTAKE)/route.INTAKE,0.,1.);prefix=prefix*prefix*(3-2*prefix)
            fade=np.clip(shift/(12*LINK),0.,1.);fade=fade*fade*(3-2*fade)
            captured*=((1-fade)*prefix)[:,None]
        p+=captured
    # Pressure alters this same route, with a smooth entry blend. No joint patch.
    w=np.clip((v+route.INTAKE)/route.INTAKE,0.,1.);w=w*w*(3-2*w)
    radial=p[:,[0,2]]-[1.1,1.]
    radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
    p[:,[0,2]]+=w[:,None]*radius_delta*radial
    return p,v

class WholeCurveMapper:
    def __init__(self,model):
        self.m=model;self.d=mujoco.MjData(model)
        self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.va=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.head=model.body('head').id
        self.robot_geom=np.array([next(g for g in range(model.ngeom) if model.geom_bodyid[g]==self.head+j) for j in range(21)])
        self.root=np.zeros(3);self.q=np.zeros(20);self.rotation=np.eye(3)
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv))

    def set(self,root,rotation,q):
        self.d.qpos[:3]=root
        self.d.qpos[3:7]=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        self.d.qpos[self.qa]=q
        mujoco.mj_fwdPosition(self.m,self.d)

    def collision_rows(self,jacobian=False):
        values=np.zeros(42);jac=np.zeros((42,23))
        depth=np.zeros(21);selfdepth=np.zeros(21)
        for contact in self.d.contact:
            b1,b2=map(int,self.m.geom_bodyid[[contact.geom1,contact.geom2]])
            one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
            if one!=two:
                pipe=contact.geom2 if one else contact.geom1
                if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                body=b1 if one else b2;i=body-self.head
                depth[i]=max(depth[i],-float(contact.dist))
                violation=max(0.,-float(contact.dist)-.0004)
                if violation<=values[i]:continue
                values[i]=violation
                if jacobian:
                    mujoco.mj_jac(self.m,self.d,self.jp,self.jr,contact.pos,body)
                    jp=np.column_stack([np.eye(3),self.jp[:,self.va]])
                    jac[i]=(1. if one else -1.)*contact.frame[:3]@jp
            elif one and two and abs(b1-b2)>1:
                i=min(b1,b2)-self.head
                selfdepth[i]=max(selfdepth[i],-float(contact.dist))
                violation=max(0.,-float(contact.dist)-.0002)
                if violation<=values[21+i]:continue
                values[21+i]=violation
                if jacobian:
                    mujoco.mj_jac(self.m,self.d,self.jp,self.jr,contact.pos,b1);a=self.jp[:,self.va].copy()
                    mujoco.mj_jac(self.m,self.d,self.jp,self.jr,contact.pos,b2)
                    jac[21+i,3:]=contact.frame[:3]@(a-self.jp[:,self.va])
        return values,jac,depth,selfdepth

    def initialize(self,guide):
        bone=SpatialBishopBackbone(guide*1000,np.full(21,LINK*1000))
        q=np.clip(bone.joint_angles(),-LIMIT,LIMIT)
        z=guide[0]-guide[1];z/=np.linalg.norm(z)
        # joint1's child has a 180-degree Z rotation: its world hinge is -head X.
        # The head normal must therefore oppose the backbone's first normal.
        x=-bone.start_frame[:,1].copy();x-=x@z*z;x/=np.linalg.norm(x)
        rotation=np.column_stack([x,np.cross(z,x),z])
        root=guide[0]-LINK*z
        base=np.r_[root,np.zeros(3),q]
        units=np.r_[np.full(3,.03),np.full(3,.2),np.full(20,.2)]
        lo=np.r_[np.full(3,-4.),np.full(3,-8.),(-LIMIT-base[6:])/.2]
        hi=np.r_[np.full(3,4.),np.full(3,8.),(LIMIT-base[6:])/.2]
        weights=np.ones(22);weights[:10]=2.
        cache={};collision_enabled=[False]
        def evaluate(zv):
            if 'z' in cache and np.array_equal(cache['z'],zv) and cache['collision']==collision_enabled[0]:return cache
            x=base+units*zv
            self.set(x[:3],Rotation.from_rotvec(x[3:6]).as_matrix()@rotation,x[6:])
            theta=np.linalg.norm(x[3:6]);cross=skew(x[3:6])
            left=np.eye(3)+(1-math.cos(theta))/max(theta**2,1e-12)*cross+(theta-math.sin(theta))/max(theta**3,1e-12)*(cross@cross)
            def point_jac(point,body):
                mujoco.mj_jac(self.m,self.d,self.jp,self.jr,point,body)
                return np.column_stack([np.eye(3),-skew(point-x[:3])@left,self.jp[:,self.va]])*units
            p=nodes(self.d);jac=np.array([point_jac(point,self.head+min(i,20)) for i,point in enumerate(p)])
            penalty=np.zeros(42);penjac=np.zeros((42,26))
            if collision_enabled[0]:
                for contact in self.d.contact:
                    b1,b2=map(int,self.m.geom_bodyid[[contact.geom1,contact.geom2]])
                    one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                    if one!=two:
                        pipe=contact.geom2 if one else contact.geom1
                        if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                        body=b1 if one else b2;i=body-self.head
                        violation=max(0.,-float(contact.dist)-.0004)
                        if violation>penalty[i]:
                            penalty[i]=violation;penjac[i]=(1 if one else -1)*contact.frame[:3]@point_jac(contact.pos,body)
                    elif one and two and abs(b1-b2)>1:
                        i=21+min(b1,b2)-self.head;violation=max(0.,-float(contact.dist)-.0002)
                        if violation>penalty[i]:
                            penalty[i]=violation;penjac[i]=contact.frame[:3]@(point_jac(contact.pos,b1)-point_jac(contact.pos,b2))
            cache.clear();cache.update(z=zv.copy(),collision=collision_enabled[0],
                residual=np.r_[(weights[:,None]*(p-guide)/.012).ravel(),penalty/.00035],
                jac=np.vstack([(weights[:,None,None]*jac/.012).reshape(66,26),penjac/.00035]))
            return cache
        result=least_squares(lambda z:evaluate(z)['residual'],np.zeros(26),jac=lambda z:evaluate(z)['jac'],bounds=(lo,hi),
            max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        collision_enabled[0]=True
        result=least_squares(lambda z:evaluate(z)['residual'],result.x,jac=lambda z:evaluate(z)['jac'],bounds=(lo,hi),
            max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        x=base+units*result.x
        self.root=x[:3].copy();self.rotation=Rotation.from_rotvec(x[3:6]).as_matrix()@rotation;self.q=x[6:].copy()
        self.set(self.root,self.rotation,self.q)
        _,_,pd,sd=self.collision_rows()
        p=nodes(self.d);c=.5*(p[:-1]+p[1:])
        tr=np.linalg.norm(c[:9,[0,2]]-[1.1,1.],axis=1)
        sr=np.linalg.norm(c[15:,[1,2]]-[0.,1.],axis=1)
        phi=np.unwrap(np.arctan2(c[:9,2]-1,c[:9,0]-1.1))
        report=dict(solver_success=bool(result.success),solver_message=str(result.message),evaluations=int(result.nfev),
            guide_rms_mm=float(np.sqrt(np.mean(np.sum((p-guide)**2,axis=1)))*1000),
            target_center_radii_mm=(tr*1000).tolist(),source_center_radii_mm=(sr*1000).tolist(),
            target_wrap_span_rad=float(np.ptp(phi)),maximum_joint_deg=float(np.degrees(abs(self.q)).max()),
            pipe_depth_mm=float(pd.max()*1000),self_depth_mm=float(sd.max()*1000),
            physical_steps=0,root_placement='one artificial initial placement before first physics step')
        report['ready']=bool(pd.max()<=.0008 and sd.max()<=.0005 and report['guide_rms_mm']<.035*1000
            and np.count_nonzero((tr>.065)&(tr<.125))>=7 and np.count_nonzero((sr>.065)&(sr<.13))>=4 and np.ptp(phi)>4.5)
        return self.d.qpos.copy(),report

    def solve(self,guide,before_guide,data,forces,previous,max_nfev=18):
        weights=np.ones(22);weights[:10]=2.
        # No free mathematical-root translation or rotation. Register every
        # candidate to the same measured carrying rod and its physical frame.
        self.d.qpos[:]=data.qpos;mujoco.mj_fwdPosition(self.m,self.d)
        actual_points=nodes(self.d);actual_centers=.5*(actual_points[:-1]+actual_points[1:])
        target_ids=np.flatnonzero(forces[1]>.1)
        source_ids=np.flatnonzero(forces[0]>.1)
        side=1 if len(target_ids) else 0 if len(source_ids) else None
        ids=target_ids if side==1 else source_ids if side==0 else np.array([0])
        old=getattr(self,'bearing_module',None)
        module=int(old) if old in ids else int(ids[np.argmax(forces[side,ids])]) if side is not None else 0
        self.bearing_module=module
        actual_rotation=self.d.xmat[self.head+module].reshape(3,3).copy()
        actual_root=data.qpos[:7].copy()
        root_rotation=Rotation.from_quat(actual_root[3:7][[1,2,3,0]]).as_matrix()
        requested=.5*(guide[module]+guide[module+1]-before_guide[module]-before_guide[module+1])
        if side is None:requested[:]=0.
        requested*=min(1.,.008*DT/max(np.linalg.norm(requested),1e-12))
        desired_center=actual_centers[module]+requested
        omega=np.zeros(3);lever=None
        matching=[]
        for con in data.contact:
            for pipe,robot in [(con.geom1,con.geom2),(con.geom2,con.geom1)]:
                if int(self.m.geom_bodyid[robot])!=self.head+module:continue
                name=self.m.geom(pipe).name or ''
                s=0 if name in {'pipe_straight_01','pipe_sharp_in_02'} else 1 if name in {'pipe_straight_04','pipe_sharp_out_03'} else None
                if s==side and abs(con.frame[side])<.35:matching.append((float(con.dist),con.pos.copy()))
        if matching:
            lever=min(matching,key=lambda x:x[0])[1]-actual_centers[module]
            radial=actual_centers[module]-([1.1,0.,1.] if side==1 else [0.,0.,1.]);radial[side]=0.;radial/=max(np.linalg.norm(radial),1e-12)
            pressure=radial*float(radial@requested)/DT
            rolling=skew(lever)
            # Center velocity = pressure + lever x angular velocity.
            hessian=rolling.T@rolling+.003**2*np.eye(3)
            omega=np.linalg.solve(hessian,rolling.T@(requested/DT-pressure))
            row=rolling[side];inverse=np.linalg.solve(hessian,row);den=float(row@inverse)
            if den>1e-12:omega+=inverse*((requested/DT-pressure)[side]-float(row@omega))/den
            spin=float(actual_rotation[:,2]@omega);perpendicular=omega-spin*actual_rotation[:,2]
            omega*=min(1.,.12/max(abs(spin),1e-12),.15/max(np.linalg.norm(perpendicular),1e-12))
        desired_rotation=Rotation.from_rotvec(omega*DT).as_matrix()@actual_rotation
        lower=np.maximum(-LIMIT,previous-STEP);upper=np.minimum(LIMIT,previous+STEP)
        seed=np.clip(previous,lower,upper)
        cache={}
        def evaluate(q):
            if 'q' in cache and np.array_equal(cache['q'],q):return cache
            self.d.qpos[:]=data.qpos;self.d.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,self.d)
            raw=nodes(self.d);center=.5*(raw[module]+raw[module+1])
            register=desired_rotation@self.d.xmat[self.head+module].reshape(3,3).T
            translation=desired_center-register@center
            self.set(register@actual_root[:3]+translation,register@root_rotation,q)
            p=nodes(self.d)
            mujoco.mj_jac(self.m,self.d,self.jp,self.jr,desired_center,self.head+module)
            center_jac=self.jp[:,self.va].copy();angular_jac=self.jr[:,self.va].copy()
            def point_jac(point,body):
                mujoco.mj_jac(self.m,self.d,self.jp,self.jr,point,body)
                return self.jp[:,self.va]-center_jac+skew(point-desired_center)@angular_jac
            jac=[]
            for i,point in enumerate(p):
                jac.append(point_jac(point,self.head+min(i,20)))
            penetration=np.zeros(42);penjac=np.zeros((42,20));pd=np.zeros(21);sd=np.zeros(21)
            for con in self.d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]])
                one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                if one!=two:
                    pipe=con.geom2 if one else con.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                    body=b1 if one else b2;i=body-self.head;pd[i]=max(pd[i],-float(con.dist))
                    violation=max(0.,-float(con.dist)-.0006)
                    if violation>penetration[i]:
                        penetration[i]=violation;penjac[i]=(1 if one else -1)*con.frame[:3]@point_jac(con.pos,body)
                elif one and two and abs(b1-b2)>1:
                    i=min(b1,b2)-self.head;sd[i]=max(sd[i],-float(con.dist));violation=max(0.,-float(con.dist)-.0002)
                    if violation>penetration[21+i]:
                        penetration[21+i]=violation;penjac[21+i]=con.frame[:3]@(point_jac(con.pos,b1)-point_jac(con.pos,b2))
            residual=np.r_[(weights[:,None]*(p-guide)/.012).ravel(),penetration/.00035,.015*(q-previous)]
            fulljac=np.vstack([(weights[:,None,None]*np.array(jac)/.012).reshape(66,20),penjac/.00035,.015*np.eye(20)])
            cache.clear();cache.update(q=q.copy(),residual=residual,jac=fulljac,points=p.copy(),pipe_depth=pd,self_depth=sd,root=self.d.qpos[:7].copy())
            return cache
        started=time.perf_counter()
        result=least_squares(lambda x:evaluate(x)['residual'],seed,jac=lambda x:evaluate(x)['jac'],
            bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-5)
        state=evaluate(result.x)
        info=dict(solver_success=bool(result.success),evaluations=int(result.nfev),elapsed_wall_s=time.perf_counter()-started,
            curve_fit_rms_mm=float(np.sqrt(np.mean(np.sum((state['points']-guide)**2,axis=1)))*1000),
            predicted_pipe_depth_mm=float(state['pipe_depth'].max()*1000),predicted_self_depth_mm=float(state['self_depth'].max()*1000),
            registered_math_root=state['root'].tolist(),physical_root_assigned=False,
            anchor_module=module,anchor_pipe_side=side,anchor_displacement_mm=(requested*1000).tolist(),
            anchor_omega_rad_s=omega.tolist(),anchor_contact_lever_mm=(lever*1000).tolist() if lever is not None else None,
            free_root_variables=0,maximum_candidate_step_deg=float(np.degrees(abs(result.x-previous)).max()),geometry_q_rad=result.x.tolist())
        return state['root'].copy(),result.x.copy(),info

class SuspendedSupplyTransfer:
    def __init__(self,model,initial_pose):
        self.mapper=WholeCurveMapper(model)
        self.mapper.root=initial_pose[:3].copy()
        self.orientation=Rotation.from_quat(initial_pose[3:7][[1,2,3,0]]).as_matrix()
        self.mapper.rotation=self.orientation.copy();self.last=initial_pose[self.mapper.qa].copy()
        p,v=desired_curve('peel',0.)
        self.mapper.set(initial_pose[:3],self.orientation,self.last)
        actual=nodes(self.mapper.d)
        self.initial_field=CubicSpline(v[::-1],(actual-p)[::-1],axis=0,bc_type='clamped')
        self.initial_curve_residual=actual-p
        p=actual.copy()
        self.reference_tangent=p[0]-p[1];self.reference_tangent/=np.linalg.norm(self.reference_tangent)
        self.last_guide=p.copy()
        self.stage='peel';self.parameter=0.;self.shift=0.;self.radius_delta=0.
        self.feed_origin=None;self.events=[];self.support_age=0.;self.bad_candidates=0
        self.phase_start=0.;self.last_progress=0.;self.exited=set();self.received=set()
        self.last_stage='peel';self.failure=None
        self.handoff_tangents=None;self.hanging_best=None

    def release_guide(self,actual_points,forces,exited,parameter):
        """Keep only the still-carried source suffix; accept its free bow.

        The fixed target prefix is never replaced by a retreating pose. The
        liberated portion is one continuous curve through the observed bow,
        joined to the remaining measured source suffix. Its outlet receives
        the active radial peeling field.
        """
        base,v=desired_curve('peel',0.,0.,self.radius_delta,self.initial_field)
        prefix=0
        while 15+prefix in exited and prefix<6:prefix+=1
        boundary=15+prefix
        free=actual_points[9:boundary+1].copy()
        start_error=base[9]-free[0]
        blend=np.linspace(1.,0.,len(free));blend=blend*blend*(3-2*blend)
        free+=blend[:,None]*start_error
        base[9:boundary+1]=free
        # Preserve only the source suffix that has not yet physically exited.
        if boundary<21:
            base[boundary:]=actual_points[boundary:]
            opening=min(.035,.050*parameter)
            count=min(3,22-boundary)
            weights=np.array([1.,.6,0.])[:count]
            for k,w in enumerate(weights):
                j=boundary+k;radial=base[j]-[0.,0.,1.];radial[0]=0.;radial/=max(np.linalg.norm(radial),1e-12)
                base[j]+=opening*w*radial
        return base,v

    def unfolding_guide(self,parameter):
        base,v=desired_curve('unfold',1.,0.,self.radius_delta,self.initial_field)
        target,_=desired_curve('peel',0.,0.,self.radius_delta,self.initial_field)
        end_t=base[9:-1]-base[10:];end_t/=np.linalg.norm(end_t,axis=1)[:,None]
        h=parameter*parameter*(3-2*parameter)
        tangent=(1-h)*self.handoff_tangents+h*end_t
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
        # One curve with twelve fixed-length tangent spans. The actual robot
        # still supplies every rigid rod length and both hinge families.
        target[10:]=target[9]-np.cumsum(LINK*tangent,axis=0)
        return target,v

    def command(self,data,t,forces,actual,exited,received):
        old_shift=self.shift
        if self.feed_origin is None and self.stage=='haul':self.feed_origin=actual.copy()
        contact_sum=float(forces[1,:max(9,len(received))].sum())
        self.support_age=self.support_age+DT if contact_sum>3. else 0.
        if len(exited)>len(self.exited) or len(received)>len(self.received):self.last_progress=t
        self.exited=set(exited);self.received=set(received)
        target_load=28.
        pressure_step=np.clip(.000015*(contact_sum-target_load),-.00030,.00030)*DT
        self.radius_delta=float(np.clip(self.radius_delta+pressure_step,-.0015,.0015))
        mismatch=float(np.degrees(np.abs(self.last-data.qpos[self.mapper.qa])).max())
        stress=float(np.mean(np.abs(data.qfrc_actuator[self.mapper.va])>2.2))
        scale=float(np.clip((12.-mismatch)/9.,.08,1.)*np.clip(1.-stress,.3,1.))
        proposal=self.parameter;shift=self.shift
        support_ready=self.support_age>=.3
        if self.stage=='peel':
            if support_ready:proposal=min(1.,self.parameter+DT/50.*scale)
            elif t<1.:proposal=min(.02,self.parameter+DT/50.*scale)
        elif self.stage=='unfold':proposal=min(1.,self.parameter+DT/55.*scale) if support_ready else self.parameter
        else:shift=min(12*LINK,self.shift+.0025*DT*scale) if support_ready else self.shift
        if self.stage=='peel' and len(exited)==6 and support_ready:
            # Actual complete source clearance is a phase event. It advances
            # the predefined macro route; the global route is not refitted to it.
            actual_nodes=nodes(data)
            self.handoff_tangents=(actual_nodes[9:-1]-actual_nodes[10:])/LINK
            self.transition('unfold',t);proposal=0.
            self.hanging_best=float(np.linalg.norm(actual[-1]-route.receiving([-12*LINK])[0]))
        if self.stage=='peel':guide,locations=self.release_guide(nodes(data),forces,exited,proposal)
        elif self.stage=='unfold' and self.handoff_tangents is not None:guide,locations=self.unfolding_guide(proposal)
        else:guide,locations=desired_curve(self.stage,proposal,shift,self.radius_delta,self.initial_field)
        wanted=guide[0]-guide[1];wanted/=np.linalg.norm(wanted)
        before=self.reference_tangent;cross=np.cross(before,wanted);sine=np.linalg.norm(cross)
        rot=Rotation.from_rotvec(cross*math.atan2(sine,float(np.clip(before@wanted,-1,1)))/max(sine,1e-12)).as_matrix()
        orientation=rot@self.orientation
        root,candidate,solve=self.mapper.solve(guide,self.last_guide,data,forces,self.last)
        delta=float(np.abs(candidate-self.last).max())
        fraction=min(1.,STEP/max(delta,1e-12))
        if fraction<.999 and (proposal!=self.parameter or shift!=self.shift):
            proposal=self.parameter+fraction*(proposal-self.parameter);shift=self.shift+fraction*(shift-self.shift)
            guide,locations=desired_curve(self.stage,proposal,shift,self.radius_delta,self.initial_field)
            wanted=guide[0]-guide[1];wanted/=np.linalg.norm(wanted)
            cross=np.cross(before,wanted);sine=np.linalg.norm(cross)
            orientation=Rotation.from_rotvec(cross*math.atan2(sine,float(np.clip(before@wanted,-1,1)))/max(sine,1e-12)).as_matrix()@self.orientation
            root,candidate,solve=self.mapper.solve(guide,self.last_guide,data,forces,self.last)
        self.mapper.d.qpos[:]=data.qpos;mujoco.mj_fwdPosition(self.mapper.m,self.mapper.d)
        _,_,actual_pd,actual_sd=self.mapper.collision_rows()
        contact_allowance=min(2.5,max(1.2,float(actual_pd.max()*1000)+.8))
        geometric_pass=solve['predicted_pipe_depth_mm']<=contact_allowance and solve['predicted_self_depth_mm']<=max(.8,float(actual_sd.max()*1000)+.4)
        if geometric_pass:
            target=candidate.copy()
            self.orientation=orientation
            self.reference_tangent=wanted.copy()
            self.last_guide=guide.copy()
            self.parameter=proposal;self.shift=shift;self.bad_candidates=0
        else:
            target=self.last.copy();self.bad_candidates+=1
        if t>8. and self.stage=='peel' and self.parameter<.02 and not support_ready:self.failure='initial_target_bearing_not_established'
        if self.bad_candidates*DT>=8.:self.failure='continuous_curve_chain_unavailable_8s'
        if self.stage=='peel' and self.parameter>=1.-1e-8:
            if len(exited)<4:self.failure='peel_reference_finished_without_four_actual_exits'
            else:
                actual_nodes=nodes(data);self.handoff_tangents=(actual_nodes[9:-1]-actual_nodes[10:])/LINK
                self.transition('unfold',t)
        elif self.stage=='unfold' and self.parameter>=1.-1e-8:
            if len(exited)<6:self.failure='unfold_reference_finished_without_full_source_exit'
            elif np.linalg.norm(actual[-1]-route.receiving([-12*LINK])[0])>.14:self.failure='unfold_reference_finished_without_hanging_tail'
            elif np.mean(np.abs((nodes(data)[9:-1]-nodes(data)[10:])[:,2])/LINK)<.80:self.failure='unfold_reference_finished_without_vertical_supply'
            else:self.transition('haul',t);self.feed_origin=actual.copy()
        if self.stage=='unfold':
            distance=float(np.linalg.norm(actual[-1]-route.receiving([-12*LINK])[0]))
            if self.hanging_best is None or distance<self.hanging_best-.005:self.hanging_best=distance;self.last_progress=t
        elif self.stage=='haul':
            stroke=int(self.shift/(2*LINK))
            expected=min(21,9+2*stroke)
            if stroke>=1 and not all(j in received for j in range(9,expected)):
                self.failure='two_rod_haul_reference_finished_without_retained_receivers'
            if t-self.last_progress>30. and self.shift>=2*LINK:self.failure='no_actual_haul_progress_30s'
        if self.stage!='haul' and t-self.last_progress>75. and self.parameter>.33:self.failure='no_actual_source_exit_or_hanging_progress_75s'
        self.last=target.copy()
        info=dict(stage=self.stage,shape_parameter=self.parameter,shift_mm=self.shift*1000,
            shift_speed_mm_s=(self.shift-old_shift)/DT*1000,curve_radius_delta_mm=self.radius_delta*1000,
            support_normal_n=contact_sum,support_ready=support_ready,tracking_deg=mismatch,
            rate_scale=scale,reference_points_mm=(guide*1000).tolist(),material_locations_mm=(locations*1000).tolist(),
            candidate_accepted=geometric_pass,maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),
            predicted_contact_allowance_mm=contact_allowance,
            actual_exited_modules=sorted(exited),actual_received_modules=sorted(received),failure=self.failure,mapper=solve)
        return target,info

    def transition(self,stage,t):
        self.events.append(dict(t_s=t,old_stage=self.stage,new_stage=stage,actual_exited=sorted(self.exited),actual_received=sorted(self.received)))
        self.stage=stage;self.parameter=0.;self.phase_start=t;self.last_progress=t
