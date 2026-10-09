"""A growing horizontal-pipe helix, driven by a common material roll phase."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from snake_control.suspended_transfer_route import receiving,ALPHA,BETA,R,INTAKE
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,STEP,nodes,skew
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone

def backbone(shift):
    material=9*LINK+shift-np.arange(22)*LINK
    return receiving(material),material

class HingePhaseMapper:
    """Project a common roll phase through all twenty physical hinge axes."""
    def __init__(self,model,data):
        self.m=model;self.d=mujoco.MjData(model);self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]];self.va=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.head=model.body('head').id;self.rotation=Rotation.from_quat(data.qpos[3:7][[1,2,3,0]]).as_matrix()
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv));self.root=data.qpos[:3].copy();self.tilt=np.zeros(3)

    def solve(self,guide,shift,phase,previous,rate_bound=True,passive_rear=False,active_prefix=9,observed_q=None,node_weights=None,orientation_override=None,command_step=None):
        screw=Rotation.from_rotvec([0.,BETA*shift/R,0.]).as_matrix()
        orientation=screw@self.rotation@Rotation.from_rotvec([0.,0.,-phase]).as_matrix()
        if orientation_override is not None:orientation=np.asarray(orientation_override)
        # The root is private FK geometry only; actual data is never assigned.
        step=STEP if command_step is None else command_step
        qlo=np.maximum(-LIMIT,previous-step) if rate_bound else np.full(20,-LIMIT)
        qhi=np.minimum(LIMIT,previous+step) if rate_bound else np.full(20,LIMIT)
        if passive_rear and observed_q is not None:
            count=min(20,active_prefix+3);held=np.clip(observed_q[count:],-LIMIT+1e-7,LIMIT-1e-7)
            qlo[count:]=held-1e-8;qhi[count:]=held+1e-8
        guess=np.r_[self.root,self.tilt,np.clip(previous,qlo,qhi)]
        lo=np.r_[np.full(3,-4.),[-.5,-.5,-.07],qlo];hi=np.r_[np.full(3,4.),[.5,.5,.07],qhi]
        weights=np.ones(22);weights[:10]=3.
        if passive_rear:
            weights[:active_prefix+1]=3.;weights[active_prefix+1:active_prefix+5]=.5;weights[active_prefix+5:]=0.
        if node_weights is not None:weights=np.asarray(node_weights)
        cache={}
        def evaluate(x):
            if 'x' in cache and np.array_equal(x,cache['x']):return cache
            rotation=orientation@Rotation.from_rotvec(x[3:6]).as_matrix()
            self.d.qpos[:3]=x[:3];self.d.qpos[3:7]=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]];self.d.qpos[self.qa]=x[6:];mujoco.mj_fwdPosition(self.m,self.d)
            theta=np.linalg.norm(x[3:6]);cross=skew(x[3:6])
            left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-12)*cross+(theta-np.sin(theta))/max(theta**3,1e-12)*(cross@cross)
            def point_jac(point,body):
                mujoco.mj_jac(self.m,self.d,self.jp,self.jr,point,body)
                return np.column_stack([np.eye(3),-skew(point-x[:3])@orientation@left,self.jp[:,self.va]])
            p=nodes(self.d);jac=[]
            for i,point in enumerate(p):
                jac.append(point_jac(point,self.head+min(i,20)))
            penalty=np.zeros(42);penjac=np.zeros((42,26));depth=0.
            for con in self.d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]]);one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                if one!=two:
                    pipe=con.geom2 if one else con.geom1
                    if not (self.m.geom(pipe).name or '').startswith('pipe_'):continue
                    body=b1 if one else b2;index=body-self.head;value=max(0.,-con.dist-.0006);depth=max(depth,-con.dist)
                    if value>penalty[index]:
                        penalty[index]=value;penjac[index]=(1 if one else -1)*con.frame[:3]@point_jac(con.pos,body)
                elif one and two and abs(b1-b2)>1:
                    index=21+min(b1,b2)-self.head;value=max(0.,-con.dist-.0002)
                    if value>penalty[index]:
                        penalty[index]=value;a=point_jac(con.pos,b1);penjac[index]=con.frame[:3]@(a-point_jac(con.pos,b2))
            regular=np.column_stack([np.zeros((20,6)),.015*np.eye(20)])
            cache.clear();cache.update(x=x.copy(),residual=np.r_[(weights[:,None]*(p-guide)/.012).ravel(),penalty/.0004,.015*(x[6:]-previous)],
                jac=np.vstack([(weights[:,None,None]*np.asarray(jac)/.012).reshape(66,26),penjac/.0004,regular]),
                points=p.copy(),depth=float(depth))
            return cache
        result=least_squares(lambda x:evaluate(x)['residual'],guess,jac=lambda x:evaluate(x)['jac'],bounds=(lo,hi),max_nfev=20 if rate_bound else 100,ftol=1e-6,xtol=1e-6,gtol=1e-5)
        a=evaluate(result.x);self.root=result.x[:3].copy();self.tilt=result.x[3:6].copy();p=a['points']
        return result.x[6:].copy(),dict(front_fit_rms_mm=float(np.sqrt(np.mean(np.sum((p[:10]-guide[:10])**2,axis=1)))*1000),
            whole_fit_rms_mm=float(np.sqrt(np.mean(np.sum((p-guide)**2,axis=1)))*1000),predicted_pipe_depth_mm=a['depth']*1000,
            phase_rad=phase,shift_mm=shift*1000,private_root=self.root.copy(),private_head_tilt_rad=self.tilt.copy(),physical_root_assigned=False,evaluations=int(result.nfev),points_mm=p*1000)

class HorizontalPipeRoll:
    def __init__(self,model,data,phase_sign=1.,speed=.0025,front_passive=False,true_hinge=False,connected_intake=False):
        self.m=model;self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]];self.va=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.last=data.ctrl.copy();self.shift=0.;self.phase=0.;self.sign=phase_sign;self.speed=speed
        self.initial=data.qpos[self.qa].copy();self.kp=np.full(20,127.);self.received_age=np.zeros(21);self.received=set()
        p,_=backbone(0);bone=SpatialBishopBackbone(p*1000,np.full(21,LINK*1000))
        self.nominal0=bone.joint_angles();self.residual=self.initial-self.nominal0
        self.phi0=np.arctan2((.5*(nodes(data)[:-1]+nodes(data)[1:]))[:,2]-1.,(.5*(nodes(data)[:-1]+nodes(data)[1:]))[:,0]-1.1)
        self.phi=self.phi0.copy();self.previous_raw=self.phi0.copy();self.engaged=set();self.best_received=0;self.last_progress=0.;self.failure=None
        self.front_passive=front_passive
        self.true_hinge=true_hinge;self.mapper=HingePhaseMapper(model,data) if true_hinge else None
        self.connected_intake=connected_intake

    def command(self,data,t,forces):
        p=nodes(data);c=.5*(p[:-1]+p[1:]);rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        raw=np.arctan2(c[:,2]-1,c[:,0]-1.1);self.phi+=np.arctan2(np.sin(raw-self.previous_raw),np.cos(raw-self.previous_raw));self.previous_raw=raw
        winding=self.phi0-self.phi
        near=(rr<.12)&(rr>.065)&(c[:,1]>.15)
        spatial_phi=np.unwrap(raw);da=np.diff(spatial_phi);dy=np.diff(c[:,1])
        connection=np.zeros(21,dtype=bool);connection[:9]=near[:9]
        connection[1:]=(near[1:]&near[:-1]&(da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)&(np.abs(dy+.180/(2*np.pi)*da)<.025))
        condition=connection if self.connected_intake else near
        self.received_age=np.where(condition,self.received_age+DT,0.)
        self.received={j for j in range(9,21) if self.received_age[j]>=1. and (self.connected_intake or winding[j]>.35)}
        if len(self.received)>self.best_received:self.best_received=len(self.received);self.last_progress=t
        self.engaged.update(self.received)
        prefix=9
        while prefix in self.engaged:prefix+=1
        error=float(np.degrees(abs(self.last[:min(20,prefix+3)]-data.qpos[self.qa][:min(20,prefix+3)])).max()) if self.front_passive else float(np.degrees(abs(self.last-data.qpos[self.qa])).max())
        scale=float(np.clip((20.-error)/15.,.2,1.))
        increment=self.speed*DT*scale;self.shift=min(12*LINK,self.shift+increment)
        # One phase controls every active curvature component. Discrete hinge
        # targets are produced from the growing route, not absolute world fitting.
        self.phase+=self.sign*increment/(.029/BETA)
        guide,material=backbone(self.shift)
        bone=SpatialBishopBackbone(guide*1000,np.full(21,LINK*1000))
        q=bone.joint_angles(self.phase)
        # Preserve initial reconstruction discrepancy only in the already
        # wrapped prefix; taper it as material advances one complete turn.
        fade=max(0.,1-self.shift/(2*np.pi*R/BETA))
        q[:8]+=fade*self.residual[:8]*np.cos(self.phase)
        q=np.clip(q,-LIMIT,LIMIT)
        projection=None
        if self.true_hinge:q,projection=self.mapper.solve(guide,self.shift,self.phase,self.last,passive_rear=self.front_passive,active_prefix=prefix,observed_q=data.qpos[self.qa])
        if self.front_passive:
            kp=np.zeros(20);kp[:min(20,prefix-1)]=127.
            kp[max(8,prefix-1):min(20,prefix+3)]=63.5
            ramp=np.clip(t/1.5,0,1);kp=(1-ramp)*127+ramp*kp
            self.kp+=np.clip(kp-self.kp,-127*DT/1.5,127*DT/1.5)
            q[self.kp<.5]=np.clip(data.qpos[self.qa][self.kp<.5],-LIMIT,LIMIT)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        target=self.last+np.clip(q-self.last,-STEP,STEP);self.last=target.copy()
        if self.shift>=2*LINK and not self.received and t>90:self.failure='two_rod_action_without_retained_intake'
        if t-self.last_progress>90 and self.shift>=3*LINK:self.failure='no_new_retained_intake_90s'
        info=dict(stage='horizontal_pipe_material_roll',shape_parameter=self.phase,shift_mm=self.shift*1000,tracking_deg=error,
            curve_radius_delta_mm=0.,reference_points_mm=guide*1000,kp_nm_per_rad=self.kp.copy(),active_joint_mask=self.kp>.5,
            region_codes=np.where(self.kp<.5,2,np.where(self.kp>126,0,1)),actual_new_winding_rad=winding,
            actual_new_retained_modules=sorted(self.received),material_coordinate_mm=material*1000,phase_rad=self.phase,
            phase_sign=self.sign,desired_joint_rad=q.copy(),nominal_bishop_joint_rad=bone.joint_angles(),
            true_twenty_hinge_projection=self.true_hinge,projection=projection,
            connected_intake=self.connected_intake,current_spatial_connection_mask=connection,
            maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),failure=self.failure)
        return target,info
