"""Feedback-driven joint-only alternating transfer; SI internally.

Simulation pose/contact inputs are explicit. The free-root columns in the
kinematic solve are predictions, never commands. Existing MuJoCo drives remain
unchanged. Hardware use requires replacing these observations.
"""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from snake_control.three_segment_compliance import LocalSection
from snake_control.pole_compliance import ComplianceParameters

DT = .05
class AlternatingHelixTransfer:
    def __init__(self, model, data, variant=1):
        self.m=model; self.d=mujoco.MjData(model); self.head=model.body('head').id
        self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.va=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.last=data.ctrl.copy();self.preload=data.ctrl-data.qpos[self.qa]
        self.phase='form_front';self.phase_start=0.;self.events=[];self.variant=variant
        self.front=7;self.rear=11;self.cycle=0;self.mark=None;self.history=[];self.reference=None
        self.config=dict(joint_count=20,head_length_mm=64.5,joints=[dict(joint_index=i+1,axis='Yaw' if i%2==0 else 'Pitch',link_length_mm=64.5) for i in range(20)])
    def observe(self,data):
        self.d.qpos[:]=data.qpos; mujoco.mj_fwdPosition(self.m,self.d)
        p=np.vstack([self.d.site('s_head').xpos,*[self.d.joint('joint'+str(i)).xanchor for i in range(1,21)],self.d.site('s_tail').xpos])
        return p,(p[:-1]+p[1:])/2
    def command(self,data,t,forces):
        p,c=self.observe(data);q=data.qpos[self.qa];n=self.m.nv
        if self.reference is None:self.reference=c.copy()
        # Ordered material boundary follows actual position, never scheduled IDs.
        good=(c[:,1]>.085)&(np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)<.125)
        front=0
        while front<20 and good[front+1]:front+=1
        self.front=max(self.front,front) if self.variant>=4 else max(5,front)
        sources=np.flatnonzero((np.arange(21)>self.front)&(c[:,0]<1.02)&(np.linalg.norm(c[:,[1,2]]-[0,1.],axis=1)<.14))
        self.rear=max(self.rear,int(sources[0])) if len(sources) and self.variant>=4 else int(sources[0]) if len(sources) else 21
        if self.mark is None:self.mark=c.copy()
        elapsed=t-self.phase_start
        progress=float((c[:self.front+1,1]-self.mark[:self.front+1,1]).mean())
        rear_progress=float((c[self.rear:,0]-self.mark[self.rear:,0]).mean()) if self.rear<21 else 1.
        self.history.append((t,float(c[:6,1].mean())))
        while self.history and t-self.history[0][0]>2:self.history.pop(0)
        rate=(self.history[-1][1]-self.history[0][1])/max(.05,t-self.history[0][0])
        reason=None
        if self.phase=='form_front' and elapsed>=12:reason='initial helix formation interval complete'
        elif self.phase=='front_roll' and elapsed>=3 and (rate<.0002 or elapsed>=10):reason='front blocked' if rate<.0002 else 'front increment complete'
        elif self.phase=='rear_feed' and elapsed>=8:reason='rear reshape/feed interval complete'
        if reason:
            old=self.phase;self.phase='front_roll' if old!='front_roll' else 'rear_feed'
            self.events.append(dict(t_s=t,old=old,new=self.phase,reason=reason,front_progress_mm=progress*1000,rear_progress_mm=rear_progress*1000))
            self.phase_start=t;self.mark=c.copy();elapsed=0
            if self.phase=='front_roll':self.cycle+=1
        shape_velocity=np.zeros_like(c)
        if self.variant>=3:
            # Fit a regular helix to each actual cloud. This removes the local
            # kinks observed in A02, rather than preserving every kink forever.
            for ids,radial_dims,axial_dim,origin,radius in [
                (np.arange(self.front+1),[0,2],1,np.array([1.1,1.]),.091),
                (np.arange(self.rear,21),[1,2],0,np.array([0.,1.]),.087)]:
                if len(ids)<3:continue
                uv=c[ids][:,radial_dims]-origin
                theta=np.unwrap(np.arctan2(uv[:,1],uv[:,0]))
                winding=np.sign(np.mean(np.diff(theta)))
                step=winding*2*np.arcsin(.0645*np.sqrt(1-.32**2)/(2*radius))
                phase=float(np.mean(theta-step*np.arange(len(ids))))
                angles=phase+step*np.arange(len(ids))
                axial=c[ids,axial_dim].mean()+.32*.0645*((len(ids)-1)/2-np.arange(len(ids)))
                want=c[ids].copy();want[:,radial_dims]=origin+radius*np.column_stack([np.cos(angles),np.sin(angles)])
                want[:,axial_dim]=axial
                shape_velocity[ids]=np.clip(.18*(want-c[ids]),-.004,.004)
        A=[];b=[];jp=np.zeros((3,n));jr=np.zeros((3,n))
        def add(j,v,w):A.append(w*j);b.append(w*np.atleast_1d(v))
        for k,center in enumerate(c):
            mujoco.mj_jac(self.m,self.d,jp,jr,center,self.head+k)
            u=self.d.xmat[self.head+k].reshape(3,3)[:,2]
            side='target' if k<=self.front else 'source' if k>=self.rear else 'middle'
            if side!='middle':
                axis=np.array([0.,1.,0.]) if side=='target' else np.array([1.,0.,0.])
                origin=np.array([1.1,0.,1.]) if side=='target' else np.array([0.,0.,1.])
                radial=center-origin;radial-=axis*(radial@axis);radius=np.linalg.norm(radial);normal=radial/radius
                active=(side=='target' and self.phase in ('form_front','front_roll')) or (side=='source' and self.phase=='rear_feed')
                roll=.075 if active else (.035 if side=='target' and self.variant>=2 else .006)
                sign=(-1. if side=='target' else 1.) if self.variant>=4 else np.sign(np.cross(u,normal)@axis)
                omega=sign*roll
                velocity=.029*omega*np.cross(u,normal)
                orbital=(velocity@np.cross(axis,normal))/radius
                angular=omega*u+orbital*axis
                # Gradually open axial pitch at the front without world anchoring.
                if side=='target' and (self.phase=='form_front' or self.variant>=2):
                    desired=c[self.front,1]+.32*.0645*(self.front-k)
                    velocity+=axis*np.clip((desired-center[1])*.10,-.0015,.003)
                if self.variant>=2:
                    # Restore radial grasp and retain achieved axial position.
                    velocity+=normal*np.clip(((.087 if side=='source' and self.variant>=3 else .091)-radius)*.25,-.002,.002)
                    if self.variant>=3:velocity+=shape_velocity[k]
                    coordinate=1 if side=='target' else 0
                    self.reference[k,coordinate]=max(self.reference[k,coordinate],center[coordinate])
                    velocity+=axis*np.clip((self.reference[k,coordinate]-center[coordinate])*.5,0,.004)
                add(jp,velocity,2. if self.variant>=2 else 1.)
                if self.variant>=5:
                    # Rolling constrains material spin only. Bending rates must
                    # remain available to reform a helix and feed the connector.
                    add(u@jr,float(u@angular),.06)
                else:add(jr,angular,.025 if self.variant>=4 and k>=19 else .06)
                add(normal@jp,normal@velocity,2.)
            else:
                # Connector carries shape change; no fixed world pose.
                if self.variant>=2:
                    tangent=p[k]-p[k+1];tangent/=np.linalg.norm(tangent)
                    add(jp,.003*tangent,.8)
                else:add(jp,np.zeros(3),.08)
        # Joint regularization, soft recentering at stops, bounded velocity.
        J=np.zeros((20,n));J[np.arange(20),self.va]=1
        limit_push=-.3*np.sign(q)*np.maximum(abs(q)-np.radians(78),0)
        add(J,limit_push,.04 if self.variant>=3 else .012)
        aa=np.vstack(A);bb=np.concatenate(b)
        lo=np.full(n,-1.);hi=np.full(n,1.)
        lo[self.va]=np.maximum(-np.radians(10),(np.radians(-88)-q)/DT)
        hi[self.va]=np.minimum(np.radians(10),(np.radians(88)-q)/DT)
        dq=lsq_linear(aa,bb,bounds=(lo,hi),tol=1e-6,max_iter=40).x[self.va]
        delta=dq*DT
        compliance=np.zeros(20);compliance_status='inactive'
        if self.phase=='rear_feed' and self.rear<17:
            start=self.rear;stop=20
            geometry=dict(config=self.config,points_mm=p*1000,curve=dict(gamma_min_rad=np.radians(1)))
            pitch=float(np.arcsin(np.clip(abs((p[-1]-p[start])[0])/np.sum(np.linalg.norm(np.diff(p[start:],axis=0),axis=1)),.1,.6)))
            try:
                zero=LocalSection(geometry,ComplianceParameters(0.,pitch),start,stop,np.array([1.,0,0]),np.array([0.,0,1000.]))
                shaped=LocalSection(geometry,ComplianceParameters(.01,pitch),start,stop,np.array([1.,0,0]),np.array([0.,0,1000.]))
                gauge=zero.align(q[start:stop],0)
                correction=shaped.angles(gauge)-zero.angles(gauge)
                compliance[start:stop]=np.clip(correction,-np.radians(2),np.radians(2))*.05
                compliance_status='A-D local section'
            except (ValueError,FloatingPointError) as exc:compliance_status=str(exc)
        raw=self.last+delta+compliance
        if self.variant>=4:
            # A03 J11/J19/J20 remained at stops: actively unload tracking
            # effort there, leaving the physical servo gain unchanged.
            near=abs(q)>np.radians(82)
            raw[near]-=DT*.6*(self.last[near]-q[near])
            raw[near]-=DT*.08*np.sign(q[near])*(abs(q[near])-np.radians(78))
        # Retain the saved entry clamp, but do not accumulate unlimited tracking error.
        budget=np.maximum(abs(self.preload),np.radians(3))
        raw=np.clip(raw,q-budget,q+budget)
        target=np.clip(raw,np.maximum(-np.radians(88),self.last-np.radians(10)*DT),np.minimum(np.radians(88),self.last+np.radians(10)*DT))
        if t==0:target=self.last.copy()
        self.last=target.copy()
        return target,dict(stage=self.phase,front_last_module=self.front,rear_first_module=self.rear,front_speed_mm_s=rate*1000,cycle=self.cycle,compliance=compliance.tolist(),compliance_status=compliance_status,requested_joint_velocity=dq.tolist())
