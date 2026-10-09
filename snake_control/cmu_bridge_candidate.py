"""Offline whole-prefix sine family for the single C03 adaptation candidate.

Seven target-wave rods + five bridge rods + nine retained source rods.
Only eight global parameters; no independent per-hinge inverse fit.
"""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from snake_control.cmu_window_gait import wave,placed_pose,CMUWindowControl
from snake_control.suspended_supply_transfer import nodes,LINK

class BridgeCandidate:
    bend=12;front=7
    def __init__(self,model,A,w,source_phase):
        self.m=model;self.A=A;self.w=w;self.ph=source_phase;self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]];self.head=model.body('head').id
        self.initial=mujoco.MjData(model);self.d=mujoco.MjData(model);self.robot_bodies=np.arange(self.head,self.head+21)
        self.source={model.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']};self.target={model.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']}
        self.basepose,self.basepoints,_=placed_pose(model,A,w,source_phase,self.bend,0.)
        self.mapper=CMUWindowControl(model,self.initial,A,w,source_phase,self.bend)
    def start(self,x):
        # Global source exit placement and polar angle only; before dynamics.
        roll=Rotation.from_rotvec([x[5],0,0]).as_matrix();q=self.basepose.copy();q[:3]=roll@(q[:3]-[0,0,1])+[x[4],0,1];q[3:7]=Rotation.from_matrix(roll@Rotation.from_quat(q[3:7][[1,2,3,0]]).as_matrix()).as_quat()[[3,0,1,2]]
        self.initial.qpos[:]=q;mujoco.mj_fwdPosition(self.m,self.initial);return q.copy()
    def joint_wave(self,x,s):
        # x: head offset, bridge A, bridge spatial, bridge phase, source X,
        # source polar angle, head mid-loosening, bridge mid-loosening.
        h=self.front;b=self.bend;k=np.arange(20);loose=np.sin(np.pi*s)
        q=wave(self.A,self.w,self.ph);q[:h]=wave(self.A*(1-x[6]*loose),self.w,self.ph+x[0]*s)[:h]
        B=((1-s)*self.A+s*x[1])*(1-x[7]*loose);wb=(1-s)*self.w+s*x[2];pb=(1-s)*(self.ph+self.w*h)+s*x[3]
        q[h:b]=B*np.sin(wb*(k[h:b]-h)+pb+(k[h:b]%2)*np.pi/2)
        return q
    def pose(self,x,s,contact=True):
        q=self.joint_wave(x,s);p=self.mapper.forward_reference(self.initial,q,self.bend);self.d.qpos[:]=self.mapper.fk.qpos;mujoco.mj_fwdPosition(self.m,self.d)
        c=(p[:-1]+p[1:])/2;pd=np.zeros(21);sd=np.zeros(21);target=source=0
        if contact:
            for ct in self.d.contact:
                b1,b2=map(int,self.m.geom_bodyid[[ct.geom1,ct.geom2]]);one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
                if one!=two and ({ct.geom1,ct.geom2}&(self.source|self.target)):
                    j=(b1 if one else b2)-self.head;pd[j]=max(pd[j],-ct.dist)
                    if ct.dist<=0:target+=bool({ct.geom1,ct.geom2}&self.target);source+=bool({ct.geom1,ct.geom2}&self.source)
                if one and two and abs(b1-b2)>1:sd[[b1-self.head,b2-self.head]]=np.maximum(sd[[b1-self.head,b2-self.head]],-ct.dist)
        # Gravity moments of the released prefix about its real hinge axes.
        tau=[]
        for j in range(1,self.bend+1):
            jt=self.m.joint('joint'+str(j)).id;moment=np.cross(c[:j]-p[j],self.m.opt.gravity*.136).sum(0);tau.append(float(self.d.xaxis[jt]@moment))
        return dict(q=q,qpos=self.d.qpos.copy(),points=p,centers=c,pipe_depth=pd,self_depth=sd,target_contact_count=target,source_contact_count=source,unsupported_gravity_torque_nm=np.array(tau))
    def residual(self,x):
        self.start(x);out=[];h=self.front
        for s in [0.,.25,.5,.75,1.]:
            r=self.pose(x,s);p=r['points'];c=r['centers'];out.extend(np.maximum(r['pipe_depth']-.0012,0)/.003);out.extend(np.maximum(r['self_depth']-.0005,0)/.003)
            if s==0:
                # Whole initial source and no target preload.
                out.extend(np.maximum(c[:,0]-1.085,0)/.008);out.append(r['target_contact_count']*2.)
            if s==1:
                cc=c[:h];rr=np.linalg.norm(cc[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(cc[:,2]-1,cc[:,0]-1.1));dy=np.diff(cc[:,1]);dp=np.diff(phi)
                out.extend((rr-.088)/.003);out.extend((dy+.18/(2*np.pi)*dp)/.003);out.extend(np.maximum(.3-dp,0)/.1);out.extend(np.maximum(dy+.002,0)/.01)
                entry=np.array([1.1+.088*np.cos(5*np.pi/4),.18,1+.088*np.sin(5*np.pi/4)]);out.extend((cc[-1]-entry)/.01);out.extend(np.maximum(.15-cc[:,1],0)/.01)
            if s==.5:
                # The head must reach the lower inner side before wrapping up.
                out.extend(np.maximum([p[0,0]-1.065,.10-p[0,1],p[0,2]-.955,.90-p[0,2]],0)/.02)
        return np.asarray(out)
