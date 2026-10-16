"""Robot-only kinematics, source-stance drawing and fore gravity compensation.

No scene distance, receiving geom or private root pose reaches the physical data.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from physics import mujoco,nodes
from rough_planner import CurvatureCoordinates

class NativeRobot:
    def __init__(self,m,d):
        self.m=m;self.d=mujoco.MjData(m);self.head=m.body('head').id
        self.qa=m.jnt_qposadr[m.actuator_trnid[:,0]];self.va=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.initial_q=d.qpos[self.qa].copy();self.initial_ctrl=d.ctrl.copy();self.given=nodes(d).copy();self.coordinates=CurvatureCoordinates();self.observe(d,12)
    def observe(self,d,anchor):
        self.anchor=min(20,int(anchor));b=self.head+self.anchor
        self.state=d.qpos.copy();self.anchorP=d.xpos[b].copy();self.anchorR=d.xmat[b].reshape(3,3).copy();self.rootR=d.xmat[self.head].reshape(3,3).copy()
    def fk(self,q):
        d=self.d;m=self.m;d.qpos[:]=self.state;d.qpos[self.qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        b=self.head+self.anchor;R=self.anchorR@d.xmat[b].reshape(3,3).T;v=self.anchorP-R@d.xpos[b]
        d.qpos[:3]=R@self.state[:3]+v;d.qpos[3:7]=Rotation.from_matrix(R@self.rootR).as_quat()[[3,0,1,2]]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return nodes(d).copy()
    def gravity_torque(self,q,n):
        self.fk(q);out=np.zeros(20)
        for j in range(n):
            jid=self.m.actuator_trnid[j,0];moment=np.zeros(3)
            for i in range(j+1):moment+=np.cross(self.d.xipos[self.head+i]-self.d.xanchor[jid],self.m.body_mass[self.head+i]*self.m.opt.gravity)
            out[j]=self.d.xaxis[jid]@moment
        return out
