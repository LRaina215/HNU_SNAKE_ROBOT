"""Exact original alternating-hinge projection of geometric guide, no actual feedback."""
from physics import *
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class CurveProjector:
    def __init__(self,m,initial_qpos,initial_reference_frame):
        self.m=m;self.d=mujoco.MjData(m);self.d.qpos[:]=initial_qpos
        self.qa=m.jnt_qposadr[m.actuator_trnid[:,0]];self.va=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.initial_rotation=Rotation.from_quat(initial_qpos[3:7][[1,2,3,0]]).as_matrix();self.initial_frame=initial_reference_frame.copy();self.last=initial_qpos[self.qa].copy()
        self.joints=[m.joint('joint'+str(i)).id for i in range(1,21)]

    def command(self,guide,frame):
        R=frame@self.initial_frame.T@self.initial_rotation;self.d.qpos[:3]=guide[0]-R@self.m.site('s_head').pos;self.d.qpos[3:7]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        previous=self.last.copy();cache={}
        def evaluate(q):
            key=q.tobytes()
            if key in cache:return cache[key]
            self.d.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,self.d);p=nodes(self.d);jac=[]
            for point,body in zip(p,[self.m.site('s_head').bodyid,*[self.m.jnt_bodyid[j] for j in self.joints],self.m.site('s_tail').bodyid]):
                jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(self.m,self.d,jp,jr,point,int(body));jac.append(jp[:,self.va])
            residual=np.r_[((p-guide)/.0645).ravel(),.025*(q-previous)]
            derivative=np.vstack([np.asarray(jac).reshape(-1,20)/.0645,.025*np.eye(20)])
            cache[key]=(residual,derivative,p);return cache[key]
        result=least_squares(lambda q:evaluate(q)[0],previous,jac=lambda q:evaluate(q)[1],bounds=(-np.radians(88),np.radians(88)),max_nfev=10,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        self.last=result.x.copy();p=evaluate(result.x)[2]
        return result.x,dict(projected_points=p.copy(),projection_rms_m=float(np.sqrt(np.mean(np.sum((p-guide)**2,axis=1)))),projection_max_m=float(np.max(np.linalg.norm(p-guide,axis=1))),projection_iterations=int(result.nfev),virtual_root_qpos=self.d.qpos[:7].copy())
