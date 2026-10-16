"""Prescribed source boundary hinge's outward departure directions; no stepping."""
import json
from physics import *
from scipy.spatial.transform import Rotation
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);p=nodes(d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
for n in [11,12,13,14]:
    T=(p[n-1]-p[n])/.0645;rad=np.r_[0.,p[n,1],p[n,2]-1];rad/=np.linalg.norm(rad);axis=d.xaxis[m.actuator_trnid[n-1,0]]
    changes=np.linspace(-np.radians(88)-d.qpos[qa[n-1]],np.radians(88)-d.qpos[qa[n-1]],501);directions=Rotation.from_rotvec(-changes[:,None]*axis).apply(T)
    score=directions@rad-.10*abs(directions[:,0]-.3);k=int(np.argmax(score))
    print(json.dumps(dict(held_start=n,initial_T=T.tolist(),axis=axis.tolist(),source_outward_normal=rad.tolist(),best_T=directions[k].tolist(),best_joint_deg=float((d.qpos[qa[n-1]]+changes[k])*180/np.pi),outward_dot=float(directions[k]@rad))))
