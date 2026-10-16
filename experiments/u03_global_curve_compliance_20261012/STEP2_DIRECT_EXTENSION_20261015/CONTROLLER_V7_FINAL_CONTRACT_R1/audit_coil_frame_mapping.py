"""Necessary native mirror-frame identity check; no contacts or integration."""
import json
from scipy.spatial.transform import Rotation
from physics import *
from geometry import Geometry
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);cal=np.load(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.npz');qa=m.jnt_qposadr[m.actuator_trnid[:,0]];g=Geometry([1.1,.2,1],[0,1,0],.0565);e,b=g.frame();Rmap=np.column_stack([g.axis,e,b]);J=np.diag([1.,-1.,1.]);origin=np.array([cal['points'][0,0],0,1.]);goal=g.entry+.4*g.axis+(cal['points']-origin)@Rmap.T;headR=Rmap@cal['body_xmat'][0]@J;headP=g.entry+.4*g.axis+Rmap@(cal['body_xpos'][0]-origin)
records=[]
for parity in [0,1]:
 q=cal['joint_curvature_integrals'].copy();q[parity::2]*=-1;d.qpos[:3]=headP;d.qpos[3:7]=Rotation.from_matrix(headR).as_quat()[[3,0,1,2]];d.qpos[qa]=q;mujoco.mj_kinematics(m,d);p=nodes(d);c=.5*(p[:-1]+p[1:]);records.append(dict(flipped_zero_based_parity=parity,maximum_node_identity_error_mm=float(np.linalg.norm(p-goal,axis=1).max()*1000),head_error_mm=float(np.linalg.norm(p[0]-goal[0])*1000),front6_span_deg=float(np.ptp(np.unwrap(np.arctan2(c[:6,2]-1,c[:6,0]-1.1)))*180/np.pi)))
out=dict(simulation_steps=0,map_determinant=float(np.linalg.det(Rmap)),proper_head_frame_determinant=float(np.linalg.det(headR)),records=records);(BASE/'COIL_FRAME_MAPPING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
