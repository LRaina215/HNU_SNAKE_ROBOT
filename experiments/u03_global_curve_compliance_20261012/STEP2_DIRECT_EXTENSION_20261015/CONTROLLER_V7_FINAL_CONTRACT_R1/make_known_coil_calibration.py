"""Freeze an immutable prescribed source-coil shape, not any action snapshot."""
import json
from physics import *
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(BASE.parent/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);p=nodes(d).copy();c=.5*(p[:-1]+p[1:]);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,1]));qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
head=m.body('head').id
np.savez_compressed(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.npz',points=p,body_xpos=d.xpos[head:head+21].copy(),body_xmat=d.xmat[head:head+21].reshape(21,3,3).copy(),joint_curvature_integrals=d.qpos[qa].copy(),source_axis=np.array([1.,0.,0.]),source_axis_point=np.array([0.,0.,1.]),source_pipe_radius=.0565)
report=dict(role='Immutable prescribed source geometry and native-model coiling calibration; no live state or target information',simulation_steps=0,source='Reviewed static given source setup',front6_source_span_deg=float(np.ptp(phi[:6])*180/np.pi),front6_center_radii_m=np.linalg.norm(c[:6,1:]-[0,1],axis=1).tolist(),material_chord_error_mm=float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-.0645).max()*1000))
(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
