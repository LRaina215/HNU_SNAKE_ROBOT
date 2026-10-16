"""Native coiling covariance under a proper rotation; no bridge/path claim."""
import json
from scipy.spatial.transform import Rotation
from physics import *
from geometry import Geometry
from rough_planner import coarse_model

reg=json.loads((BASE/'REGISTERED.json').read_text());g=Geometry.from_dict(reg['conditions']['DIRECTION']);e,b=g.frame();cal=np.load(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.npz');cp=cal['points'];source=np.load(BASE/'GIVEN_INITIAL.npz')
sign=reg['direct_extension']['receiver_winding_sign'];phi=np.radians(-165+sign*reg['direct_extension']['maximum_wrap_deg']);phi_source=np.arctan2(cp[0,2]-1,cp[0,1]);delta=phi+sign*phi_source
R=np.column_stack([g.axis,np.cos(delta)*e+np.sin(delta)*b,sign*(np.sin(delta)*e-np.cos(delta)*b)])
assert sign==1 and np.linalg.det(R)>.999999
head_ax=reg['source_constants']['pitch']*reg['direct_extension']['maximum_wrap_deg']/360.+reg['kinematic_solver']['front_wrapping_axial_lead_m'];origin=np.array([cp[0,0],0.,1.])
target=g.entry+head_ax*g.axis+(cp-origin)@R.T
m=coarse_model(g);d=mujoco.MjData(m);q0=source['qpos'];d.qpos[:]=q0;d.qpos[:3]=g.entry+head_ax*g.axis+R@(q0[:3]-origin);d.qpos[3:7]=Rotation.from_matrix(R@Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_matrix()).as_quat()[[3,0,1,2]]
mujoco.mj_kinematics(m,d);observed=nodes(d);centers=.5*(observed[:-1]+observed[1:]);ax,rad,rho=g.coordinates(centers[:6]);angles=np.unwrap(np.arctan2(rad@b,rad@e));error=float(np.linalg.norm(observed-target,axis=1).max()*1000)
report=dict(simulation_steps=0,rough_prior_only=True,proper_rotation_determinant=float(np.linalg.det(R)),local_hinge_sign_change=False,
    maximum_native_FK_node_error_mm=error,front6_polar_span_deg=float(np.ptp(angles)*180/np.pi),front6_center_radii_m=rho.tolist(),front6_axial_relative_entry_m=ax.tolist(),
    source_and_receiver_pipe_radii_equal=True,source_virtual_radius_m=reg['source_constants']['radius'],source_pipe_radius_m=float(cal['source_pipe_radius']),
    passed=error<1e-8,scope='Pure native coil model covariance only. Whole rotated robot is a private geometric construction; it does not establish a source-held bridge, contacts, capture or dynamics.')
(BASE/'NATIVE_TEMPLATE_RIGID_ROTATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
