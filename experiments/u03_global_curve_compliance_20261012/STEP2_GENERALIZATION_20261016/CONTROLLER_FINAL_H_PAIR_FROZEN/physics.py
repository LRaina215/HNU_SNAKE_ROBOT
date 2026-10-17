from pathlib import Path
import numpy as np,mujoco
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'mujoco_model').is_dir() and (p/'snake_control').is_dir())
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
FRAME=np.eye(3);ORIGIN=np.array([0.,0.,1.])
def set_control_frame(matrix,origin):
    global FRAME,ORIGIN
    FRAME=np.asarray(matrix,float);ORIGIN=np.asarray(origin,float)
    assert np.max(abs(FRAME.T@FRAME-np.eye(3)))<1e-10 and np.linalg.det(FRAME)>.99999
def to_local_points(p):return ORIGIN+(np.asarray(p)-ORIGIN)@FRAME
def to_world_points(p):return ORIGIN+(np.asarray(p)-ORIGIN)@FRAME.T
def to_local_vectors(v):return np.asarray(v)@FRAME
def to_world_vectors(v):return np.asarray(v)@FRAME.T
def world_nodes(d):return np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
def nodes(d):return to_local_points(world_nodes(d))
def world_info(info):
    out=info.copy()
    for k in ['reference_points','nominal_curve_points','nominal_curve_dense','geometric_reference_points','global_reference_points','global_reference_dense','native_exact_C_points','estimated_entry_m','applied_global_entry_m']:
        if k in out:out[k]=to_world_points(out[k])
    for k in ['estimated_axis','applied_global_axis']:
        if k in out:out[k]=to_world_vectors(out[k])
    out['reference_frame']=FRAME@out['reference_frame']
    return out
