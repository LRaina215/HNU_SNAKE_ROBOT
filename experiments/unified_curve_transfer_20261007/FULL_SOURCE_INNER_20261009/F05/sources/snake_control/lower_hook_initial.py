"""Construct a lower-entry hooked pose before dynamics, in unchanged gravity.

This is an artificial initial placement, not an active hook acquisition.
World Z reflection is realized by a proper head rotation and real hinge signs.
No reflected trajectory or reflected gravity is used for subsequent dynamics.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from snake_control.suspended_supply_transfer import nodes


def construct_lower_entry(model,data):
    old_points=nodes(data).copy();old_qpos=data.qpos.copy();old_ctrl=data.ctrl.copy()
    qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
    world=np.diag([1.,1.,-1.]);local=np.diag([1.,-1.,1.])
    initial_rotation=data.xmat[model.body('head').id].reshape(3,3).copy()
    rotation=world@initial_rotation@local
    signs=np.where(np.abs(model.jnt_axis[model.actuator_trnid[:,0],0])>.5,-1.,1.)
    data.qpos[2]=2.-old_qpos[2]
    data.qpos[3:7]=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
    data.qpos[qa]=signs*old_qpos[qa];data.ctrl[:]=signs*old_ctrl
    data.qvel[:]=0.;data.qacc_warmstart[:]=0.;data.time=0.
    mujoco.mj_forward(model,data)
    expected=old_points.copy();expected[:,2]=2.-old_points[:,2]
    actual=nodes(data);error=float(np.max(np.abs(actual-expected))*1000)
    phi=np.unwrap(np.arctan2((actual[:-1]+actual[1:])[:9,2]/2-1,(actual[:-1]+actual[1:])[:9,0]/2-1.1))
    report=dict(kind='artificial mirrored CLOSE1B geometry before first mj_step',
        active_upward_hook_acquisition_demonstrated=False,physical_steps=0,
        world_reflection='Z -> 2m - Z; pipe axes and gravity remain original',
        proper_head_rotation_determinant=float(np.linalg.det(rotation)),hinge_signs=signs,
        maximum_node_mirror_error_mm=error,old_upper_points_mm=old_points*1000,
        new_lower_points_mm=actual*1000,bridge_tail_to_head_nodes=[10,9,8],
        bridge_tail_to_head_z_mm=actual[[10,9,8],2]*1000,
        bridge_entry_z_relative_to_axis_mm=(actual[8,2]-1)*1000,
        bridge_tail_to_head_tangent=actual[8]-actual[9],
        head_to_tail_winding_sign=-1,front9_wrap_deg=float(np.degrees(np.ptp(phi))),
        initial_velocity='zero artificial rest state; old qvel and warmstart are not reused',
        initial_ctrl='mirrored original all-joint preload, no per-joint manual adjustment')
    if error>1e-6:raise ValueError('real hinge mirror did not reproduce requested lower geometry')
    return report
