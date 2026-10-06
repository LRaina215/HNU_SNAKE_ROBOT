"""Exact initial placement and first R2 command check, without physics steps."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import ast
import argparse
import numpy as np
import mujoco
from curve_local_path import BASE,OUT,LocalPathFit
from snake_control.local_material_reference import CoupledMaterialCurve,BoundaryMaterialCurve
from snake_control.curve_grasp_revision import FittedMaterialCurve
from snake_control.executable_grasp_state import EquilibriumGraspController
from curve_guided_geometry import points


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def methods(path):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return {(c.name,m.name):ast.dump(m,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef)
        for m in c.body if isinstance(m,ast.FunctionDef)}


def main(boundary=False):
    global OUT
    if boundary:OUT=BASE/'local_path_boundary'
    fit=LocalPathFit()
    with np.load(BASE/('initial_boundary_pose.npz' if boundary else 'initial_path_pose.npz')) as z:
        fit.d.qpos[:]=z['qpos'];fit.d.qvel[:]=z['qvel'];fit.d.ctrl[:]=z['target']
    mujoco.mj_forward(fit.m,fit.d)
    geometry_points=points(fit.m,fit.d).copy()
    kinematic=mujoco.MjData(fit.m)
    def encoder_fk(q):
        kinematic.qpos[:]=0.;kinematic.qpos[3]=1.;kinematic.qpos[fit.qadr]=q
        mujoco.mj_kinematics(fit.m,kinematic)
        return points(fit.m,kinematic)
    config=json.loads((BASE/'R2_C0/config.json').read_text(encoding='utf8'))
    curve=(BoundaryMaterialCurve if boundary else CoupledMaterialCurve)(OUT/'reference_ready.npz')
    registered=fit.d.body('head').xmat.reshape(3,3).copy()
    with np.load(BASE/'R2_C0/trace.npz') as saved:
        prior=mujoco.MjData(fit.m);prior.qpos[:]=saved['qpos'][0];mujoco.mj_kinematics(fit.m,prior)
        old_registered=prior.body('head').xmat.reshape(3,3).copy()
    if boundary:registered=old_registered
    controller=EquilibriumGraspController(curve,config,fit.d.ctrl.copy(),encoder_fk,
        registered)
    age=45.25 if boundary else 0.
    target,control=controller.compute(fit.d.qpos[fit.qadr].copy(),0.,age,.05)
    controller.accept(target,0.)
    current=methods(ROOT/'snake_control/executable_grasp_state.py')
    frozen=methods(BASE/'R2_C0/sources/snake_control/executable_grasp_state.py')
    # Docstrings can differ; only executable statements of the relevant methods.
    same=all(current[k]==frozen[k] for k in current)
    length=np.linalg.norm(np.diff(geometry_points,axis=0),axis=1)
    row=dict(physics_steps=0,pose_source='saved actual R2 t45.25s',
        qpos_identity_maximum=float(abs(fit.d.qpos-fit.held_qpos).max()),
        acknowledged_target_identity_maximum=float(abs(fit.d.ctrl-fit.held_ctrl).max()),qvel_zero=bool(np.all(fit.d.qvel==0)),
        maximum_length_error_mm=float(abs(length-64.5).max()),total_length_mm=float(length.sum()),
        first_command_step_maximum_deg=float(np.degrees(abs(target-fit.d.ctrl)).max()),
        first_raw_target_maximum_deg=float(np.degrees(abs(np.asarray(control['raw_target_rad']))).max()),
        material_axes=control['material_axes'],first_residual_deg=np.degrees(controller.shape).tolist(),
        state_ack_identity_error_rad=float(abs(controller.shape-(target-curve.joint_angles(0.))).max()),
        R2_executable_method_AST_same=same,
        interface_check_pass=bool(same and abs(fit.d.qpos-fit.held_qpos).max()==0 and
            np.all(abs(target)<=np.radians(90)) and abs(target-fit.d.ctrl).max()<=np.radians(.500001)),
        limitation='Forward/first-command check only; matching C0 still required')
    if boundary:
        old_curve=FittedMaterialCurve(BASE.parent/'structural_revision_v3/fitted_reference.npz')
        old_controller=EquilibriumGraspController(old_curve,config,fit.d.ctrl.copy(),encoder_fk,old_registered)
        old_target,old_control=old_controller.compute(fit.d.qpos[fit.qadr].copy(),0.,45.25,.05)
        difference=float(abs(old_target-target).max())
        nominal_difference=float(abs(curve.joint_angles(0.)-old_curve.joint_angles(0.)).max())
        tangent_difference=float(abs(curve.feed_tangent(0.)-old_curve.feed_tangent(0.)).max())
        axis_difference=float(abs(np.array([a['direction_head'] for a in control['material_axes']])-
            np.array([a['direction_head'] for a in old_control['material_axes']])).max())
        row.update(warm_start_next_command_identity_error_rad=difference,
            nominal_identity_error_rad=nominal_difference,tangent_identity_error_rad_per_mm=tangent_difference,
            material_axis_identity_error=axis_difference,controller_age_s=age,
            warm_start_next_command_identity_pass=bool(max(difference,nominal_difference,tangent_difference,axis_difference)<1e-12))
        row['interface_check_pass'] &= row['warm_start_next_command_identity_pass']
    write('interface_check.json',row)
    if not row['interface_check_pass']:raise RuntimeError('Initial interface check failed')
    print(json.dumps({k:row[k] for k in ['interface_check_pass','first_command_step_maximum_deg',
        'first_raw_target_maximum_deg','R2_executable_method_AST_same']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--boundary',action='store_true')
    main(parser.parse_args().boundary)
