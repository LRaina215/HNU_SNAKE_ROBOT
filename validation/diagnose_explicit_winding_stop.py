"""Read-only final curve/integral occupancy diagnostic; no mj_step calls."""
from pathlib import Path
import argparse
import copy
import json
import math
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snake_control.explicit_winding_transfer import ExplicitWindingTransfer
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.unified_curve_transfer import BODY_RADIUS


def minimum(curve,geometry,shift):
    enabled=curve.clearance_enabled;curve.clearance_enabled=False
    try:nodes,_,_=curve.reference(shift)
    finally:curve.clearance_enabled=enabled
    fractions=np.linspace(0.,1.,65)
    samples=nodes[:-1,None]*(1-fractions[None,:,None])+nodes[1:,None]*fractions[None,:,None]
    gaps=geometry.signed(samples);rod,station=np.unravel_index(gaps.argmin(),gaps.shape)
    return nodes,dict(minimum_mm=float(gaps[rod,station]*1000),limiting_rod=int(rod),point_mm=(samples[rod,station]*1000).tolist())


def pipe_penetration(model,data):
    depths=[]
    for contact in data.contact:
        one=(model.geom(contact.geom1).name or '').startswith('pipe_')
        two=(model.geom(contact.geom2).name or '').startswith('pipe_')
        if one != two:depths.append(max(0.,-contact.dist*1000))
    return max(depths,default=0.)


def diagnose(case):
    model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    actual=mujoco.MjData(model)
    restore_saved_state(model,actual,ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz')
    controller=ExplicitWindingTransfer(model,actual)
    controller.load_state(case/'controller_state.json')
    restore_saved_state(model,actual,case/'final_state.npz')
    geometry=controller.pipe_geometry
    saved=copy.deepcopy(controller.curve.__dict__)
    current,current_stats=minimum(controller.curve,geometry,controller.shift)
    actual_nodes,_=controller.observe(actual)
    increment=.0001
    controller.curve.increment(increment)
    pure,pure_stats=minimum(controller.curve,geometry,controller.shift+increment)
    correction_error=None
    try:controller.route_outside_bridge(controller.shift+increment)
    except ValueError as exc:correction_error=str(exc)
    corrected,corrected_stats=minimum(controller.curve,geometry,controller.shift+increment)
    axis=corrected[0]-corrected[1];axis/=np.linalg.norm(axis)
    cross=np.cross(controller.root_orientation[:,2],axis);sine=np.linalg.norm(cross)
    transport=Rotation.from_rotvec(cross*math.atan2(sine,float(controller.root_orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()
    source_step=-np.sign(controller.curve.theta_rates[0])*controller.curve.axial_sines[0]*increment/(BODY_RADIUS*math.sqrt(1-controller.curve.axial_sines[0]**2))
    target_step=np.sign(controller.curve.theta_rates[1])*controller.curve.axial_sines[1]*increment/(BODY_RADIUS*math.sqrt(1-controller.curve.axial_sines[1]**2))
    orientation=Rotation.from_rotvec(axis*target_step).as_matrix()@transport@controller.root_orientation
    prior,_,_=controller.integral_targets(corrected,orientation,controller.pipe_roll_phase+np.array([source_step,target_step]))
    candidate=mujoco.MjData(model);candidate.qpos[:]=actual.qpos
    candidate.qpos[controller.qa]=np.clip(prior,-np.radians(88),np.radians(88))
    mujoco.mj_fwdPosition(model,candidate)
    candidate_nodes,_=controller.observe(candidate)
    report=dict(name=case.name,no_physics_steps=True,requested_window_increment_mm=increment*1000,
        current_abstract_curve=current_stats,pure_increment_abstract_curve=pure_stats,
        bounded_clearance_increment_curve=corrected_stats,bounded_clearance_rejection=correction_error,
        maximum_bridge_clearance_field_mm=float(np.linalg.norm(controller.curve.bridge_clearance_offsets,axis=1).max()*1000),
        candidate_maximum_node_step_mm=float(np.linalg.norm(corrected-current,axis=1).max()*1000),
        actual_rod_centerline_clearance_mm=float(geometry.link_minimum(actual_nodes,65).min()*1000),
        candidate_exact_chain_centerline_clearance_mm=float(geometry.link_minimum(candidate_nodes,65).min()*1000),
        actual_shell_pipe_contact_penetration_mm=pipe_penetration(model,actual),
        candidate_exact_integral_shell_pipe_contact_penetration_mm=pipe_penetration(model,candidate),
        candidate_maximum_integral_goal_change_deg=float(np.degrees(abs(np.clip(prior,-np.radians(88),np.radians(88))-controller.joint_geometry)).max()),
        distinction='candidate is the full integral goal at measured root, not the next rate-limited command; contact distances are geometry-only; neither this nor positive centreline clearance proves executable motion')
    controller.curve.__dict__.clear();controller.curve.__dict__.update(saved)
    (case/'stop_geometry_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);args=p.parse_args();diagnose(args.case)
