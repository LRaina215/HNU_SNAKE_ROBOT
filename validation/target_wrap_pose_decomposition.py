"""Saved-pose component swaps, offline only; not executable trajectories."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.target_capture_reference import TargetCaptureReference

BASE=ROOT/'experiments/curve_guided_transfer_20261006/target_wrap_round2'


def main():
    case=BASE/'CAPTURE1_B';z=np.load(case/'trace.npz');curve=TargetCaptureReference(case/'reference.npz')
    fit=LocalPathFit();rows=[]
    for moment in [5,34.9,50,56.5,61.8,66.8]:
        i=int(abs(z['sim_time']-moment).argmin());feed=float(z['feed_mm'][i]);state=curve.state(feed)
        nominal=np.r_[state[:3]/1000,Rotation.from_rotvec(state[3:6]).as_quat()[[3,0,1,2]],state[6:]]
        actual=z['qpos'][i].copy();combinations={
            'actual_root_actual_joints':actual,
            'nominal_root_nominal_joints':nominal,
            'actual_root_nominal_joints':np.r_[actual[:7],nominal[7:]],
            'nominal_root_actual_joints':np.r_[nominal[:7],actual[7:]]}
        poses={}
        for label,qpos in combinations.items():
            fit.d.qpos[:]=qpos;mujoco.mj_kinematics(fit.m,fit.d);fit.last=None
            p=points(fit.m,fit.d);relative=p-[1100,0,1000];angle=np.unwrap(np.arctan2(relative[:9,2],relative[:9,0]))
            poses[label]=dict(M9_target_shell_gap_mm=float(fit.radial_gap(9,1)),
                M9_endpoint_Y_mm=p[9:11,1].tolist(),M9_endpoint_radial_mm=np.linalg.norm(relative[9:11][:,[0,2]],axis=1).tolist(),
                front_net_polar_turn_deg=float(abs(np.degrees(angle[-1]-angle[0]))))
        rows.append(dict(time_s=float(z['sim_time'][i]),feed_mm=feed,poses=poses))
    report=dict(physics_steps=0,control_dispatches=0,source_case='CAPTURE1_B',rows=rows,
        interpretation='Component swaps diagnose sensitivity to relative root pose and joint shape. They can violate contact/collision constraints and are not physically reachable counterfactuals or causal proof. Errors may compensate; effects are not additive.',
        actual_joint_deviation_includes='Both original R2 compliance and added receiving correction; cannot attribute all joint deviation to the new window')
    (BASE/'saved_pose_decomposition.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(rows[-1],indent=2),flush=True)


if __name__=='__main__':main()
