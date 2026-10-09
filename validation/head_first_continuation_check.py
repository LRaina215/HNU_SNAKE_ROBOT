"""Private true-hinge continuation review after bounded F07, zero physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco,json
from snake_control.head_first_inner_route import HeadFirstInnerRoute
from snake_control.suspended_supply_transfer import WholeCurveMapper
from validation.suspended_supply_experiment import write,default
from validation.full_source_inner_experiment import BASE

def run():
    case=BASE/'F07';out=BASE/'POST_F07_CONTINUATION_UNRUN';out.mkdir(exist_ok=False)
    z=np.load(case/'trace.npz');i=int(np.argmin(abs(z['sim_time']-111.65)));p=z['points_mm'][i]/1000
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'));route=HeadFirstInnerRoute(p,smooth_source=False);last=z['qpos'][i].copy();checks=[];states=[]
    for s in np.linspace(0,route.total,35):
        g,v=route.window(s);mapper=WholeCurveMapper(m);last,fit=mapper.initialize(g,initial_pose=last);states.append(last.copy())
        row=dict(shift_mm=s*1000,rod_error_mm=float(abs(np.linalg.norm(np.diff(g,axis=0),axis=1)-.0645).max()*1000),fit_rms_mm=fit['guide_rms_mm'],pipe_depth_mm=fit['pipe_depth_mm'],self_depth_mm=fit['self_depth_mm'],maximum_joint_deg=fit['maximum_joint_deg']);checks.append(row);print(json.dumps(row),flush=True)
    ready=all(x['fit_rms_mm']<35 and x['pipe_depth_mm']<2 and x['self_depth_mm']<.5 for x in checks)
    write(out/'summary.json',dict(physical_steps=0,seed_actual_time_s=float(z['sim_time'][i]),purpose='Distinguish independently initialized optimizer basins from a continued private20-hinge solution; not physical execution or a global infeasibility proof',ready=ready,maximum_fit_rms_mm=max(x['fit_rms_mm'] for x in checks),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in checks),maximum_self_depth_mm=max(x['self_depth_mm'] for x in checks),remaining_formal_trial_allowance=0))
    write(out/'checks.json',checks);np.savez_compressed(out/'private_kinematics.npz',qpos=states)

if __name__=='__main__':run()
