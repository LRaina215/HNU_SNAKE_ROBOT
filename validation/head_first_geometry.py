"""Zero-step exact-rod and true-hinge checks before a head-first trial."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,argparse,time
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.head_first_inner_route import HeadFirstInnerRoute
from snake_control.horizontal_pipe_roll import HingePhaseMapper
from snake_control.suspended_supply_transfer import LINK,nodes
from snake_control.suspended_transfer_route import BETA
from validation.suspended_supply_experiment import write

BASE=ROOT/'experiments/unified_curve_transfer_20261007/FULL_SOURCE_INNER_20261009'


def run(name,left):
    out=BASE/name;out.mkdir(exist_ok=False)
    z=np.load(BASE/'F04/trace.npz');i=int(np.argmin(abs(z['sim_time']-14.75)))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m);d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];mujoco.mj_forward(m,d)
    route=HeadFirstInnerRoute(nodes(d),left=left);mapper=HingePhaseMapper(m,d);last=d.qpos[mapper.qa].copy();reports=[];points=[];states=[];started=time.monotonic()
    for s in np.linspace(0,route.bridge_length+8*LINK*1.04,41):
        guide,v=route.window(s);phase=s*BETA/.029*.328
        q,info=mapper.solve(guide,s,phase,last,rate_bound=False,node_weights=np.full(22,3.),orientation_override=route.head_orientation(mapper.rotation,s,phase));last=q;points.append(guide);states.append(mapper.d.qpos.copy())
        actual=nodes(mapper.d);minrod=np.linalg.norm(np.diff(guide,axis=0),axis=1).min();maxrod=np.linalg.norm(np.diff(guide,axis=0),axis=1).max();maxq=np.degrees(abs(q).max())
        reports.append(dict(shift_mm=s*1000,rod_min_mm=minrod*1000,rod_max_mm=maxrod*1000,maximum_joint_deg=maxq,whole_fit_rms_mm=info['whole_fit_rms_mm'],front_fit_rms_mm=info['front_fit_rms_mm'],predicted_pipe_depth_mm=info['predicted_pipe_depth_mm'],guide_points_mm=guide*1000,projected_actual_points_mm=actual*1000,q_rad=q,curve_stations_mm=v*1000))
        print(json.dumps({k:reports[-1][k] for k in ['shift_mm','whole_fit_rms_mm','front_fit_rms_mm','maximum_joint_deg','predicted_pipe_depth_mm']}),flush=True)
    grid=np.linspace(route.knots[-1],route.total,4001);allp=route.sample(grid);fig=plt.figure(figsize=(15,8));ax=fig.add_subplot(121,projection='3d');ax.plot(*allp.T,label='One continuous route');ax.plot(*nodes(d).T,'o-',label='Actual junction state used for offline design');ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)');ax.view_init(27,140);ax.legend()
    ax=fig.add_subplot(122);ax.plot([r['shift_mm'] for r in reports],[r['whole_fit_rms_mm'] for r in reports],label='True20-hinge curve fit RMS mm');ax.plot([r['shift_mm'] for r in reports],[r['predicted_pipe_depth_mm'] for r in reports],label='Private predicted pipe depth mm');ax.set(xlabel='Shared head shift(mm)',ylabel='mm');ax.legend();fig.tight_layout();fig.savefig(out/'geometry_review.png',dpi=155);plt.close(fig)
    write(out/'complete_curve.json',dict(stations_mm=grid*1000,points_mm=allp*1000,bridge_length_mm=route.bridge_length*1000,source_to_bridge_at_mm=0,target_join_mm=route.target_join*1000))
    summary=dict(physical_steps=0,seed='read-only F04 actual junction sample, not a future physical reset',seed_time_s=float(z['sim_time'][i]),curve='actual source spline, one inner Bezier bridge, one target helix; nearest previous exact64.5mm chord intersection',bridge_length_mm=route.bridge_length*1000,bridge_target_node_clearance_mm=float(np.linalg.norm(route.connector[:,[0,2]]-[1.1,1],axis=1).min()*1000),bridge_source_node_clearance_mm=float(np.linalg.norm(route.connector[:,[1,2]]-[0,1],axis=1).min()*1000),maximum_rod_length_error_mm=max(max(abs(r['rod_min_mm']-64.5),abs(r['rod_max_mm']-64.5)) for r in reports),maximum_true_hinge_fit_rms_mm=max(r['whole_fit_rms_mm'] for r in reports),maximum_predicted_pipe_depth_mm=max(r['predicted_pipe_depth_mm'] for r in reports),maximum_joint_deg=max(r['maximum_joint_deg'] for r in reports),wall_s=time.monotonic()-started,ready=max(r['whole_fit_rms_mm'] for r in reports)<25 and max(r['predicted_pipe_depth_mm'] for r in reports)<2.)
    write(out/'geometry_summary.json',summary);write(out/'projection_windows.json',reports);np.savez_compressed(out/'kinematic_projection.npz',shift_mm=np.array([r['shift_mm'] for r in reports]),reference_points=points,qpos=states)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='HEAD_FIRST_GEOMETRY_01');ap.add_argument('--left',type=float,default=.935);args=ap.parse_args();run(args.name,args.left)
