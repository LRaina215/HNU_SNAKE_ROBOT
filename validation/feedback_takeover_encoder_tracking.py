"""Encoder-frame CAD tracker replay; floating pose is scoring-only.

Relative body registration predicts the next hypothesis. CAD/load-envelope
refinement corrects that prediction. This does not assume that the chosen
body points are physically stationary or that rolling is no-slip; residual
axis drift is measured explicitly against held-out simulator state columns.
"""
import time
import argparse
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json,pipe_frame
from mujoco_checkpoint import load_checkpoint
from encoder_pipe_geometry import right_angle_hypothesis
from encoder_pipe_proximity import EncoderPipeProximity
from feedback_takeover_cad_refinement import refine

BASE=ROOT/'experiments/segmented_pole_20261005/support_transfer'
OUT=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'


def transport(previous,current,estimate):
    x=previous-previous.mean(axis=0);y=current-current.mean(axis=0)
    u,_,vt=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@fix@u.T
    shift=current.mean(axis=0)-rotation@previous.mean(axis=0)
    return dict(source=rotation@estimate['source'],target=rotation@estimate['target'],
        corner_mm=rotation@estimate['corner_mm']+shift,
        pipe_radius_mm=estimate['pipe_radius_mm'],geometric_candidate=True)


def local_transport(previous,current,estimate,front_point_count=8):
    front=transport(previous[:front_point_count],current[:front_point_count],estimate)
    rear=transport(previous[-8:],current[-8:],estimate)
    target=front['target'];source=rear['source']
    source=source-(source@target)*target;source/=np.linalg.norm(source)
    # Translation along either individual pipe is unobservable but does not
    # alter its line. Their closest intersection combines the two line poses.
    shift=front['corner_mm']-rear['corner_mm']
    uv=np.linalg.lstsq(np.column_stack((source,-target)),shift,rcond=None)[0]
    corner=.5*(rear['corner_mm']+uv[0]*source+front['corner_mm']+uv[1]*target)
    return dict(source=source,target=target,corner_mm=corner,
        pipe_radius_mm=estimate['pipe_radius_mm'],geometric_candidate=True)


def run(front_registration=False,contact_memory=False,local_registration=False):
    all_rows=[]
    suffix='_front_fixed_radius' if front_registration else ''
    if contact_memory:suffix+='_memory'
    if local_registration:suffix+='_local'
    for name,cp in [('radial_front_relative_braking',HORIZONTAL),('vertical_initial_front_reference',VERTICAL)]:
        m,d,meta,_=load_checkpoint(cp,verify_hashes=False)
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        robot=mujoco.MjData(m);robot.qpos[:7]=[0,0,0,1,0,0,0]
        query=EncoderPipeProximity(m)
        with np.load(BASE/name/'trace.npz') as z:
            qpos=z['qpos'][::20];ctrl=z['ctrl'][::20];times=z['time'][::20]-z['time'][0]
        estimate=previous=None;rows=[];candidate_age=np.full((2,21),10.)
        corner,source,target,r=pipe_frame(m)
        for q,command,at in zip(qpos,ctrl,times):
            measured=q[qadr];robot.qpos[qadr]=measured;mujoco.mj_kinematics(m,robot)
            points=1000*np.vstack([robot.site('s_head').xpos,
                *[robot.joint('joint'+str(i)).xanchor for i in range(1,21)],robot.site('s_tail').xpos])
            load=np.clip(m.actuator_gainprm[:,0]*(command-measured),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
            started=time.perf_counter()
            predicted=right_angle_hypothesis(points,meta['settings']['radius']) if estimate is None else transport(
                previous[:8] if front_registration else previous,
                points[:8] if front_registration else points,estimate)
            if estimate is not None and local_registration:predicted=local_transport(previous,points,estimate)
            estimate=refine(query,measured,points,predicted,load,
                fixed_radius=front_registration and estimate is not None,
                candidate_age=candidate_age if contact_memory else None)
            if contact_memory:candidate_age=estimate['candidate_age']
            previous=points.copy();elapsed=time.perf_counter()-started
            # No floating-pose values enter the estimator above.
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q[3:7]);rot=rot.reshape(3,3)
            scores=dict(source_axis_error_deg=float(np.degrees(np.arccos(np.clip((rot@estimate['source'])@source,-1,1)))),
                target_axis_error_deg=float(np.degrees(np.arccos(np.clip((rot@estimate['target'])@target,-1,1)))),
                corner_error_mm=float(np.linalg.norm(rot@estimate['corner_mm']+1000*q[:3]-corner)),
                radius_error_mm=float(estimate['pipe_radius_mm']-r))
            row=dict(run=name,time=float(at),inference_seconds=elapsed,estimate=estimate,scoring=scores)
            rows.append(row)
            if len(rows)%50==1:print(name,round(at,1),scores,flush=True)
        all_rows+=rows
        write_json(OUT/(name+'_encoder_tracking'+suffix+'.json'),dict(rows=rows,
            inputs='Encoded joints, recorded command, robot CAD, nominal R and right-angle prior only.',
            front_registration_and_constant_fitted_radius=front_registration,
            limitation='Body-shape registration predicts a gauge transform; rolling/slip can bias it. No truth correction.'))
    write_json(OUT/('encoder_tracking_summary'+suffix+'.json'),dict(runs=[dict(run=name,
        target_error_p95_deg=float(np.percentile([r['scoring']['target_axis_error_deg'] for r in all_rows if r['run']==name],95)),
        maximum_corner_error_mm=float(max(r['scoring']['corner_error_mm'] for r in all_rows if r['run']==name)))
        for name in dict.fromkeys(r['run'] for r in all_rows)]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front-registration',action='store_true')
    parser.add_argument('--contact-memory',action='store_true')
    parser.add_argument('--local-registration',action='store_true')
    args=parser.parse_args();run(args.front_registration,args.contact_memory,args.local_registration)
