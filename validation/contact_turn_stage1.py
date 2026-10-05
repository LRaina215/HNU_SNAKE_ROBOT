"""Privileged known-direction transfer, gated by independent stage-0 audit."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
import time

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'validation')]
from contact_turn_experiment import Simulation,BASE,write_json,sha
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
from snake_control.local_bend_backbone import LocalBendBackbone
from snake_control.contact_turn_controller import JointObservation,KnownDirectionOracle,KnownDirectionTransfer


def points(d):
    return np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{i}').xanchor for i in range(1,21)],d.site('s_tail').xpos])


def route(m):
    line=m.numeric('pipe_centerline').data.reshape(-1,4)[:,:3]
    source=line[1]-line[0];source/=np.linalg.norm(source)
    for i in range(1,len(line)-1):
        axis=line[i+1]-line[i];axis/=np.linalg.norm(axis)
        if axis@source<.9:
            return line[i],source,axis
    raise ValueError('No sharp target branch in model')


def oracle(sim,direction):
    m,d=sim.model,sim.data
    corner,source,target=route(m)
    pts=points(d)
    along=(pts-corner)@target
    perpendicular=source
    second=np.cross(target,perpendicular)
    radial=pts-corner-np.outer(along,target)
    angle=np.arctan2(radial@second,radial@perpendicular)
    angle_delta=np.arctan2(np.sin(np.diff(angle)),np.cos(np.diff(angle)))
    distance=np.linalg.norm(radial,axis=1)
    mask=((along[:-1]+along[1:])/2>.03)&((distance[:-1]+distance[1:])/2<.11)
    wrap=float(abs(angle_delta[mask].sum()))
    source_modules=set();target_modules=set()
    head=m.body('head').id
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if (b1>=head)!=(b2>=head):
            other=c.geom2 if b1>=head else c.geom1
            if not (m.geom(other).name or '').startswith('pipe'):
                continue
            module=(b1 if b1>=head else b2)-head
            name=m.geom(other).name or ''
            if ('sharp_out' in name or name=='pipe_straight_04') and (c.pos-corner)@target>.025:
                target_modules.add(module)
            else:
                source_modules.add(module)
    return KnownDirectionOracle(bool((pts[0]-corner)@source>=-.08),tuple(sorted(target_modules)),
                                tuple(sorted(source_modules)),wrap,
                                bool(np.all(along>.04) and not source_modules),float(direction),
                                bool(sim.records[-1]['ground']))


def choose_direction(sim,policy):
    """Oracle FK search: rigidly anchor posterior points, score predicted head.

    This only selects a direction for the privileged K feasibility reference.
    It uses true world geometry and must never be reported as sensing success.
    """
    m,d=sim.model,sim.data
    corner,source,target=route(m)
    before=points(d)
    goal=corner+.15*target
    test=mujoco.MjData(m)
    scores=[]
    for direction in np.linspace(0,2*math.pi,16,endpoint=False):
        shape=LocalBendBackbone(policy.helix,np.full(21,64.5),policy.mu,policy.width,
                                 math.radians(10),direction,policy.release,policy.reference,policy.initial_phase)
        test.qpos[:]=d.qpos;test.qpos[7:]=shape.joint_angles(policy.phase)
        mujoco.mj_forward(m,test)
        after=points(test)
        a,b=after[-10:],before[-10:]
        u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
        rot=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        moved=(after-a.mean(0))@rot+b.mean(0)
        loss=float(np.linalg.norm(moved[0]-goal))
        scores.append(dict(direction_rad=float(direction),head_goal_distance_m=loss,predicted_head=moved[0].tolist()))
    return min(scores,key=lambda x:x['head_goal_distance_m'])['direction_rad'],scores


def trial(args):
    gate=json.loads((BASE/'stage0_independent_audit.json').read_text(encoding='utf8'))
    assert gate['passed']
    # Geometry checks precede dynamics; an invalid request is a recorded reject.
    label=f'f{args.front}_w{args.width:g}_turn{args.turn:g}_p{args.phase:g}_{args.attempt}'
    directory=args.out/'stage1'/label
    if directory.exists():
        raise RuntimeError('refusing to overwrite stage-1 attempt')
    model=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
    manifest=dict(method='K_nominal',privileged_geometry=True,autonomous=False,
                  front=args.front,width_mm=args.width,maximum_turn_deg=args.turn,initial_phase=args.phase,
                  physical_budget_s=args.budget,hold_gate_s=10,hold_slip_limit_mm=20,
                  torque_limit_nm=2.355,kp=127,gravity=True,dt=.001,
                  rolling_phase_rate_rad_s=-.75,initialization='round2: one 50% measured-error relaxation',
                  code_hash={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'snake_control/local_bend_backbone.py',
                      ROOT/'snake_control/contact_turn_controller.py',ROOT/'validation/contact_turn_experiment.py']},
                  model_hash=sha(model),exe_hash=sha(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe'))
    sim=None;policy=None
    result=dict(outcome='engineering_failure',success=False,milestones={},label=label)
    try:
        sim=Simulation(model,directory,args.phase)
        write_json(directory/'manifest.json',manifest)
        helix=HelixBackbone.from_radius_pitch(81.5,220.)
        nominal=helix.joint_angles(20,64.5,args.phase)
        sim.stage='initialization';sim.command(nominal);sim.advance(2.)
        before=sim.data.ctrl.copy();q=sim.data.qpos[sim.qadr].copy()
        bound=2.355*.5/127
        reference=q+np.clip(before-q,-bound,bound)
        for weight in np.linspace(0,1,11)[1:]:
            sim.command(before*(1-weight)+reference*weight);sim.advance(.05)
        sim.advance(2.)
        result['common_initial_qpos']=sim.data.qpos.copy().tolist()
        result['common_initial_ctrl']=reference.tolist()
        policy=KnownDirectionTransfer(helix,reference,args.phase,args.front,args.width,args.turn)
        direction,scores=choose_direction(sim,policy)
        write_json(directory/'oracle_direction_search.json',scores)
        start=float(sim.data.time);last_progress=start;best=0.;transfer_start=None
        corner,source,target=route(sim.model)
        start_head=float((points(sim.data)[0]-corner)@source)
        result['initial_head_distance_to_joint_m']=-start_head
        result['oracle_direction_rad']=direction
        saved_shapes={}
        sim.stage='motion'
        while sim.data.time-start<args.budget:
            row=sim.records[-1]
            observation=JointObservation(float(sim.data.time),row['wall_unix_ms'],time.time()*1000,
                                          tuple(row['q']),tuple(row['velocity']),tuple(row['ctrl']),tuple(row['torque']),
                                          int(row['applied_command_id']))
            truth=oracle(sim,direction)
            if policy.state=='approach' and truth.head_near_joint:
                direction,scores=choose_direction(sim,policy)
                write_json(directory/'oracle_direction_search_at_turn.json',scores)
                truth=oracle(sim,direction)
            if truth.target_contact_modules and 'target_contact_s' not in result['milestones']:
                result['milestones']['target_contact_s']=float(sim.data.time-start)
            if truth.target_wrap_rad>=math.pi/2 and len(truth.target_contact_modules)>=3 and 'target_wrap_s' not in result['milestones']:
                result['milestones']['target_wrap_s']=float(sim.data.time-start)
            if truth.tail_in_target and 'tail_transfer_s' not in result['milestones']:
                result['milestones']['tail_transfer_s']=float(sim.data.time-start)
                transfer_start=points(sim.data).mean(0)@target
            if truth.ground_contact:
                result['outcome']='ground_contact';break
            if row['self_contact']:
                result['outcome']='self_contact';break
            goal,shape=policy.compute(observation,truth)
            sim.phase=policy.phase
            if policy.state not in saved_shapes:
                saved_shapes[policy.state]=dict(time_s=float(sim.data.time),q=goal.tolist(),
                    curve=shape.sample(policy.phase,5.)['points_mm'].tolist())
            try:
                sim.command(goal)
            except ValueError as error:
                result['outcome']='target_limit_reject';result['error']=str(error);break
            sim.advance(.05)
            pts=points(sim.data)
            progress=max(float((pts[0]-corner)@source)-start_head,0.) if policy.state=='approach' else float((pts[0]-corner)@target)
            if progress>best+.005:
                best=progress;last_progress=float(sim.data.time)
            if sim.data.time-last_progress>15 and policy.state!='target_rolling':
                result['outcome']='no_progress_or_no_target_support';break
            if transfer_start is not None and pts.mean(0)@target-transfer_start>=.2:
                sim.stage='final_hold';p0=pts.mean(0)@target;hold_begin=sim.data.time
                sim.advance(10.)
                hold=[r for r in sim.records if r['sim_time']>hold_begin+1e-8]
                retreat=max(p0-r['points'].mean(0)@target for r in hold)
                result['final_hold_retreat_mm']=float(retreat*1000)
                result['success']=bool(retreat<=.02 and not any(r['ground'] or r['self_contact'] for r in hold))
                result['outcome']='success' if result['success'] else 'final_hold_failed'
                break
        else:
            result['outcome']='physical_budget_exhausted'
        result.update(physical_end_s=float(sim.data.time),motion_duration_s=float(sim.data.time-start),
                      final_state=policy.state,max_target_wrap_rad=max(x['target_wrap_rad'] for x in policy.records),
                      maximum_target_contact_modules=max(len(x['target_contact_modules']) for x in policy.records),
                      bend_window_progress_mm=float(policy.mu),max_turn_reached_deg=math.degrees(policy.turn),
                      paused_phase_fraction=float(np.mean([x['state']=='turn' for x in policy.records])))
        write_json(directory/'shape_snapshots.json',saved_shapes)
    except Exception as error:
        result['error']=repr(error)
    finally:
        if sim:
            sim.close()
        if not directory.exists():
            directory.mkdir(parents=True)
        write_json(directory/'result.json',result)
        if policy:
            write_json(directory/'control.json',policy.records)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('common_initial')},ensure_ascii=False),flush=True)
    return result
