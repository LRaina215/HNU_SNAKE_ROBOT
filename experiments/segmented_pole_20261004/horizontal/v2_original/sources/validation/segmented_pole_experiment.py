"""Paired post-contact rolling experiment, isolated EXE relay, MuJoCo 3.2.7.

Restores complete user checkpoints. Never controls a serial port or changes
friction, gravity, gains, timestep, actuators, or the initial physical state.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'codes'), str(ROOT/'visualizer')]
from mujoco_checkpoint import load_checkpoint, json_value, sha
from mujoco_execution_bridge import MujocoExecutionBridge
from python_client import SnakeControlClient
from pole_compliance import build_backbone
from curved_axis_compliance import local_sections, joint_segments
from segmented_pole_backbone import (SegmentedOptions, MovingAxisBackbone, BoundaryTracker,
                                      SwitchContinuity, unit, smoothstep)

BASE = ROOT/'experiments/segmented_pole_20261004'
CAPTURES = ROOT/'validation/results/manual-captures'
HORIZONTAL = BASE/'inputs/horizontal'
VERTICAL = BASE/'inputs/vertical_down'
POLICY = dict(max_steps=80, advance_mm=64.5, new_modules=2, step_seconds=.2, hold_seconds=3.,
              transfer_persistence_steps=3, region_margin_mm=10., region_clearance_mm=15.,
              hold_retreat_mm=10., hold_root_motion_mm=20., minimum_contact_force_N=.1,
              lost_support_seconds=.2)
SOURCES = ['codes/segmented_pole_backbone.py', 'codes/pole_compliance.py',
           'codes/curved_axis_compliance.py', 'codes/mujoco_checkpoint.py',
           'codes/mujoco_execution_bridge.py', 'codes/mujoco_visualizer_bridge.py',
           'visualizer/python_client.py', 'validation/segmented_pole_experiment.py',
           'validation/SegmentedAxisReplay.cs']


class AxisReplay:
    def __init__(self):
        folder=Path(tempfile.mkdtemp(prefix='segmented-axis-'))
        exe=folder/'axis.exe'
        compiler=Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        subprocess.run([str(compiler),'/nologo','/r:System.Web.Extensions.dll','/out:'+str(exe),
                        str(ROOT/'validation/SegmentedAxisReplay.cs')],check=True,capture_output=True)
        self.process=subprocess.Popen([str(exe),str(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        self.seeded=False

    def geometry(self,g,seed):
        packet=dict(points_mm=g['points_mm'])
        if not self.seeded:packet['seed']=seed;self.seeded=True
        self.process.stdin.write(json.dumps(packet)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(self.process.stderr.read())
        axis=json.loads(line)
        geometry=true_axis_geometry(g,np.array(axis['direction']),np.array(axis['point_mm']))
        geometry['feedback_usable']=axis['valid'] and not axis['fallback']
        geometry['replayed_auto_axis']=axis
        return geometry

    def close(self):
        self.process.stdin.close();self.process.wait(timeout=5)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_value,
                                    allow_nan=False), encoding='utf-8')


def pipe_frame(model):
    line = model.numeric('pipe_centerline').data.reshape(-1,4)
    directions = np.diff(line[:,:3],axis=0)
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    index = np.flatnonzero(directions @ directions[0] < .99)[0]
    return line[index,:3]*1000, directions[0], directions[index], float(line[0,3]*1000)


def true_axis_geometry(g, direction, corner):
    """Same A-D construction as baseline, but project onto known source axis."""
    result = copy.deepcopy(g)
    direction = unit(direction)
    horizontal = np.array([0.,1.,0.])
    if abs(horizontal @ direction)>.9: horizontal = np.array([0.,0.,1.])
    horizontal = unit(horizontal-(horizontal@direction)*direction)
    vertical = np.cross(direction,horizontal)
    points = np.asarray(g['points_mm'])
    xy = (points-corner) @ np.column_stack([horizontal,vertical])
    gamma_min = g['curve']['gamma_min_rad']
    sections = local_sections(points, np.tile(direction,(len(points)-1,1)),gamma_min)
    winding = math.copysign(1.,sum(row[2] for row in sections))
    groups = [[(sections[0][0],0.,0.)]]
    groups += [joint_segments(a,b,gamma,winding,gamma_min) for a,b,gamma,_,_ in sections]
    groups += [[(sections[-1][1],0.,0.)]]
    segments=[]
    for j,group in enumerate(groups):
        for length,radius,turn in group:
            segments.append(dict(kind='arc' if radius else 'line',length_mm=length,radius_mm=radius,
                turn_rad=turn,joint_index=j if 0<j<len(groups)-1 else None,
                start_mm=xy[0].tolist(),tangent=unit(xy[1]-xy[0]).tolist()))
    result['projection_axis'] = dict(point_mm=corner.tolist(),direction=direction.tolist(),
        horizontal=horizontal.tolist(),vertical=vertical.tolist(),valid=True,confidence=1.,fallback=False)
    result['projected_mm']=xy.tolist()
    result['curve']=dict(valid=True,segments=segments,gamma_min_rad=gamma_min)
    result['feedback_usable']=True
    return result


class Trial:
    def __init__(self, checkpoint, folder):
        self.folder = Path(folder); self.folder.mkdir(parents=True,exist_ok=False)
        for name in SOURCES:
            dest=self.folder/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        self.m,self.d,self.meta,self.original = load_checkpoint(checkpoint)
        self.scratch=mujoco.MjData(self.m)
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION))
        self.initial_state=self.get_state().copy()
        self.records=[];self.commands=[];self.controls=[];self.geometry_log=[]
        self.corner,self.source,self.target,self.radius=pipe_frame(self.m)
        self.jids=self.m.actuator_trnid[:,0];self.qadr=self.m.jnt_qposadr[self.jids]
        self.vadr=self.m.jnt_dofadr[self.jids]
        self.phase=self.meta['settings']['phase'];self.boundary=6.
        self.stage='initial';self.app=self.bridge=self.client=None
        self.old_port=os.environ.get('SNAKE_VISUALIZER_PORT')
        self.snapshot()
        appdir=Path(tempfile.mkdtemp(prefix='segmented-relay-'))
        exe=appdir/'SnakeRobotVisualizer.exe'
        shutil.copy2(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe',exe)
        shutil.copy2(ROOT/'validation/fixtures/app-state.json',appdir/'app-state.json')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        os.environ['SNAKE_VISUALIZER_PORT']=str(port)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        self.app=subprocess.Popen([str(exe)],cwd=appdir,env=os.environ.copy(),startupinfo=startup)
        self.bridge=MujocoExecutionBridge(self.m,self.d);self.bridge.period=.001
        mujoco.mj_setState(self.m,self.d,self.initial_state,mujoco.mjtState.mjSTATE_INTEGRATION)
        self.client=SnakeControlClient(port=port,timeout=.5)
        self.fresh(timeout=20.)

    def get_state(self):
        mujoco.mj_getState(self.m,self.d,self.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        return self.state

    def fresh(self,timeout=5.):
        start=time.time()*1000;deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.bridge.publish(self.d)
            try:
                g=self.client.get_geometry()
                if g['sample_unix_ms']>=start and g['target_state'].get('mujoco_executor_ready'):
                    return g
            except (OSError,RuntimeError,KeyError): pass
            time.sleep(.005)
        raise RuntimeError('isolated relay feedback timeout')

    def snapshot(self):
        # Compute diagnostic contacts in a copy: do not perturb warm starts.
        m=self.m;d=self.scratch
        mujoco.mj_setState(m,d,self.get_state(),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m,d)
        points=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{i}').xanchor for i in range(1,21)],d.site('s_tail').xpos])*1000
        forces=np.zeros((3,21));ground=False
        pipe_names=['pipe_straight_01','pipe_straight_03','pipe_unused_branch_02']
        for k,ct in enumerate(d.contact):
            names=[m.geom(int(x)).name for x in ct.geom]
            for side in (0,1):
                body=m.body(int(m.geom_bodyid[ct.geom[1-side]])).name
                module=0 if body=='head' else int(body[4:]) if body.startswith('link') else None
                if module is None:continue
                ground |= names[side]=='floor'
                if names[side] in pipe_names:
                    f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f)
                    forces[pipe_names.index(names[side]),module]+=max(0.,f[0])
        centers=(points[:-1]+points[1:])/2;delta=centers-self.corner
        ds=np.linalg.norm(delta-(delta@self.source)[:,None]*self.source,axis=1)
        dt=np.linalg.norm(delta-(delta@self.target)[:,None]*self.target,axis=1)
        # Independent transfer audit. Clearance includes the 29 mm shell radius.
        region=(dt+POLICY['region_margin_mm']<ds)&(delta@self.target>self.radius)&(dt<self.radius+29+POLICY['region_clearance_mm'])
        row=dict(time=float(d.time),qpos=self.d.qpos.copy(),qvel=self.d.qvel.copy(),ctrl=self.d.ctrl.copy(),
                 points_mm=points,normal_force=forces,ground=ground,region=region,
                 phase=self.phase,boundary_links=self.boundary,stage=self.stage,
                 error_deg=np.degrees(d.ctrl-d.qpos[self.qadr]),torque=d.qfrc_actuator[self.vadr].copy(),
                 applied_id=self.bridge.applied_command_id if self.bridge else 0)
        self.records.append(row)
        return row

    def command(self,targets,geometry):
        ids=self.jids;bounds=self.m.jnt_range[ids]
        if not np.isfinite(targets).all() or np.any(targets<bounds[:,0]) or np.any(targets>bounds[:,1]):
            raise ValueError('invalid execution target')
        record=dict(id=len(self.commands)+1,phase=self.phase,time=float(self.d.time),target=targets.tolist(),applied=False)
        self.commands.append(record)
        # Re-fetch if geometry computation consumed the protocol freshness window.
        if time.time()*1000-geometry['sample_unix_ms']>200: geometry=self.fresh()
        self.client.submit_targets(geometry,np.degrees(targets*self.bridge.signs),record['id'],execution='mujoco')
        deadline=time.monotonic()+5
        while not self.bridge.apply_received(self.d,expected_targets=targets):
            self.bridge.publish(self.d)
            if time.monotonic()>deadline:raise RuntimeError('relay target timeout')
            time.sleep(.002)
        record.update(applied=True,delivery_id=self.bridge.applied_delivery_id)

    def advance(self,duration):
        for i in range(round(duration/self.m.opt.timestep)):
            mujoco.mj_step(self.m,self.d)
            if (i+1)%10==0:self.snapshot()

    def close(self):
        if self.client:self.client.close()
        if self.bridge:self.bridge.close()
        if self.app and self.app.poll() is None:self.app.terminate();self.app.wait(timeout=10)
        if self.old_port is None:os.environ.pop('SNAKE_VISUALIZER_PORT',None)
        else:os.environ['SNAKE_VISUALIZER_PORT']=self.old_port
        np.savez_compressed(self.folder/'trace.npz',**{k:np.asarray([r[k] for r in self.records]) for k in self.records[0]})
        np.savez_compressed(self.folder/'initial_state.npz',state=self.initial_state)
        write_json(self.folder/'commands.json',self.commands)
        write_json(self.folder/'controls.json',self.controls)
        write_json(self.folder/'geometry.json',self.geometry_log)


def run(checkpoint, folder, method, options, max_steps=80):
    sim=None;axis_replay=None
    result=dict(method=method,options=asdict(options),checkpoint=str(Path(checkpoint).resolve()),
                policy=POLICY,max_steps=max_steps,outcome='engineering_failure',passed=False)
    try:
        sim=Trial(checkpoint,folder)
        m,d=sim.m,sim.d
        settings=sim.meta['settings'];links=np.array(settings['links']);link=links.mean()
        tracker=BoundaryTracker(options,link);switch=None
        start=sim.records[0];phase0=sim.phase
        if method=='original':axis_replay=AxisReplay()
        transfer_runs=np.zeros(21,int);initial_region=start['region']
        # Front prefix already belongs to the target side by the agreed manual
        # initial partition. A geometric hole inside it is NOT a new transfer.
        eligible=np.arange(21)>=math.ceil(options.initial_boundary_links)
        supported_losses=0
        result.update(mujoco=mujoco.__version__,checkpoint_hash=sha(Path(checkpoint)/'checkpoint.json'),
                      source_hashes={p:sha(ROOT/p) for p in SOURCES},
                      exe_sha256=sha(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe'),
                      initial_state_sha256=hashlib.sha256(sim.initial_state.tobytes()).hexdigest(),
                      settings=settings,initial_region=initial_region.tolist(),transfer_eligible=eligible.tolist(),
                      step0_target_gap_deg=None,switch_raw_gap_deg=None)
        for step in range(max_steps):
            sim.stage='rolling'
            g=sim.fresh()
            # Start every method with the user's exact captured geometric frame.
            if step==0:
                saved=copy.deepcopy(sim.meta['geometry_at_capture'])
                for key in ('run_id','config_id','sequence','sample_unix_ms','age_ms'):saved[key]=g[key]
                control_g=saved
            else:control_g=g
            if axis_replay is not None:
                replayed=axis_replay.geometry(control_g,sim.meta['geometry_at_capture']['projection_axis'])
                if step:control_g=replayed
            sim.geometry_log.append(control_g)
            phase=phase0-(step+1)*settings['phase_step']
            info=dict(step=step+1,phase=phase,boundary_links=tracker.value)
            if method=='original':
                if step==0:sim.original.key=(control_g['run_id'],control_g['config_id'],-1)
                targets=sim.original.compute(control_g,phase,d.ctrl.copy(),reference_phase=sim.phase)
                if targets is None:raise ValueError(sim.original.status)
                raw=sim.original.backbone.joint_angles(phase+sim.original.phase_offset)
            else:
                if method=='source_axis':
                    bone=build_backbone(true_axis_geometry(control_g,sim.source,sim.corner),sim.original.parameters)
                else:
                    row=sim.records[-1]
                    centers=(row['points_mm'][:-1]+row['points_mm'][1:])/2
                    if step:tracker.update(centers,sim.corner,sim.source,sim.target,row['normal_force'][1]>.1)
                    bone=MovingAxisBackbone(np.asarray(control_g['points_mm']),links,sim.original.parameters,
                        sim.source,sim.target,tracker.value*link,options.transition_links*link,
                        control_g['curve']['gamma_min_rad'])
                    info['boundary_links']=tracker.value
                if switch is None:
                    switch=SwitchContinuity(bone,d.ctrl.copy(),phase0)
                    result['switch_raw_gap_deg']=float(np.max(np.abs(bone.joint_angles(phase0+switch.offset)-d.ctrl))*180/np.pi)
                    result['step0_target_gap_deg']=float(np.max(np.abs(switch.angles(bone,phase0,0)-d.ctrl))*180/np.pi)
                weight=float(smoothstep((step+1)/options.switch_steps))
                raw=switch.angles(bone,phase,weight)
                # Same joint clipping policy as original ComplianceController.
                targets=np.clip(raw,m.jnt_range[sim.jids,0]+.01,m.jnt_range[sim.jids,1]-.01)
                info.update(switch_weight=weight,phase_offset=switch.offset)
                if method=='segmented':
                    nodes=np.linspace(0,links.sum(),120)
                    info.update(curve_phase0_mm=bone.evaluate(nodes)[:,:3].tolist(),
                                axis_directions=bone.axis_field.frames(nodes)[0][:,:,2].tolist())
            info.update(raw_targets_rad=raw.tolist(),targets_rad=targets.tolist(),
                        clipped_count=int(np.count_nonzero(abs(raw-targets)>1e-9)))
            sim.controls.append(info);sim.phase=phase;sim.boundary=tracker.value
            record_start=len(sim.records)
            sim.command(targets,g);sim.advance(POLICY['step_seconds'])
            row=sim.records[-1]
            transfer_runs=np.where(row['region']&eligible&~initial_region,transfer_runs+1,0)
            progress=float((row['points_mm'][0]-start['points_mm'][0])@sim.target)
            new=int(np.sum(transfer_runs>=POLICY['transfer_persistence_steps']))
            lost_during_step=False
            for sample in sim.records[record_start:]:
                both=np.any(sample['normal_force'][0]>.1) and np.any(sample['normal_force'][1]>.1)
                supported_losses=0 if both else supported_losses+1
                lost_during_step |= supported_losses*.01>=POLICY['lost_support_seconds']
            result.update(steps=step+1,progress_mm=progress,new_modules=new)
            print(method,step+1,round(progress,2),new,round(tracker.value,2),flush=True)
            if any(r['ground'] for r in sim.records[record_start:]) or lost_during_step:
                result['outcome']='support_lost';break
            if progress>=POLICY['advance_mm'] and new>=POLICY['new_modules']:
                result['outcome']='movement_gate';break
        else:result['outcome']='movement_budget'
        sim.stage='hold';hold_start=len(sim.records);before=sim.records[-1]
        if method=='segmented' and switch is not None:
            write_json(sim.folder/'last_command_curve.json',bone.sample_command(sim.phase+switch.offset,switch,weight))
        sim.advance(POLICY['hold_seconds'])
        hold=sim.records[hold_start:]
        head_progress=np.array([(r['points_mm'][0]-before['points_mm'][0])@sim.target for r in hold])
        root_motion=np.linalg.norm(np.array([r['qpos'][:3]-before['qpos'][:3] for r in hold]),axis=1)*1000
        hold_ok=(all(not r['ground'] and np.any(r['normal_force'][0]>.1) and np.any(r['normal_force'][1]>.1) for r in hold)
                 and min(head_progress)>=-POLICY['hold_retreat_mm'] and max(root_motion)<=POLICY['hold_root_motion_mm'])
        result.update(hold_passed=bool(hold_ok),hold_max_retreat_mm=float(max(0,-min(head_progress))),
                      hold_max_root_motion_mm=float(max(root_motion)),
                      max_error_deg=float(max(np.max(abs(r['error_deg'])) for r in sim.records)),
                      clipped_joint_steps=sum(x['clipped_count'] for x in sim.controls),
                      boundary_final_links=tracker.value,
                      passed=bool(result['outcome']=='movement_gate' and hold_ok))
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}')
        import traceback
        result['traceback']=traceback.format_exc()
    finally:
        if axis_replay:axis_replay.close()
        if sim:sim.close()
        Path(folder).mkdir(parents=True,exist_ok=True)
        write_json(Path(folder)/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HORIZONTAL)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['original','source_axis','segmented'],required=True)
    parser.add_argument('--transition-links',type=float,default=4.)
    parser.add_argument('--steps',type=int,default=80)
    args=parser.parse_args()
    result=run(args.checkpoint,args.output,args.method,SegmentedOptions(transition_links=args.transition_links),args.steps)
    print(json.dumps(result,ensure_ascii=False,default=json_value),flush=True)
