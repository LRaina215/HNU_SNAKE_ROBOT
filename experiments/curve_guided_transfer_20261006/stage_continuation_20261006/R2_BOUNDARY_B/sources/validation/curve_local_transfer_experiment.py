"""Paired interface holding and local material transfer with frozen R2 logic.

Requires separately checked local reference/timing gate. No online true pose.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation'),str(ROOT/'visualizer')]
import argparse
import csv
import json
import shutil
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import curve_guided_experiment as infrastructure
from v3_static_holding_experiment import StaticTrial
from curve_grasp_revision_experiment import holding
from snake_control.local_material_reference import CoupledMaterialCurve,BoundaryMaterialCurve
from snake_control.executable_grasp_state import EquilibriumGraspController
from snake_control.local_transfer_score import LocalTransferScore

PARENT=ROOT/'experiments/curve_guided_transfer_20261006'
BASE=PARENT/'stage_continuation_20261006'
PATH=BASE/'local_path'
VERSION='R2_PATH'
BOUNDARY=False
infrastructure.BASE=BASE
write=infrastructure.write


def replay_local(case):
    """Saved dynamics only, with scored material units visible in the overlay."""
    cv2=infrastructure.cv2
    with np.load(case/'trace.npz') as z:
        model=mujoco.MjModel.from_xml_path(str(infrastructure.MODEL));data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=480,width=720);camera=mujoco.MjvCamera()
        camera.lookat[:]=[1.05,.15,.98];camera.distance=1.6;camera.azimuth=135;camera.elevation=-25
        video=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(720,480))
        if not video.isOpened():raise RuntimeError('Replay video writer unavailable')
        for i in range(0,len(z['sim_time']),5):
            data.qpos[:]=z['qpos'][i];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            gl=renderer.scene.camera
            eye=(gl[0].pos+gl[1].pos)/2
            forward=gl[0].forward/np.linalg.norm(gl[0].forward);up=gl[0].up/np.linalg.norm(gl[0].up)
            right=np.cross(forward,up);focal=480/(2*np.tan(np.radians(model.vis.global_.fovy)/2))
            for module,color in [(8,(20,20,255)),(9,(255,80,20))]:
                delta=z['centers_mm'][i,module]/1000-eye;depth=float(delta@forward)
                if depth>0:
                    xy=(int(360+focal*(delta@right)/depth),int(240-focal*(delta@up)/depth))
                    cv2.circle(frame,xy,5,color,2)
                    cv2.putText(frame,'M'+str(module),(xy[0]+7,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            cv2.putText(frame,f'{case.name} t={z["sim_time"][i]:.2f}s {z["stage"][i]} d={z["feed_mm"][i]:.2f}mm',
                (10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.putText(frame,'RED M8 / BLUE M9; ACTUAL SAVED PHYSICS, 1x',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            video.write(frame)
            if i in (0,5*((len(z['sim_time'])//2)//5),5*((len(z['sim_time'])-1)//5)):
                cv2.imwrite(str(case/f'frame_{i:05d}.png'),frame)
        video.release();renderer.close()
        write(case/'replay_metadata.json',dict(fps=20,sampled_physics_interval_s=.05,speed=1.,
            last_physics_time_s=float(z['sim_time'][-1]),initial_frame_included=True,marked_modules=[8,9]))


class TransferTrial(StaticTrial):
    def __init__(self,condition,perturb=False):
        self.exact_contacts=[]
        self.condition=condition
        prefix='initial_boundary_pose' if BOUNDARY else 'initial_path_pose'
        name=prefix+('_perturb' if perturb else '')+'.npz'
        infrastructure.Trial.__init__(self,condition,VERSION,name)
        shutil.copy2(PATH/'reference_ready.npz',self.case/'reference.npz')

    def initialize_relay(self):
        infrastructure.Trial.initialize_relay(self)
        registered=self.d.body('head').xmat.reshape(3,3).copy()
        if BOUNDARY:
            with np.load(BASE/'R2_C0/trace.npz') as prior:self.kinematic.qpos[:]=prior['qpos'][0]
            mujoco.mj_kinematics(self.m,self.kinematic)
            registered=self.kinematic.body('head').xmat.reshape(3,3).copy()
        self.controller=EquilibriumGraspController((BoundaryMaterialCurve if BOUNDARY else CoupledMaterialCurve)(self.case/'reference.npz'),
            self.controller.config,self.d.ctrl.copy(),self.encoder_fk,
            registered)
        write(self.case/'config.json',self.controller.config)


def window_displacement(rows,seconds=2.):
    end=rows[-1]['sim_time']
    selected=[r for r in rows if r['sim_time']>=end-seconds-1e-8]
    centers=np.array([r['centers_mm'] for r in selected])
    return float(np.linalg.norm(centers-centers[0],axis=2).max())


def holding_result(rows,controls,total):
    t=np.array([r['sim_time'] for r in rows]);p=np.array([r['centers_mm'] for r in rows])
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
    counts=np.count_nonzero(np.array([r['contact_force_N'] for r in rows])>.1,axis=2)
    ct=np.array([r['time_s'] for r in controls]);h=np.degrees([r['retained_shape_rad'] for r in controls])
    tail=t>=total-10-1e-8;pt=p[tail]
    drift=float(np.linalg.norm(pt-pt[0],axis=2).max())
    state=float(np.ptp(h[ct>=total-10-1e-8],axis=0).max())
    ctrl=np.array([r['ctrl'] for r in rows])
    margin=float(90-np.degrees(abs(ctrl[tail])).max())
    lastspeed=float(speed[t[1:]>=total-5-1e-8].max())
    qq=np.array([r['qpos'] for r in rows])[tail][:,[4,5,6,3]];rots=Rotation.from_quat(qq)
    turn=float(np.degrees((rots*rots[0].inv()).magnitude()).max())
    screen,maximum=holding(rows)
    continuous=bool(np.all(counts[t>=5-1e-8]>=2))
    passed=screen and continuous and drift<=6.45 and state<=1 and margin>=1 and lastspeed<5 and turn<=5
    return dict(full_holding_pass=bool(passed),final_screen=screen,final_maximum_speed_mm_s=maximum,
        last10s_maximum_module_drift_mm=drift,last10s_state_range_deg=state,
        last10s_minimum_command_angle_margin_deg=margin,last5s_maximum_speed_mm_s=lastspeed,
        last10s_head_rotation_deg=turn,continuous_two_pipe_support_after_entry=continuous)


def run(condition,boundary=False):
    global PATH,VERSION,BOUNDARY
    if boundary:
        PATH=BASE/'local_path_boundary';VERSION='R2_BOUNDARY';BOUNDARY=True
    gate=json.loads((PATH/'path_gate.json').read_text(encoding='utf8'))
    if not gate.get('geometry_and_command_timing_pass',False):
        raise RuntimeError('Continuous local geometry and command timing gate is closed')
    if BOUNDARY:
        check=json.loads((PATH/'interface_check.json').read_text(encoding='utf8'))
        if not check.get('warm_start_next_command_identity_pass'):raise RuntimeError('Inherited R2 boundary is inconsistent')
    with (PARENT/'trial_ledger.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;previous=list(reader)
    if len(previous)>=12:raise RuntimeError('Cumulative dynamics limit reached')
    if condition not in ('PAIR_C0','B','PAIR_C0_PERTURB','B_FREEZE'):raise ValueError('Unknown phase condition')
    is_hold=condition.startswith('PAIR_C0')
    if not is_hold:
        c0=json.loads((BASE/(VERSION+'_PAIR_C0')/'result.json').read_text(encoding='utf8'))
        if not c0['full_holding_pass']:raise RuntimeError('Interface paired holding must pass before B')
    if condition in ('PAIR_C0_PERTURB','B_FREEZE'):
        b=json.loads((BASE/(VERSION+'_B')/'result.json').read_text(encoding='utf8'))
        if not b.get('local_transfer_pass'):raise RuntimeError('Primary actual transfer must pass before freeze check')
    if condition=='B_FREEZE':
        pc0=json.loads((BASE/(VERSION+'_PAIR_C0_PERTURB')/'result.json').read_text(encoding='utf8'))
        if not pc0['full_holding_pass']:raise RuntimeError('Same-perturbation holding is required for freeze comparison')
    trial=TransferTrial(condition,perturb=condition in ('B_FREEZE','PAIR_C0_PERTURB'))
    schedule=np.load(PATH/'command_schedule.npz')
    planned_times=schedule['time_s'];planned_feed=schedule['feed_mm'];stages=schedule['stage'];active=schedule['active_time_s']
    total=float(gate['paired_C0_duration_s'] if is_hold else planned_times[-1])
    n=int(round(total/.05))
    result=dict(outcome='engineering_failure',condition=condition,full_holding_pass=False,local_transfer_pass=False,
        entry_screen_pass=False,controller='frozen R2',old_V3_full_path='NOT_READY')
    write(trial.case/'plan.json',dict(condition=condition,total_s=total,previous_trials=len(previous),budget=12,
        distinguish='Can the held shape enter the checked local reference without losing support?' if is_hold else
            'Do M8/M9 migrate in order beyond the junction with pauses, beyond matched C0 passive drift?',
        changed='reference/interface and recorded held initial placement; frozen R2 algorithm/gains/physics',
        boundary='saved V3 nominal/tangent, inherited old R2 initial axis prior and mature controller age' if BOUNDARY else 'near-held nominal with new registration',
        initial='R2 held qpos and acknowledged ctrl, qvel zero before first physical step',
        perturb='root x +1mm before first step' if condition in ('B_FREEZE','PAIR_C0_PERTURB') else 'none',
        scored_modules=[8,9],region='both endpoints beyond y75, radial 56.5..120, actual target force >.1N for .5s',
        score_confirmation='crossing time recorded at first sample of sustained .5s occupancy',
        stall_stop='actual mean station of nodes8/9/10 fails to exceed last highwater by 1mm for 10 active seconds',
        pause_and_end='last 1s original holding screen plus last2s all-module displacement<=6.45mm',
        active_endpoint='both ordered entries during feeding, final persistence, resume station advance>=5mm and each body module y advance beyond paired C0 by>=6.45mm',
        feedback='encoders/acknowledged targets only; exact positions and contacts only scoring/stops',
        no_pose_or_velocity_reset_after_start=True,real_connection=False))
    for name in ['snake_control/executable_grasp_state.py','snake_control/local_material_reference.py','snake_control/local_transfer_score.py',
                 'validation/curve_local_transfer_experiment.py']:
        dest=trial.case/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    scorer=LocalTransferScore(trial.rows[0]['points_mm'],trial.rows[0]['contact_force_N'])
    trial.score_rows=[]
    start=time.monotonic();lost=fast=0
    try:
        trial.initialize_relay()
        previous_stage='entry';pause_pass=None
        for k in range(n):
            t=k*.05
            if is_hold:
                feed=0.;stage='entry' if t<5 else 'hold_pair_C0';elapsed_active=0.
            else:
                feed=float(planned_feed[k]);stage=str(stages[k]);elapsed_active=float(active[k])
            trial.feed=feed;trial.stage=stage
            if stage=='feed_2' and previous_stage=='pause':
                valid,speed=holding(trial.rows);drift=window_displacement(trial.rows)
                pause_pass=bool(valid and drift<=6.45)
                result.update(pause_screen_pass=pause_pass,pause_maximum_speed_mm_s=speed,pause_last2s_module_displacement_mm=drift)
                if not pause_pass:result['outcome']='cannot_hold_at_middle_pause';break
            geometry=trial.fresh();measured=np.radians(np.asarray(geometry['angles_deg'])*trial.bridge.signs)
            target,control=trial.controller.compute(measured,feed,t+(45.25 if BOUNDARY else 0.),.05)
            control.update(time_s=t,stage=stage,measured_rad=measured.tolist(),applied=False)
            trial.controls.append(control);trial.dispatch(target,geometry);trial.controller.accept(target,feed);control['applied']=True
            if k==0:write(trial.case/'physics_started.json',dict(cumulative_trial=len(previous)+1,never_restart=True))
            stalled=False
            for j in range(50):
                mujoco.mj_step(trial.m,trial.d)
                if (j+1)%10==0:
                    trial.snapshot();r=trial.rows[-1]
                    if not is_hold:
                        stalled|=scorer.record(r['sim_time'],stage,feed,r['points_mm'],r['contact_force_N'],elapsed_active)
            last=trial.rows[-1];recent=trial.rows[-6:]
            speed=float(np.linalg.norm((recent[-1]['centers_mm']-recent[0]['centers_mm'])/.05,axis=1).max())
            lost=lost+1 if not np.any(last['contact_force_N']>.1) else 0;fast=fast+1 if speed>500 else 0
            if last['ground'] or last['self_contact']:result['outcome']='ground_or_nonadjacent_self_contact';break
            if t>=1 and (lost>=4 or fast>=4):result['outcome']='support_loss_or_uncontrolled_motion';break
            if stalled:result['outcome']='actual_material_stall_10_active_seconds';break
            if k==99:
                valid,maximum=holding(trial.rows);result.update(entry_screen_pass=valid,entry_maximum_speed_mm_s=maximum)
                if not valid:result['outcome']='initial_cannot_hold_on_both_pipes';break
            previous_stage=stage
            if k%100==99:print(json.dumps(dict(condition=condition,time_s=float(trial.d.time),feed_mm=feed,stage=stage,speed_mm_s=speed)),flush=True)
        else:
            if is_hold:
                result.update(holding_result(trial.rows,trial.controls,total));result['outcome']='interface_holding_pass' if result['full_holding_pass'] else 'interface_holding_not_converged'
            else:
                end_ok,maximum=holding(trial.rows);drift=window_displacement(trial.rows)
                score=scorer.result();result.update(material_score=score,final_screen=end_ok,final_maximum_speed_mm_s=maximum,end_last2s_module_displacement_mm=drift)
                paired=VERSION+('_PAIR_C0_PERTURB' if condition=='B_FREEZE' else '_PAIR_C0')
                with np.load(BASE/paired/'trace.npz') as c0:
                    actual=np.array([r['centers_mm'] for r in trial.rows])
                    ci=int(np.abs(c0['sim_time']-trial.d.time).argmin())
                    net_extra=(actual[-1]-actual[0])[[8,9],1]-(c0['centers_mm'][ci]-c0['centers_mm'][0])[[8,9],1]
                second=[r for r in scorer.rows if r['stage']=='feed_2']
                resume=float(second[-1]['actual_station_y_mm']-second[0]['actual_station_y_mm']) if second else 0.
                result.update(extra_module_y_advance_over_paired_C0_mm=net_extra.tolist(),resume_actual_station_advance_mm=resume)
                passed=(score['ordered_active_entries'] and score['final_modules_in_target'] and
                    bool(pause_pass) and end_ok and drift<=6.45 and resume>=5 and np.all(net_extra>=6.45))
                result.update(local_transfer_pass=bool(passed),outcome='local_transfer_pass' if passed else 'local_endpoint_or_pause_or_C0_discrimination_failed')
    except Exception as error:
        result.update(outcome='controller_or_interface_stop',error=repr(error))
    finally:
        result.update(physical_seconds=float(trial.d.time),wall_seconds=time.monotonic()-start,
            new_dynamics_trials=int(trial.d.time>0),cumulative_dynamics_trials=len(previous)+int(trial.d.time>0),
            material_score=scorer.result() if not is_hold else None)
        trial.close();write(trial.case/'material_score.json',scorer.rows)
    p=np.array([r['centers_mm'] for r in trial.rows]);tt=np.array([r['sim_time'] for r in trial.rows])
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(tt)[:,None,None],axis=2)
    result.update(maximum_module_speed_mm_s=float(speed.max()) if len(speed) else 0.,
        maximum_tracking_error_deg=float(np.degrees(abs(np.array([r['q']-r['ctrl'] for r in trial.rows]))).max()),
        root_displacement_mm=float(np.linalg.norm(trial.d.qpos[:3]-trial.initial_qpos[:3])*1000),
        commands=len(trial.commands),all_commands_applied=all(c['applied'] for c in trial.commands))
    write(trial.case/'result.json',result)
    if result['new_dynamics_trials']:
        row={f:'' for f in fields};row.update(trial_id=trial.case.name,stage='local_path_interface_and_transfer',version=VERSION,condition=condition,
            initial_usable=result['full_holding_pass'] if is_hold else result['entry_screen_pass'],
            physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],outcome=result['outcome'],
            evidence=trial.case.relative_to(ROOT).as_posix(),initial_file=trial.initial_name,
            comparability='same_held_initial_and_R2_local_reference' if condition not in ('B_FREEZE','PAIR_C0_PERTURB') else 'frozen_version_root_x_plus1mm_matched_pair',
            maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s','NOT_REACHED'),
            maximum_tracking_error_deg=result['maximum_tracking_error_deg'],root_displacement_mm=result['root_displacement_mm'],
            commanded_feed_mm=trial.feed,metrics_evidence=(trial.case/'result.json').relative_to(PARENT).as_posix())
        with (PARENT/'trial_ledger.csv').open('a',encoding='utf8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
        replay_local(trial.case)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--condition',required=True);parser.add_argument('--boundary',action='store_true')
    args=parser.parse_args();run(args.condition,args.boundary)
