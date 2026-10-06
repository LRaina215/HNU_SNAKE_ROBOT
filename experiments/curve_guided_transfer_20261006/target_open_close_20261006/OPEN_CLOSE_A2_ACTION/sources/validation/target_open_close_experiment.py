"""Bounded actual open-insert-close action; no true pose/contact feedback."""
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
from snake_control.local_transfer_score import LocalTransferScore
from snake_control.target_open_close_reference import TargetOpenCloseReference
from snake_control.target_open_close_control import TargetOpenCloseController
from target_open_close_geometry import BASE,PARENT,OLD,write

infrastructure.BASE=BASE


def pipe_inside_projection(points,feed):
    """Diagnostic open-chain projection closed by a chord; no topology proof."""
    sigma=451.5+feed-np.arange(22)*64.5
    ids=np.flatnonzero(sigma>=0)
    xy=(np.asarray(points)[ids]-[1100.,0.,1000.])[:,[0,2]]
    angles=np.arctan2(xy[:,1],xy[:,0])
    turns=(np.roll(angles,-1)-angles+np.pi)%(2*np.pi)-np.pi
    return bool(abs(sum(turns))>np.pi)


class OpenCloseTrial(StaticTrial):
    def __init__(self,version):
        self.exact_contacts=[]
        self.action=0.;self.opening=0.
        infrastructure.Trial.__init__(self,'ACTION',version,'initial_boundary_pose.npz')
        shutil.copy2(BASE/'reference_ready.npz',self.case/'reference.npz')
        shutil.copy2(BASE/'original_reference.npz',self.case/'original_reference.npz')

    def snapshot(self):
        super().snapshot()
        self.rows[-1].update(action_mm=self.action,opening_mm=self.opening,
            reference_plateau_radius_mm=95.+self.opening)

    def initialize_relay(self):
        infrastructure.Trial.initialize_relay(self)
        with np.load(OLD/'R2_C0/trace.npz') as prior:
            self.kinematic.qpos[:]=prior['qpos'][0]
        mujoco.mj_kinematics(self.m,self.kinematic)
        registered=self.kinematic.body('head').xmat.reshape(3,3).copy()
        self.controller=TargetOpenCloseController(
            TargetOpenCloseReference(self.case/'reference.npz',self.case/'original_reference.npz'),
            self.controller.config,self.d.ctrl.copy(),self.encoder_fk,registered)
        write(self.case/'config.json',self.controller.config)


def stopped_holding(rows):
    end=rows[-1]['sim_time'];last2=[r for r in rows if r['sim_time']>=end-2-1e-8]
    p=np.array([r['centers_mm'] for r in last2])
    drift=float(np.linalg.norm(p-p[0],axis=2).max())
    selected=[r for r in rows if r['sim_time']>=end-5-1e-8]
    pp=np.array([r['centers_mm'] for r in selected]);tt=np.array([r['sim_time'] for r in selected])
    speed=float(np.linalg.norm(np.diff(pp,axis=0)/np.diff(tt)[:,None,None],axis=2).max())
    screen,maximum=holding(rows)
    recent=[r for r in rows if r['sim_time']>=end-1-1e-8]
    inside=np.mean([pipe_inside_projection(r['points_mm'],r['feed_mm']) for r in recent])
    passed=bool(screen and drift<=6.45 and speed<5 and inside>=.95)
    return dict(stable_front_grasp=passed,last2s_all_module_drift_mm=drift,
        last5s_module_maximum_speed_mm_s=speed,original_last1s_two_pipe_screen=screen,
        last1s_screen_speed_mm_s=float(maximum),last1s_pipe_inside_projected_front_polygon_fraction=float(inside),
        polygon_limitation='closure chord is a descriptive projection, not closed-body topology or sufficient grip evidence')


def run(version='OPEN_CLOSE_A1'):
    if version not in ('OPEN_CLOSE_A1','OPEN_CLOSE_A2'):
        raise ValueError('Only registered action/stop-interface repair allowed')
    action_only_stall_log=version=='OPEN_CLOSE_A2'
    registry=json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    if not registry['further_dynamics_authorized']:raise RuntimeError('Action study stopped')
    gate=json.loads((BASE/'path_gate.json').read_text(encoding='utf8'))
    if not gate['geometry_and_command_timing_pass']:raise RuntimeError('Geometry/timing not admitted')
    interface=json.loads((BASE/'interface_check.json').read_text(encoding='utf8'))
    if not interface['interface_and_nominal_opening_pass']:raise RuntimeError('Opening/interface not admitted')
    with (PARENT/'trial_ledger.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;previous=list(reader)
    if len(previous)>=19 or len(previous)-15>=4:raise RuntimeError('Action budget exhausted')
    with (BASE/'trial_ledger.csv').open(encoding='utf8',newline='') as f:
        local=list(csv.DictReader(f))
    if len(local)!=len(previous)-15:raise RuntimeError('Total/local ledger mismatch')
    trial=OpenCloseTrial(version)
    schedule=np.load(BASE/'command_schedule.npz');total=float(schedule['time_s'][-1])
    write(trial.case/'plan.json',dict(version=version,total_s=total,previous_trials=len(previous),budget=19,
        radius_mm=[95,130,95],material_feed_mm=[0,161.25],open_insert_endpoint_mm=80.625,
        action='5s entry,12s opening,45s open insertion,12s closing,10s held,45s resume closed,10s end held',
        target_AD_gain='1-opening/35: initial1, fully open0, closing restores to1; rear gain1 unchanged',
        actuator_stiffness='original kp127 all stages; gain gate modifies front AD update, not torque or actuator stiffness',
        control_inputs='20encoders/ACK/history/initial prior only; no world/contact/true-axis input',
        opening_target_support='may be zero temporarily; rear actual>=2contact modules required after entry, four .05s misses stop',
        stop='original ground/nonadjacent selfcontact, total loss/highspeed, interface invalid; postclose/end hold gates',
        stationary_gate='last1s original contact/speed screen, last2s <=6.45mm drift, last5s module speed<5mm/s, projected front polygon contains pipe axis>=95%last1s',
        grading='opening demonstration separated from original failed interval and complete M8/M9 transfer score',
        M8_M9_stall_rule='record legacy failure; not an action interruption; rear/physical danger stops unchanged' if action_only_stall_log else 'legacy10 active seconds interruption',
        stop_interface_repair='A1 opened but legacy transfer stall interrupted before scheduled close; A2 executes same139s geometry/control with old score failure retained' if action_only_stall_log else None,
        old_C0='starting complete d0 branch equivalent to old R2; old66.8s C0 not extrapolated to139s',
        no_body_reset_after_first_step=True,no_virtual_support=True,real_hardware=False,full_V3_path='NOT_READY'))
    for name in ['snake_control/target_open_close_reference.py','snake_control/target_open_close_control.py',
        'snake_control/executable_grasp_state.py','snake_control/local_material_reference.py',
        'snake_control/curve_grasp_revision.py','snake_control/local_transfer_score.py','snake_control/three_segment_compliance.py',
        'snake_control/pole_compliance.py','snake_control/mujoco_execution_bridge.py','snake_control/mujoco_visualizer_bridge.py',
        'validation/target_open_close_experiment.py','validation/target_open_close_geometry.py','validation/target_open_close_review.py',
        'validation/curve_guided_experiment.py','validation/v3_static_holding_experiment.py',
        'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml']:
        dest=trial.case/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    result=dict(outcome='engineering_failure',action_completed=False,post_close_grasp=False,
        original_failed_interval_reached=False,end_grasp=False,full_local_handoff=False,
        initial_holding_reused='R2_SPEED3_PAIR_C0 complete branch equivalence',
        old_V3_full_path='NOT_READY')
    scorer=LocalTransferScore(trial.rows[0]['points_mm'],trial.rows[0]['contact_force_N'])
    started=time.monotonic();lost=fast=rear_lost=0;active=0.
    try:
        trial.initialize_relay();previous_stage='entry'
        for k in range(len(schedule['time_s'])-1):
            t=float(schedule['time_s'][k]);stage=str(schedule['stage'][k])
            action=float(schedule['action_mm'][k]);feed=float(schedule['feed_mm'][k]);opening=float(schedule['opening_mm'][k])
            if stage=='feed_closed' and previous_stage=='post_close_hold':
                stats=stopped_holding(trial.rows);result.update(post_close_diagnostics=stats,
                    post_close_grasp=stats['stable_front_grasp'],action_completed=True)
                if not stats['stable_front_grasp']:
                    result['outcome']='cannot_hold_after_open_insert_close';break
            trial.action,trial.feed,trial.opening,trial.stage=action,feed,opening,stage
            geometry=trial.fresh();measured=np.radians(np.asarray(geometry['angles_deg'])*trial.bridge.signs)
            target,control=trial.controller.compute(measured,action,45.25+t,.05)
            control.update(time_s=t,stage=stage,action_mm=action,feed_mm=feed,opening_mm=opening,
                reference_plateau_radius_mm=95.+opening,measured_rad=measured.tolist(),applied=False)
            trial.controls.append(control);trial.dispatch(target,geometry);trial.controller.accept(target,action);control['applied']=True
            if k==0:write(trial.case/'physics_started.json',dict(cumulative_trial=len(previous)+1,never_restart=True))
            if stage.startswith('feed_'):active+=.05
            stalled=False
            for j in range(50):
                mujoco.mj_step(trial.m,trial.d)
                if (j+1)%10==0:
                    trial.snapshot();r=trial.rows[-1]
                    stalled |= scorer.record(r['sim_time'],stage,feed,r['points_mm'],r['contact_force_N'],active)
            last=trial.rows[-1];recent=trial.rows[-6:]
            speed=float(np.linalg.norm((recent[-1]['centers_mm']-recent[0]['centers_mm'])/.05,axis=1).max())
            lost=lost+1 if not np.any(last['contact_force_N']>.1) else 0
            fast=fast+1 if speed>500 else 0
            rear_lost=rear_lost+1 if np.count_nonzero(last['contact_force_N'][0]>.1)<2 else 0
            if last['ground'] or last['self_contact']:result['outcome']='ground_or_nonadjacent_self_contact';break
            if t>=1 and (lost>=4 or fast>=4):result['outcome']='support_loss_or_uncontrolled_motion';break
            if t>=5 and rear_lost>=4:result['outcome']='rear_support_below_two_modules_0p2s';break
            if stalled:
                if 'first_legacy_material_stall_time_s' not in result:
                    result['first_legacy_material_stall_time_s']=float(trial.d.time)
                    result['legacy_transfer_stall_pass']=False
                if not action_only_stall_log:
                    result['outcome']='actual_material_stall_10_active_seconds';break
            if k==99:
                valid,maximum=holding(trial.rows);result.update(entry_screen_pass=valid,entry_maximum_speed_mm_s=maximum)
                if not valid:result['outcome']='initial_cannot_hold';break
            if feed>125.:result['original_failed_interval_reached']=True
            previous_stage=stage
            if k%100==99:
                print(json.dumps(dict(physical_time_s=float(trial.d.time),stage=stage,feed_mm=feed,radius_mm=95+opening,
                    source_contacts=int(np.count_nonzero(last['contact_force_N'][0]>.1)),
                    target_contacts=int(np.count_nonzero(last['contact_force_N'][1]>.1)),max_module_speed_mm_s=speed)),flush=True)
        else:
            stats=stopped_holding(trial.rows)
            result.update(action_completed=True,end_grasp=stats['stable_front_grasp'],end_hold_diagnostics=stats,
                outcome='open_close_and_original_interval_grasp_pass' if stats['stable_front_grasp'] else 'end_grasp_failed',
                full_local_handoff='NOT_ESTABLISHED_WITHOUT_DURATION_MATCHED_C0')
    except Exception as e:
        result.update(outcome='controller_or_interface_stop',error=repr(e))
    finally:
        result.update(physical_seconds=float(trial.d.time),wall_seconds=time.monotonic()-started,
            material_score=scorer.result(),new_dynamics_trials=int(trial.d.time>0),
            cumulative_dynamics_trials=len(previous)+int(trial.d.time>0),commands=len(trial.commands),
            all_commands_applied=all(c['applied'] for c in trial.commands))
        trial.close();write(trial.case/'material_score.json',scorer.rows)
    t=np.array([r['sim_time'] for r in trial.rows]);p=np.array([r['centers_mm'] for r in trial.rows])
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
    result.update(maximum_module_speed_mm_s=float(speed.max()) if len(speed) else 0.,
        maximum_tracking_error_deg=float(np.degrees(abs(np.array([r['q']-r['ctrl'] for r in trial.rows]))).max()),
        root_displacement_mm=float(np.linalg.norm(trial.d.qpos[:3]-trial.initial_qpos[:3])*1000),
        last_applied_feed_mm=float(trial.rows[-1]['feed_mm']))
    write(trial.case/'result.json',result)
    if trial.d.time>0:
        row={f:'' for f in fields};row.update(trial_id=trial.case.name,stage='target_open_close_action',version=version,condition='ACTION',
            initial_usable=result.get('entry_screen_pass',False),physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],
            outcome=result['outcome'],evidence=trial.case.relative_to(ROOT).as_posix(),initial_file=trial.initial_name,
            comparability='same_R2_held_initial_target_radius_open_insert_close_original_capabilities',
            maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s','NOT_REACHED'),
            maximum_tracking_error_deg=result['maximum_tracking_error_deg'],root_displacement_mm=result['root_displacement_mm'],
            commanded_feed_mm=result['last_applied_feed_mm'],metrics_evidence=(trial.case/'result.json').relative_to(PARENT).as_posix())
        for path in [PARENT/'trial_ledger.csv',BASE/'trial_ledger.csv']:
            with path.open('a',encoding='utf8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
        registry.update(status='EVALUATING',new_dynamics=len(previous)-14,total_dynamics=len(previous)+1,
            last_case=trial.case.name,last_outcome=result['outcome'])
        write(BASE/'revision_registry.json',registry)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='OPEN_CLOSE_A1')
    run(parser.parse_args().version)
