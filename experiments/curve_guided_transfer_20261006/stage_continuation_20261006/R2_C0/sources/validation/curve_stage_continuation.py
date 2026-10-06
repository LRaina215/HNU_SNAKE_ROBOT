"""Authorized bounded continuation. Old trial entry points and gates untouched."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation'), str(ROOT/'visualizer')]
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
from snake_control.curve_grasp_revision import FittedMaterialCurve
from snake_control.executable_grasp_state import ExecutableGraspController, EquilibriumGraspController

PARENT = ROOT/'experiments/curve_guided_transfer_20261006'
BASE = PARENT/'stage_continuation_20261006'
V3 = PARENT/'structural_revision_v3'
LEDGER = PARENT/'trial_ledger.csv'
infrastructure.BASE = BASE
write = infrastructure.write


class PhaseTrial(StaticTrial):
    def __init__(self, revision):
        self.exact_contacts = []
        self.revision = revision
        infrastructure.Trial.__init__(self, 'C0', 'R'+str(revision), 'initial_pose.npz')

    def initialize_relay(self):
        infrastructure.Trial.initialize_relay(self)
        controller_type = ExecutableGraspController if self.revision==1 else EquilibriumGraspController
        self.controller = controller_type(FittedMaterialCurve(V3/'fitted_reference.npz'),
            self.controller.config, self.d.ctrl.copy(), self.encoder_fk,
            self.d.body('head').xmat.reshape(3, 3).copy())
        write(self.case/'config.json', self.controller.config)


def read_ledger():
    with LEDGER.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def late_metrics(trial):
    times = np.array([r['sim_time'] for r in trial.rows])
    centers = np.array([r['centers_mm'] for r in trial.rows])
    speed = np.linalg.norm(np.diff(centers, axis=0)/np.diff(times)[:, None, None], axis=2)
    counts = np.count_nonzero(np.array([r['contact_force_N'] for r in trial.rows])>.1, axis=2)
    q = np.array([r['q'] for r in trial.rows]); ctrl = np.array([r['ctrl'] for r in trial.rows])
    result = dict(maximum_module_speed_mm_s=float(speed.max()),
        maximum_tracking_error_deg=float(np.degrees(abs(q-ctrl)).max()),
        root_displacement_mm=float(np.linalg.norm(trial.d.qpos[:3]-trial.initial_qpos[:3])*1000),
        maximum_actual_torque_Nm=float(abs(np.array([r['torque'] for r in trial.rows])).max()))
    if times[-1] < 45.25-1e-8:
        return result
    late = times >= 35.25-1e-8
    late_centers = centers[late]
    drift = float(np.linalg.norm(late_centers-late_centers[0], axis=2).max())
    after_entry = times >= 5-1e-8
    continued = bool(np.all(counts[after_entry]>=2))
    ct = np.array([r['time_s'] for r in trial.controls])
    shapes = np.degrees(np.array([r['retained_shape_rad'] for r in trial.controls]))
    ptp = float(np.ptp(shapes[ct>=35.25-1e-8], axis=0).max())
    angular_margin = float(90-np.degrees(abs(ctrl[late])).max())
    last5_speed = float(speed[times[1:]>=40.25-1e-8].max())
    qq = np.array([r['qpos'] for r in trial.rows])[late][:, [4,5,6,3]]
    rotations = Rotation.from_quat(qq)
    turn = float(np.degrees((rotations*rotations[0].inv()).magnitude()).max())
    final_screen, final_speed = holding(trial.rows)
    convergence = (drift<=6.45 and ptp<=1. and angular_margin>=1. and last5_speed<5. and turn<=5.)
    result.update(last10s_maximum_module_drift_mm=drift, last10s_state_range_deg=ptp,
        last10s_minimum_command_angle_margin_deg=angular_margin, last5s_maximum_speed_mm_s=last5_speed,
        last10s_head_rotation_deg=turn, continuous_two_pipe_support_after_entry=continued,
        final_screen=final_screen, final_maximum_speed_mm_s=final_speed,
        holding_converged=bool(convergence), full_holding_pass=bool(final_screen and continued and convergence))
    return result


def run(revision):
    fields, previous = read_ledger()
    if len(previous)>=12:
        raise RuntimeError('Cumulative dynamics budget exhausted')
    registry=json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    if revision not in [r['round'] for r in registry['registered_rounds']] or revision not in (1,2):
        raise RuntimeError('Mechanism revision must be registered before physics')
    if (BASE/('R'+str(revision)+'_C0')).exists():
        raise FileExistsError('Preserve the existing attempt; do not repeat unchanged failures')
    if not (BASE/'initial_pose.npz').exists():
        shutil.copy2(PARENT/'v3_static_holding_20261006/initial_pose.npz', BASE/'initial_pose.npz')
    trial = PhaseTrial(revision)
    # Necessary saved-pose/model interface check, no physical evolution.
    initial = json.loads((V3/'geometry_result.json').read_text(encoding='utf8'))['rows'][0]
    if not np.allclose(trial.d.qpos, initial['qpos'], rtol=0, atol=1e-12):
        raise ValueError('Saved V3 d=0 initial state mismatch')
    if trial.m.nu!=20 or trial.m.opt.timestep!=.001 or trial.d.time!=0 or np.any(trial.d.qvel):
        raise ValueError('Original model/initial velocity interface mismatch')
    write(trial.case/'plan.json', dict(revision=revision, condition='C0', total_s=45.25,
        feed_mm=0., roll_rad=0., entry_s=5., previous_trials=len(previous), cumulative_limit=12,
        distinguish=('Does executable-state bookkeeping plus transported material gauge remove continuing deformation growth?' if revision==1 else
            'Does finite reference-restoring equilibrium preserve support while stopping free shape drift?'),
        changed=['state stores only acknowledged feasible targets', 'per-window encoder material-cloud gauge'] if revision==1 else
            ['add r*dt/0.5s reference-restoring term; R1 state/coordinates retained'],
        unchanged=['V3 d=0 placement', 'K .01/.02', 'physics', '90 degree and 10 degree/s limits'],
        original_screens='V3 last 1 s at 5 and 45.25; >=2 modules each pipe >.1N and speed <50mm/s',
        original_stops='ground/nonadjacent self; after 1s four control periods unsupported or speed >500mm/s; invalid feedback/geometry',
        added_convergence=dict(last10s_module_displacement_mm=6.45, last10s_retained_state_range_deg=1.,
            last5s_maximum_speed_mm_s=5., last10s_head_rotation_deg=5., last10s_command_limit_margin_deg=1.,
            continuous_two_pipe_support_after_entry=True),
        criteria_basis='0.1 link drift, 1 degree ~1.13mm per link; low late speed and state range demonstrate convergence; no late limit-wall holding',
        state_bound='r=q_ack-q_nom, bounded by original joint and per-command boxes; no latent rejected increment',
        coordinate_guard='noncollinear same-material-node transport; RMS<=link/4; not an observed pipe/world axis',
        feedback=['encoders','acknowledged target history'], score_only=['pose','contacts','velocities']))
    for name in ['snake_control/executable_grasp_state.py','validation/curve_stage_continuation.py']:
        dest=trial.case/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    result=dict(outcome='engineering_failure', full_holding_pass=False, condition='C0', revision=revision,
        entry_screen_pass=False, full_V3_path='NOT_READY', B_run=False, physical_parameters_changed=False)
    start=time.monotonic()
    try:
        trial.initialize_relay()
        lost=fast=0
        for k in range(905):
            t=k*.05;trial.feed=0.;trial.stage='entry' if t<5 else 'hold_C0'
            geometry=trial.fresh()
            measured=np.radians(np.asarray(geometry['angles_deg'])*trial.bridge.signs)
            target, control=trial.controller.compute(measured,0.,t,.05)
            control.update(time_s=t,stage=trial.stage,measured_rad=measured.tolist(),applied=False)
            trial.controls.append(control)
            trial.dispatch(target,geometry)
            trial.controller.accept(target,0.);control['applied']=True
            if k==0:
                write(trial.case/'physics_started.json',dict(cumulative_trial=len(previous)+1, never_restart=True))
            for j in range(50):
                mujoco.mj_step(trial.m,trial.d)
                if (j+1)%10==0:trial.snapshot()
            recent=trial.rows[-6:];last=recent[-1]
            speed=float(np.linalg.norm((recent[-1]['centers_mm']-recent[0]['centers_mm'])/.05,axis=1).max())
            lost=lost+1 if not np.any(last['contact_force_N']>.1) else 0
            fast=fast+1 if speed>500 else 0
            if last['ground'] or last['self_contact']:
                result['outcome']='ground_or_nonadjacent_self_contact';break
            if t>=1 and (lost>=4 or fast>=4):
                result['outcome']='support_loss_or_uncontrolled_motion';break
            if k==99:
                passed, maximum=holding(trial.rows)
                result.update(entry_screen_pass=passed,entry_maximum_speed_mm_s=maximum)
                if not passed:result['outcome']='initial_cannot_hold_on_both_pipes';break
            if k%100==99:
                print(json.dumps(dict(revision=revision,time_s=float(trial.d.time),speed_mm_s=speed,
                    max_state_deg=float(np.degrees(abs(trial.controller.shape)).max()))),flush=True)
        else:
            result['outcome']='completed_duration_pending_convergence_review'
    except Exception as error:
        result.update(outcome='controller_or_interface_stop',error=repr(error),
            failed_measured_rad=measured.tolist() if 'measured' in locals() else None,
            last_acknowledged_target_rad=trial.d.ctrl.tolist(), retained_state_rad=trial.controller.shape.tolist() if hasattr(trial,'controller') else None)
    finally:
        result.update(physical_seconds=float(trial.d.time),wall_seconds=time.monotonic()-start,
            new_dynamics_trials=int(trial.d.time>0),cumulative_dynamics_trials=len(previous)+int(trial.d.time>0))
        trial.close()
    result.update(late_metrics(trial))
    if result['outcome']=='completed_duration_pending_convergence_review':
        result['outcome']='full_holding_pass' if result['full_holding_pass'] else 'full_duration_not_converged_or_not_supported'
    write(trial.case/'result.json',result)
    if result['new_dynamics_trials']:
        row={key:'' for key in fields}
        row.update(trial_id=trial.case.name,stage='authorized_stage_continuation',version='R'+str(revision),condition='C0',
            initial_usable=result['full_holding_pass'],physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],
            outcome=result['outcome'],evidence=trial.case.relative_to(ROOT).as_posix(),
            initial_file='stage_continuation_20261006/initial_pose.npz',comparability='same_saved_V3_d0_changed_mechanism',
            maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],
            entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s','NOT_REACHED'),
            maximum_tracking_error_deg=result['maximum_tracking_error_deg'],root_displacement_mm=result['root_displacement_mm'],
            commanded_feed_mm=0.,metrics_evidence=(trial.case/'result.json').relative_to(PARENT).as_posix())
        with LEDGER.open('a',encoding='utf8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
        infrastructure.replay(trial.case)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=int,required=True)
    run(parser.parse_args().revision)
