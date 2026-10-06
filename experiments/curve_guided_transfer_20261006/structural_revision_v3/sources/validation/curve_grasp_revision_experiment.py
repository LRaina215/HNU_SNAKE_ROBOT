"""One V3 C0. B is deliberately deferred until holding is established."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation'),str(ROOT/'visualizer')]
import csv
import json
import shutil
import time
import numpy as np
import mujoco
import curve_guided_experiment as infrastructure
from curve_grasp_revision_geometry import BASE
from snake_control.curve_grasp_revision import FittedMaterialCurve,RetainedGraspController

infrastructure.BASE=BASE


class RevisionTrial(infrastructure.Trial):
    def initialize_relay(self):
        super().initialize_relay()
        curve=FittedMaterialCurve(BASE/'fitted_reference.npz')
        self.controller=RetainedGraspController(curve,self.controller.config,self.d.ctrl.copy(),self.encoder_fk,
            self.d.body('head').xmat.reshape(3,3).copy())


def holding(rows):
    tail=rows[-100:]
    speed=max(float(np.linalg.norm((b['centers_mm']-a['centers_mm'])/.01,axis=1).max())
        for a,b in zip(tail[:-1],tail[1:]))
    contacts=all(np.count_nonzero(r['contact_force_N'][0]>.1)>=2 and
                 np.count_nonzero(r['contact_force_N'][1]>.1)>=2 for r in tail)
    return bool(contacts and speed<50),speed


def run_c0():
    trial=RevisionTrial('C0','V3','initial_pose.npz')
    result=dict(condition='C0',version='V3',outcome='engineering_failure',initial_usable=False,
        local_transfer=False,mujoco=mujoco.__version__,real_hardware=False,body_feed_mm=0.)
    start=time.monotonic()
    try:
        trial.initialize_relay()
        infrastructure.write(trial.case/'plan.json',dict(total_seconds=45.25,entry_seconds=5.,feed_mm=0.,
            holding_speed_limit_mm_s=50.,holding_modules_per_pipe=2,normal_force_threshold_N=.1,
            physics_parameters_changed=False,controller='retained A-D deformation / same gains; no added F',
            previous_dynamics_trials=2,this_round_maximum_dynamics_trials=2,
            start_pose_only='artificial checked finite-shell fit; no support or online repositioning'))
        lost=fast=0
        for k in range(905):
            t=k*.05
            trial.feed=0.;trial.stage='entry' if t<5 else 'hold_C0'
            geometry=trial.fresh()
            measured=np.radians(np.asarray(geometry['angles_deg'])*trial.bridge.signs)
            target,control=trial.controller.compute(measured,0.,t,.05)
            control.update(time_s=t,stage=trial.stage)
            trial.controls.append(control)
            trial.dispatch(target,geometry)
            for step in range(50):
                mujoco.mj_step(trial.m,trial.d)
                if (step+1)%10==0:
                    trial.snapshot()
            recent=trial.rows[-6:];last=recent[-1]
            speed=float(np.linalg.norm((recent[-1]['centers_mm']-recent[0]['centers_mm'])/.05,axis=1).max())
            lost=lost+1 if not np.any(last['contact_force_N']>.1) else 0
            fast=fast+1 if speed>500 else 0
            if last['ground'] or last['self_contact']:
                result['outcome']='ground_or_nonadjacent_self_contact';break
            if t>=1 and (lost>=4 or fast>=4):
                result['outcome']='support_loss_or_uncontrolled_motion';break
            if k==99:
                usable,maximum=holding(trial.rows)
                result.update(initial_usable=usable,entry_maximum_speed_mm_s=maximum)
                if not usable:
                    result['outcome']='initial_cannot_hold_on_both_pipes';break
            if k%100==99:
                print(json.dumps(dict(time_s=float(trial.d.time),C0=True,speed_mm_s=speed)),flush=True)
        else:
            usable,maximum=holding(trial.rows)
            result.update(initial_usable=usable,final_maximum_speed_mm_s=maximum,
                outcome='completed_stable_C0' if usable else 'final_C0_cannot_hold')
    except ValueError as error:
        result.update(outcome='controller_target_or_geometry_stop',error=str(error),initial_usable=False)
    except Exception as error:
        result.update(error=str(error),initial_usable=False)
    finally:
        result.update(physical_seconds=float(trial.d.time),wall_seconds=time.monotonic()-start)
        if result['outcome']!='completed_stable_C0':
            result['initial_usable']=False
        trial.close()
    centers=np.array([r['centers_mm'] for r in trial.rows])
    q=np.array([r['q'] for r in trial.rows]);ctrl=np.array([r['ctrl'] for r in trial.rows])
    result.update(maximum_module_speed_mm_s=float(np.linalg.norm(np.diff(centers,axis=0)/.01,axis=2).max()),
        maximum_tracking_error_deg=float(np.degrees(abs(q-ctrl)).max()),
        root_displacement_mm=float(np.linalg.norm(trial.rows[-1]['qpos'][:3]-trial.rows[0]['qpos'][:3])*1000),
        commands=len(trial.commands),all_commands_applied=all(r['applied'] for r in trial.commands))
    infrastructure.write(trial.case/'result.json',result)
    ledger=BASE.parent/'trial_ledger.csv'
    with ledger.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;previous=list(reader)
    if len(previous)!=2:
        raise RuntimeError('Budget ledger changed; preserve trials and reconcile before any further run')
    row={field:'' for field in fields}
    row.update(trial_id='V3_C0',stage='structural_revision',version='V3',condition='C0',initial_usable=result['initial_usable'],
        physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],outcome=result['outcome'],
        evidence=trial.case.relative_to(ROOT).as_posix(),initial_file='structural_revision_v3/initial_pose.npz',
        comparability='new_structural_version_not_paired_with_V1_or_V2',
        maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s','NOT_REACHED'),
        maximum_tracking_error_deg=result['maximum_tracking_error_deg'],root_displacement_mm=result['root_displacement_mm'],
        commanded_feed_mm=0.,metrics_evidence='structural_revision_v3/V3_C0/result.json')
    with ledger.open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(row)
    infrastructure.replay(trial.case)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    infrastructure.write(BASE/'simulation_gate.json',dict(previous_trials=2,new_trials=1,total_dynamics_trials=3,
        C0_pass=result['initial_usable'],B_run=False,F_run=False,real_trials=0,
        next_step='implement B material endpoint and stall scorer before one B' if result['initial_usable'] else 'stop this round after failed C0'))
    return result


def main():
    if (BASE/'V3_C0').exists():
        raise FileExistsError('The one V3 C0 has already been registered')
    geometry=json.loads((BASE/'geometry_result.json').read_text(encoding='utf8'))
    if geometry['design_status']!='READY_FOR_MINIMAL_SIM':
        raise RuntimeError('Revision geometry is not ready; do not step physics')
    review=json.loads((BASE/'interpolation_review.json').read_text(encoding='utf8'))
    if review['design_status']!='READY_FOR_MINIMAL_SIM':
        raise RuntimeError('Continuous geometry/rate review is not ready; do not step physics')
    ledger=BASE.parent/'trial_ledger.csv'
    with ledger.open(encoding='utf-8-sig',newline='') as f:
        if len(list(csv.DictReader(f)))!=2:
            raise RuntimeError('Expected two existing dynamics trials; do not reset budget')
    for name in ['snake_control/curve_grasp_revision.py','validation/curve_grasp_revision_geometry.py',
                 'validation/curve_grasp_revision_experiment.py','validation/curve_guided_experiment.py']:
        path=BASE/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    run_c0()


if __name__=='__main__':
    main()
