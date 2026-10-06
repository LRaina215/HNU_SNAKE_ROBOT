"""Bounded real local rolling, fixed rear nominal and original compliance.

Truth is only snapshot/scoring/danger stop. No physical reset after first step.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation'),str(ROOT/'visualizer')]
import argparse,csv,json,shutil,time
import numpy as np
import mujoco
import curve_guided_experiment as infrastructure
from v3_static_holding_experiment import StaticTrial
from curve_grasp_revision_experiment import holding
from target_open_close_experiment import pipe_inside_projection
from front_local_rolling_geometry import BASE,PARENT,OLD,ORIGINAL,write
from front_local_rolling_fit import material_twist,pipe_relative_twist
from snake_control.front_local_rolling import FrontLocalRollingReference,FrontLocalRollingController
infrastructure.BASE=BASE


class RollingTrial(StaticTrial):
    def __init__(self,version,condition,initial):
        self.exact_contacts=[];self.phase=0.
        infrastructure.Trial.__init__(self,condition,version,initial)
        shutil.copy2(BASE/'reference_ready.npz',self.case/'reference.npz')
        shutil.copy2(BASE/'original_reference.npz',self.case/'original_reference.npz')

    def snapshot(self):
        super().snapshot();psi,curvature=material_twist(self.m,self.scratch);pipepsi,radial=pipe_relative_twist(self.m,self.scratch)
        self.rows[-1].update(front_phase_rad=self.phase,rear_phase_rad=0.,material_twist_rad=psi,
            discrete_curvature_quality=curvature,body_orientation=self.scratch.xmat.copy(),pipe_relative_twist_rad=pipepsi,pipe_radial_quality_mm=radial)

    def initialize_relay(self):
        infrastructure.Trial.initialize_relay(self)
        with np.load(OLD/'R2_C0/trace.npz') as prior:self.kinematic.qpos[:]=prior['qpos'][0]
        mujoco.mj_kinematics(self.m,self.kinematic);registered=self.kinematic.body('head').xmat.reshape(3,3).copy()
        self.controller=FrontLocalRollingController(
            FrontLocalRollingReference(self.case/'reference.npz',self.case/'original_reference.npz'),
            self.controller.config,self.d.ctrl.copy(),self.encoder_fk,registered)
        write(self.case/'config.json',self.controller.config)


def wrapdelta(a):return (np.asarray(a)+np.pi)%(2*np.pi)-np.pi


def metrics(rows,is_C0=False,expected_total_s=None,required_end_hold_s=10.):
    t=np.array([r['sim_time'] for r in rows]);p=np.array([r['centers_mm'] for r in rows]);f=np.array([r['contact_force_N'] for r in rows])
    # M13..M20 is the fixed rear nominal grasp, not the entire source window.
    rear=p[:,13:];displacement=np.linalg.norm(rear-rear[0],axis=2)
    azi=np.arctan2(rear[:,:,2]-1000,rear[:,:,1]);azi_change=np.degrees(wrapdelta(azi-azi[0]))
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2) if len(t)>1 else np.zeros((1,21))
    end=float(t[-1]);last2=t>=end-2-1e-8;last5=t>=end-5-1e-8;last1=t>=end-1-1e-8
    drift=float(np.linalg.norm(p[last2]-p[last2][0],axis=2).max());tail_speed=float(speed[last5[1:]].max()) if np.any(last5[1:]) else 0.
    screen,screen_speed=holding(rows) if len(rows)>2 else (False,0.)
    inside=float(np.mean([pipe_inside_projection(r['points_mm'],0.) for r in rows if r['sim_time']>=end-1-1e-8]))
    rear_distributed=np.count_nonzero(f[:,0,13:]>.1,axis=1);front_distributed=np.count_nonzero(f[:,1,:8]>.1,axis=1)
    source_ok=bool(rear_distributed.min()>=2 and displacement.max()<=6.45 and abs(azi_change).max()<=5.)
    expected_total_s=float(expected_total_s if expected_total_s is not None else end)
    stages=np.array([r['stage'] for r in rows])
    hold_elapsed=float(np.diff(t)[stages[1:]=='end_hold'].sum()) if not is_C0 else max(0.,end-5.)
    end_grasp=bool(end>=expected_total_s-1e-8 and hold_elapsed>=required_end_hold_s-1e-8 and screen and drift<=6.45 and tail_speed<5 and inside>=.95 and front_distributed[last1].min()>=2)
    psi=np.array([r['pipe_relative_twist_rad'][:6] for r in rows]);change=np.degrees(wrapdelta(psi-psi[0]))
    mean_end=change[last1].mean(0)
    roll=bool(np.count_nonzero(mean_end<=-7.5)>=4 and np.median(mean_end)<=-7.5)
    front_y=(p[last1,:7,1].mean(axis=0)-p[0,:7,1]);local_forward=bool(np.median(front_y)>1.)
    tracking=np.degrees(abs(np.array([r['q']-r['ctrl'] for r in rows])))
    torque=abs(np.array([r['torque'] for r in rows]));saturated=torque>=2.355-1e-6
    return dict(rear_anchor_pass=source_ok,rear_maximum_module_displacement_mm=float(displacement.max()),
        rear_maximum_azimuth_change_deg=float(abs(azi_change).max()),rear_minimum_distributed_source_contacts=int(rear_distributed.min()),
        end_grasp_pass=end_grasp,expected_total_s=expected_total_s,credited_end_hold_s=hold_elapsed,
        last2s_all_module_drift_mm=drift,last5s_maximum_module_speed_mm_s=tail_speed,
        original_last1s_two_pipe_screen=screen,last1s_screen_speed_mm_s=screen_speed,last1s_target_inside_fraction=inside,
        last1s_front_minimum_distributed_target_contacts=int(front_distributed[last1].min()),
        actual_front_pipe_relative_twist_change_deg=mean_end.tolist(),actual_front_rolling_pass=roll,
        actual_front_axial_displacement_mm=front_y.tolist(),actual_local_forward_pass=local_forward,
        maximum_tracking_error_deg=float(tracking.max()),connector_maximum_tracking_error_deg=float(tracking[:,6:12].max()),
        torque_saturation_sample_fraction=float(saturated.mean()),connector_torque_saturation_sample_fraction=float(saturated[:,6:12].mean()),
        maximum_module_speed_mm_s=float(speed.max()),root_displacement_mm=float(np.linalg.norm(rows[-1]['qpos'][:3]-rows[0]['qpos'][:3])*1000),
        combination_pass=bool(end_grasp and source_ok and (roll or is_C0)),
        full_transfer=False,material_feed_mm=0.,rolling_metric_limitation='material-y against truth pipe radial frame diagnoses relative roll; internal curvature frame separately saved; true frame never controls')


def run(condition):
    if condition not in ['PRIMARY','C0_PERTURB','FREEZE']:raise ValueError('Only preregistered cases')
    registry_path=BASE/'active_registry.json'
    if not registry_path.exists():registry_path=BASE/'revision_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf8'))
    if not registry['further_dynamics_authorized']:raise RuntimeError('Study stopped')
    if not json.loads((BASE/'path_gate.json').read_text(encoding='utf8'))['geometry_and_command_timing_pass']:raise RuntimeError('Geometry not ready')
    if not json.loads((BASE/'interface_check.json').read_text(encoding='utf8'))['interface_pass']:raise RuntimeError('Interface not ready')
    with (PARENT/'trial_ledger.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;previous=list(reader)
    with (BASE/'trial_ledger.csv').open(encoding='utf8',newline='') as f:local=list(csv.DictReader(f))
    if len(previous)>=21 or len(previous)-17>=4 or len(local)!=len(previous)-17:raise RuntimeError('Budget/ledger guard')
    if condition!='PRIMARY':
        primary=json.loads((BASE/'ROLL1_PRIMARY/result.json').read_text(encoding='utf8'))
        if not primary.get('combination_pass'):raise RuntimeError('Freeze requires primary combination pass')
    if condition=='FREEZE':
        matched=json.loads((BASE/'ROLL1_C0_PERTURB/result.json').read_text(encoding='utf8'))
        if not matched.get('combination_pass'):raise RuntimeError('Freeze requires matching perturbed C0')
    initial='initial_boundary_pose.npz' if condition=='PRIMARY' else 'initial_boundary_pose_perturb.npz'
    trial=RollingTrial('ROLL1',condition,initial)
    schedule=np.load(BASE/'command_schedule.npz');is_C0=condition=='C0_PERTURB';total=float(schedule['time_s'][-1])
    write(trial.case/'plan.json',dict(condition=condition,total_s=total,front_phase_deg=[0,0 if is_C0 else 15],rear_phase_deg=0,
        rear_active_feed_mm=0,full_body_feed_mm=0,entry_s=5,rolling_s=total-15,end_hold_s=10,
        rear_range=[13,20],connector_range=[7,12],front_range=[1,6],
        rear_anchor_gate='M13..20 displacement<=6.45mm and source azimuth<=5deg throughout, >=2source contacts .2s stop',
        actual_rolling_gate='>=4 of six front pipe-relative material twists and median <=-7.5deg relative actual start, end last1s',
        end_gate='Original contact/speed, last2s drift<=6.45mm,last5s speed<5mm/s,target inside>=95%,>=2front target contacts',
        front_advance='median actual M0..6 axial displacement>1mm reported separately, not necessary to claim local rolling/grasp',
        gains=[.01,.02],restoring_tau_s=.5,rate_limit_deg_s=10,angle_limit_deg=90,torque_limit_Nm=2.355,
        control_inputs='encoders/ACK/history/initial prior; world, exact contact, material twist diagnostics never enter control',
        original_model_unchanged=True,no_reset_after_first_step=True,no_virtual_support=True,old_V3_path='NOT_READY',
        inherited_C0='complete arbitrary-state phi0 equivalence to R2_SPEED3_PAIR_C0',prior_trials=len(previous),budget=21))
    sources=['snake_control/front_local_rolling.py','validation/front_local_rolling_geometry.py','validation/front_local_rolling_fit.py',
        'validation/front_local_rolling_interface.py','validation/front_local_rolling_experiment.py',
        'validation/front_local_rolling_continue.py','validation/front_local_rolling_review.py','snake_control/executable_grasp_state.py',
        'snake_control/local_material_reference.py','snake_control/curve_grasp_revision.py','snake_control/three_segment_compliance.py',
        'snake_control/pole_compliance.py','snake_control/mujoco_execution_bridge.py','snake_control/mujoco_visualizer_bridge.py',
        'validation/curve_guided_experiment.py','validation/v3_static_holding_experiment.py','mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml']
    for name in sources:
        dest=trial.case/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    result=dict(outcome='engineering_failure',condition=condition,combination_pass=False,schedule_completed=False)
    started=time.monotonic();lost=fast=rear_lost=0;stop=None
    try:
        trial.initialize_relay()
        for k in range(len(schedule['time_s'])-1):
            t=float(schedule['time_s'][k]);phase=0. if is_C0 else float(schedule['phase_rad'][k]);stage='hold_C0' if is_C0 and t>=5 else str(schedule['stage'][k])
            trial.phase,trial.feed,trial.stage=phase,0.,stage
            g=trial.fresh();measured=np.radians(np.asarray(g['angles_deg'])*trial.bridge.signs)
            target,control=trial.controller.compute(measured,phase,45.25+t,.05)
            control.update(time_s=t,stage=stage,front_phase_rad=phase,feed_mm=0.,rear_phase_rad=0.,measured_rad=measured.tolist(),applied=False)
            trial.controls.append(control);trial.dispatch(target,g);trial.controller.accept(target,phase);control['applied']=True
            if k==0:write(trial.case/'physics_started.json',dict(cumulative_trial=len(previous)+1,never_restart=True))
            for j in range(50):
                mujoco.mj_step(trial.m,trial.d)
                if (j+1)%10==0:
                    trial.snapshot();r=trial.rows[-1];rear=r['centers_mm'][13:];initialrear=trial.rows[0]['centers_mm'][13:]
                    drift=float(np.linalg.norm(rear-initialrear,axis=1).max())
                    azi=np.degrees(wrapdelta(np.arctan2(rear[:,2]-1000,rear[:,1])-np.arctan2(initialrear[:,2]-1000,initialrear[:,1])))
                    if drift>6.45 or abs(azi).max()>5:
                        stop='rear_anchor_displacement_or_azimuth_exceeded';result['first_anchor_violation_s']=float(trial.d.time);break
            if stop:result['outcome']=stop;break
            last=trial.rows[-1];recent=trial.rows[-6:]
            speed=float(np.linalg.norm((recent[-1]['centers_mm']-recent[0]['centers_mm'])/.05,axis=1).max())
            lost=lost+1 if not np.any(last['contact_force_N']>.1) else 0;fast=fast+1 if speed>500 else 0
            rear_lost=rear_lost+1 if np.count_nonzero(last['contact_force_N'][0,13:]>.1)<2 else 0
            if last['ground'] or last['self_contact']:result['outcome']='ground_or_nonadjacent_self_contact';break
            if t>=1 and (lost>=4 or fast>=4):result['outcome']='support_loss_or_uncontrolled_motion';break
            if t>=5 and rear_lost>=4:result['outcome']='rear_support_below_two_0p2s';break
            if k==99:
                valid,maximum=holding(trial.rows);result.update(entry_screen_pass=valid,entry_maximum_speed_mm_s=maximum)
                if not valid:result['outcome']='initial_cannot_hold';break
            if k%100==99:print(json.dumps(dict(physical_time_s=float(trial.d.time),stage=stage,phase_deg=float(np.degrees(phase)),rear_contacts=int(np.count_nonzero(last['contact_force_N'][0,13:]>.1)),target_contacts=int(np.count_nonzero(last['contact_force_N'][1,:8]>.1)),speed_mm_s=speed)),flush=True)
        else:result.update(outcome='schedule_completed',schedule_completed=True)
    except Exception as e:result.update(outcome='controller_or_interface_stop',error=repr(e))
    finally:
        result.update(physical_seconds=float(trial.d.time),wall_seconds=time.monotonic()-started,commands=len(trial.commands),
            all_commands_applied=all(c['applied'] for c in trial.commands),cumulative_dynamics=len(previous)+int(trial.d.time>0))
        trial.close()
    if trial.d.time>0:
        result.update(metrics(trial.rows,is_C0,total,10.))
        result['combination_pass']=bool(result['combination_pass'] and result['schedule_completed'])
        result['outcome']='combination_pass' if result['combination_pass'] else result['outcome']
        row={f:'' for f in fields};row.update(trial_id=trial.case.name,stage='rear_hold_front_local_rolling',version='ROLL1',condition=condition,
            initial_usable=result.get('entry_screen_pass',False),physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],outcome=result['outcome'],
            evidence=trial.case.relative_to(ROOT).as_posix(),initial_file=initial,comparability='same_R2_initial_fixed_rear_local_phase',
            maximum_module_speed_mm_s=result['maximum_module_speed_mm_s'],entry_final_1s_maximum_speed_mm_s=result.get('entry_maximum_speed_mm_s','NOT_REACHED'),
            maximum_tracking_error_deg=result['maximum_tracking_error_deg'],root_displacement_mm=result['root_displacement_mm'],commanded_feed_mm=0,
            metrics_evidence=(trial.case/'result.json').relative_to(PARENT).as_posix())
        for path in [PARENT/'trial_ledger.csv',BASE/'trial_ledger.csv']:
            with path.open('a',encoding='utf8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
        registry.update(status='EVALUATING',new_dynamics=len(previous)-16,total_dynamics=len(previous)+1,last_case=trial.case.name,last_outcome=result['outcome'])
        write(registry_path,registry)
    write(trial.case/'result.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',default='PRIMARY');run(p.parse_args().condition)
