"""Two authorized short open-loop runs at most; frozen commands, no refitting."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import csv
import json
import time
import shutil
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from curve_guided_geometry import MODEL, CORNER, points
from hook_pull_experiment import encloses, replay_video
from hook_pull_close import wrap_angle_deg

BASE = ROOT / 'experiments/hook_pull_review_20261007'
OUT = BASE / 'short_physics'
INITIAL = ROOT / 'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
FIELDS = ['trial_id', 'attempt_number', 'status', 'physical_steps', 'physical_seconds', 'outcome', 'two_pipe_holding_pass']


def write(path, value):
    def native(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=native) + '\n', encoding='utf8')


def ledger_rows():
    path = OUT / 'trial_ledger.csv'
    if not path.exists(): return []
    with path.open(encoding='utf8', newline='') as f: return list(csv.DictReader(f))


def ledger_save(rows):
    with (OUT / 'trial_ledger.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def reserve(name):
    rows = ledger_rows()
    expected = [] if name == 'C0_HOLD' else ['C0_HOLD']
    if [r['trial_id'] for r in rows] != expected:
        raise RuntimeError('Preserve attempted runs; no retry or renamed budget')
    if name == 'P1_PREFIX':
        c0 = json.loads((OUT / 'C0_HOLD/result.json').read_text(encoding='utf8'))
        if not c0['completed'] or not c0['two_pipe_holding']['passed']:
            raise RuntimeError('C0 gate did not pass; P1 prohibited')
    case = OUT / name
    if case.exists(): raise FileExistsError('Attempt directory already exists')
    # Reservation counts even if initialization fails before the first step.
    rows.append(dict(trial_id=name, attempt_number=len(rows)+1, status='ENTRY_STARTED',
                     physical_steps=0, physical_seconds=0, outcome='pending', two_pipe_holding_pass=False))
    ledger_save(rows); case.mkdir()
    write(case / 'entry_started.json', dict(attempt_number=len(rows), counts_even_if_no_step=True,
        this_dispatch_max_attempts=2, review_total_cap=4, previous_new_attempts=0, utc_unix_s=time.time()))
    return case


class Trial:
    def __init__(self, name, case):
        self.name, self.case = name, case
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d, self.diag = mujoco.MjData(self.m), mujoco.MjData(self.m)
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.head, self.floor = self.m.body('head').id, self.m.geom('floor').id
        self.source = {self.m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
        self.target = {self.m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
        self.robot = set(range(self.head, self.head + 21))
        self.raw = np.empty(mujoco.mj_stateSize(self.m, self.spec))
        with np.load(INITIAL) as f:
            self.initial_state, self.held = f['integration_state'].copy(), f['ctrl'].copy()
        mujoco.mj_setState(self.m, self.d, self.initial_state, self.spec)
        # No physical mj_forward. All pose/force diagnostics use self.diag.
        if self.m.nu != 20 or self.m.opt.timestep != .001 or not np.array_equal(self.d.ctrl, self.held):
            raise ValueError('Saved state/model interface does not match registration')
        self.start_time = float(self.d.time)
        with np.load(BASE / 'entry_prefix_path.npz') as f:
            self.command_times, self.prefix = f['dispatch_time_s'].copy(), f['dispatch_ctrl_rad'].copy()
        if self.prefix.shape != (41, 20) or not np.allclose(self.command_times, np.arange(41)*.05, rtol=0, atol=1e-12):
            raise ValueError('Frozen 50 ms, 2 s command interface required')
        if not np.array_equal(self.prefix[0], self.held):
            raise ValueError('Frozen first target must equal saved ctrl exactly')
        self.rows, self.commands, self.contacts = [], [], []
        self.steps = 0; self.source_lost = self.target_lost = self.fast = 0
        self.roll = np.zeros(21); self.previous_frames = None
        self.maximum_sampler_state_change = 0.
        self.snapshot()
        write(case / 'initial_interface.json', dict(
            integration_max_difference=float(abs(self.raw-self.initial_state).max()),
            t0_ctrl_max_difference=float(abs(self.d.ctrl-self.held).max()),
            source=str(INITIAL.relative_to(ROOT)), no_physical_forward=True,
            timestep_s=float(self.m.opt.timestep), controller='frozen open-loop targets only'))

    def snapshot(self):
        mujoco.mj_getState(self.m, self.d, self.raw, self.spec)
        before = self.raw.copy()
        mujoco.mj_setState(self.m, self.diag, before, self.spec)
        mujoco.mj_forward(self.m, self.diag)
        s = self.diag
        p = points(self.m, s); centers = (p[:-1]+p[1:])/2
        frames = s.xmat[self.head:self.head+21].reshape(21,3,3).copy()
        if self.previous_frames is not None:
            for k in range(21):
                angular_delta = (Rotation.from_matrix(frames[k])*Rotation.from_matrix(self.previous_frames[k]).inv()).as_rotvec()
                axis = frames[k,:,2]+self.previous_frames[k,:,2]; axis /= np.linalg.norm(axis)
                self.roll[k] += float(axis@angular_delta)
        self.previous_frames = frames.copy()
        force = np.zeros((2,21)); ground = selfhit = False; deepest = 0.; exact = []
        for i,c in enumerate(s.contact):
            b1,b2 = [int(self.m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            ground |= (c.geom1==self.floor and b2 in self.robot) or (c.geom2==self.floor and b1 in self.robot)
            selfhit |= b1 in self.robot and b2 in self.robot and abs(b1-b2)>1 and c.dist<-.0005
            f = np.zeros(6); mujoco.mj_contactForce(self.m,s,i,f)
            for geom, body in [(c.geom1,b2),(c.geom2,b1)]:
                if body not in self.robot: continue
                if geom in self.source or geom in self.target: deepest=min(deepest,float(c.dist))
                group=0 if geom in self.source else 1 if geom in self.target and c.pos[1]>.075 else None
                if group is not None: force[group,body-self.head]+=max(0.,f[0])
            exact.append(dict(geom1=self.m.geom(c.geom1).name,geom2=self.m.geom(c.geom2).name,
                module1=b1-self.head if b1 in self.robot else None,
                module2=b2-self.head if b2 in self.robot else None,
                distance_mm=float(c.dist*1000),position_mm=c.pos*1000,force_contact_frame_N_Nm=f))
        t=self.steps*.001
        interval=t-self.rows[-1]['t_s'] if self.rows else 0.
        velocity=(centers-self.rows[-1]['centers_mm'])/interval if self.rows else np.zeros((21,3))
        self.rows.append(dict(t_s=t,sim_time=float(self.d.time),integration_state=before,
            qpos=self.d.qpos.copy(),qvel=self.d.qvel.copy(),ctrl=self.d.ctrl.copy(),q=self.d.qpos[self.qadr].copy(),
            torque=self.d.qfrc_actuator[self.vadr].copy(),diagnostic_actuator_torque=s.qfrc_actuator[self.vadr].copy(),
            actuator_force=self.d.actuator_force.copy(),points_mm=p,centers_mm=centers,
            module_xpos_m=s.xpos[self.head:self.head+21].copy(),module_xquat_wxyz=s.xquat[self.head:self.head+21].copy(),
            module_xmat=frames,axial_roll_deg=np.degrees(self.roll.copy()),module_velocity_mm_s=velocity,
            maximum_module_speed_mm_s=float(np.linalg.norm(velocity,axis=1).max()),contact_force_N=force,
            pipe_penetration_mm=-deepest*1000,ground=bool(ground),self_contact=bool(selfhit),
            wrap_deg=wrap_angle_deg(centers),pipe_enclosed=encloses((centers[:9]-CORNER)[:,[0,2]]),
            preload_angle_difference_rad=self.d.ctrl-self.d.qpos[self.qadr],
            xfrc_applied=self.d.xfrc_applied.copy(),qfrc_applied=self.d.qfrc_applied.copy(),
            stage='C0_hold' if self.name=='C0_HOLD' else 'prefix' if t<2 else 'final_hold'))
        self.contacts.append(dict(t_s=t,contacts=exact))
        mujoco.mj_getState(self.m,self.d,self.raw,self.spec)
        delta=float(abs(self.raw-before).max()); self.maximum_sampler_state_change=max(self.maximum_sampler_state_change,delta)
        if delta!=0: raise RuntimeError('Private sampler altered physical integration state')

    def danger(self):
        r=self.rows[-1]; failures=[]
        numeric=[r[k] for k in ['integration_state','ctrl','centers_mm','contact_force_N','axial_roll_deg']]
        if not all(np.isfinite(x).all() for x in numeric): failures.append('nonfinite_state_or_diagnosis')
        if np.degrees(abs(r['q'])).max()>88.+1e-10: failures.append('actual_joint_above_88deg')
        if np.degrees(abs(r['ctrl'])).max()>88.+1e-10: failures.append('command_joint_above_88deg')
        if r['pipe_penetration_mm']>3.: failures.append('pipe_penetration_above_3mm')
        if r['ground'] or r['self_contact']: failures.append('ground_or_nonadjacent_self_contact')
        if r['wrap_deg']<180.: failures.append('front_wrap_below_180deg')
        src,tgt=np.count_nonzero(r['contact_force_N']>.1,axis=1)
        self.source_lost=self.source_lost+1 if src<2 else 0
        self.target_lost=self.target_lost+1 if tgt==0 else 0
        self.fast=self.fast+1 if r['maximum_module_speed_mm_s']>500 else 0
        if self.source_lost>=20: failures.append('source_support_below_two_0p2s')
        if self.target_lost>=20: failures.append('target_support_zero_0p2s')
        if self.fast>=20: failures.append('module_speed_above_500mm_s_0p2s')
        if np.any(r['xfrc_applied']) or np.any(r['qfrc_applied']): failures.append('unexpected_external_force')
        return failures

    def run(self):
        failure=[]; status='completed_12s'; last=self.held.copy()
        initial_failure=self.danger()
        self.source_lost=self.target_lost=self.fast=0  # Duration counters begin after t=0.
        if initial_failure: return ','.join(initial_failure)
        for tick in range(240):
            target=self.held if self.name=='C0_HOLD' else self.prefix[min(tick,40)]
            if not np.isfinite(target).all(): failure=['nonfinite_target']; break
            if np.degrees(abs(target)).max()>88.+1e-10: failure=['command_joint_above_88deg']; break
            if np.degrees(abs(target-last)).max()>.5+1e-10: failure=['frozen_command_rate_violation']; break
            if tick==0 and not np.array_equal(target,self.held): failure=['first_target_changed']; break
            self.d.ctrl[:]=target
            self.commands.append(dict(t_s=tick*.05,absolute_sim_time=float(self.d.time),target_rad=target.copy(),
                frozen_index=min(tick,40) if self.name=='P1_PREFIX' else None,applied=True))
            last=target.copy()
            for _ in range(50):
                mujoco.mj_step(self.m,self.d); self.steps+=1
                if self.steps%10==0:
                    self.snapshot(); failure=self.danger()
                    if failure: break
            if failure: break
            if tick%40==39:
                r=self.rows[-1]
                print(json.dumps(dict(trial=self.name,t_s=r['t_s'],contacts=np.count_nonzero(r['contact_force_N']>.1,axis=1).tolist(),speed=r['maximum_module_speed_mm_s'])),flush=True)
        if failure: status=','.join(failure)
        return status

    def save(self):
        if self.steps%10: self.snapshot()  # Preserve a partial interval on exceptions.
        np.savez_compressed(self.case/'trace.npz',**{key:np.array([r[key] for r in self.rows]) for key in self.rows[0]})
        write(self.case/'commands.json',self.commands); write(self.case/'exact_contacts.json',self.contacts)
        mujoco.mj_getState(self.m,self.d,self.raw,self.spec)
        np.savez_compressed(self.case/'final_state.npz',integration_state=self.raw,ctrl=self.d.ctrl)


def holding(rows,completed):
    result=dict(passed=False,name='two-pipe jointly supported holding',completed_12s=completed)
    if not rows or rows[-1]['t_s']<12.-1e-9: return result
    t=np.array([r['t_s'] for r in rows]);c=np.array([r['centers_mm'] for r in rows]);f=np.array([r['contact_force_N'] for r in rows])
    tail2=c[t>=10.-1e-9];tail5=c[t>=7.-1e-9];dt=np.diff(t[t>=7.-1e-9])
    counts=np.count_nonzero(f[t>=11.-1e-9]>.1,axis=2)
    result.update(last2s_all_module_max_3d_drift_mm=float(np.linalg.norm(tail2-tail2[0],axis=2).max()),
        last5s_maximum_module_speed_mm_s=float(np.linalg.norm(np.diff(tail5,axis=0)/dt[:,None,None],axis=2).max()),
        last1s_min_source_contacts=int(counts[:,0].min()),last1s_min_target_contacts=int(counts[:,1].min()),
        last1s_enclosed_axis_fraction=float(np.mean([r['pipe_enclosed'] for r in rows if r['t_s']>=11.-1e-9])),
        final_wrap_deg=rows[-1]['wrap_deg'])
    result['passed']=bool(completed and result['last2s_all_module_max_3d_drift_mm']<=6.45
        and result['last5s_maximum_module_speed_mm_s']<5 and result['last1s_min_source_contacts']>=2
        and result['last1s_min_target_contacts']>=3 and result['last1s_enclosed_axis_fraction']>=.95 and result['final_wrap_deg']>180)
    return result


def run_one(name):
    case=reserve(name);trial=None;started=time.monotonic();outcome='entry_exception';error=None
    try:
        trial=Trial(name,case);outcome=trial.run()
    except BaseException as exc:
        error=repr(exc);outcome='entry_or_runtime_exception'
    finally:
        if trial is not None: trial.save()
    completed=trial is not None and trial.steps==12000 and outcome=='completed_12s'
    result=dict(trial=name,outcome=outcome,error=error,completed=completed,
        physical_steps=trial.steps if trial else 0,physical_seconds=trial.steps*.001 if trial else 0,
        entry_attempt_counts=1,two_pipe_holding=holding(trial.rows,completed) if trial else dict(passed=False),
        sampler_physical_state_max_difference=trial.maximum_sampler_state_change if trial else None,
        wall_s=time.monotonic()-started,feedback_added=False,real_hardware=False,
        interpretation='Open-loop two-pipe diagnosis, not independent grasp/feedback/crossing success')
    write(case/'result.json',result)
    rows=ledger_rows();rows[-1].update(status='FINISHED',physical_steps=result['physical_steps'],physical_seconds=result['physical_seconds'],
        outcome=outcome,two_pipe_holding_pass=result['two_pipe_holding']['passed']);ledger_save(rows)
    if trial is not None and trial.rows:
        try:
            replay_rows=[dict(r,sim_time=r['t_s'],psi_rad=np.radians(r['axial_roll_deg'][:8].mean())) for r in trial.rows]
            replay_video(case,replay_rows)
            cap=cv2.VideoCapture(str(case/'replay_1x.mp4'))
            result['replay_available']=bool(cap.isOpened())
            write(case/'replay_metadata.json',dict(opened=cap.isOpened(),fps=cap.get(cv2.CAP_PROP_FPS),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),saved_physical_seconds=result['physical_seconds'],
                frame_interval_s=.05,playback_speed=1.,roll_label='Actual front M0-M7 mean axial rotation from FK poses',
                extra_initial_frame=True,source='actual saved physical qpos, not predicted root/shape'))
            cap.release()
        except Exception as exc:
            result.update(replay_available=False,replay_error=repr(exc))
        write(case/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    if ledger_rows(): raise FileExistsError('This dispatch already began; never restart physical runs')
    registration=json.loads((OUT/'registration.json').read_text(encoding='utf8'))
    if registration['this_dispatch_maximum_attempts']!=2: raise RuntimeError('Unexpected registered budget')
    shutil.copy2(__file__,OUT/'runner_source.py')
    c0=run_one('C0_HOLD')
    if c0['completed'] and c0['two_pipe_holding']['passed'] and c0.get('replay_available'):
        run_one('P1_PREFIX')
    write(OUT/'dispatch_status.json',dict(attempted_trials=len(ledger_rows()),previous_new_attempts=0,
        this_review_new_total=len(ledger_rows()),review_cap=4,dispatch_cap=2,
        next='STOP: preserve all failures and deliver for review; no additional budget use'))


if __name__=='__main__': main()
