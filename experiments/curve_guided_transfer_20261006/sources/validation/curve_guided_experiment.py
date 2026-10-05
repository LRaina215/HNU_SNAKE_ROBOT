"""P1b: one C0, then B only if C0 establishes a usable paired initial state.

No serial calls, no checkpoint hashes, no physics/controller parameter sweep.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'visualizer'), str(ROOT / 'validation')]
import csv
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import cv2

from curve_guided_geometry import BASE, MODEL, CORNER, points
from snake_control.curve_guided_transfer import FiveSegmentCurve, feed_schedule
from snake_control.curve_transfer_controller import CurveTransferController
from snake_control.mujoco_execution_bridge import MujocoExecutionBridge
from python_client import SnakeControlClient


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def replay(case):
    with np.load(case / 'trace.npz') as trace:
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, height=480, width=720)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.05, .15, .98]
        camera.distance = 1.6
        camera.azimuth = 135
        camera.elevation = -25
        video = cv2.VideoWriter(str(case / 'replay_1x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (720, 480))
        assert video.isOpened()
        for index in range(0, len(trace['sim_time']), 5):
            d.qpos[:] = trace['qpos'][index]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera)
            p = points(m, d) / 1000
            for module, color in [(7, [1., .05, .05, 1.]), (8, [.05, .25, 1., 1.])]:
                scene = renderer.scene
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.012, 0, 0]), (p[module] + p[module + 1]) / 2,
                                   np.eye(3).ravel(), np.array(color, dtype=np.float32))
                scene.ngeom += 1
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            text = f'{case.name}  t={trace["sim_time"][index]:.2f}s  {trace["stage"][index]}  feed={trace["feed_mm"][index]:.1f}mm'
            cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(frame, 'RED=M7  BLUE=M8; saved physics, 1x replay', (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            video.write(frame)
            if index in (0, 5 * ((len(trace['sim_time']) // 2) // 5), 5 * ((len(trace['sim_time']) - 1) // 5)):
                cv2.imwrite(str(case / f'frame_{index:05d}.png'), frame)
        video.release()
        renderer.close()


class Trial:
    def __init__(self, condition, version='V1', initial_name='initial_pose.npz'):
        self.condition = condition
        self.version, self.initial_name = version, initial_name
        self.case = BASE / (version + '_' + condition)
        self.case.mkdir(exist_ok=False)
        self.rows, self.commands, self.controls = [], [], []
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d, self.scratch, self.kinematic = [mujoco.MjData(self.m) for _ in range(3)]
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        with np.load(BASE / initial_name) as initial:
            self.d.qpos[:] = initial['qpos']
            self.d.qvel[:] = initial['qvel']
            self.d.ctrl[:] = initial['target']
        mujoco.mj_forward(self.m, self.d)
        self.initial_qpos = self.d.qpos.copy()
        self.state = np.empty(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.app = self.bridge = self.client = None
        self.old_port = os.environ.get('SNAKE_VISUALIZER_PORT')
        self.feed, self.stage = 0., 'entry'
        self.command_id = 0
        self.head = self.m.body('head').id
        self.pipe_source = {self.m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
        self.pipe_target = {self.m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
        self.floor = self.m.geom('floor').id
        self.snapshot()

    def initialize_relay(self):
        appdir = Path(tempfile.mkdtemp(prefix='curve-transfer-relay-'))
        exe = appdir / 'SnakeRobotVisualizer.exe'
        shutil.copy2(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', exe)
        shutil.copy2(ROOT / 'validation/fixtures/app-state.json', appdir / 'app-state.json')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        os.environ['SNAKE_VISUALIZER_PORT'] = str(port)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        self.app = subprocess.Popen([str(exe)], cwd=appdir, env=os.environ.copy(), startupinfo=startup)
        self.bridge = MujocoExecutionBridge(self.m, self.d)
        self.bridge.period = .001
        self.client = SnakeControlClient(port=port, timeout=.5)
        g = self.fresh(20)
        self.controller = CurveTransferController(FiveSegmentCurve(), g['config'], self.d.ctrl.copy(),
            self.encoder_fk, self.d.body('head').xmat.reshape(3, 3).copy())

    def encoder_fk(self, angles):
        self.kinematic.qpos[:] = 0
        self.kinematic.qpos[3] = 1
        self.kinematic.qpos[self.qadr] = angles
        mujoco.mj_kinematics(self.m, self.kinematic)
        return points(self.m, self.kinematic)

    def fresh(self, timeout=3):
        captured = time.time() * 1000
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.bridge.publish(self.d)
            try:
                g = self.client.get_geometry()
                if g['sample_unix_ms'] >= captured and g['age_ms'] < 200 and g['target_state'].get('mujoco_executor_ready'):
                    g['received_unix_ms'] = time.time() * 1000
                    return g
            except (OSError, RuntimeError, KeyError):
                pass
            time.sleep(.005)
        raise RuntimeError('fresh EXE feedback missing')

    def dispatch(self, target, geometry):
        self.command_id += 1
        record = dict(command_id=self.command_id, sim_time=float(self.d.time),
            sample_unix_ms=geometry['sample_unix_ms'], received_unix_ms=geometry['received_unix_ms'],
            submitted_unix_ms=time.time() * 1000, sequence=geometry['sequence'], target_rad=target.tolist(), applied=False)
        self.commands.append(record)
        record['ack'] = self.client.submit_targets(geometry, np.degrees(target * self.bridge.signs),
                                                 self.command_id, execution='mujoco')
        deadline = time.monotonic() + 3
        while not self.bridge.apply_received(self.d, expected_targets=target):
            self.bridge.publish(self.d)
            if time.monotonic() >= deadline:
                raise RuntimeError('target application timeout')
            time.sleep(.002)
        record.update(applied=True, applied_unix_ms=time.time() * 1000,
                      applied_sim_time=float(self.d.time))

    def snapshot(self):
        mujoco.mj_getState(self.m, self.d, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_setState(self.m, self.scratch, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m, self.scratch)
        d = self.scratch
        p = points(self.m, d)
        centers = (p[:-1] + p[1:]) / 2
        force = np.zeros((2, 21))
        ground = self_contact = False
        for index, c in enumerate(d.contact):
            b1, b2 = [int(self.m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
            ground |= bool((c.geom1 == self.floor and b2 >= self.head) or (c.geom2 == self.floor and b1 >= self.head))
            self_contact |= bool(b1 >= self.head and b2 >= self.head and abs(b1 - b2) > 1)
            for pipe, body in [(c.geom1, b2), (c.geom2, b1)]:
                if body < self.head:
                    continue
                group = 0 if pipe in self.pipe_source else 1 if pipe in self.pipe_target and c.pos[1] > .075 else None
                if group is not None:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.m, d, index, f)
                    force[group, body - self.head] += max(0., f[0])
        delta = centers - CORNER
        region = (delta[:, 1] > 75) & (np.linalg.norm(delta[:, [0, 2]], axis=1) < 121.5)
        self.rows.append(dict(sim_time=float(self.d.time), wall_unix_ms=time.time() * 1000,
            qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(), ctrl=self.d.ctrl.copy(),
            q=self.d.qpos[self.qadr].copy(), torque=d.qfrc_actuator[self.vadr].copy(),
            points_mm=p, centers_mm=centers, contact_force_N=force, target_region=region,
            ground=ground, self_contact=self_contact, feed_mm=self.feed, stage=self.stage,
            command_id=self.bridge.applied_command_id if self.bridge else 0))

    def close(self):
        if self.client:
            self.client.close()
        if self.bridge:
            self.bridge.close()
        if self.app and self.app.poll() is None:
            self.app.terminate()
            self.app.wait(timeout=5)
        if self.old_port is None:
            os.environ.pop('SNAKE_VISUALIZER_PORT', None)
        else:
            os.environ['SNAKE_VISUALIZER_PORT'] = self.old_port
        if self.rows:
            np.savez_compressed(self.case / 'trace.npz', **{key:np.array([r[key] for r in self.rows]) for key in self.rows[0]})
        write(self.case / 'commands.json', self.commands)
        write(self.case / 'control.json', self.controls)


def run(condition, version='V1', initial_name='initial_pose.npz'):
    trial = Trial(condition, version, initial_name)
    result = dict(condition=condition, version=version, outcome='engineering_failure',
                  initial_usable=False, local_transfer=False, mujoco=mujoco.__version__, real_hardware=False)
    started = time.monotonic()
    try:
        trial.initialize_relay()
        write(trial.case / 'plan.json', dict(condition=condition, total_seconds=45.25,
            feed_mm=161.25 if condition == 'B' else 0., paired_initial=BASE.joinpath(initial_name).as_posix(),
            physics_parameters_changed=False, motion_definition=BASE.joinpath('motion_definition.md').as_posix(),
            operator_real_experiment=False))
        lost = fast = 0
        for k in range(905):
            t = k * .05
            trial.feed, trial.stage = feed_schedule(t, condition)
            geometry = trial.fresh()
            measured = np.radians(np.array(geometry['angles_deg']) * trial.bridge.signs)
            target, control = trial.controller.compute(measured, trial.feed, t, .05)
            control.update(time_s=t, stage=trial.stage)
            trial.controls.append(control)
            trial.dispatch(target, geometry)
            for step in range(50):
                mujoco.mj_step(trial.m, trial.d)
                if (step + 1) % 10 == 0:
                    trial.snapshot()
            last = trial.rows[-1]
            recent = trial.rows[-6:]
            speed = float(np.linalg.norm((recent[-1]['centers_mm'] - recent[0]['centers_mm']) / .05, axis=1).max())
            lost = lost + 1 if not np.any(last['contact_force_N'] > .1) else 0
            fast = fast + 1 if speed > 500 else 0
            if last['ground'] or last['self_contact']:
                result['outcome'] = 'ground_or_nonadjacent_self_contact'
                break
            if t >= 1 and (lost >= 4 or fast >= 4):
                result['outcome'] = 'support_loss_or_uncontrolled_motion'
                break
            if k == 99:
                tail = trial.rows[-100:]
                maximum_speed = max(np.linalg.norm((b['centers_mm'] - a['centers_mm']) / .01, axis=1).max()
                                    for a, b in zip(tail[:-1], tail[1:]))
                usable = all(np.count_nonzero(r['contact_force_N'][0] > .1) >= 2 and
                             np.count_nonzero(r['contact_force_N'][1] > .1) >= 2 for r in tail)
                result.update(initial_usable=bool(usable and maximum_speed < 50),
                              entry_maximum_speed_mm_s=float(maximum_speed))
                if not result['initial_usable']:
                    result['outcome'] = 'initial_cannot_hold_on_both_pipes'
                    break
            if k % 100 == 99:
                print(json.dumps(dict(condition=condition, physical_time=float(trial.d.time), speed_mm_s=speed,
                    source_contacts=int(np.count_nonzero(last['contact_force_N'][0] > .1)),
                    target_contacts=int(np.count_nonzero(last['contact_force_N'][1] > .1)))), flush=True)
        else:
            result['outcome'] = 'completed_schedule'
        result['physical_seconds'] = float(trial.d.time)
        result['wall_seconds'] = time.monotonic() - started
    except Exception as error:
        result['error'] = str(error)
        result['physical_seconds'] = float(trial.d.time)
        result['wall_seconds'] = time.monotonic() - started
    finally:
        trial.close()
    write(trial.case / 'result.json', result)
    with (BASE / 'trial_ledger.csv').open('a', encoding='utf-8-sig', newline='') as f:
        fields = ['trial_id', 'stage', 'version', 'condition', 'initial_usable', 'physical_seconds', 'wall_seconds', 'outcome', 'evidence']
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(dict(trial_id=trial.case.name, stage='P1b', version=version, condition=condition,
            initial_usable=result['initial_usable'], physical_seconds=result['physical_seconds'],
            wall_seconds=result['wall_seconds'], outcome=result['outcome'], evidence=trial.case.relative_to(ROOT).as_posix()))
    replay(trial.case)
    print(json.dumps(result), flush=True)
    return result


def main():
    assert json.loads((BASE / 'geometry_result.json').read_text(encoding='utf8'))['design_status'] == 'READY_FOR_MINIMAL_SIM'
    (BASE / 'sources').mkdir(exist_ok=True)
    for name in ['snake_control/curve_guided_transfer.py', 'snake_control/curve_transfer_controller.py',
                 'validation/curve_guided_geometry.py', 'validation/curve_guided_experiment.py']:
        target = BASE / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact-initial', action='store_true', help='one recorded artificial initial repair after V1 C0')
    args = parser.parse_args()
    version = 'V2' if args.contact_initial else 'V1'
    initial_name = 'initial_pose_contact.npz' if args.contact_initial else 'initial_pose.npz'
    result = run('C0', version, initial_name)
    if result['initial_usable'] and result['outcome'] == 'completed_schedule':
        run('B', version, initial_name)
    else:
        write(BASE / 'simulation_gate.json', dict(B_run=False, reason=result['outcome'], simulation_trials=2 if args.contact_initial else 1,
            real_trials=0, F_run=False, pause='C0 did not establish a usable stable paired initial state'))


if __name__ == '__main__':
    main()
