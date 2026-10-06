"""钩挂回拉 第2步：从开合 A2 张开状态出发，冻结形状整身反向滚动，把钩子拉回套住目标管。

起点：按 A2 记录的已下发指令（trace 中 ctrl，逐位）从 A2 初态重放 17 s，不重置任何状态。
动作：settle 1 s → pull（整身滚动表 psi 平滑增加）→ hold 5 s。
停止回拉：钩子内侧 M0–M6 与目标管持续接触 0.5 s（仿真接触真值，仅本次可行性验证使用，不作为实机控制输入）。
控制：纯关节位置目标，50 ms 一条，目标变化 ≤10°/s，±90°；不经 EXE 转发，直接写 ctrl。
后部不叠加顺应修正（冻结的是 A2 第 17 s 已含顺应修正的指令形状）。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse
import csv
import json
import shutil
import time

import cv2
import mujoco
import numpy as np

from curve_guided_geometry import MODEL, CORNER, points
from snake_control.hook_pull_motion import RollTable

OC = ROOT / 'experiments/curve_guided_transfer_20261006/target_open_close_20261006'
A2 = OC / 'OPEN_CLOSE_A2_ACTION'
BASE = ROOT / 'experiments/hook_pull_20261006'
TABLE = BASE / 'offline/roll_table.npz'
DT, STEPS = .05, 50
RATE = np.radians(10.) * DT
HOOK = list(range(0, 7))
PIPE_RADIUS, BODY_RADIUS = 55., 29.


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3 - 2 * x)


def encloses(polygon, point=(0., 0.)):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


class Sim:
    def __init__(self, case):
        self.case = case
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d, self.scratch = mujoco.MjData(self.m), mujoco.MjData(self.m)
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.head = self.m.body('head').id
        self.source = {self.m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
        self.target = {self.m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
        self.floor = self.m.geom('floor').id
        self.state = np.empty(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.rows, self.commands = [], []
        self.stage, self.psi = 'replay_A2', 0.

    def load_A2_start(self):
        with np.load(OC / 'initial_boundary_pose.npz') as initial:
            self.d.qpos[:] = initial['qpos']
            self.d.qvel[:] = initial['qvel']
            self.d.ctrl[:] = initial['target']
        mujoco.mj_forward(self.m, self.d)

    def step_command(self, target, record=True):
        self.d.ctrl[:] = target
        for j in range(STEPS):
            mujoco.mj_step(self.m, self.d)
            if record and (j + 1) % 10 == 0:
                self.snapshot()

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
                group = 0 if pipe in self.source else 1 if pipe in self.target and c.pos[1] > .075 else None
                if group is not None:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.m, d, index, f)
                    force[group, body - self.head] += max(0., f[0])
        uv = (centers[:9] - CORNER)[:, [0, 2]]
        self.rows.append(dict(sim_time=float(self.d.time), qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(),
            ctrl=self.d.ctrl.copy(), q=self.d.qpos[self.qadr].copy(), torque=d.qfrc_actuator[self.vadr].copy(),
            points_mm=p, centers_mm=centers, contact_force_N=force, ground=ground, self_contact=self_contact,
            pipe_enclosed=encloses(uv), stage=self.stage, psi_rad=self.psi))

    def save(self):
        if self.rows:
            np.savez_compressed(self.case / 'trace.npz',
                                **{k: np.array([r[k] for r in self.rows]) for k in self.rows[0]})
        write(self.case / 'commands.json', self.commands)
        mujoco.mj_getState(self.m, self.d, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez(self.case / 'final_state.npz', integration_state=self.state.copy(), ctrl=self.d.ctrl.copy(),
                 psi_rad=self.psi, sim_time=self.d.time)


def replay_video(case, rows):
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=480, width=720)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.08, .12, .98]
    camera.distance = 1.3
    camera.azimuth = 150
    camera.elevation = -20
    video = cv2.VideoWriter(str(case / 'replay_1x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (720 + 300, 480))
    for index in range(0, len(rows), 5):
        r = rows[index]
        d.qpos[:] = r['qpos']
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=camera)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        panel = np.full((480, 300, 3), 30, np.uint8)
        scale, origin = .9, np.array([150, 240])
        cv2.circle(panel, tuple(origin), int(PIPE_RADIUS * scale), (90, 160, 90), -1)
        cv2.circle(panel, tuple(origin), int((PIPE_RADIUS + BODY_RADIUS) * scale), (90, 160, 90), 1)
        uv = (r['centers_mm'][:9] - CORNER)[:, [0, 2]]
        pix = (origin + uv * [scale, -scale]).astype(int)
        for a, b in zip(pix[:-1], pix[1:]):
            cv2.line(panel, tuple(a), tuple(b), (0, 200, 255), 2)
        for k, p in enumerate(pix):
            color = (0, 0, 255) if r['contact_force_N'][1, k] > .1 else (255, 255, 255)
            cv2.circle(panel, tuple(p), 4, color, -1)
        cv2.putText(panel, 'M0', tuple(pix[0] + 5), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(panel, 'target-pipe section (M0-M8)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(panel, f'enclosed: {r["pipe_enclosed"]}', (8, 455), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(panel, 'red = touching target pipe', (8, 475), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 255), 1)
        src = int(np.count_nonzero(r['contact_force_N'][0] > .1))
        tgt = int(np.count_nonzero(r['contact_force_N'][1] > .1))
        text = f'{case.name} t={r["sim_time"]:.2f}s {r["stage"]} roll={np.degrees(r["psi_rad"]):.0f}deg  src/tgt contact {src}/{tgt}'
        cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(frame, 'saved physics, 1x replay', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        video.write(np.hstack([frame, panel]))
    video.release()
    renderer.close()


def hold_metrics(rows, seconds):
    end = rows[-1]['sim_time']
    window = [r for r in rows if r['sim_time'] >= end - seconds - 1e-9]
    p = np.array([r['centers_mm'] for r in window])
    t = np.array([r['sim_time'] for r in window])
    last2 = p[t >= end - 2 - 1e-9]
    last1 = [r for r in window if r['sim_time'] >= end - 1 - 1e-9]
    return dict(window_s=seconds,
                last2s_drift_mm=float(np.linalg.norm(last2 - last2[0], axis=2).max()),
                window_max_module_speed_mm_s=float(np.linalg.norm(np.diff(p, axis=0) / np.diff(t)[:, None, None], axis=2).max()),
                last1s_min_source_contacts=int(min(np.count_nonzero(r['contact_force_N'][0] > .1) for r in last1)),
                last1s_min_target_contacts=int(min(np.count_nonzero(r['contact_force_N'][1] > .1) for r in last1)),
                last1s_min_hook_contacts=int(min(np.count_nonzero(r['contact_force_N'][1, HOOK] > .1) for r in last1)),
                last1s_pipe_enclosed_fraction=float(np.mean([r['pipe_enclosed'] for r in last1])),
                final_target_contact_modules=np.flatnonzero(rows[-1]['contact_force_N'][1] > .1).tolist(),
                final_source_contact_modules=np.flatnonzero(rows[-1]['contact_force_N'][0] > .1).tolist())


def run(name, psi_cap_deg, pull_seconds):
    case = BASE / name
    case.mkdir(parents=True, exist_ok=False)
    for src in ['snake_control/hook_pull_motion.py', 'validation/hook_pull_offline.py', 'validation/hook_pull_experiment.py']:
        dest = case / 'sources' / src
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, dest)
    shutil.copy2(TABLE, case / 'roll_table.npz')
    table = RollTable.load(TABLE)
    with np.load(TABLE) as data:
        sign = float(data['roll_sign_for_pull'])
        start_ctrl = data['start_ctrl_rad']
    psi_cap = sign * np.radians(psi_cap_deg)
    plan = dict(start='A2 recorded ctrl replay to 17.00 s (340 commands), no reset', settle_s=1., pull_s=pull_seconds,
                psi_cap_deg=float(np.degrees(psi_cap)), hold_s=5., rate_limit_deg_s=10., joint_limit_deg=90.,
                engagement_stop='hook M0-M6 target-pipe contact >0.1 N continuously 0.5 s (simulation truth, feasibility only)',
                safety_stop='ground / nonadjacent self contact; source contacts <2 for 0.2 s; module >500 mm/s for 0.2 s',
                control='pure joint position targets from roll table; no EXE relay; no added compliance',
                physics='model, friction, kp, torque limit unchanged')
    write(case / 'plan.json', plan)
    sim = Sim(case)
    sim.load_A2_start()
    started = time.monotonic()
    result = dict(name=name, outcome='engineering_failure')
    try:
        with np.load(A2 / 'trace.npz') as a2:
            a2_ctrl, a2_qpos = a2['ctrl'], a2['qpos']
        for k in range(340):
            sim.step_command(a2_ctrl[5 * k + 1], record=False)
        result['replay_match'] = dict(time_s=float(sim.d.time),
                                      max_qpos_difference=float(abs(sim.d.qpos - a2_qpos[1700]).max()),
                                      ctrl_equals_start_table=bool(np.array_equal(a2_ctrl[1700], start_ctrl)))
        last = start_ctrl.copy()
        sim.snapshot()
        engaged_since = None
        psi = 0.
        lost = fast = 0
        schedule = [('settle', 1.), ('pull', pull_seconds), ('hold', 5.)]
        t_stage = 0.
        stage_index = 0
        pull_frozen_at = None
        while stage_index < len(schedule):
            stage, duration = schedule[stage_index]
            sim.stage = stage
            if stage == 'pull' and pull_frozen_at is None:
                psi = psi_cap * smooth((t_stage + DT) / pull_seconds)
            sim.psi = psi
            target = table(psi)
            target = np.clip(target, last - RATE, last + RATE)
            target = np.clip(target, -np.pi / 2, np.pi / 2)
            sim.commands.append(dict(time_s=float(sim.d.time), stage=stage, psi_rad=float(psi), target_rad=target.tolist()))
            sim.step_command(target)
            last = target
            r = sim.rows[-1]
            recent = sim.rows[-6:]
            speed = float(np.linalg.norm((recent[-1]['centers_mm'] - recent[0]['centers_mm']) / DT, axis=1).max())
            fast = fast + 1 if speed > 500 else 0
            lost = lost + 1 if np.count_nonzero(r['contact_force_N'][0] > .1) < 2 else 0
            if r['ground'] or r['self_contact']:
                result['outcome'] = 'ground_or_nonadjacent_self_contact'
                break
            if lost >= 4:
                result['outcome'] = 'source_support_below_two_modules_0p2s'
                break
            if fast >= 4:
                result['outcome'] = 'uncontrolled_motion'
                break
            hook_touch = np.count_nonzero(r['contact_force_N'][1, HOOK] > .1) > 0
            if stage == 'pull':
                if hook_touch:
                    engaged_since = engaged_since if engaged_since is not None else float(sim.d.time)
                    if pull_frozen_at is None and sim.d.time - engaged_since >= .5 - 1e-9:
                        pull_frozen_at = dict(time_s=float(sim.d.time), psi_deg=float(np.degrees(psi)),
                                              hook_modules=np.flatnonzero(r['contact_force_N'][1, HOOK] > .1).tolist())
                        result['first_hook_contact_time_s'] = engaged_since
                        t_stage, stage_index = 0., stage_index + 1
                        continue
                else:
                    engaged_since = None
            t_stage += DT
            if t_stage >= duration - 1e-9:
                t_stage, stage_index = 0., stage_index + 1
            if len(sim.commands) % 40 == 0:
                print(json.dumps(dict(t=round(float(sim.d.time), 2), stage=stage, psi_deg=round(float(np.degrees(psi)), 1),
                    src=int(np.count_nonzero(r['contact_force_N'][0] > .1)),
                    tgt=np.flatnonzero(r['contact_force_N'][1] > .1).tolist(), enclosed=r['pipe_enclosed'],
                    speed=round(speed, 1))), flush=True)
        else:
            result['outcome'] = 'hold_completed'
        result['pull_frozen'] = pull_frozen_at
    except Exception as error:
        result.update(outcome='controller_or_interface_stop', error=repr(error))
    finally:
        sim.save()
    rows = sim.rows
    if rows:
        first, last_row = rows[0], rows[-1]
        rear = slice(10, 21)
        dx_rear = float((last_row['centers_mm'][rear, 0] - first['centers_mm'][rear, 0]).mean())
        head0 = first['centers_mm'][0] - [0, 0, 1000.]
        head1 = last_row['centers_mm'][0] - [0, 0, 1000.]
        rotation = float(np.degrees(np.arctan2(head1[2], head1[1]) - np.arctan2(head0[2], head0[1])))
        err = np.degrees(abs(np.array([r['ctrl'] - r['q'] for r in rows])))
        torque = abs(np.array([r['torque'] for r in rows]))
        result.update(physical_seconds_after_branch=float(last_row['sim_time'] - first['sim_time']),
                      wall_seconds=time.monotonic() - started,
                      rear_mean_x_displacement_mm=dx_rear,
                      head_rotation_about_source_axis_deg=rotation,
                      final_psi_deg=float(np.degrees(last_row['psi_rad'])),
                      maximum_tracking_error_deg=float(err.max()),
                      hook_joint_maximum_tracking_error_deg=float(err[:, :8].max()),
                      torque_saturation_fraction=float(np.mean(torque > 2.355 * .999)),
                      maximum_module_speed_mm_s=float(max(np.linalg.norm(np.diff([r['centers_mm'] for r in rows], axis=0), axis=2).max() / .01, 0)),
                      start_enclosed=bool(first['pipe_enclosed']),
                      start_target_contacts=np.flatnonzero(first['contact_force_N'][1] > .1).tolist(),
                      start_source_contacts=np.flatnonzero(first['contact_force_N'][0] > .1).tolist())
        if result['outcome'] == 'hold_completed':
            result['hold'] = hold_metrics(rows, 5.)
    write(case / 'result.json', result)
    ledger = BASE / 'trial_ledger.csv'
    fields = ['trial_id', 'step', 'outcome', 'physical_seconds_after_branch', 'final_psi_deg', 'evidence']
    new = not ledger.exists()
    with ledger.open('a', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(dict(trial_id=name, step='2_pull', outcome=result['outcome'],
                             physical_seconds_after_branch=result.get('physical_seconds_after_branch', 0),
                             final_psi_deg=result.get('final_psi_deg', 0), evidence=case.relative_to(ROOT).as_posix()))
    if rows:
        replay_video(case, rows)
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='PULL1')
    parser.add_argument('--psi-cap-deg', type=float, default=150.)
    parser.add_argument('--pull-seconds', type=float, default=50.)
    args = parser.parse_args()
    run(args.name, args.psi_cap_deg, args.pull_seconds)
