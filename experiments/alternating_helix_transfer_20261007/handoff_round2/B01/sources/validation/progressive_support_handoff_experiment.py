"""At most two actual runs for the reviewed support-handoff dispatch."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
import argparse
import json
import shutil
import time
import cv2
import mujoco
import numpy as np

from hook_pull_experiment import Sim
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.progressive_support_handoff import ProgressiveSupportHandoff, DT

BASE = ROOT/'experiments/alternating_helix_transfer_20261007/handoff_round2'
INITIAL = ROOT/'experiments/alternating_helix_transfer_20261007/A07/final_state.npz'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf8')


class HandoffSim(Sim):
    def __init__(self, case):
        super().__init__(case)
        self.info = {}

    def snapshot(self):
        super().snapshot()
        deepest = selfdeep = 0.
        for contact in self.scratch.contact:
            b1, b2 = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            one = self.head <= b1 < self.head+21
            two = self.head <= b2 < self.head+21
            if one != two:
                names = [self.m.geom(g).name or '' for g in [contact.geom1, contact.geom2]]
                if any(x.startswith('pipe_') for x in names):
                    deepest = min(deepest, float(contact.dist))
            if one and two and abs(b1-b2) > 1:
                selfdeep = min(selfdeep, float(contact.dist))
        mask = np.zeros(21, dtype=bool)
        mask[self.info.get('protected_source_modules', [])] = True
        self.rows[-1].update(front_last_module=self.info.get('front_last_module', 9),
            candidate_module=self.info.get('candidate_module', 10),
            protected_source_mask=mask,
            candidate_distance_mm=self.info.get('candidate_distance_mm', 0.),
            pipe_penetration_mm=-1000*deepest,
            nonadjacent_self_penetration_mm=-1000*selfdeep)


def video(case, rows):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=800)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.04, .20, .9]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 145, -25
    writer = cv2.VideoWriter(str(case/'replay_1x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1120, 540))
    if not writer.isOpened():
        raise RuntimeError('Video writer did not open')
    for i in range(0, len(rows), 5):
        r = rows[i]
        data.qpos[:] = r['qpos']
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        t = r['sim_time']-rows[0]['sim_time']
        cv2.putText(frame, f'{case.name} {t:.2f}s  {r["stage"]}', (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.putText(frame, 'actual saved dynamics, continuous 1x', (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        panel = np.full((540, 320, 3), 24, np.uint8)
        texts = [f'front through M{r["front_last_module"]}',
                 f'receiving M{r["candidate_module"]}',
                 f'goal distance {r["candidate_distance_mm"]:.1f} mm',
                 'Module: source / target contact',
                 'blue = protected source', 'orange = receiving one module']
        for line, text in enumerate(texts):
            cv2.putText(panel, text, (10, 23+line*22), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)
        for module in range(21):
            col, line = module//11, module%11
            x, y = 12+col*150, 180+line*29
            color = (60, 180, 255) if module == r['candidate_module'] else (255, 170, 60) if r['protected_source_mask'][module] else (200, 200, 200)
            cv2.putText(panel, f'M{module:02d}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1)
            for side in range(2):
                force = r['contact_force_N'][side, module]
                cv2.circle(panel, (x+60+side*35, y-5), 6, (80, 210, 90) if force > .1 else (65, 65, 65), -1)
        writer.write(np.hstack([frame, panel]))
    writer.release()
    renderer.close()
    cap = cv2.VideoCapture(str(case/'replay_1x.mp4'))
    write(case/'video.json', dict(fps=cap.get(cv2.CAP_PROP_FPS), frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),
        physical_seconds=rows[-1]['sim_time']-rows[0]['sim_time'], speed=1.,
        source='saved actual qpos only; one initial frame'))
    cap.release()


def run(name, revision, initial, note):
    BASE.mkdir(parents=True, exist_ok=True)
    ledger = BASE/'attempts.jsonl'
    attempted = [json.loads(x) for x in ledger.read_text(encoding='utf8').splitlines()] if ledger.exists() else []
    if len(attempted) >= 2:
        raise RuntimeError('This dispatch permits at most two new actual runs')
    case = BASE/name
    case.mkdir(exist_ok=False)
    with ledger.open('a', encoding='utf8') as out:
        out.write(json.dumps(dict(name=name, revision=revision, initial=str(initial), note=note), ensure_ascii=False)+'\n')
    for source in ['snake_control/progressive_support_handoff.py', 'validation/progressive_support_handoff_experiment.py']:
        dest = case/'sources'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, dest)
    write(case/'plan.json', dict(initial=str(initial), revision=revision, change=note,
        maximum_dispatch_runs=2, admitted_contact_seconds=.75, maximum_no_admission_seconds=90.,
        completion='all modules around target, tail-region support, no source contact, sustained 2 seconds',
        physics='original XML, joint ctrl only; rate 10 deg/s, target 88 deg, drive 2.355 Nm',
        LocalSection='fixed 0.01 parameter shape objective, not force-adaptive compliance'))
    sim = HandoffSim(case)
    restore_saved_state(sim.m, sim.d, initial)
    sim.snapshot()
    controller = ProgressiveSupportHandoff(sim.m, sim.d, sim.rows[-1]['contact_force_N'], revision)
    t0 = float(sim.d.time)
    outcome, error = 'horizon_limit', None
    source_lost = target_lost = complete_ticks = 0
    actual_source_loss_events = []
    last_admitted = controller.front
    started = time.monotonic()
    try:
        for tick in range(round(2400/DT)):
            t = tick*DT
            target, info = controller.command(sim.d, t, sim.rows[-1]['contact_force_N'])
            sim.info, sim.stage = info, info['stage']
            sim.commands.append(dict(t_s=t, target_rad=target.tolist(), **info))
            sim.step_command(target)
            r = sim.rows[-1]
            c, forces = r['centers_mm'], r['contact_force_N']
            if controller.front > last_admitted:
                checkpoint = case/f'admitted_M{controller.front:02d}.npz'
                mujoco.mj_getState(sim.m, sim.d, sim.state, mujoco.mjtState.mjSTATE_INTEGRATION)
                np.savez_compressed(checkpoint, integration_state=sim.state.copy(), ctrl=sim.d.ctrl.copy(), sim_time=sim.d.time)
                last_admitted = controller.front
            source_count = np.count_nonzero(forces[0] > .1)
            target_count = np.count_nonzero(forces[1] > .1)
            needs_rear = controller.front < 16
            source_lost = source_lost+1 if needs_rear and source_count < 2 else 0
            target_lost = target_lost+1 if target_count < 2 else 0
            if source_lost == 1:
                actual_source_loss_events.append(dict(t_s=t+DT, source_modules=np.flatnonzero(forces[0] > .1).tolist()))
            if not np.isfinite(sim.d.qpos).all(): outcome = 'nonfinite'; break
            if r['ground']: outcome = 'ground_contact'; break
            if np.max(abs(r['q'])) > np.radians(90.5): outcome = 'actual_joint_limit'; break
            if r['nonadjacent_self_penetration_mm'] > 3.: outcome = 'nonadjacent_self_interference'; break
            if source_lost >= 20: outcome = 'necessary_source_support_lost_1s'; break
            if target_lost >= 20: outcome = 'front_support_below_two_1s'; break
            radii = np.linalg.norm(c[:, [0, 2]]-[1100, 1000], axis=1)
            complete = ((c[:, 1] > 80).all() and (radii < 125).all() and source_count == 0
                        and target_count >= 4 and np.any(forces[1, 17:] > .1))
            complete_ticks = complete_ticks+1 if complete else 0
            if complete_ticks >= 40: outcome = 'complete_transfer'; break
            if t-controller.last_admission_time >= 90.:
                outcome = 'no_sustained_next_module_admission_90s'; break
            if tick % 100 == 99:
                print(json.dumps(dict(t_s=round(t+DT, 2), stage=sim.stage, front=controller.front,
                    receiving=controller.candidate, source=np.flatnonzero(forces[0] > .1).tolist(),
                    protected=info['protected_source_modules'], target=np.flatnonzero(forces[1] > .1).tolist(),
                    goal_mm=round(info['candidate_distance_mm'], 2),
                    qmax_deg=round(float(np.degrees(abs(r['q'])).max()), 2))), flush=True)
    except Exception as exc:
        outcome, error = 'controller_exception', repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        sim.save()
    r = sim.rows[-1]
    result = dict(name=name, revision=revision, outcome=outcome, complete=outcome == 'complete_transfer',
        error=error, physical_seconds=sim.d.time-t0, wall_seconds=time.monotonic()-started,
        initial_front_module=controller.initial_front, final_front_module=controller.front,
        newly_admitted_modules=sorted(controller.acquired-set(range(controller.initial_front+1))),
        final_source_modules=np.flatnonzero(r['contact_force_N'][0] > .1),
        final_target_modules=np.flatnonzero(r['contact_force_N'][1] > .1),
        final_centers_mm=r['centers_mm'], final_q_deg=np.degrees(r['q']), tail_center_mm=r['centers_mm'][-1],
        first_ctrl_error_rad=float(abs(sim.rows[0]['ctrl']-sim.commands[0]['target_rad']).max()) if sim.commands else None,
        max_dispatch_step_deg=float(np.degrees(abs(np.diff([x['ctrl'] for x in sim.rows], axis=0))).max()),
        max_actual_joint_deg=float(np.degrees(abs(np.array([x['q'] for x in sim.rows]))).max()),
        max_actuator_torque_Nm=float(abs(np.array([x['torque'] for x in sim.rows])).max()),
        max_pipe_penetration_mm=max(x['pipe_penetration_mm'] for x in sim.rows),
        max_self_penetration_mm=max(x['nonadjacent_self_penetration_mm'] for x in sim.rows),
        source_loss_events=actual_source_loss_events, events=controller.events)
    write(case/'result.json', result)
    print(json.dumps({k: result[k] for k in ['name', 'outcome', 'physical_seconds', 'newly_admitted_modules', 'error']}), flush=True)
    video(case, sim.rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--revision', type=int, choices=[1, 2], default=1)
    parser.add_argument('--initial', type=Path, default=INITIAL)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    run(args.name, args.revision, args.initial, args.note)
