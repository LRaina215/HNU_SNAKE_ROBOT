"""Actual dynamics, saved material/reference mapping, and labelled 1x replay."""
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
from snake_control.unified_curve_transfer import UnifiedCurveTransfer, DT, physical_wrap_masks, cylindrical_contact


BASE = ROOT/'experiments/unified_curve_transfer_20261007'
INITIAL = ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
SOURCES = ['snake_control/unified_curve_transfer.py',
           'snake_control/pipe_clearance.py',
           'validation/unified_curve_transfer_experiment.py',
           'validation/recover_unified_curve_state.py',
           'validation/recover_unified_checkpoint.py',
           'snake_control/spatial_bishop_backbone.py']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf8')


class UnifiedSim(Sim):
    def __init__(self, case):
        super().__init__(case)
        self.info = {}

    def snapshot(self):
        super().snapshot()
        deepest = selfdeep = 0.
        cylinder_force = np.zeros((2, 21))
        for index, contact in enumerate(self.scratch.contact):
            b1, b2 = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            one, two = self.head <= b1 < self.head+21, self.head <= b2 < self.head+21
            if one != two and any((self.m.geom(g).name or '').startswith('pipe_') for g in [contact.geom1, contact.geom2]):
                deepest = min(deepest, float(contact.dist))
            if one and two and abs(b1-b2) > 1:
                selfdeep = min(selfdeep, float(contact.dist))
            for pipe, body in [(contact.geom1, b2), (contact.geom2, b1)]:
                if not self.head <= body < self.head+21:
                    continue
                side = 0 if pipe in self.source else 1 if pipe in self.target else None
                if side is not None and cylindrical_contact(side, contact.pos, contact.frame[:3]):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.m, self.scratch, index, f)
                    cylinder_force[side, body-self.head] += max(0., f[0])
        r = self.rows[-1]
        source_wrap, target_wrap = physical_wrap_masks(r['centers_mm']/1000)
        reference = np.asarray(self.info.get('reference_points_mm', r['points_mm']))
        region = np.asarray(self.info.get('material_region_codes', [2]*8+[1]*5+[0]*8))
        r.update(shift_mm=self.info.get('shift_mm', 0.),
            shift_speed_mm_s=self.info.get('shift_speed_mm_s', 0.),
            reference_points_mm=reference, material_region_codes=region,
            material_locations_mm=np.asarray(self.info.get('material_locations_mm', np.zeros(21))),
            reference_lead_mm=self.info.get('reference_lead_mm', 0.),
            reference_pipe_clearance_mm=self.info.get('reference_pipe_clearance_mm') if self.info.get('reference_pipe_clearance_mm') is not None else np.nan,
            actual_material_estimate_mm=self.info.get('actual_material_estimate_mm', 0.),
            tracking_error_deg=self.info.get('tracking_error_deg', 0.),
            body_curve_rms_error_mm=self.info.get('body_curve_rms_error_mm', 0.),
            radius_delta_mm=np.array([self.info.get('curve_parameters', {}).get('source_radius_delta_mm', 0.),
                                      self.info.get('curve_parameters', {}).get('target_radius_delta_mm', 0.)]),
            reference_node_arclength_mm=np.asarray(self.info.get('reference_node_arclength_mm', np.zeros(22))),
            maximum_reference_link_error_mm=self.info.get('maximum_reference_link_error_mm', 0.),
            dynamic_grip_bias_deg=np.asarray(self.info.get('dynamic_grip_bias_deg', np.zeros(20))),
            source_grip_region_mask=source_wrap, target_grip_region_mask=target_wrap,
            cylindrical_contact_force_N=cylinder_force,
            reference_source_region_mask=np.asarray(self.info.get('reference_source_region_mask', np.arange(21) >= 13), dtype=bool),
            pipe_penetration_mm=-1000*deepest, nonadjacent_self_penetration_mm=-1000*selfdeep)


def video(case, rows):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=880)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.02, .25, .85]
    camera.distance, camera.azimuth, camera.elevation = 1.85, 145, -25
    writer = cv2.VideoWriter(str(case/'replay_1x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1240, 600))
    if not writer.isOpened():
        raise RuntimeError('Video writer did not open')
    sample_frames = {0, (len(rows)//2)//5, (len(rows)-1)//5}
    for frame_index, i in enumerate(range(0, len(rows), 5)):
        r = rows[i]
        data.qpos[:] = r['qpos']
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cam = renderer.scene.camera[0]
        position = np.mean([c.pos for c in renderer.scene.camera], axis=0)
        forward, up = np.array(cam.forward), np.array(cam.up)
        right = np.cross(forward, up)
        tangent = (cam.frustum_top-cam.frustum_bottom)/(2*cam.frustum_near)
        def mark(world, text, color, offset):
            d = world-position
            depth = d@forward
            if depth <= 0:
                return
            xy = np.array([440+300*(d@right)/(depth*tangent), 300-300*(d@up)/(depth*tangent)])
            xy = np.round(xy).astype(int)
            if not (0 <= xy[0] < 880 and 0 <= xy[1] < 600):
                return
            label = np.clip(xy+offset, [8, 80], [730, 575])
            cv2.circle(frame, tuple(xy), 5, color, -1)
            cv2.line(frame, tuple(xy), tuple(label), color, 1)
            cv2.putText(frame, text, tuple(label), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3)
            cv2.putText(frame, text, tuple(label), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 1)
        mark(r['centers_mm'][0]/1000, 'HEAD M0', (70, 240, 255), np.array([30, -28]))
        mark(r['centers_mm'][20]/1000, 'TAIL M20', (255, 160, 70), np.array([-90, 35]))
        mark(np.array([1.1, .56, 1.]), 'TARGET +Y', (110, 255, 130), np.array([20, -35]))
        mark(np.array([.65, 0., 1.]), 'SOURCE +X', (255, 190, 120), np.array([-90, -30]))
        t = r['sim_time']-rows[0]['sim_time']
        for line, value in enumerate([f'{case.name}  t={t:.2f}s  {r["stage"]}',
                                     'Actual joint-driven dynamics / continuous 1x']):
            cv2.putText(frame, value, (12, 25+line*24), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 1)
        panel = np.full((600, 360, 3), 24, np.uint8)
        texts = [f'Shared shift {r["shift_mm"]:.1f} mm',
                 f'Shift speed {r["shift_speed_mm_s"]:.2f} mm/s',
                 f'Reference lead {r["reference_lead_mm"]:.1f} mm',
                 f'Tracking error {r["tracking_error_deg"]:.1f} deg',
                 'Row colour: reference curve region',
                 'BLUE source / ORANGE bridge / GREEN target',
                 'Dots: ACTUAL source / target contact']
        for line, value in enumerate(texts):
            cv2.putText(panel, value, (10, 23+line*24), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1)
        for module in range(21):
            col, line = module//11, module%11
            x, y = 12+col*170, 225+line*29
            color = [(255, 170, 60), (60, 180, 255), (90, 220, 100)][r['material_region_codes'][module]]
            cv2.putText(panel, f'M{module:02d}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1)
            for side in range(2):
                cv2.circle(panel, (x+65+side*38, y-5), 6,
                    (80, 210, 90) if r['contact_force_N'][side, module] > .1 else (65, 65, 65), -1)
        cv2.putText(panel, 'Region colour is planned, not a success score', (10, 578), cv2.FONT_HERSHEY_SIMPLEX, .4, (180, 180, 180), 1)
        whole = np.hstack([frame, panel])
        writer.write(whole)
        if frame_index in sample_frames:
            cv2.imwrite(str(case/f'frame_{frame_index:05d}.jpg'), whole)
    writer.release()
    renderer.close()
    cap = cv2.VideoCapture(str(case/'replay_1x.mp4'))
    write(case/'video.json', dict(fps=cap.get(cv2.CAP_PROP_FPS), frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),
        physical_seconds=rows[-1]['sim_time']-rows[0]['sim_time'], speed=1.,
        source='actual saved qpos only; one initial frame; reference is recorded separately in trace'))
    cap.release()


def run(name, initial, horizon, note, controller_state=None, preserve_grip_clearance=False):
    BASE.mkdir(parents=True, exist_ok=True)
    case = BASE/name
    case.mkdir(exist_ok=False)
    with (BASE/'attempts.jsonl').open('a', encoding='utf8') as out:
        out.write(json.dumps(dict(name=name, initial=str(initial), controller_state=str(controller_state) if controller_state else None, change=note), ensure_ascii=False)+'\n')
    for source in SOURCES:
        dest = case/'sources'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, dest)
    write(case/'plan.json', dict(initial=str(initial), controller_state=str(controller_state) if controller_state else None, change=note, horizon_s=horizon,
        physics='original horizontal L-pipe XML, 20 joint ctrl only, 2.355 Nm drives',
        software_limits='88 degrees, 10 degrees per second; first saved ctrl retained, symmetric encoder +/-3deg anti-windup with current-curve pressure-dependent clamp',
        model='current curve arclength map; one head shift; every following node exactly one 64.5mm chord; projected tangent rate including feedforward and separate axial spin',
        feedback='encoder target error/load plus explicitly labelled world-pose/contact simulation feedback',
        stops='ground, >3 mm nonadjacent self penetration, actual q>90.5 deg, required support lost 1s, no real progress 90s',
        completion='all centres on target, tail contacts target, no source contact, sustained 2s'))
    sim = UnifiedSim(case)
    restore_saved_state(sim.m, sim.d, initial)
    sim.stage = 'saved_two_pipe_support'
    sim.snapshot()
    controller = UnifiedCurveTransfer(sim.m, sim.d)
    time_offset = controller.load_state(controller_state) if controller_state else 0.
    if not np.array_equal(controller.last, sim.d.ctrl):
        raise ValueError('Controller checkpoint last target differs from physical saved ctrl')
    if preserve_grip_clearance:
        controller.retain_observed_grip_clearance(sim.d, time_offset, sim.rows[0]['cylindrical_contact_force_N'])
    controller.save_state(case/'initial_controller_state.json', time_offset)
    initial_reference, initial_reference_centers, initial_locations = controller.curve.reference(controller.shift)
    sim.info = dict(reference_points_mm=initial_reference*1000,
        material_locations_mm=initial_locations*1000,
        material_region_codes=controller.curve.regions(initial_locations),
        reference_source_region_mask=controller.source_span(initial_locations),
        shift_mm=controller.shift*1000, shift_speed_mm_s=controller.speed*1000,
        reference_lead_mm=(controller.shift-controller.physical_progress_origin_shift-controller.observed_shift)*1000
            if controller.physical_observer_ready else 0.,
        actual_material_estimate_mm=controller.observed_shift*1000 if controller.physical_observer_ready else 0.,
        tracking_error_deg=float(np.degrees(np.abs(controller.last-sim.d.qpos[controller.qa])).max()),
        body_curve_rms_error_mm=float(np.sqrt(np.mean((initial_reference_centers-sim.rows[0]['centers_mm']/1000)**2))*1000),
        reference_node_arclength_mm=controller.curve.reference_node_arc*1000,
        maximum_reference_link_error_mm=float(np.abs(np.linalg.norm(np.diff(initial_reference, axis=0),axis=1)-.0645).max()*1000),
        curve_parameters=controller.curve.parameters())
    sim.rows.pop()
    sim.snapshot()
    write(case/'continuation.json', dict(controller_elapsed_s=time_offset,
        initial_physical_time=float(sim.d.time), first_saved_ctrl_rad=sim.d.ctrl.tolist(),
        initial_reference_points_mm=initial_reference*1000,
        acceptance='continuations are intermediate evidence; final version must run from CLOSE1B'))
    write(case/'initial_curve.json', dict(source_bound_mm=controller.curve.source_bound*1000,
        target_bound_mm=controller.curve.target_bound*1000, body_length_mm=controller.curve.length*1000,
        axis_radii_mm=np.array(controller.curve.radii)*1000,
        axial_sines=controller.curve.axial_sines, polar_angle_rates_rad_m=controller.curve.theta_rates,
        initial_spline_node_parameters_mm=controller.curve.node_s0*1000,
        initial_node_arclength_mm=controller.curve.initial_node_arc*1000,
        initial_head_arclength_mm=controller.curve.initial_head_arc*1000))
    started, t0 = time.monotonic(), float(sim.d.time)
    outcome, error = 'horizon_limit', None
    source_lost = target_lost = complete_ticks = 0
    source0, target0 = [np.flatnonzero(x > .1).tolist() for x in sim.rows[0]['contact_force_N']]
    initial_target_end = max(target0)
    age = np.zeros(21)
    admitted = set()
    last_progress_time = 0.
    best_progress = controller.last_body_progress
    admissions = []
    try:
        for tick in range(round(horizon/DT)):
            t = tick*DT
            controller_t = time_offset+t
            r = sim.rows[-1]
            target, info = controller.command(sim.d, controller_t, r['contact_force_N'], r['torque'], r['cylindrical_contact_force_N'])
            sim.info, sim.stage = info, info['stage']
            sim.commands.append(dict(t_s=t, controller_t_s=controller_t, target_rad=target.tolist(), **info))
            sim.step_command(target)
            r = sim.rows[-1]
            c, f = r['centers_mm'], r['contact_force_N']
            age = np.where(f[1] > .1, age+DT, 0.)
            for module in np.flatnonzero(age >= .75):
                if module > initial_target_end and module not in admitted:
                    admitted.add(int(module))
                    admissions.append(dict(t_s=t+DT, module=int(module), center_mm=c[module].tolist()))
                    mujoco.mj_getState(sim.m, sim.d, sim.state, mujoco.mjtState.mjSTATE_INTEGRATION)
                    np.savez_compressed(case/f'admitted_M{module:02d}.npz',
                        integration_state=sim.state.copy(), ctrl=sim.d.ctrl.copy(), sim_time=sim.d.time)
                    controller.save_state(case/f'admitted_M{module:02d}_controller.json', time_offset+t+DT)
                    last_progress_time = t
            if info.get('physical_progress_observer_initialized'):
                best_progress = controller.last_body_progress
            elif controller.last_body_progress > best_progress+.005:
                best_progress = controller.last_body_progress
                last_progress_time = t
            source_count, target_count = [int(np.count_nonzero(x > .1)) for x in f]
            needs_source = c[-1, 1] < 80 or np.linalg.norm(c[-1, [0, 2]]-[1100, 1000]) > 130
            cylinder_force = r['cylindrical_contact_force_N']
            posterior_count = int(np.count_nonzero((cylinder_force[0] > .1) & r['source_grip_region_mask']))
            receiving_count = int(np.count_nonzero((cylinder_force[1] > .1) & r['target_grip_region_mask']))
            source_lost = source_lost+1 if needs_source and posterior_count < 2 else 0
            target_lost = target_lost+1 if receiving_count < 2 else 0
            if not np.isfinite(sim.d.qpos).all(): outcome = 'nonfinite'; break
            if info.get('maximum_reference_link_error_mm', 0.) > .001:
                outcome = 'reference_link_length_invalid'; break
            if r['ground']: outcome = 'ground_contact'; break
            if np.max(abs(r['q'])) > np.radians(90.5): outcome = 'actual_joint_limit'; break
            if r['nonadjacent_self_penetration_mm'] > 3.: outcome = 'nonadjacent_self_interference'; break
            if source_lost >= 20: outcome = 'posterior_source_support_below_two_1s'; break
            if target_lost >= 20: outcome = 'target_support_below_two_1s'; break
            radii = np.linalg.norm(c[:, [0, 2]]-[1100, 1000], axis=1)
            complete = ((c[:, 1] > 80).all() and (radii < 125).all() and source_count == 0
                        and target_count >= 4 and np.any(f[1, 17:] > .1))
            complete_ticks = complete_ticks+1 if complete else 0
            if complete_ticks >= 40: outcome = 'complete_transfer'; break
            if t-last_progress_time >= 90.:
                outcome = 'no_actual_body_progress_90s'; break
            if tick % 100 == 99:
                print(json.dumps(dict(t_s=round(t+DT, 2), stage=sim.stage,
                    shift_mm=round(info['shift_mm'], 2), speed=round(info['shift_speed_mm_s'], 2),
                    front_y=round(info['fixed_front_y_mm'], 2), rear_x=round(info['fixed_rear_x_mm'], 2),
                    tracking=round(info['tracking_error_deg'], 2),
                    source=np.flatnonzero(f[0] > .1).tolist(), target=np.flatnonzero(f[1] > .1).tolist(),
                    posterior_support=posterior_count,
                    admitted=sorted(admitted))), flush=True)
    except Exception as exc:
        outcome, error = 'controller_exception', repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        sim.save()
        controller.save_state(case/'controller_state.json', time_offset+sim.d.time-t0)
    r = sim.rows[-1]
    force = np.array([x['contact_force_N'] for x in sim.rows])
    c = np.array([x['centers_mm'] for x in sim.rows])
    result = dict(name=name, outcome=outcome, complete=outcome == 'complete_transfer', error=error,
        physical_seconds=sim.d.time-t0, dynamics_wall_seconds=time.monotonic()-started,
        controller_initial_elapsed_s=time_offset, controller_final_elapsed_s=time_offset+sim.d.time-t0,
        final_shift_mm=controller.shift*1000, final_curve_parameters=controller.curve.parameters(),
        initial_source_modules=source0, initial_target_modules=target0,
        final_source_modules=np.flatnonzero(force[-1, 0] > .1), final_target_modules=np.flatnonzero(force[-1, 1] > .1),
        new_sustained_target_modules=sorted(admitted), admissions=admissions,
        maximum_new_module_target_force_N=force[:, 1, initial_target_end+1:].max(axis=0),
        fixed_front_M0_M6_mean_y_advance_mm=float((c[-1, :7, 1]-c[0, :7, 1]).mean()),
        fixed_rear_M14_M20_mean_x_advance_mm=float((c[-1, 14:, 0]-c[0, 14:, 0]).mean()),
        final_centers_mm=c[-1], tail_center_mm=c[-1, -1], final_q_deg=np.degrees(r['q']),
        max_actual_joint_deg=float(np.degrees(abs(np.array([x['q'] for x in sim.rows]))).max()),
        max_actuator_torque_Nm=float(abs(np.array([x['torque'] for x in sim.rows])).max()),
        max_dispatch_step_deg=float(np.degrees(abs(np.diff([x['ctrl'] for x in sim.rows], axis=0))).max()),
        first_ctrl_error_rad=float(abs(sim.rows[0]['ctrl']-sim.commands[0]['target_rad']).max()) if sim.commands else None,
        max_pipe_penetration_mm=max(x['pipe_penetration_mm'] for x in sim.rows),
        max_self_penetration_mm=max(x['nonadjacent_self_penetration_mm'] for x in sim.rows),
        max_reference_link_error_mm=max(x['maximum_reference_link_error_mm'] for x in sim.rows),
        minimum_reference_pipe_clearance_mm=min((x['reference_pipe_clearance_mm'] for x in sim.rows if np.isfinite(x['reference_pipe_clearance_mm'])), default=None),
        final_actual_source_grip_modules=np.flatnonzero((r['cylindrical_contact_force_N'][0] > .1) & r['source_grip_region_mask']),
        final_actual_target_grip_modules=np.flatnonzero((r['cylindrical_contact_force_N'][1] > .1) & r['target_grip_region_mask']),
        drive_sign_counts=[dict(joint=j+1, positive_frames=int(np.count_nonzero(np.array([x['torque'][j] for x in sim.rows]) > .01)),
            negative_frames=int(np.count_nonzero(np.array([x['torque'][j] for x in sim.rows]) < -.01))) for j in range(20)],
        controller_events=controller.events, curve_feedback_updates=controller.n_adaptations)
    write(case/'result.json', result)
    print(json.dumps({k: result[k] for k in ['name', 'outcome', 'physical_seconds', 'final_shift_mm', 'new_sustained_target_modules', 'error']}), flush=True)
    video(case, sim.rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--initial', type=Path, default=INITIAL)
    parser.add_argument('--controller-state', type=Path)
    parser.add_argument('--preserve-grip-clearance', action='store_true')
    parser.add_argument('--horizon', type=float, default=1800.)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    run(args.name, args.initial, args.horizon, args.note, args.controller_state, args.preserve_grip_clearance)
