"""Actual dynamics, saved material/reference mapping, and labelled 1x replay."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
import argparse
import json
import shutil
import time
from datetime import datetime

import cv2
import mujoco
import numpy as np

from hook_pull_experiment import Sim
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.unified_curve_transfer import UnifiedCurveTransfer, DT, physical_wrap_masks, cylindrical_contact


BASE = ROOT/'experiments/unified_curve_transfer_20261007'
INITIAL = ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
SOURCES = ['snake_control/unified_curve_transfer.py',
           'snake_control/coordinated_curve_transfer.py',
           'snake_control/direct_curve_transfer.py',
           'snake_control/articulated_shift_transfer.py',
           'snake_control/explicit_winding_transfer.py',
           'snake_control/adaptive_front_winding_transfer.py',
           'snake_control/unfolding_transfer.py',
           'snake_control/moving_compliance_transfer.py',
           'snake_control/fixed_spatial_shift_transfer.py',
           'snake_control/pipe_clearance.py',
           'snake_control/global_transfer_route.py',
           'validation/unified_curve_transfer_experiment.py',
           'validation/recover_unified_curve_state.py',
           'validation/recover_unified_checkpoint.py',
           'snake_control/spatial_bishop_backbone.py']


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def write(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, default=json_default)+'\n'
    path.write_text(encoded, encoding='utf8')


def snake_pipe_parameters(sim):
    rows=[]
    for contact in sim.d.contact:
        for pipe,robot in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
            body=int(sim.m.geom_bodyid[robot])
            if pipe not in sim.source|sim.target or not sim.head<=body<sim.head+21:
                continue
            rows.append(dict(pipe=sim.m.geom(pipe).name,module=body-sim.head,
                geom_priorities=sim.m.geom_priority[[pipe,robot]].tolist(),
                geom_friction=sim.m.geom_friction[[pipe,robot]].tolist(),
                combined_contact_friction=contact.friction.tolist()))
    return rows


class UnifiedSim(Sim):
    def __init__(self, case):
        super().__init__(case)
        self.info = {}

    def save(self):
        # Each artifact is independent: a JSON failure must not prevent the
        # authoritative trajectory and full MuJoCo integration state saving.
        errors = []
        def attempt(label, operation):
            try:
                operation()
            except Exception as exc:
                errors.append(dict(artifact=label,error=repr(exc)))
        def physical_state():
            mujoco.mj_getState(self.m,self.d,self.state,mujoco.mjtState.mjSTATE_INTEGRATION)
            np.savez(self.case/'final_state.npz',integration_state=self.state.copy(),
                     ctrl=self.d.ctrl.copy(),psi_rad=self.psi,sim_time=self.d.time)
        attempt('final_state.npz',physical_state)
        if self.rows:
            attempt('trace.npz',lambda: np.savez_compressed(self.case/'trace.npz',
                **{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]}))
        attempt('commands.json',lambda: write(self.case/'commands.json',self.commands))
        return errors

    def snapshot(self):
        super().snapshot()
        deepest = selfdeep = 0.
        cylinder_force = np.zeros((2, 21))
        sliding_sum = np.zeros((2,21))
        contact_sliding_max = np.zeros((2,21))
        effective_sliding_friction = np.full((2,21),np.nan)
        jp,jr = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
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
                    normal_force=max(0.,f[0]);module=body-self.head
                    cylinder_force[side,module] += normal_force
                    mujoco.mj_jac(self.m,self.scratch,jp,jr,contact.pos,int(body))
                    velocity=jp@self.scratch.qvel
                    normal=contact.frame[:3]
                    sliding=float(np.linalg.norm(velocity-normal*(normal@velocity))*1000)
                    sliding_sum[side,module] += normal_force*sliding
                    contact_sliding_max[side,module] = max(contact_sliding_max[side,module],sliding)
                    effective_sliding_friction[side,module] = contact.friction[0]
        r = self.rows[-1]
        source_wrap, target_wrap = physical_wrap_masks(r['centers_mm']/1000)
        front_phi=np.unwrap(np.arctan2(-(r['centers_mm'][:8,2]-1000),r['centers_mm'][:8,0]-1100))
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
            force_weighted_contact_sliding_mm_s=sliding_sum/np.maximum(cylinder_force,1e-12),
            maximum_contact_sliding_mm_s=contact_sliding_max,
            effective_sliding_friction=effective_sliding_friction,
            actual_front_target_winding_span_rad=float(front_phi[0]-front_phi[-1]),
            moving_compliance_softness=np.asarray(self.info.get('moving_compliance',{}).get('softness',np.zeros(20))),
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


def run(name, initial, horizon, note, controller_state=None, preserve_grip_clearance=False, localized_yield=False, soft_positive_clearance=False, alternating_yield=False, global_route=None, coordinated=False, direct=False, articulated=False, sliding_friction=None, contact_tangent_scale=None, explicit_winding=False, night_policy=False, deadline_utc=None,adaptive_front_winding=False,unfolding_transfer=False,moving_compliance=False,fixed_spatial_shift=False):
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
        model='one head shift; planned local guide uses ordered continuous material arc correspondence with recorded discrete chord error; original articulated model enforces actual64.5mm rods; projected tangent rate and separate axial spin',
        feedback='encoder target error/load plus explicitly labelled world-pose/contact simulation feedback',
        stops='ground, >3 mm nonadjacent self penetration, actual q>90.5 deg, required support lost 1s, no real progress 90s',
        completion='all centres on target, tail contacts target, no source contact, sustained 2s'))
    sim = UnifiedSim(case)
    restore_saved_state(sim.m, sim.d, initial)
    before_contact=snake_pipe_parameters(sim)
    if sliding_friction is not None:
        before_friction=sim.m.geom_friction.copy()
        original_state=np.empty_like(sim.state)
        mujoco.mj_getState(sim.m,sim.d,original_state,mujoco.mjtState.mjSTATE_INTEGRATION)
        ids=sorted(sim.source|sim.target)
        sim.m.geom_friction[ids,0]=sliding_friction
        restore_saved_state(sim.m,sim.d,initial)
        mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        after_contact=snake_pipe_parameters(sim)
        if not np.array_equal(original_state,sim.state):
            raise RuntimeError('Friction comparison changed initial integration state')
        if not np.array_equal(before_friction[:,1:],sim.m.geom_friction[:,1:]):
            raise RuntimeError('Torsional or rolling friction changed')
        if not after_contact or any(not np.allclose(row['combined_contact_friction'][:2],sliding_friction) for row in after_contact):
            raise RuntimeError('Requested effective sliding friction was not obtained in actual contacts')
        write(case/'runtime_friction.json',dict(original_xml_untouched=True,
            initial_full_integration_state_identical=True,requested_effective_sliding_friction=sliding_friction,
            modified_runtime_pipe_geom_ids=ids,original_geom_friction=before_friction[ids],
            runtime_geom_friction=sim.m.geom_friction[ids],before_actual_contacts=before_contact,
            after_actual_contacts=after_contact,rule_source='https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters'))
    sim.stage = 'saved_two_pipe_support'
    sim.snapshot()
    if fixed_spatial_shift:
        from snake_control.fixed_spatial_shift_transfer import FixedSpatialShiftTransfer
        controller=FixedSpatialShiftTransfer(sim.m,sim.d)
        coordinated=adaptive_front_winding=True
    elif moving_compliance:
        from snake_control.moving_compliance_transfer import MovingComplianceTransfer
        controller=MovingComplianceTransfer(sim.m,sim.d)
        coordinated=adaptive_front_winding=True
    elif unfolding_transfer:
        from snake_control.unfolding_transfer import UnfoldingTransfer
        controller=UnfoldingTransfer(sim.m,sim.d)
        coordinated=adaptive_front_winding=True
    elif adaptive_front_winding:
        from snake_control.adaptive_front_winding_transfer import AdaptiveFrontWindingTransfer
        controller=AdaptiveFrontWindingTransfer(sim.m,sim.d)
        coordinated=True
    elif explicit_winding:
        from snake_control.explicit_winding_transfer import ExplicitWindingTransfer
        controller=ExplicitWindingTransfer(sim.m,sim.d)
        coordinated=True
    elif articulated:
        from snake_control.articulated_shift_transfer import ArticulatedShiftTransfer
        controller = ArticulatedShiftTransfer(sim.m, sim.d)
        coordinated = True
    elif direct:
        from snake_control.direct_curve_transfer import DirectCurveTransfer
        controller = DirectCurveTransfer(sim.m, sim.d)
        coordinated = True
    elif coordinated:
        from snake_control.coordinated_curve_transfer import CoordinatedCurveTransfer
        controller = CoordinatedCurveTransfer(sim.m, sim.d)
    else:
        controller = UnifiedCurveTransfer(sim.m, sim.d)
    if contact_tangent_scale is not None:
        controller.contact_tangent_constraint_scale=contact_tangent_scale
    time_offset = controller.load_state(controller_state) if controller_state else 0.
    if global_route is not None:
        if controller_state is not None:
            raise ValueError('New manual global route starts from CLOSE1B; do not reset old continuation correspondence')
        controller.enable_planned_route(json.loads(global_route.read_text(encoding='utf8')))
    if not np.array_equal(controller.last, sim.d.ctrl):
        raise ValueError('Controller checkpoint last target differs from physical saved ctrl')
    controller.soft_positive_clearance_enabled = soft_positive_clearance
    controller.curve.soft_positive_clearance = soft_positive_clearance
    controller.curve.arc_cache = None
    controller.curve.shape_revision += 1
    if preserve_grip_clearance:
        controller.retain_observed_grip_clearance(sim.d, time_offset, sim.rows[0]['cylindrical_contact_force_N'])
    controller.localized_yield_enabled = localized_yield
    controller.alternating_yield_enabled = alternating_yield
    controller.save_state(case/'initial_controller_state.json', time_offset)
    initial_reference, initial_reference_centers, initial_locations = controller.curve.reference(controller.shift)
    initial_local_pipe_progress = (controller.physical_pipe_progress-controller.local_reference_pipe_progress_origin)/controller.physical_axial_sines
    initial_local_progress = float(initial_local_pipe_progress[1] if physical_wrap_masks(sim.rows[0]['centers_mm']/1000)[1][-1] else min(initial_local_pipe_progress))
    sim.info = dict(reference_points_mm=initial_reference*1000,
        material_locations_mm=initial_locations*1000,
        material_region_codes=controller.curve.regions(initial_locations),
        reference_source_region_mask=controller.source_span(initial_locations),
        shift_mm=controller.shift*1000, shift_speed_mm_s=controller.speed*1000,
        reference_lead_mm=(controller.shift-controller.local_reference_shift_origin-initial_local_progress)*1000
            if controller.local_reference_meter_ready else 0.,
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
    initial_target_end = max(controller.transferred_material_modules,default=max(target0))
    age = np.zeros(21)
    admitted = set()
    last_progress_time = 0.
    best_progress = controller.last_body_progress
    best_actual_front = best_actual_rear = 0.
    admissions = []
    action_checkpoints = []
    last_action_stage = controller.yield_state['stage'] if controller.yield_state else None
    last_continuation_checkpoint = 0
    tail_motion_history=[]
    uncontrolled_tail_fall_ticks=0
    detached_front_ticks=0
    opportunity_started=False
    deadline_timestamp=datetime.fromisoformat(deadline_utc).timestamp() if deadline_utc else None
    best_winding_span=sim.rows[0]['actual_front_target_winding_span_rad']
    try:
        for tick in range(round(horizon/DT)):
            t = tick*DT
            if deadline_timestamp is not None and time.time()>=deadline_timestamp:
                outcome='night_deadline_saved';break
            if (case/'stop_and_save.request').exists():
                outcome = 'requested_diagnostic_stop_saved'
                break
            controller_t = time_offset+t
            r = sim.rows[-1]
            target, info = controller.command(sim.d, controller_t, r['contact_force_N'], r['torque'], r['cylindrical_contact_force_N'])
            sim.info, sim.stage = info, info['stage']
            sim.commands.append(dict(t_s=t, controller_t_s=controller_t, target_rad=target.tolist(), **info))
            sim.step_command(target)
            r = sim.rows[-1]
            action = info.get('localized_yield')
            continuous_checkpoint = action.get('continuation_checkpoint',0) if action else 0
            if action and ((action['stage'] != last_action_stage and action['stage'] in ['feed', 'completed'])
                    or continuous_checkpoint>last_continuation_checkpoint):
                label = f'action_t{t+DT:.2f}_{action["stage"]}'.replace('.', 'p')
                mujoco.mj_getState(sim.m, sim.d, sim.state, mujoco.mjtState.mjSTATE_INTEGRATION)
                np.savez_compressed(case/f'{label}.npz', integration_state=sim.state.copy(),
                    ctrl=sim.d.ctrl.copy(), sim_time=sim.d.time)
                controller.save_state(case/f'{label}_controller.json', time_offset+t+DT)
                action_checkpoints.append(dict(t_s=t+DT, stage=action['stage'],
                    joint=action['selected_joint'], physical_state=f'{label}.npz',
                    controller_state=f'{label}_controller.json'))
            last_action_stage = action['stage'] if action else None
            last_continuation_checkpoint = continuous_checkpoint
            c, f = r['centers_mm'], r['contact_force_N']
            if night_policy and t>=getattr(controller,'action_opportunity_time',90.) and not opportunity_started:
                opportunity_started=True
                last_progress_time=max(last_progress_time,t)
            if coordinated:
                real_front = float(np.mean(c[:7,1]-sim.rows[0]['centers_mm'][:7,1]))
                real_rear = float(np.mean(c[14:,0]-sim.rows[0]['centers_mm'][14:,0]))
                credible_front=(not night_policy or bool(np.any(r['cylindrical_contact_force_N'][1]>.1)))
                credible_rear=(not night_policy or bool(np.any((r['cylindrical_contact_force_N'][0]>.1)&r['source_grip_region_mask'])))
                if not adaptive_front_winding and ((credible_front and real_front>best_actual_front+5.) or (credible_rear and real_rear>best_actual_rear+5.)):
                    best_actual_front=max(best_actual_front,real_front)
                    best_actual_rear=max(best_actual_rear,real_rear)
                    last_progress_time=t
                if night_policy and credible_front and r['actual_front_target_winding_span_rad']>best_winding_span+.25:
                    best_winding_span=r['actual_front_target_winding_span_rad']
                    last_progress_time=t
            admission_contact=(r['cylindrical_contact_force_N'][1]>.1)&r['target_grip_region_mask'] if adaptive_front_winding else f[1]>.1
            age = np.where(admission_contact, age+DT, 0.)
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
            elif not adaptive_front_winding and controller.last_body_progress > best_progress+.005:
                best_progress = controller.last_body_progress
                last_progress_time = t
            source_count, target_count = [int(np.count_nonzero(x > .1)) for x in f]
            needs_source = c[-1, 1] < 80 or np.linalg.norm(c[-1, [0, 2]]-[1100, 1000]) > 130
            cylinder_force = r['cylindrical_contact_force_N']
            posterior_count = int(np.count_nonzero((cylinder_force[0] > .1) & r['source_grip_region_mask']))
            receiving_count = int(np.count_nonzero((cylinder_force[1] > .1) & r['target_grip_region_mask']))
            carried_by_target=controller.target_carries_transfer(c/1000,cylinder_force)
            handover_verified=max(controller.transferred_material_modules,default=-1)>=15
            source_lost=source_lost+1 if needs_source and posterior_count<2 and not handover_verified else 0
            reference_centers=(r['reference_points_mm'][:-1]+r['reference_points_mm'][1:])/2
            tail_error=c[-1]-reference_centers[-1]
            tail_motion_history.append((t,float(c[-1,2]),float(tail_error[2])))
            tail_motion_history=[row for row in tail_motion_history if row[0]>=t-1.]
            old_t,old_z,old_error=tail_motion_history[0]
            falling_uncontrolled=(carried_by_target and t-old_t>=.95 and
                c[-1,2]-old_z < -10. and tail_error[2]-old_error < -10. and
                tail_error[2] < -20. and np.linalg.norm(tail_error)>64.5)
            uncontrolled_tail_fall_ticks=uncontrolled_tail_fall_ticks+1 if falling_uncontrolled else 0
            target_lost = target_lost+1 if receiving_count < 2 else 0
            if not np.isfinite(sim.d.qpos).all(): outcome = 'nonfinite'; break
            if info.get('localized_yield') and info['localized_yield']['stage'] == 'failed':
                outcome = info['localized_yield']['failure']; break
            # Planned execution nodes are a soft ordered guide. Actual rods
            # remain exactly fixed by the original MuJoCo articulated model.
            if controller.curve.planned_route_spec is None and info.get('maximum_reference_link_error_mm', 0.) > .001:
                outcome = 'reference_link_length_invalid'; break
            if r['ground']: outcome = 'ground_contact'; break
            if np.max(abs(r['q'])) > np.radians(90.5): outcome = 'actual_joint_limit'; break
            if r['nonadjacent_self_penetration_mm'] > 3.: outcome = 'nonadjacent_self_interference'; break
            if info.get('adaptive_front_winding',{}).get('failure'):
                outcome=info['adaptive_front_winding']['failure'];break
            if not night_policy and source_lost >= 20: outcome = 'posterior_source_support_below_two_1s'; break
            if not night_policy and target_lost >= 20: outcome = 'target_support_below_two_1s'; break
            if night_policy:
                initial_c=sim.rows[0]['centers_mm']
                front_radius=np.linalg.norm(c[:8,[0,2]]-[1100,1000],axis=1)
                front_fallen=(c[:8,2].mean()<initial_c[:8,2].mean()-80.)
                lost_hook=bool(r['cylindrical_contact_force_N'][1].sum()<.1 and np.median(front_radius)>160. and front_fallen)
                detached_front_ticks=detached_front_ticks+1 if lost_hook else 0
                if detached_front_ticks>=20:outcome='target_hook_detached_and_falling_1s';break
            if uncontrolled_tail_fall_ticks>=20: outcome='target_carried_uncontrolled_tail_fall_1s'; break
            radii = np.linalg.norm(c[:, [0, 2]]-[1100, 1000], axis=1)
            complete = ((c[:, 1] > 80).all() and (radii < 125).all() and source_count == 0
                        and target_count >= 4 and np.any(f[1, 17:] > .1))
            if controller.curve.planned_route_spec is not None:
                # Check the chosen target helix, which continues the inherited
                # front hook. Source handedness is not a completion condition.
                phi_y = np.unwrap(np.arctan2(-(c[:, 2]-1000), c[:, 0]-1100))
                dy, da = np.diff(c[:, 1]), np.diff(phi_y)
                meaningful = np.abs(dy) > 2.
                sign = np.sign(controller.curve.planned_route_spec['target_right_hand_dphi_daxis'])
                target_axial = controller.curve.planned_route_spec.get('pipe_axial_sines',
                    [controller.curve.planned_route_spec['axial_sine']]*2)[1]
                minimum_target_span_mm = .65*21*.0645*target_axial*1000
                same_hand = bool(np.count_nonzero(meaningful)>=16 and
                    np.mean(np.sign(da[meaningful]*dy[meaningful]) == sign)>=.9 and
                    np.max(dy)<3. and c[0,1]-c[-1,1]>minimum_target_span_mm)
                complete = complete and same_hand
            complete_ticks = complete_ticks+1 if complete else 0
            if complete_ticks >= 40: outcome = 'complete_transfer'; break
            receiver_window_end=info.get('fixed_spatial_shift',{}).get('receiver_action_opportunity_end_s',0.)
            no_progress_since=max(last_progress_time,receiver_window_end) if fixed_spatial_shift else last_progress_time
            if (night_policy and opportunity_started and t-no_progress_since>=45.) or (not night_policy and t-last_progress_time>=90.):
                outcome='no_real_progress_45s_after_action_window' if night_policy else 'no_actual_body_progress_90s';break
            if tick % 100 == 99:
                print(json.dumps(dict(t_s=round(t+DT, 2), stage=sim.stage,
                    shift_mm=round(info['shift_mm'], 2), speed=round(info['shift_speed_mm_s'], 2),
                    front_y=round(info['fixed_front_y_mm'], 2), rear_x=round(info['fixed_rear_x_mm'], 2),
                    tracking=round(info['tracking_error_deg'], 2),
                    source=np.flatnonzero(f[0] > .1).tolist(), target=np.flatnonzero(f[1] > .1).tolist(),
                    posterior_support=posterior_count,
                    contact_sliding_mm_s=[round(float(np.sum(r['force_weighted_contact_sliding_mm_s'][side]*cylinder_force[side])/max(cylinder_force[side].sum(),1e-12)),3) for side in range(2)],
                    front_winding_span_rad=round(r['actual_front_target_winding_span_rad'],3),
                    local_front_winding=({k:info['adaptive_front_winding'].get(k) for k in [
                        'front_local_material_mm','rear_local_material_mm','front_requested_speed_mm_s','front_accepted_speed_mm_s',
                        'front_integral_tracking_error_deg','rear_integral_tracking_error_deg',
                        'common_speed_guard_error_deg','failure']} if info.get('adaptive_front_winding') else None),
                    curve_rejection_summary=({key:[item['reason'].split(':')[0] for item in info['adaptive_front_winding'][key]]
                        for key in ['curve_shape_update_rejections','local_rejections','common_shift_rejections']}
                        if info.get('adaptive_front_winding') else None),
                    source_unfolding=info.get('source_unfolding'),
                    fixed_spatial_shift=info.get('fixed_spatial_shift'),
                    moving_compliance=({k:info['moving_compliance'].get(k) for k in ['material_joint_members','ordered_incoming_target_modules',
                        'actual_bearing_recovery_allowed']} if info.get('moving_compliance') else None),
                    global_guidance=info.get('global_route_guidance'),
                    admitted=sorted(admitted))), flush=True)
    except Exception as exc:
        outcome, error = 'controller_exception', repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        save_errors = sim.save()
        try:
            controller.save_state(case/'controller_state.json', time_offset+sim.d.time-t0)
        except Exception as exc:
            save_errors.append(dict(artifact='controller_state.json',error=repr(exc)))
    r = sim.rows[-1]
    force = np.array([x['contact_force_N'] for x in sim.rows])
    c = np.array([x['centers_mm'] for x in sim.rows])
    result = dict(name=name, outcome=outcome, complete=outcome == 'complete_transfer', error=error,
        save_errors=save_errors,
        night_policy=night_policy,deadline_utc=deadline_utc,
        actual_front_winding_span_initial_rad=sim.rows[0]['actual_front_target_winding_span_rad'],
        actual_front_winding_span_final_rad=r['actual_front_target_winding_span_rad'],
        action_opportunity_window_s=getattr(controller,'action_opportunity_time',None),
        action_opportunity_reference_shift_mm=getattr(controller,'action_opportunity_shift',0.)*1000,
        physical_seconds=sim.d.time-t0, dynamics_wall_seconds=time.monotonic()-started,
        controller_initial_elapsed_s=time_offset, controller_final_elapsed_s=time_offset+sim.d.time-t0,
        final_shift_mm=controller.shift*1000, final_curve_parameters=controller.curve.parameters(),
        initial_source_modules=source0, initial_target_modules=target0,
        final_source_modules=np.flatnonzero(force[-1, 0] > .1), final_target_modules=np.flatnonzero(force[-1, 1] > .1),
        new_sustained_target_modules=sorted(admitted), admissions=admissions,
        action_checkpoints=action_checkpoints,
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
        final_actual_source_grip_modules=np.flatnonzero(r['cylindrical_contact_force_N'][0] > .1),
        final_actual_target_grip_modules=np.flatnonzero(r['cylindrical_contact_force_N'][1] > .1),
        final_geometrically_filtered_source_grip_modules=np.flatnonzero((r['cylindrical_contact_force_N'][0] > .1) & r['source_grip_region_mask']),
        final_geometrically_filtered_target_grip_modules=np.flatnonzero((r['cylindrical_contact_force_N'][1] > .1) & r['target_grip_region_mask']),
        drive_sign_counts=[dict(joint=j+1, positive_frames=int(np.count_nonzero(np.array([x['torque'][j] for x in sim.rows]) > .01)),
            negative_frames=int(np.count_nonzero(np.array([x['torque'][j] for x in sim.rows]) < -.01))) for j in range(20)],
        controller_events=controller.events, curve_feedback_updates=controller.n_adaptations)
    result['final_adaptive_front_winding_action']=getattr(controller,'local_action',None)
    result['final_source_unfolding']=getattr(controller.curve,'source_release',None)
    result['ordered_target_handover_modules']=controller.transferred_material_modules if getattr(controller,'ordered_handover_required',False) else None
    result['final_moving_compliance']=getattr(controller,'compliance_feedback',None)
    result['final_fixed_spatial_shift']=getattr(controller,'fixed_spatial_action',None)
    sliding=np.array([x['force_weighted_contact_sliding_mm_s'] for x in sim.rows])
    cylinder_forces=np.array([x['cylindrical_contact_force_N'] for x in sim.rows])
    result['runtime_effective_sliding_friction']=sliding_friction
    result['contact_tangent_constraint_scale']=contact_tangent_scale
    result['force_time_weighted_contact_sliding_mm_s']=(np.sum(sliding*cylinder_forces,axis=(0,2))/np.maximum(np.sum(cylinder_forces,axis=(0,2)),1e-12)).tolist()
    result['maximum_supported_contact_sliding_mm_s']=[float(sliding[:,side][cylinder_forces[:,side]>.1].max()) for side in range(2)]
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
    parser.add_argument('--localized-yield', action='store_true')
    parser.add_argument('--soft-positive-clearance', action='store_true')
    parser.add_argument('--alternating-yield', action='store_true')
    parser.add_argument('--global-route', type=Path)
    parser.add_argument('--coordinated', action='store_true')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--articulated', action='store_true')
    parser.add_argument('--sliding-friction',type=float)
    parser.add_argument('--contact-tangent-scale',type=float)
    parser.add_argument('--explicit-winding',action='store_true')
    parser.add_argument('--adaptive-front-winding',action='store_true')
    parser.add_argument('--unfolding-transfer',action='store_true')
    parser.add_argument('--moving-compliance',action='store_true')
    parser.add_argument('--fixed-spatial-shift',action='store_true')
    parser.add_argument('--night-policy',action='store_true')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--horizon', type=float, default=1800.)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    run(args.name, args.initial, args.horizon, args.note, args.controller_state, args.preserve_grip_clearance, args.localized_yield, args.soft_positive_clearance, args.alternating_yield, args.global_route, args.coordinated, args.direct, args.articulated,args.sliding_friction,args.contact_tangent_scale,args.explicit_winding,args.night_policy,args.deadline_utc,args.adaptive_front_winding,args.unfolding_transfer,args.moving_compliance,args.fixed_spatial_shift)
