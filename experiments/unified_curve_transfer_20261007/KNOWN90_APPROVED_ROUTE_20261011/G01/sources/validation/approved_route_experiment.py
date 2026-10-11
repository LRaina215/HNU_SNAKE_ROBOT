"""At most two actual approved-route candidates; success permits one control."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
import shutil
import traceback
from datetime import datetime, timezone
import numpy as np
import mujoco
from snake_control.approved_route_transfer import geometry, ExecutionCurve, ApprovedRouteTransfer, initial_fit
from snake_control.suspended_supply_transfer import DT, LINK, nodes, WholeCurveMapper
from validation.inner_ccw_compliance_experiment import Sim, write, default, shape, review

BASE = ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_APPROVED_ROUTE_20261011'
MODEL = ROOT/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'
START = datetime.fromisoformat('2026-10-11T13:55:12+00:00').timestamp()
DEADLINE = START+90*60


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    route = geometry.Route()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    # Necessary setup fit only: no physics or separate static-hold experiment.
    pose, guide, initial = initial_fit(model, route)
    np.savez(BASE/'prepared_initial.npz', qpos=pose, guide_m=guide)
    write(BASE/'initial_fit.json', initial)
    checks = []
    for label, shift in [('initial', 0.), ('leaving', .55*route.bridge_length),
                         ('upward', route.bridge_length+geometry.R*np.pi/(3*geometry.BETA)),
                         ('final', route.final_shift)]:
        g, stations = ExecutionCurve(route).window(shift)
        mapper = WholeCurveMapper(model)
        _, report = mapper.initialize(g)
        actual = nodes(mapper.d)
        checks.append(dict(stage=label, shift_mm=shift*1000, guide_rms_mm=report['guide_rms_mm'],
                           pipe_depth_mm=report['pipe_depth_mm'], self_depth_mm=report['self_depth_mm'],
                           maximum_joint_deg=report['maximum_joint_deg'], physical_steps=0,
                           final_helix_diagnostic=shape(actual) if label == 'final' else None))
        np.savez(BASE/f'fit_{label}.npz', qpos=mapper.d.qpos, reference_m=g, actual_m=actual, stations_m=stations)
    write(BASE/'mechanism_fit_checks.json', checks)
    print(json.dumps(dict(initial={k: initial[k] for k in ['ready', 'guide_rms_mm', 'pipe_depth_mm', 'self_depth_mm', 'maximum_joint_deg', 'initial_source_contact_modules', 'initial_target_contact_modules']},
                          stages=[{k:v for k,v in c.items() if k != 'final_helix_diagnostic'} for c in checks]), default=default), flush=True)


def freeze(case):
    paths = ['snake_control/approved_route_transfer.py', 'validation/approved_route_experiment.py',
             'snake_control/known90_global_local.py', 'snake_control/inner_ccw_compliance.py',
             'snake_control/suspended_supply_transfer.py', 'snake_control/suspended_transfer_route.py',
             'snake_control/spatial_bishop_backbone.py', 'snake_control/horizontal_pipe_roll.py',
             'validation/inner_ccw_compliance_experiment.py',
             'experiments/unified_curve_transfer_20261007/KNOWN90_ROUTE_CONFIRM_20261011/draw_global_route.py',
             'mujoco_model/Snake_Robot_o20_full_source_inner.xml',
             'mujoco_model/asset/c_link_H.stl', 'mujoco_model/asset/c_link_V.stl']
    for source in paths:
        destination = case/'sources'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, destination)
    write(case/'sources.json', dict(paths=paths, hash_check_performed=False))


def run(args):
    BASE.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding='utf-8')) for p in BASE.glob('*/physics_started.json')]
    count = sum(not r['local_shape_off'] for r in records)
    controls = sum(r['local_shape_off'] for r in records)
    if time.time() >= DEADLINE:
        raise RuntimeError('The authorized implementation/run deadline has been reached')
    if args.local_shape_off:
        success = any(json.loads(p.read_text(encoding='utf-8')).get('full_success') for p in BASE.glob('*/result.json'))
        if not success or controls >= 1:
            raise RuntimeError('A successful candidate is required before the one authorized control')
    elif count >= 2:
        raise RuntimeError('Both authorized actual candidates have been consumed')
    case = BASE/args.name
    case.mkdir(exist_ok=False)
    route = geometry.Route()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    initial = json.loads((BASE/'initial_fit.json').read_text(encoding='utf-8'))
    fits = json.loads((BASE/'mechanism_fit_checks.json').read_text(encoding='utf-8'))
    pose = np.load(BASE/'prepared_initial.npz')['qpos'].copy()
    if not initial['ready']:
        write(case/'result.json', dict(physical_run=False, outcome='initial_fit_rejected', full_success=False))
        raise RuntimeError('Initial source contact/fit is not ready')
    if any(c['guide_rms_mm'] > 50 or c['pipe_depth_mm'] > 1.5 or c['self_depth_mm'] > 1 for c in fits):
        write(case/'result.json', dict(physical_run=False, outcome='keyframe_fit_conflict', full_success=False))
        raise RuntimeError('A keyframe has a concrete mechanism conflict; inspect before dynamics')
    sim = Sim(model, pose)
    ctrl = ApprovedRouteTransfer(model, sim.d, route, not args.local_shape_off, args.speed, args.variant)
    g, stations = route.window(0.)
    execution, _ = ctrl.curve().window(0.)
    sim.info = dict(global_points_mm=g*1000, execution_points_mm=execution*1000, estimated_points_mm=nodes(sim.d)*1000)
    sim.snapshot()
    np.savez(case/'initial_pose.npz', qpos=pose, ctrl=sim.d.ctrl, points_m=execution)
    write(case/'initial_geometry.json', initial)
    write(case/'route.json', dict(source_exit_m=geometry.Q, target_entry_m=geometry.E, target_phi_rad=geometry.TARGET_PHI,
                                bridge_length_mm=route.bridge_length*1000, final_shift_mm=route.final_shift*1000,
                                approved_route_intent='inner lower-to-upper progressive wrap; reverse hook is not active'))
    s = np.linspace(stations[-1], route.final_shift+.08, 7001)
    np.savez(case/'full_global_route.npz', stations_m=s, points_m=route.sample(s))
    freeze(case)
    plan = dict(name=args.name, variant=args.variant, speed_m_s=args.speed, local_shape_off=args.local_shape_off,
                registered_utc=datetime.now(timezone.utc).isoformat(), round_start_utc=datetime.fromtimestamp(START, timezone.utc).isoformat(),
                round_deadline_utc=datetime.fromtimestamp(DEADLINE, timezone.utc).isoformat(), maximum_actual_seconds=600,
                maximum_candidates=2, maximum_controls_after_success=1,
                curve='approved entire route, 21 exact 64.5mm rods, one common head curve station',
                local_feedback='same material segment actual pose - execution reference, joint tracking, actuator load; bounded C1 bridge deflection; receiving radius only after wrap',
                source_sizing='fixed -1.5mm node radius change for initial mechanism contact; no online source-radius changes',
                observations='read-only MuJoCo actual pose allowed; contact identities/forces used by evaluator only',
                physical_root='artificial placement before first mj_step; thereafter joint targets only',
                direction='actual source centers theta=atan2(Z-1,-Y), tail looks +X; positive = CCW',
                stops='600s;30s no new actual departure/upward/intake despite local adjustment;ground;joint or actuator limits;nonfinite;self collision >3mm;90-minute round deadline',
                physics='original gravity, friction, masses, timestep and joint/torque limits; Kp127 with bridge-only load admittance to95.25',
                progress='head on fixed approved route within50mm advances3mm, fixed E distance decreases3mm, sustained new wrap0.10rad or target prefix; no reference-only progress',
                comparator='same initial pose, speed coordination, rolling, actuator limits and stopping; disable only feedback local curve changes')
    write(case/'plan.json', plan)
    record = dict(name=args.name, local_shape_off=args.local_shape_off, start_utc=datetime.now(timezone.utc).isoformat(), variant=args.variant)
    write(case/'physics_started.json', record)
    write(BASE/'ledger.json', dict(actual_candidates=count+int(not args.local_shape_off), controls=controls+int(args.local_shape_off), runs=records+[record]))
    path_s = np.linspace(0, route.final_shift, 6001)
    path = route.sample(path_s)
    commands, events = [], []
    last_progress = best_station = best_wrap = 0.
    best_entry = float(np.linalg.norm(nodes(sim.d)[0]-geometry.E))
    best_prefix = maximum_prefix = 0
    maximum_wrap = 0.
    inner_lower = upward = captured = full_success = False
    capture_age = wrap_age = prefix_age = complete_age = 0.
    captured_time = None
    outcome, error = 'maximum_actual_duration', None
    started = time.monotonic()
    try:
        while sim.d.time < 600:
            if time.time() >= DEADLINE:
                outcome = 'round_90minute_deadline'; break
            target, info = ctrl.command(sim.d, float(sim.d.time))
            if abs(target).max() > np.radians(88)+1e-8:
                outcome = 'proposed_command_limit'; break
            if info['maximum_dispatch_step_deg'] > 1.00001:
                outcome = 'proposed_command_rate_limit'; break
            sim.info = info
            commands.append(dict(t_s=float(sim.d.time), ctrl_rad=target.copy(), **info))
            before = len(sim.rows)
            sim.command(target)
            r = sim.rows[-1]
            t, p, c, forces = r['t_s'], r['actual_points_mm']/1000, r['actual_centers_mm']/1000, r['diagnostic_contact_force_N']
            prefix, wrap = r['actual_target_prefix'], r['actual_target_wrap_rad']
            maximum_prefix, maximum_wrap = max(maximum_prefix, prefix), max(maximum_wrap, wrap)
            # Progress is actual geometry measured against a fixed route/entry.
            distance = np.linalg.norm(path-p[0], axis=1)
            nearest = int(np.argmin(distance))
            station = float(path_s[nearest])
            entry = float(np.linalg.norm(p[0]-geometry.E))
            if distance[nearest] < .05 and station > best_station+.003:
                best_station, last_progress = station, t
                events.append(dict(t_s=t, event='actual_fixed_route_advance', station_mm=station*1000))
            if entry < best_entry-.003:
                best_entry, last_progress = entry, t
                events.append(dict(t_s=t, event='actual_fixed_entry_approach', distance_mm=entry*1000))
            wrap_age = wrap_age+DT if prefix >= 2 and wrap > best_wrap+.10 else 0.
            if wrap_age >= .25:
                best_wrap, last_progress, wrap_age = wrap, t, 0.
                events.append(dict(t_s=t, event='actual_upward_wrap_increment', wrap_rad=wrap))
            prefix_age = prefix_age+DT if prefix > best_prefix and forces[1, :prefix].sum() > .5 else 0.
            if prefix_age >= .5:
                best_prefix, last_progress, prefix_age = prefix, t, 0.
                events.append(dict(t_s=t, event='actual_new_bearing_prefix', prefix=prefix))
            if 1.0 < p[0, 0] < 1.10 and .15 < p[0, 1] < .30 and .88 < p[0, 2] < 1.0:
                inner_lower = True
            if inner_lower and p[0, 2] > 1.02 and prefix >= 2:
                upward = True
            condition = prefix >= 5 and wrap > 2.3 and forces[1, :prefix].sum() > .5
            capture_age = capture_age+DT if condition else 0.
            if capture_age >= 1. and not captured:
                captured, captured_time, last_progress = True, t, t
                events.append(dict(t_s=t, event='actual_active_wrap_capture', prefix=prefix, wrap_rad=wrap))
            final_shape = shape(p)
            ccw = np.degrees(r['actual_source_ccw_rad'][12:])
            direction_ok = bool(np.median(ccw) >= 5 and np.mean(ccw > 0) >= .70)
            complete = final_shape['strict_single_helix'] and not r['diagnostic_source_contact_any'].any() and forces[1, 18:].sum() > .2
            complete_age = complete_age+DT if complete else 0.
            if complete_age >= 1.:
                full_success = bool(captured and inner_lower and upward and direction_ok)
                outcome = 'complete_actual_inner_ccw_single_target_helix' if full_success else 'target_helix_but_entry_or_direction_unconfirmed'
                break
            recent = sim.rows[before:]
            if any(x['diagnostic_ground'] for x in recent):
                outcome = 'ground_contact'; break
            if any(abs(x['q']).max() > np.radians(90.5) for x in recent):
                outcome = 'actual_joint_limit'; break
            if any(abs(x['torque_nm']).max() > 2.35501 for x in recent):
                outcome = 'actual_actuator_limit'; break
            if any(x['diagnostic_self_depth_mm'] > 3 for x in recent):
                outcome = 'actual_self_interference'; break
            if not np.isfinite(sim.d.qpos).all() or not np.isfinite(sim.d.qvel).all():
                outcome = 'nonfinite'; break
            if t-last_progress >= 30:
                outcome = 'no_actual_progress_30s'; break
            if len(commands) % 100 == 0:
                message = dict(t_s=round(t, 2), wall_s=round(time.monotonic()-started, 1), shift_mm=round(ctrl.shift*1000, 1),
                               actual_prefix=prefix, actual_wrap=round(wrap, 3), source_contact_modules=r['source_support_count'],
                               head_mm=p[0]*1000, head_execution_error_mm=round(info['head_execution_error_mm'], 1),
                               tracking_deg=round(info['tracking_deg'], 1), projection_rms_mm=round(info['projection_rms_mm'], 1),
                               best_entry_mm=round(best_entry*1000, 1), best_station_mm=round(best_station*1000, 1),
                               last_progress_s=round(last_progress, 2), source_ccw_median_deg=float(np.median(ccw)), coeff_mm=ctrl.coeff*1000)
                print(json.dumps(message, default=default), flush=True)
                write(case/'progress.json', message)
    except Exception:
        error, outcome = traceback.format_exc(), 'implementation_exception'
        print(error, flush=True)
    finally:
        np.savez_compressed(case/'trace.npz', **{k:np.asarray([r[k] for r in sim.rows]) for k in sim.rows[0]})
        write(case/'commands.json', commands)
        np.savez(case/'final_state.npz', qpos=sim.d.qpos, qvel=sim.d.qvel, ctrl=sim.d.ctrl, time=sim.d.time)
        r = sim.rows[-1]
        result = dict(name=args.name, variant=args.variant, local_shape_off=args.local_shape_off,
                      full_success=full_success, physical_run=True, outcome=outcome,
                      actual_seconds=float(sim.d.time), wall_seconds=time.monotonic()-started, error=error,
                      maximum_actual_target_prefix=maximum_prefix, maximum_actual_wrap_rad=maximum_wrap,
                      actual_registered_progress_mm=best_station*1000, best_actual_entry_distance_mm=best_entry*1000,
                      last_actual_progress_s=last_progress, inner_lower_entry_observed=inner_lower, upward_wrap_observed=upward,
                      active_capture=captured, active_capture_first_s=captured_time, progress_events=events,
                      final_shape=shape(r['actual_points_mm']/1000), source_ccw_deg=np.degrees(r['actual_source_ccw_rad']),
                      final_global_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['global_points_mm'])**2, axis=1)))),
                      final_execution_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['execution_points_mm'])**2, axis=1)))),
                      maximum_actual_joint_deg=float(max(np.degrees(abs(x['q']).max()) for x in sim.rows)),
                      maximum_actual_torque_nm=float(max(abs(x['torque_nm']).max() for x in sim.rows)),
                      maximum_pipe_depth_mm=float(max(x['diagnostic_pipe_depth_mm'] for x in sim.rows)),
                      maximum_self_depth_mm=float(max(x['diagnostic_self_depth_mm'] for x in sim.rows)),
                      ground=any(x['diagnostic_ground'] for x in sim.rows), physical_root_assigned_after_initialization=False,
                      maximum_reference_rod_length_error_mm=max(x['maximum_reference_rod_length_error_mm'] for x in commands),
                      maximum_local_deflection_mm=float(max(abs(x['bridge_coeff_mm']).max() for x in commands)),
                      local_feedback='read-only actual MuJoCo pose; not an encoder-only or hardware observation claim',
                      source_radius_feedback=False, independent_roll_while_shift_stopped=False)
        write(case/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('progress_events', 'final_shape')}, default=default), flush=True)
    review(case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--name', default='G01')
    parser.add_argument('--variant', type=int, default=1)
    parser.add_argument('--speed', type=float, default=.005)
    parser.add_argument('--local-shape-off', action='store_true')
    parser.add_argument('--review', action='store_true')
    arguments = parser.parse_args()
    if arguments.prepare: prepare()
    elif arguments.review: review(BASE/arguments.name)
    else: run(arguments)
