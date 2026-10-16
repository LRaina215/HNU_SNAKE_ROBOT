"""Offline truth evaluation of saved servo-only runs; never imports a simulator.

G is an unregistered virtual guide. No world-space G tracking error is computed.
All continuous-duration claims refer to consecutive saved samples, not unseen
states between samples. Signed spin is integrated from actual body angular rate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BASE = Path(__file__).resolve().parent
CONTACT_THRESHOLD_N = .02
REQUIRED_TARGET_DURATION_S = 50.
REQUIRED_ADVANCE_M = .04
REQUIRED_SPIN_RAD = np.pi / 2
COMPONENTS = ('source_straight', 'inlet_connector', 'target_straight', 'outlet_connector')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def integrate(values, time):
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values, dtype=float)
    if len(time) > 1:
        width = np.diff(time).reshape((-1,) + (1,) * (values.ndim - 1))
        result[1:] = np.cumsum(.5 * (values[:-1] + values[1:]) * width, axis=0)
    return result


def intervals(mask, time, max_gap):
    answer = []
    start = None
    for i, valid in enumerate(mask):
        gap = i > 0 and time[i] - time[i - 1] > max_gap
        if start is not None and (not valid or gap):
            answer.append((start, i - 1))
            start = None
        if valid and start is None:
            start = i
    if start is not None:
        answer.append((start, len(time) - 1))
    return answer


def first_time(mask, time):
    found = np.flatnonzero(mask)
    return float(time[found[0]]) if len(found) else None


def movement_record(start, end, time, centers, spin_integral):
    advance = centers[end, :, 1] - centers[start, :, 1]
    spin = spin_integral[end] - spin_integral[start]
    same_sign = bool(np.all(spin >= REQUIRED_SPIN_RAD) or np.all(spin <= -REQUIRED_SPIN_RAD))
    duration = float(time[end] - time[start])
    return dict(start_t_s=float(time[start]), end_t_s=float(time[end]), duration_s=duration,
                every_rod_target_axis_advance_m=advance,
                every_rod_net_axial_spin_deg=np.degrees(spin),
                minimum_rod_advance_m=float(np.min(advance)),
                minimum_absolute_net_spin_deg=float(np.min(np.abs(np.degrees(spin)))),
                all_rods_advance_at_least_40mm=bool(np.all(advance >= REQUIRED_ADVANCE_M)),
                all_rods_net_spin_at_least_90deg_same_sign=same_sign,
                interval_passes_motion_requirements=bool(duration >= REQUIRED_TARGET_DURATION_S - 1e-7 and
                                                         np.all(advance >= REQUIRED_ADVANCE_M) and same_sign))


def qualifying_window(segments, time, centers, spin_integral):
    """Find an observed >=50s interval; later reversal cannot erase an earlier pass."""
    for start, end in segments:
        if time[end] - time[start] < REQUIRED_TARGET_DURATION_S - 1e-7:
            continue
        for last in range(start, end + 1):
            latest_start = int(np.searchsorted(time, time[last] - REQUIRED_TARGET_DURATION_S + 1e-7, side='right')) - 1
            if latest_start < start:
                continue
            firsts = np.arange(start, min(latest_start, last) + 1)
            advance = centers[last, :, 1] - centers[firsts, :, 1]
            spin = spin_integral[last] - spin_integral[firsts]
            enough_advance = np.all(advance >= REQUIRED_ADVANCE_M, axis=1)
            enough_spin = np.all(spin >= REQUIRED_SPIN_RAD, axis=1) | np.all(spin <= -REQUIRED_SPIN_RAD, axis=1)
            good = np.flatnonzero(enough_advance & enough_spin)
            if len(good):
                return movement_record(int(firsts[good[0]]), last, time, centers, spin_integral)
    return None


def composition(pipe_forces, front_only=False):
    selected = pipe_forces[:, :8] if front_only else pipe_forces
    active = np.sum(selected, axis=1) > CONTACT_THRESHOLD_N
    source, connector, target = bool(active[0]), bool(active[1] or active[3]), bool(active[2])
    names = []
    # Explicit categories keep ordinary source response separate from a target hypothesis.
    if source:
        names.append('ordinary_source')
    if connector:
        names.append('connector')
    if target:
        names.append('target_straight')
    return ('+'.join(names) if names else 'no_loaded_pipe_contact') + ('_only' if len(names) == 1 else '')


def hypothesis_audit(run, data, time, pipe_forces):
    events, warnings = [], []
    event_path = run / 'controller_events.jsonl'
    if event_path.exists():
        for number, line in enumerate(event_path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                warnings.append('unreadable event line ' + str(number))
                continue
            kind = item.get('kind', '')
            if 'hypothesis' in kind or kind == 'interaction_direction_hypothesis':
                events.append(item)
    result = read_json(run / 'result.json')
    for field in ('first_servo_interaction_hypothesis', 'first_capture_hypothesis'):
        when = result.get(field)
        if when is not None and not any(abs(float(event.get('t', -1)) - float(when)) < 1e-7 for event in events):
            events.append(dict(t=float(when), kind=field, evidence_source='result.json'))
    records = []
    for event in events:
        if 't' not in event:
            continue
        when = float(event['t'])
        index = int(np.argmin(np.abs(time - when)))
        before = int(np.clip(np.searchsorted(time, when, side='right') - 1, 0, len(time) - 1))
        after = int(np.clip(np.searchsorted(time, when, side='left'), 0, len(time) - 1))
        records.append(dict(kind=event.get('kind', ''), hypothesis_t_s=when,
                            matched_trace_t_s=float(time[index]), time_offset_s=float(time[index] - when),
                            front8_contact_composition=composition(pipe_forces[index], True),
                            all21_contact_composition=composition(pipe_forces[index]),
                            front8_before=composition(pipe_forces[before], True),
                            front8_after=composition(pipe_forces[after], True),
                            normal_N_by_component=np.sum(pipe_forces[index], axis=1),
                            event_timestamp_exactly_sampled=bool(abs(time[index] - when) < 1e-7),
                            interpretation='actual contact classification at logged samples; load hypothesis is not takeover'))
    counts = Counter(record['front8_contact_composition'] for record in records)
    mask = np.asarray(data.get('front_interaction_hypothesis', np.zeros(len(time))), dtype=bool)
    frame_counts = Counter(composition(pipe_forces[i], True) for i in np.flatnonzero(mask))
    duration_by_class = {}
    for category in frame_counts:
        classified = np.array([mask[i] and composition(pipe_forces[i], True) == category for i in range(len(time))])
        duration_by_class[category] = float(integrate(classified, time)[-1])
    return dict(events=records, event_front8_composition_counts=dict(counts),
                persistent_hypothesis_frame_counts=dict(frame_counts),
                persistent_hypothesis_duration_estimates_s=duration_by_class,
                warnings=warnings)


def depth_record(values, time, max_gap):
    values = np.asarray(values, dtype=float)
    exceeding = values > .003
    parts = intervals(exceeding, time, max_gap)
    return dict(max_depth_m=float(np.max(values)),
                estimated_time_above_3mm_s=float(integrate(exceeding, time)[-1]),
                longest_observed_span_above_3mm_s=max((float(time[b] - time[a]) for a, b in parts), default=0.))


def penetration_audit(data, time, max_gap):
    depth = data['depth']
    result = dict(source_plus_inlet_connector=depth_record(depth[:, 0], time, max_gap),
                  target_plus_outlet_connector=depth_record(depth[:, 1], time, max_gap),
                  self_contact=depth_record(depth[:, 2], time, max_gap),
                  floor=depth_record(depth[:, 3], time, max_gap))
    split = data.get('eval_pipe_depth_m')
    if split is not None:
        split = np.asarray(split)
        if split.ndim == 3:
            split = np.max(split, axis=2)
        if split.shape == (len(time), 4):
            result['separate_pipe_components_available'] = True
            result['by_component'] = {name: depth_record(split[:, i], time, max_gap) for i, name in enumerate(COMPONENTS)}
            result['connectors_combined'] = depth_record(np.max(split[:, [1, 3]], axis=1), time, max_gap)
            return result
    result.update(separate_pipe_components_available=False, connectors_combined=None,
                  limitation='saved depth combines each straight pipe with its adjacent connector; connector-only penetration cannot be reconstructed from normal force')
    return result


def torque_audit(run, data, time):
    torque = data['torque']
    model_path = run / 'assets' / 'model.xml'
    limits = None
    if model_path.exists():
        actuators = ET.parse(str(model_path)).getroot().find('actuator')
        if actuators is not None:
            ranges = [element.get('forcerange', '').split() for element in actuators]
            if len(ranges) == 20 and all(len(item) == 2 for item in ranges):
                limits = np.asarray([max(abs(float(v)) for v in item) for item in ranges])
    result = dict(max_absolute_torque_Nm=float(np.max(np.abs(torque))),
                  maximum_absolute_torque_by_joint_Nm=np.max(np.abs(torque), axis=0),
                  accumulation_method='trapezoidal integration of sampled indicators; sub-sample peaks are unavailable')
    if limits is None or np.any(limits <= 0):
        result.update(saturation_available=False, reason='explicit actuator force limits unavailable in frozen XML')
        return result
    fraction = np.abs(torque) / limits
    saturated, near = fraction >= .999, fraction >= .97
    result.update(saturation_available=True, actuator_limits_Nm=limits,
                  saturation_threshold_fraction=.999, near_saturation_threshold_fraction=.97,
                  saturated_seconds_by_joint=integrate(saturated, time)[-1],
                  any_joint_saturated_seconds=float(integrate(saturated.any(axis=1), time)[-1]),
                  near_saturated_seconds_by_joint=integrate(near, time)[-1],
                  any_joint_near_saturated_seconds=float(integrate(near.any(axis=1), time)[-1]))
    return result


def evaluate(run):
    with np.load(run / 'trace.npz', allow_pickle=False) as archive:
        keys = ('t', 'centers', 'points', 'eval_pipe_normal_N', 'actual_axial_spin_rad_s',
                'depth', 'eval_pipe_depth_m', 'torque', 'front_interaction_hypothesis', 'mode', 'S')
        data = {key: archive[key] for key in keys if key in archive.files}
    required = ('t', 'centers', 'points', 'eval_pipe_normal_N', 'actual_axial_spin_rad_s', 'depth', 'torque')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError('missing trace fields: ' + ', '.join(missing))
    time, centers = data['t'], data['centers']
    if len(time) == 0 or (len(time) > 1 and np.any(np.diff(time) <= 0)):
        raise ValueError('trace time must be nonempty and strictly increasing')
    shapes = dict(centers=(len(time), 21, 3), points=(len(time), 22, 3),
                  eval_pipe_normal_N=(len(time), 4, 21), actual_axial_spin_rad_s=(len(time), 21),
                  depth=(len(time), 4), torque=(len(time), 20))
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.all(np.isfinite(data[name])):
            raise ValueError('invalid shape or nonfinite values in ' + name)
    if not np.all(np.isfinite(time)):
        raise ValueError('nonfinite trace time')
    nominal_step = float(np.median(np.diff(time))) if len(time) > 1 else .1
    max_gap = nominal_step * 1.5 + 1e-8
    source_radius = np.linalg.norm(centers[:, :, 1:] - [0., 1.], axis=2)
    target_radius = np.linalg.norm(centers[:, :, [0, 2]] - [1.1, 1.], axis=2)
    source = ((centers[:, :, 0] >= -.2) & (centers[:, :, 0] <= .95) &
              (source_radius >= .07) & (source_radius <= .13))
    target = ((centers[:, :, 1] >= .15) & (centers[:, :, 1] <= 1.2) &
              (target_radius >= .07) & (target_radius <= .13))
    pipe = data['eval_pipe_normal_N']
    total_by_component = np.sum(pipe, axis=2)
    target_contact = total_by_component[:, 2] > CONTACT_THRESHOLD_N
    front_contact = np.sum(pipe[:, 2, :8], axis=1) > CONTACT_THRESHOLD_N
    initial_all_source = bool(np.all(source[0]))
    initial_no_target = bool(np.sum(pipe[0, 2:]) <= CONTACT_THRESHOLD_N)
    initial_only_source = bool(np.sum(pipe[0, 1:]) <= CONTACT_THRESHOLD_N)
    segments = intervals(np.all(target, axis=1), time, max_gap)
    spin_integral = integrate(data['actual_axial_spin_rad_s'], time)
    movement = [movement_record(a, b, time, centers, spin_integral) for a, b in segments]
    longest = max(movement, key=lambda item: item['duration_s']) if movement else None
    terminal = movement[-1] if segments and segments[-1][1] == len(time) - 1 else None
    qualified = qualifying_window(segments, time, centers, spin_integral)
    angles = np.unwrap(np.arctan2(centers[:, :6, 2] - 1., centers[:, :6, 0] - 1.1), axis=1)
    span = np.ptp(angles, axis=1)
    front_geometry = np.all(target[:, :6], axis=1)
    front6_contact = np.sum(pipe[:, 2, :6], axis=1) > CONTACT_THRESHOLD_N
    wrap = front_geometry & front6_contact & (span >= np.pi)
    first_front_contact = first_time(front_contact, time)
    contact_after_t0 = bool(first_front_contact is not None and first_front_contact > time[0] + 1e-7)
    chain = bool(abs(time[0]) < 1e-7 and initial_all_source and initial_no_target and initial_only_source and
                 contact_after_t0 and qualified is not None and first_front_contact <= qualified['start_t_s'] + 1e-7)
    result = read_json(run / 'result.json')
    gates = dict(trace_starts_at_t0=bool(abs(time[0]) < 1e-7), initial_all21_source=initial_all_source,
                 initial_no_target=initial_no_target, initial_only_source_contact=initial_only_source,
                 new_front_target_contact_after_t0=contact_after_t0,
                 actual_full_target_50s_and_every_rod_motion=qualified is not None,
                 front_contact_precedes_qualified_target_interval=bool(qualified is not None and first_front_contact is not None and
                                                                      first_front_contact <= qualified['start_t_s'] + 1e-7))
    report = dict(run_id=run.name, evaluation_kind='offline actual-state acceptance, independent of servo hypotheses',
                  completed_result_file_available=(run / 'result.json').exists(),
                  stop_reason=result.get('stop', 'result_file_unavailable'), final_mode=result.get('final_mode'),
                  recorded_result_action_chain_complete=result.get('action_chain_complete'),
                  action_chain_complete=chain, acceptance_gates=gates,
                  unmet_acceptance_gates=[name for name, passed in gates.items() if not passed],
                  recorded_start_t_s=float(time[0]), recorded_end_t_s=float(time[-1]), sample_count=len(time),
                  nominal_sample_step_s=nominal_step, maximum_sample_gap_s=float(np.max(np.diff(time))) if len(time) > 1 else None,
                  geometry=dict(source_axis='X', source_center_yz_m=[0., 1.], source_x_bounds_m=[-.2, .95],
                                target_axis='Y', target_center_xz_m=[1.1, 1.], target_y_bounds_m=[.15, 1.2],
                                rod_center_radius_bounds_m=[.07, .13], normal_contact_threshold_N=CONTACT_THRESHOLD_N),
                  initial=dict(all21_source_center_envelope=initial_all_source, no_target_contact=initial_no_target,
                               only_ordinary_source_pipe_contact=initial_only_source,
                               normal_N_by_component=total_by_component[0],
                               source_center_radius_min_max_m=[float(np.min(source_radius[0])), float(np.max(source_radius[0]))]),
                  actual_front=dict(first_head_endpoint_beyond_source_x_t_s=first_time(data['points'][:, 0, 0] > .95, time),
                                    first_head_rod_center_beyond_source_x_t_s=first_time(centers[:, 0, 0] > .95, time),
                                    maximum_head_endpoint_extension_x_m=float(np.max(data['points'][:, 0, 0] - .95)),
                                    first_any_target_straight_contact_t_s=first_time(target_contact, time),
                                    first_front8_target_straight_contact_t_s=first_front_contact,
                                    first_front6_target_wrap_over_180deg_t_s=first_time(wrap, time),
                                    maximum_front6_target_polar_span_deg=float(np.degrees(np.max(span))),
                                    maximum_span_when_all_front6_in_target_envelope_deg=float(np.degrees(np.max(span[front_geometry]))) if front_geometry.any() else None,
                                    final_front6_target_polar_span_deg=float(np.degrees(span[-1])),
                                    span_interpretation='angles unwrapped along first six rods; meaningful as target wrapping only with target envelope and target-straight contact'),
                  actual_full_target=dict(first_all21_target_envelope_t_s=first_time(np.all(target, axis=1), time),
                                          maximum_rods_in_target_envelope=int(np.max(np.sum(target, axis=1))),
                                          final_rods_in_target_envelope=int(np.sum(target[-1])),
                                          continuous_intervals=movement, longest_interval=longest, terminal_interval=terminal,
                                          longest_observed_duration_s=longest['duration_s'] if longest else 0.,
                                          terminal_observed_duration_s=terminal['duration_s'] if terminal else 0.,
                                          qualifying_motion_window=qualified,
                                          interval_method='consecutive in-envelope samples, no gap above 1.5 nominal sample periods; duration is last minus first timestamp'),
                  whole_record_movement=dict(every_rod_target_y_advance_m=centers[-1, :, 1] - centers[0, :, 1],
                                             every_rod_source_x_advance_m=centers[-1, :, 0] - centers[0, :, 0],
                                             every_rod_net_axial_spin_deg=np.degrees(spin_integral[-1]),
                                             every_rod_absolute_axial_spin_deg=np.degrees(integrate(np.abs(data['actual_axial_spin_rad_s']), time)[-1]),
                                             acceptance_uses='net signed spin over a full-target interval with a common sign; total absolute oscillation is not credited'),
                  penetration=penetration_audit(data, time, max_gap), torque=torque_audit(run, data, time),
                  servo_hypotheses=hypothesis_audit(run, data, time, pipe),
                  limitations=['No world error of G or C: the virtual guide has no world registration.',
                               'Contact and continuity conclusions are limited to saved sample resolution.',
                               'Servo interaction/capture hypotheses do not themselves satisfy any actual-transfer acceptance gate.'])
    (run / 'EVALUATION.json').write_text(json.dumps(report, default=serial, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--run', action='append', help='optional saved run ID; default evaluates every available trace')
    args = parser.parse_args()
    runs = args.base / 'runs'
    selected = set(args.run or [])
    reports, skipped, errors = [], [], []
    for directory in sorted(runs.iterdir()) if runs.exists() else []:
        if not directory.is_dir() or (selected and directory.name not in selected):
            continue
        if not (directory / 'trace.npz').exists():
            skipped.append(dict(run_id=directory.name, reason='no completed trace.npz available'))
            continue
        try:
            reports.append(evaluate(directory))
        except (ValueError, OSError, KeyError, ET.ParseError) as exc:
            errors.append(dict(run_id=directory.name, error=str(exc)))
    summary = dict(evaluation_kind='offline development-run summary; no dynamics executed',
                   evaluated_runs=[dict(run_id=item['run_id'], stop_reason=item['stop_reason'],
                                        recorded_end_t_s=item['recorded_end_t_s'], action_chain_complete=item['action_chain_complete'],
                                        first_target_straight_contact_t_s=item['actual_front']['first_any_target_straight_contact_t_s'],
                                        longest_all21_target_s=item['actual_full_target']['longest_observed_duration_s'],
                                        terminal_all21_target_s=item['actual_full_target']['terminal_observed_duration_s'],
                                        unmet_acceptance_gates=item['unmet_acceptance_gates']) for item in reports],
                   action_chain_complete_count=sum(item['action_chain_complete'] for item in reports),
                   skipped_runs=skipped, evaluation_errors=errors,
                   summary_scope='selected run IDs' if selected else 'all run directories with saved traces')
    (args.base / 'DEVELOPMENT_SUMMARY.json').write_text(json.dumps(summary, default=serial, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, default=serial, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
