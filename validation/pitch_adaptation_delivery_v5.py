"""Inventory / verify the completed formal delivery without running simulation.

This verifies provenance and completeness, not scientific benefit. The reviewed
stage3 decision and report carry the interpretation of the negative findings.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pitch_adaptation_formal_v5 import BASE, ROOT, verify_freeze
from pitch_adaptation_training_audit import model_path, pipe_progress


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_holds(rows):
    output = []
    for row in rows:
        case = ROOT / row['source_directory']
        result = read(case / 'result.json')
        marker = result['hold_start_s']
        model = mujoco.MjModel.from_xml_path(str(model_path(row['scene'])))
        data = mujoco.MjData(model)
        with np.load(case / 'trace.npz', allow_pickle=False) as raw:
            times = raw['sim_time']
            targets = raw['ctrl']
            held = np.flatnonzero(times >= marker - 1e-8)
            change = float(np.max(abs(targets[held] - targets[held[0]])))
            if change > 1e-12:
                raise ValueError('Hold targets changed: ' + case.name)
            points = {}
            for stamp in [2.2, marker] + [s['sim_time'] for s in result['status'] if s['sim_time'] <= marker + 1e-8]:
                index = int(np.argmin(abs(times - stamp)))
                if abs(times[index] - stamp) > 1e-8:
                    raise ValueError('Hold checkpoint lacks qpos: ' + case.name)
                data.qpos[:] = raw['qpos'][index]
                mujoco.mj_kinematics(model, data)
                points[stamp] = np.asarray(pipe_progress(model, data), float)
        end = points[marker]
        if row['scene'].startswith('pipe_straight'):
            advance = float(end[0] - points[2.2][0])
            reached = advance >= .5 - 1e-8
        else:
            feature = result['goal_feature_m']
            crossed = next((s['sim_time'] for s in result['status'] if s['sim_time'] <= marker + 1e-8 and points[s['sim_time']][1] > feature), None)
            advance = float(end[0] - points[crossed][0]) if crossed is not None else None
            reached = crossed is not None and end[1] > feature and advance >= .2 - 1e-8
        if not reached:
            raise ValueError('Hold began before the raw-data task goal: ' + case.name)
        output.append(dict(case=case.name,goal_reached=True,required_stage_advance_m=advance,
                           maximum_hold_target_change_rad=change,hold_duration_s=row['hold_duration_s']))
    return dict(trials=160,all_raw_hold_entry_goals_verified=True,
                all_hold_targets_constant=True,scope='static last-target hold; no claim of ongoing compliance feedback',rows=output)


def check_complete():
    verify_freeze(BASE / 'training_audit/parameter_freeze.json')
    analysis = BASE / 'formal_analysis'
    audit = read(analysis / 'formal_audit.json')
    if not (audit.get('complete') and audit.get('statistics_complete') and audit.get('audited') == 160
            and audit.get('complete_paired_initial_groups') == 40 and audit.get('mechanism_cases') == 160):
        raise ValueError('Formal delivery requires the complete audited design')
    if read(BASE / 'formal/formal_progress.json')['completed'] != 160:
        raise ValueError('Formal execution is incomplete')
    stats = read(analysis / 'formal_statistics.json')
    if len(stats['methods']) != 16 or any(g['trials'] != 10 for g in stats['methods']):
        raise ValueError('Statistics do not cover all sixteen groups')
    supplement = read(analysis / 'mechanism_comparison.json')
    if len(supplement['rows']) != 160 or supplement['formal_audit_sha256'] != digest(analysis / 'formal_audit.json'):
        raise ValueError('Mechanism supplement does not match the complete audit')
    review = read(analysis / 'stage3_review.json')
    if not review.get('evidence_review_complete') or review.get('stage3_required') is not False:
        raise ValueError('Conditional-stage disposition has not been completed')
    for name, expected in review['inputs_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Reviewed evidence changed: ' + name)
    for row in audit['rows']:
        for name, expected in row['raw_sha256'].items():
            if digest(ROOT / row['source_directory'] / name) != expected:
                raise ValueError('Audited raw trial changed: ' + row['source_directory'] + '/' + name)
    mechanisms = read(analysis / 'mechanism_summary.json')
    if len(mechanisms) != 160 or any(m['unmatched_feedback_times'] or m['maximum_length_partition_error_m'] > 1e-10 for m in mechanisms):
        raise ValueError('Mechanism replay is incomplete or length partition failed')
    for row in audit['rows']:
        case = analysis / 'mechanisms' / Path(row['source_directory']).name
        if any(not (case / ('mechanism.' + ext)).exists() for ext in ('json', 'npz', 'png')):
            raise ValueError('A mechanism artifact is missing')
    return audit


def inventory(manifest):
    paths = []
    for directory in (BASE / 'formal', BASE / 'formal_analysis'):
        paths.extend(p for p in directory.rglob('*') if p.is_file() and 'app' not in p.relative_to(directory).parts and p != manifest)
    paths.extend(ROOT / name for name in (
        'validation/pitch_adaptation_mechanism_summary.py',
        'validation/pitch_adaptation_delivery_v5.py'))
    return {str(p.relative_to(ROOT).as_posix()): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(set(paths))}


def main(verify):
    audit = check_complete()
    hold_report = audit_holds(audit['rows'])
    hold_path = BASE / 'formal_analysis/hold_protocol_audit.json'
    if verify:
        if read(hold_path) != hold_report:
            raise ValueError('Hold protocol evidence differs')
    else:
        hold_path.write_text(json.dumps(hold_report, indent=2), encoding='utf8')
    manifest = BASE / 'formal_analysis/delivery_manifest.json'
    current = inventory(manifest)
    if verify:
        saved = read(manifest)
        if saved['files'] != current:
            missing = sorted(set(saved['files']) - set(current))
            added = sorted(set(current) - set(saved['files']))
            changed = [p for p in set(current) & set(saved['files']) if current[p] != saved['files'][p]]
            raise ValueError('Delivery bytes differ: ' + json.dumps(dict(missing=missing, added=added, changed=sorted(changed))))
        print('PASS: complete formal delivery, reviewed conditional stage, and all saved SHA-256 values.')
    else:
        if manifest.exists():
            raise ValueError('Preserve the existing delivery manifest; use --verify')
        payload = dict(scope='formal raw data and all derived analyses, excluding private EXE app directories; new analysis sources',
                       audited=160, paired_initial_groups=40, stage3_required=False,
                       formal_audit_sha256=digest(BASE / 'formal_analysis/formal_audit.json'),
                       freeze_sha256=audit['freeze_sha256'], file_count=len(current),
                       total_bytes=sum(v['bytes'] for v in current.values()), files=current)
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf8')
        print('Saved formal delivery manifest:', len(current), 'files,', payload['total_bytes'], 'bytes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args().verify)
