"""Read-only final audit, including target contacts at every 100 Hz sample.

Does not step physics, alter the frozen controller, or rerun an experiment.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/contact_turn_20261003'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def command_audit(case):
    commands = read(case / 'commands.json')
    lookup = {x['command_id']: x for x in commands}
    assert list(lookup) == list(range(1, len(commands) + 1))
    assert all(x['applied'] and x['ack']['ok'] for x in commands)
    assert all(x['computed_sim_time'] == x['applied_sim_time'] for x in commands)
    max_error = 0.0
    audited = 0
    with np.load(case / 'trace.npz') as trace:
        for command_id in np.unique(trace['applied_command_id']):
            if command_id == 0:
                continue  # Saved initial state, before the first submitted command.
            command = lookup[int(command_id)]
            selected = trace['applied_command_id'] == command_id
            max_error = max(max_error, float(abs(trace['ctrl'][selected] - command['target_rad']).max()))
            assert np.all(trace['sim_time'][selected] >= command['applied_sim_time'] - 1e-10)
            audited += int(selected.sum())
    assert max_error < 1e-12
    return dict(commands=len(commands), raw_command_samples=audited, max_command_ctrl_error_rad=max_error)


def contact_audit(case, model):
    data = mujoco.MjData(model)
    floor = model.geom('floor').id
    head = model.body('head').id
    target = {model.geom(name).id for name in ('pipe_sharp_out_03', 'pipe_straight_04')}
    maximum = 0
    target_frames = 0
    ground_frames = 0
    self_frames = 0
    with np.load(case / 'trace.npz') as trace:
        for qpos, qvel, ctrl in zip(trace['qpos'], trace['qvel'], trace['ctrl']):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)
            modules = set()
            ground = self_contact = False
            for contact in data.contact:
                b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                ground |= bool((contact.geom1 == floor and b2 >= head) or
                               (contact.geom2 == floor and b1 >= head))
                self_contact |= bool(b1 >= head and b2 >= head and abs(b1 - b2) > 1)
                if contact.geom1 in target and b2 >= head and contact.pos[1] > .025:
                    modules.add(b2 - head)
                if contact.geom2 in target and b1 >= head and contact.pos[1] > .025:
                    modules.add(b1 - head)
            maximum = max(maximum, len(modules))
            target_frames += bool(modules)
            ground_frames += ground
            self_frames += self_contact
        samples = len(trace['sim_time'])
    assert maximum == 0 and ground_frames == 0 and self_frames == 0
    return dict(raw_contact_samples=samples, maximum_actual_target_contact_modules=maximum,
                target_contact_frames=target_frames, ground_frames=ground_frames,
                nonadjacent_self_contact_frames=self_frames)


def main():
    freeze = read(BASE / 'stage1_parameter_freeze.json')
    for relative, expected in freeze['selected_manifest']['code_hash'].items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(ROOT / 'validation/contact_turn_stage1_audit.py') == freeze['audit_sha256']
    assert sha(BASE / 'stage1_training_plan.json') == freeze['training_plan_sha256']
    assert sha(BASE / 'stage1_training_results.json') == freeze['training_results_sha256']
    snapshots = {sha(p) for p in (BASE / 'source_snapshots').glob('*.py')}
    stage0_manifest = read(BASE / 'manifest_stage0_round2_clearance.json')
    assert stage0_manifest['source_hash'] in snapshots
    assert set(freeze['selected_manifest']['code_hash'].values()) <= snapshots
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    training = read(BASE / 'stage1_training_results.json')['results']
    verification = read(BASE / 'stage1_release_gate.json')['results']
    rows = []
    for phase in range(3):
        case = BASE / 'stage0_round2_clearance' / ('phase' + str(phase))
        row = dict(directory=case.relative_to(ROOT).as_posix(), group='stage0', **command_audit(case))
        rows.append(row)
    for group, values in [('training', training), ('verification', verification)]:
        for item in values:
            case = ROOT / item['directory']
            row = dict(directory=case.relative_to(ROOT).as_posix(), group=group,
                       **command_audit(case), **contact_audit(case, model))
            if group == 'verification':
                held = read(case / 'held_out_state.json')
                assert held['freeze_sha256'] == sha(BASE / 'stage1_parameter_freeze.json')
                assert held['verification_entry_sha256'] == sha(ROOT / 'validation/contact_turn_verify.py')
            rows.append(row)
            print(json.dumps(dict(group=group, case=case.name, raw_contact_samples=row['raw_contact_samples'],
                                  target_contact_frames=row['target_contact_frames'])), flush=True)
    assert len(rows) == 20
    write(BASE / 'delivery_independent_audit.json', dict(passed=True, frozen_sources_unchanged=True,
          effective_source_snapshots_present=True, rows=rows, physics_advanced=False))
    files = [p for p in BASE.rglob('*') if p.is_file() and 'app' not in p.relative_to(BASE).parts
             and p.name != 'delivery_manifest.json']
    results = list(BASE.rglob('result.json'))
    source_paths = list((ROOT / 'validation').glob('contact_turn_*.py')) + [
        ROOT / 'codes/contact_turn_controller.py', ROOT / 'codes/local_bend_backbone.py']
    environment = dict(python=sys.executable, python_version=platform.python_version(),
                       mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__)
    assert environment['mujoco'] == '3.3.0'
    for key in ('python_version', 'mujoco', 'numpy'):
        assert environment[key] == stage0_manifest[key]
    for name, expected in stage0_manifest['asset_hash'].items():
        assert sha(ROOT / 'mujoco_model/asset' / name) == expected
    assert sha(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe') == stage0_manifest['exe_hash']
    manifest = dict(actual_run_date='2026-10-04', scheme_folder_date='2026-10-03',
        stop_stage=1, training_successes=0, verification_successes=0,
        stage2_tested=False, stage3_tested=False, real_hardware=False,
        environment_at_delivery=environment, shared_stage0_environment=stage0_manifest,
        environment_provenance_note='Stage1 used the same interpreter and checked MuJoCo 3.3.0. '
            'Its per-case manifests do not contain all runtime/asset versions; this shared linkage '
            'and current delivery check are not retroactively recorded per-case provenance.',
        attempts=dict(total=len(results), effective_stage0=3, effective_training=12,
                      effective_verification=5, exploratory_excluded=len(results)-20),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
        data_files={p.relative_to(BASE).as_posix():dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files},
        bytes_excluding_private_apps=sum(p.stat().st_size for p in files),
        exploratory_provenance_note='Some early attempts have hashes and traces but incomplete source snapshots; '
            'they are excluded from effective conclusions. Private EXE copies are ignored; the canonical EXE hash is retained.')
    write(BASE / 'delivery_manifest.json', manifest)
    print(json.dumps(dict(delivery_passed=True, attempts=len(results), files=len(files),
                          total_bytes=manifest['bytes_excluding_private_apps'])), flush=True)


if __name__ == '__main__':
    main()
