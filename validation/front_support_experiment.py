"""Bounded front-support experiment; archived baseline, unchanged physics/gates."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import importlib.util
import json
import shutil

import mujoco
import numpy as np

from three_segment_experiment import ThreeTrial, DT
from snake_control.three_segment_incremental import IncrementalThreeSegmentController
from snake_control.three_segment_front_support import FrontSupportedThreeSegmentController
from segmented_pole_experiment import BASE, ROOT, HORIZONTAL, pipe_frame, write_json
from spatial_takeover_experiment import points_for_angles
from snake_control.mujoco_checkpoint import load_checkpoint

OUT = BASE/'three_segment_front_support_v1'
BASELINE = BASE/'three_segment_repair_v1/rolling'
SOURCES = ['snake_control/three_segment_incremental.py', 'snake_control/three_segment_front_support.py',
           'snake_control/spatial_bishop_backbone.py', 'snake_control/three_segment_compliance.py',
           'validation/front_support_experiment.py', 'validation/three_segment_review.py',
           'validation/spatial_takeover_experiment.py']


def coverage(points, corner, axis):
    r = np.asarray(points)-corner
    r = r-(r@axis)[..., None]*axis
    distance = np.linalg.norm(r, axis=-1)
    u = r/np.maximum(distance[..., None], 1e-9)
    angle = np.arctan2(np.cross(u[..., :-1, :], u[..., 1:, :])@axis,
                       np.sum(u[..., :-1, :]*u[..., 1:, :], axis=-1))
    return abs(angle.sum(axis=-1)/(2*np.pi)), distance.mean(axis=-1)


def metrics(z, corner, axis):
    t = z['time']-z['time'][0]
    wrap, distance = coverage(z['points_mm'][:, :7], corner, axis)
    centers = (z['points_mm'][:, :-1]+z['points_mm'][:, 1:])/2
    advance = (centers[:, :5].mean(axis=1)-centers[0, :5].mean(axis=0))@axis
    contact = np.any(z['normal_force'][:, 1, :6]>.1, axis=1)
    rear = np.any(z['normal_force'][:, 0, 11:]>.1, axis=1)
    lost = next((float(t[i]) for i in range(19, len(t)) if not np.any(contact[i-19:i+1])), None)
    eligible = (np.arange(21)>=5) & ~z['region'][0]
    runs = np.zeros(21, int)
    persistent = np.zeros_like(z['region'])
    for i, region in enumerate(z['region']):
        runs = np.where(region & eligible, runs+1, 0)
        persistent[i] = runs>=60
    pair = np.any(persistent[:, :-1] & persistent[:, 1:], axis=1)
    # Match the existing physical-transfer gate (broader rear cohort).
    old_rear = np.any(z['normal_force'][:, 0, 6:]>.1, axis=1)
    transfer = pair & (advance>=64.5) & contact & old_rear & ~z['ground']
    support = bool(wrap.min()>=.8*wrap[0] and lost is None and
                   np.all(contact[-60:] & rear[-60:]) and advance[-1]>=64.5 and not z['ground'].any())
    summary = dict(seconds=float(t[-1]), front_wrap_initial=float(wrap[0]),
        front_wrap_final=float(wrap[-1]), front_wrap_min_fraction=float(wrap.min()/wrap[0]),
        front_advance_final_mm=float(advance[-1]), front_advance_peak_mm=float(advance.max()),
        first_continuous_front_contact_loss_s=lost, front_contact_fraction=float(contact.mean()),
        rear_contact_fraction=float(rear.mean()), front_support_gate_passed=support,
        transfer_gate_passed=bool(np.all(transfer[-60:]) and len(t)>=60),
        max_sustained_new_modules=int(persistent.sum(axis=1).max()),
        final_sustained_new_modules=int(persistent[-1].sum()),
        final_front_contact_modules=np.flatnonzero(z['normal_force'][-1, 1, :6]>.1).tolist(),
        final_partition=z['partition'][-1].tolist(), floor_contact=bool(z['ground'].any()),
        torque_saturation_fraction=float(np.mean(abs(z['torque'])>=2.355*.99)),
        front_torque_saturation_fraction=float(np.mean(abs(z['torque'][:, :5])>=2.355*.99)),
        max_tracking_error_deg=float(abs(z['error_deg']).max()))
    return summary, dict(time=t, wrap=wrap, distance=distance, advance=advance,
        contact_count=np.sum(z['normal_force'][:, 1, :6]>.1, axis=1),
        normal=z['normal_force'][:, 1, :6].sum(axis=1))


def checks(root):
    root.mkdir(parents=True, exist_ok=True)
    m, d, meta, original = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, source, target, radius = pipe_frame(m)
    geometry = json.loads((BASELINE/'geometry.json').read_text(encoding='utf8'))
    controls = json.loads((BASELINE/'controls.json').read_text(encoding='utf8'))
    z = np.load(BASELINE/'trace.npz')
    spec = importlib.util.spec_from_file_location('archived_incremental',
        BASELINE/'sources/codes/three_segment_incremental.py')
    old = importlib.util.module_from_spec(spec)
    # 历史快照仍保留原始平铺导入；只在读取快照时使用它自己的源码目录。
    sys.path.insert(0, str(BASELINE/'sources/codes'))
    try:
        spec.loader.exec_module(old)
    finally:
        sys.path.pop(0)
    make = lambda cls, **kw: cls(geometry[0], original.parameters, d.ctrl.copy(),
        meta['settings']['phase'], corner, source, target, radius,
        measured=np.array(controls[0]['measured']), forward_points=lambda q: points_for_angles(m, d, q), **kw)
    archived = make(old.IncrementalThreeSegmentController)
    base = make(IncrementalThreeSegmentController)
    zero = make(FrontSupportedThreeSegmentController, shape_gain=0.)
    shaped = make(FrontSupportedThreeSegmentController)
    errors = []; outside = []; correction = []
    diagnostic = []
    # Read-only replay of the 64 archived feedback frames, not a physics run.
    for k, (g, control) in enumerate(zip(geometry, controls)):
        i = k*20
        d.qpos[:] = z['qpos'][i]
        measured = np.array(control['measured'])
        window = z['normal_force'][max(0, i-59):i+1]
        outputs = [c.compute(g, measured, control['phase'], DT, window)
                   for c in (archived, base, zero, shaped)]
        errors.append(max(float(abs(outputs[0][0]-outputs[j][0]).max()) for j in (1, 2)))
        mask = shaped.weights[:, 0] == 0
        outside.append(float(abs(outputs[3][0][mask]-outputs[1][0][mask]).max()))
        correction.append(float(abs(outputs[3][0]-outputs[1][0]).max()))
        commanded = points_for_angles(m, d, np.array(control['target']))
        actual_wrap, actual_r = coverage(z['points_mm'][i, :7], corner, target)
        desired_wrap, desired_r = coverage(commanded[:7], corner, target)
        diagnostic.append(dict(seconds=k*DT, actual_wrap=float(actual_wrap), target_fk_wrap=float(desired_wrap),
            actual_distance_mm=float(actual_r), target_fk_distance_mm=float(desired_r),
            front_tracking_error_deg=float(abs(z['error_deg'][i, :5]).max())))
    assert max(errors)<1e-12 and max(outside)<1e-12
    assert max(correction)>1e-4
    write_json(root/'checks.json', dict(archived_and_zero_gain_error_rad=max(errors),
        same_feedback_nonfront_target_error_rad=max(outside), correction_nonzero_rad=max(correction),
        frames=len(controls), no_SHA=True))
    write_json(root/'baseline_target_diagnosis.json', dict(samples=diagnostic,
        limitation='target FK fixes the current floating base; it is not predicted dynamics or contact'))
    print(json.dumps(json.loads((root/'checks.json').read_text()), indent=2))
    print('Selected target/actual geometry:', json.dumps(diagnostic[::10], indent=2))


def run(root):
    if not (root/'checks.json').exists():
        raise RuntimeError('Run --checks first')
    folder = root/'rolling'
    sim = None
    result = dict(method='front_support', checkpoint=str(HORIZONTAL), steps=0, outcome='engineering_failure')
    try:
        sim = ThreeTrial(HORIZONTAL, folder)
        for name in SOURCES:
            dst = folder/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dst)
        shutil.copy2(ROOT/'docs/分段顺应性跨管/三段顺应性_前部包绕维持验证.md', folder/'plan_before_run.md')
        g = sim.fresh()
        controller = FrontSupportedThreeSegmentController(g, sim.original.parameters, sim.d.ctrl,
            sim.phase, sim.corner, sim.source, sim.target, sim.radius,
            measured=sim.d.qpos[sim.qadr].copy(), forward_points=lambda q: points_for_angles(sim.m, sim.d, q),
            pitch_mm=sim.meta['settings']['pitch'])
        phase0 = sim.phase
        sim.partition[:] = [controller.front, controller.rear]
        sim.records[0]['partition'] = sim.partition.copy()
        result.update(configuration=controller.configuration(), settings=sim.meta['settings'],
                      mujoco=mujoco.__version__)
        write_json(folder/'plan.json', dict(**result, short_steps=64, maximum_steps=128,
            step_seconds=DT, phase_increment=-.075, baseline=str(BASELINE),
            extension='only if BOTH front support and original transfer gates pass at 64 steps',
            no_parameter_search=True, no_SHA=True))
        lost = 0
        result['outcome'] = 'time_limit'
        for k in range(128):
            g = sim.fresh()
            measured = np.radians(g['angles_deg'])*sim.bridge.signs
            phase = phase0-(k+1)*.075
            targets, info = controller.compute(g, measured, phase, DT,
                [r['normal_force'] for r in sim.records[-60:]])
            sim.partition[:] = [controller.front, controller.rear]
            sim.phase, sim.stage = phase, 'rolling'
            info.update(step=k+1, target=targets.copy(), measured=measured.copy(), phase=phase)
            sim.controls.append(info)
            sim.geometry_log.append(g)
            start = len(sim.records)
            sim.command(targets, g)
            sim.advance(DT)
            result['steps'] = k+1
            for row in sim.records[start:]:
                lost = lost+1 if not np.any(row['normal_force']>.1) else 0
                if row['ground'] or lost>=20:
                    result['outcome'] = 'support_lost'
                    break
            if result['outcome'] != 'time_limit':
                break
            if (k+1)%16 == 0:
                print(f'front support: {k+1} steps, partition={sim.partition.tolist()}', flush=True)
            if k+1 == 64:
                arrays = {key: np.asarray([r[key] for r in sim.records]) for key in sim.records[0]}
                short, _ = metrics(arrays, sim.corner, sim.target)
                write_json(folder/'short_gate.json', short)
                if not (short['front_support_gate_passed'] and short['transfer_gate_passed']):
                    break
        result['events'] = controller.events
    except Exception as error:
        result.update(outcome='engineering_failure', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if sim is not None:
            write_json(folder/'result.json', result)
            sim.close()
    from three_segment_review import assess
    print(json.dumps(assess(folder), indent=2))


def review(root, video=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m, d, _, _ = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, _, target, _ = pipe_frame(m)
    base = np.load(BASELINE/'trace.npz')
    new = np.load(root/'rolling/trace.npz')
    assert np.array_equal(base['qpos'][0], new['qpos'][0]) and np.array_equal(base['ctrl'][0], new['ctrl'][0])
    a = np.load(BASELINE/'initial_state.npz')['state']
    b = np.load(root/'rolling/initial_state.npz')['state']
    assert np.array_equal(a, b)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    summaries = {}
    for name, z in [('baseline', base), ('front_support', new)]:
        report, series = metrics(z, corner, target)
        summaries[name] = report
        for ax, field in zip(axes, ['wrap', 'advance', 'contact_count', 'normal']):
            ax.plot(series['time'], series[field], label=name)
            ax.grid(alpha=.25)
            ax.legend()
    for ax, label in zip(axes, ['Front net winding (turns)', 'Front advance (mm)',
                               'Front target contact modules', 'Front target normal force (N)']):
        ax.set_ylabel(label)
    axes[-1].set_xlabel('Seconds from identical saved state')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=150)
    plt.close(fig)
    summaries['initial_integration_state_exact'] = True
    # In the same recorded feedback state the default target is a counterfactual
    # command, not a second simulated outcome. Only isolate the shape term.
    controls = json.loads((root/'rolling/controls.json').read_text(encoding='utf8'))
    summaries['maximum_command_step_deg'] = float(np.degrees(np.max(abs(np.diff(
        np.vstack([new['ctrl'][0], *[x['target'] for x in controls]]), axis=0)))))
    assert summaries['maximum_command_step_deg'] <= 18.+1e-8
    assessment = json.loads((root/'rolling/assessment.json').read_text(encoding='utf8'))
    assert summaries['front_support']['transfer_gate_passed'] == assessment['primary_gate_passed']
    write_json(root/'comparison.json', summaries)
    # Explain labels vs actual transfer using module centres, matching assess().
    centers = (new['points_mm'][-1, :-1]+new['points_mm'][-1, 1:])/2
    relative = centers-corner
    _, source, _, _ = pipe_frame(m)
    source_distance = np.linalg.norm(np.cross(relative, source), axis=1)
    target_distance = np.linalg.norm(np.cross(relative, target), axis=1)
    rows = [dict(module=j, initial_target_region=bool(new['region'][0, j]),
        final_target_region=bool(new['region'][-1, j]), target_distance_mm=float(target_distance[j]),
        source_distance_mm=float(source_distance[j]), target_axial_mm=float(relative[j]@target),
        target_normal_N=float(new['normal_force'][-1, 1, j])) for j in range(5, 11)]
    windows = []
    for k, row in enumerate(controls):
        if row['acquired'] and row['target_contact'] and row['source_contact']:
            windows.append(dict(seconds=k*DT, J11_source_progress_mm=float(
                (new['points_mm'][k*20, 11]-corner)@source)))
    write_json(root/'transfer_diagnosis.json', dict(final_modules=rows,
        eligible_release_windows=windows, final_gate_flags={key: controls[-1][key] for key in
            ('acquired', 'target_contact', 'source_contact', 'release_allowed', 'middle_room')}))
    print(json.dumps(summaries, indent=2))
    if video:
        from three_segment_review import render
        render(root/'rolling')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checks', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--review', action='store_true')
    p.add_argument('--video', action='store_true')
    p.add_argument('--output-root', type=Path, default=OUT)
    a = p.parse_args()
    if a.checks:
        checks(a.output_root)
    elif a.run:
        run(a.output_root)
    elif a.review:
        review(a.output_root, a.video)
    else:
        p.error('choose --checks, --run, or --review')
