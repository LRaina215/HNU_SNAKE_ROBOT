"""Single predeclared progressive-winding trial and archived matched baseline."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil

import numpy as np
import mujoco

from three_segment_experiment import ThreeTrial, DT
from segmented_pole_experiment import BASE, ROOT, HORIZONTAL, pipe_frame, write_json
from spatial_takeover_experiment import points_for_angles
from snake_control.mujoco_checkpoint import load_checkpoint
from snake_control.three_segment_front_support import FrontSupportedThreeSegmentController
from snake_control.progressive_front_winding import ProgressiveFrontWindingController
from front_support_experiment import metrics

OUT = BASE/'progressive_winding_v1'
BASELINE = BASE/'three_segment_front_support_v1/rolling'
REPORT = ROOT/'docs/分段顺应性跨管/三段顺应性_渐进包绕验证.md'
SOURCES = ['snake_control/progressive_front_winding.py', 'snake_control/three_segment_front_support.py',
           'snake_control/three_segment_incremental.py', 'snake_control/three_segment_compliance.py',
           'snake_control/spatial_bishop_backbone.py', 'validation/progressive_winding_experiment.py',
           'validation/front_support_experiment.py', 'validation/spatial_takeover_experiment.py',
           'validation/three_segment_review.py']


def checks(root, condition=False):
    root.mkdir(parents=True, exist_ok=True)
    m, d, meta, original = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, source, axis, radius = pipe_frame(m)
    geometries = json.loads((BASELINE/'geometry.json').read_text(encoding='utf8'))
    controls = json.loads((BASELINE/'controls.json').read_text(encoding='utf8'))
    z = np.load(BASELINE/'trace.npz')
    def make(cls, **kw):
        return cls(geometries[0], original.parameters, d.ctrl.copy(), meta['settings']['phase'],
            corner, source, axis, radius, measured=np.array(controls[0]['measured']),
            forward_points=lambda q: points_for_angles(m, d, q), **kw)
    base = make(FrontSupportedThreeSegmentController)
    zero = make(ProgressiveFrontWindingController, forming_gain=0.)
    saturation = actuator_error_limits(m) if condition else None
    new = make(ProgressiveFrontWindingController, saturation_error_rad=saturation)
    diagnostics = []
    errors = []
    for k, (g, c) in enumerate(zip(geometries, controls)):
        i = k*20
        d.qpos[:] = z['qpos'][i]
        measured = np.array(c['measured'])
        window = z['normal_force'][max(0, i-59):i+1]
        q0, _ = base.compute(g, measured, c['phase'], DT, window)
        q1, _ = zero.compute(g, measured, c['phase'], DT, window)
        q2, info = new.compute(g, measured, c['phase'], DT, window)
        errors.append(float(abs(q1-q0).max()))
        p = np.asarray(g['points_mm'])
        ref = info['forming_reference']
        curve = np.asarray(ref['curve'])
        anchor = ref['anchor']
        u = p[anchor]-p[anchor-1]
        v = curve[1]-curve[0]
        tangent_error = np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v), -1, 1)))
        ids = np.asarray(info['forming_joint_ids'])
        outside = np.setdiff1d(np.arange(20), ids)
        # Only same-state direct changes are tested here, not future dynamics.
        diagnostics.append(dict(seconds=k*DT, fk_feedback_error_mm=float(abs(points_for_angles(m,d,measured)-p).max()),
            start_error_mm=float(np.linalg.norm(curve[0]-p[anchor])), tangent_error_deg=float(tangent_error),
            source_axis_min_mm=float(np.linalg.norm(np.cross(curve-corner,source),axis=1).min()),
            error_before_mm=float(np.linalg.norm(info['forming_error_mm'])),
            linear_error_after_mm=float(np.linalg.norm(info['forming_linear_error_after_mm'])),
            guidance_deg=float(abs(info['forming_delta_deg']).max()),
            outside_direct_error_rad=float(abs(q2[outside]-q0[outside]).max())))
    assert max(errors) < 1e-12
    assert max(x['fk_feedback_error_mm'] for x in diagnostics) < 1e-3
    assert max(x['start_error_mm'] for x in diagnostics) < 1e-8
    assert max(x['tangent_error_deg'] for x in diagnostics) < 2.
    # The hypothetical guided controller's target history diverges; outside
    # command equality is required at the first frame only (before saturation).
    assert diagnostics[0]['outside_direct_error_rad'] < 1e-12
    assert diagnostics[0]['linear_error_after_mm'] < diagnostics[0]['error_before_mm']
    write_json(root/'checks.json', dict(zero_gain_max_error_rad=max(errors), frames=len(controls), samples=diagnostics))
    shutil.copy2(REPORT, root/'plan_before_run.md')
    print(json.dumps(dict(zero_error=max(errors), first_frame=diagnostics[0]), indent=2))


def actuator_error_limits(m):
    # Only the actual model used here is supported: gear=1, proportional
    # position servo, symmetric force limits, no velocity bias term.
    assert np.allclose(m.actuator_gear[:,0],1.)
    kp=m.actuator_gainprm[:,0]
    assert np.all(kp>0) and np.allclose(m.actuator_biasprm[:,1],-kp)
    assert np.allclose(m.actuator_biasprm[:,2],0.)
    assert np.allclose(m.actuator_forcerange[:,0],-m.actuator_forcerange[:,1])
    assert np.all(m.actuator_forcelimited)
    return m.actuator_forcerange[:,1]/kp


def run(root, condition=False):
    if not (root/'checks.json').exists():
        raise RuntimeError('Run --checks first')
    folder = root/'rolling'
    sim = None
    result = dict(method='progressive_winding', checkpoint=str(HORIZONTAL), steps=0, outcome='engineering_failure')
    try:
        sim = ThreeTrial(HORIZONTAL, folder)
        for name in SOURCES:
            dst = folder/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dst)
        shutil.copy2(root/'plan_before_run.md', folder/'plan_before_run.md')
        g = sim.fresh()
        controller = ProgressiveFrontWindingController(g, sim.original.parameters, sim.d.ctrl,
            sim.phase, sim.corner, sim.source, sim.target, sim.radius,
            measured=sim.d.qpos[sim.qadr].copy(), forward_points=lambda q: points_for_angles(sim.m, sim.d, q),
            pitch_mm=sim.meta['settings']['pitch'],
            saturation_error_rad=actuator_error_limits(sim.m) if condition else None)
        phase0 = sim.phase
        sim.partition[:] = [controller.front, controller.rear]
        sim.records[0]['partition'] = sim.partition.copy()
        result.update(configuration=controller.configuration(), settings=sim.meta['settings'], mujoco=mujoco.__version__)
        write_json(folder/'plan.json', dict(**result, maximum_steps=64, step_seconds=DT,
            phase_increment=-.075, baseline=str(BASELINE), no_parameter_search=True, no_SHA=True))
        lost = 0
        result['outcome'] = 'time_limit'
        for k in range(64):
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
                print(f'progressive: {k+1} steps, partition={sim.partition.tolist()}, forming={controller.forming_first}', flush=True)
        result.update(events=controller.events, forming_events=controller.forming_events)
    except Exception as error:
        result.update(outcome='engineering_failure', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if sim is not None:
            write_json(folder/'result.json', result)
            sim.close()
    from three_segment_review import assess
    print(json.dumps(assess(folder), indent=2))


def longest_run(mask):
    run = best = 0
    for value in mask:
        run = run+1 if value else 0
        best = max(best, run)
    return max(0, best-1)*.01


def review(root, video=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m, _, _, _ = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, _, axis, _ = pipe_frame(m)
    base = np.load(BASELINE/'trace.npz')
    new = np.load(root/'rolling/trace.npz')
    assert np.array_equal(np.load(BASELINE/'initial_state.npz')['state'],
                          np.load(root/'rolling/initial_state.npz')['state'])
    report = dict(initial_integration_state_exact=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for name, z in [('front_support',base),('progressive_winding',new)]:
        summary, series = metrics(z, corner, axis)
        contact = z['normal_force'][:,1,6:8] > .1
        summary.update(new_pair_contact_gate_passed=bool(np.all(contact[-60:])),
            module_6_longest_contact_seconds=longest_run(contact[:,0]),
            module_7_longest_contact_seconds=longest_run(contact[:,1]),
            both_new_contact_longest_seconds=longest_run(contact.all(axis=1)))
        summary['combined_gate_passed'] = bool(summary['front_support_gate_passed'] and
            summary['transfer_gate_passed'] and summary['new_pair_contact_gate_passed'])
        report[name] = summary
        for ax,key in zip(axes[:2],['wrap','advance']):
            ax.plot(series['time'],series[key], label=name)
        for j in range(2):
            axes[j+2].plot(series['time'],z['normal_force'][:,1,6+j],label=name)
    for ax,label in zip(axes,['Original front winding (turns)','Original front advance (mm)',
                              'Module 6 target normal (N)','Module 7 target normal (N)']):
        ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel('Seconds from identical saved state')
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=150); plt.close(fig)
    controls = json.loads((root/'rolling/controls.json').read_text(encoding='utf8'))
    targets = np.vstack([new['ctrl'][0], *[c['target'] for c in controls]])
    report['maximum_command_step_deg'] = float(np.degrees(abs(np.diff(targets,axis=0)).max()))
    assert report['maximum_command_step_deg'] <= 18.+1e-8
    report['forming_windows'] = sorted(set(c['forming_first_module'] for c in controls))
    report['first_and_last_forming_error_mm'] = [float(np.linalg.norm(controls[i]['forming_error_mm'])) for i in [0,-1]]
    report['guidance_saturation_fraction'] = float(np.mean([np.any(abs(np.array(c['forming_delta_deg']))>=2.-1e-8) for c in controls]))
    write_json(root/'comparison.json',report)
    print(json.dumps(report,indent=2))
    if video:
        from three_segment_review import render
        render(root/'rolling')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--review',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--condition-saturation',action='store_true')
    parser.add_argument('--output-root',type=Path,default=OUT)
    args = parser.parse_args()
    if args.checks: checks(args.output_root,args.condition_saturation)
    elif args.run: run(args.output_root,args.condition_saturation)
    elif args.review: review(args.output_root,args.video)
    else: parser.error('choose --checks, --run or --review')
