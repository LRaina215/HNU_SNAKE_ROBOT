"""Short single-factor diagnosis of the archived three-segment support failure.

The production controller is unchanged. No parameter search, no SHA checks.
Compare each intervention with the first 4 s of the archived original run.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil

import numpy as np

from three_segment_experiment import ThreeTrial, DT
from snake_control.three_segment_compliance import ThreeSegmentController, LocalSection
from segmented_pole_experiment import BASE, HORIZONTAL, write_json
from snake_control.curved_axis_compliance import window_axis
from three_segment_review import context

OUT = BASE/'three_segment_diagnosis_v1'
BASELINE = BASE/'three_segment_v1/horizontal_three'


class HeldMiddle(ThreeSegmentController):
    def compute(self, geometry, measured, phase, dt, contact_window, migrate=True):
        # Only suppress the middle reference update. The real geometry, phase,
        # servo dynamics and front/rear feedback remain unchanged.
        held = measured.copy()
        held[self.front:self.rear] = self.middle_reference[self.front:self.rear]
        return super().compute(geometry, held, phase, dt, contact_window, migrate)


def run(mode):
    folder = OUT/mode
    sim = None
    try:
        sim = ThreeTrial(HORIZONTAL, folder)
        dst = folder/'sources/validation/three_segment_diagnosis.py'
        shutil.copy2(Path(__file__), dst)
        g = sim.fresh()
        cls = HeldMiddle if mode in ('held_middle', 'held_middle_fixed_phase') else ThreeSegmentController
        controller = cls(g, sim.original.parameters, sim.d.ctrl, sim.phase,
                         sim.corner, sim.source, sim.target, sim.radius)
        phase0 = sim.phase
        initial_target = sim.d.ctrl.copy()
        sim.partition[:] = [controller.front, controller.rear]
        sim.records[0]['partition'] = sim.partition.copy()
        result = dict(method=mode, checkpoint=str(HORIZONTAL), configuration=controller.configuration(),
                      steps=0, outcome='time_limit', duration_seconds=4., settings=sim.meta['settings'])
        write_json(folder/'plan.json', dict(**result, baseline=str(BASELINE),
            intervention=mode, physical_parameters_unchanged=True,
            stop='ground or whole-body support absent for 0.2 s',
            no_parameter_search=True, no_SHA_checks=True))
        lost = 0
        for k in range(20):
            g = sim.fresh()
            measured = np.radians(g['angles_deg'])*sim.bridge.signs
            phase = phase0 if 'fixed_phase' in mode else phase0-(k+1)*.075
            target, info = controller.compute(g, measured, phase, DT,
                [r['normal_force'] for r in sim.records[-60:]])
            if mode == 'held_rear_targets':
                # Counterfactual actuator command clamp, NOT a physical anchor.
                target[controller.initial_rear:] = initial_target[controller.initial_rear:]
                controller.last_target[controller.initial_rear:] = target[controller.initial_rear:]
            sim.partition[:] = [controller.front, controller.rear]
            sim.phase, sim.stage = phase, 'rolling' if 'fixed_phase' not in mode else 'fixed_phase_feedback'
            info.update(target=target.copy(), measured=measured.copy(), phase=phase, step=k+1)
            sim.controls.append(info)
            sim.geometry_log.append(g)
            first = len(sim.records)
            sim.command(target, g)
            sim.advance(DT)
            result['steps'] = k+1
            for row in sim.records[first:]:
                lost = 0 if np.any(row['normal_force'] > .1) else lost+1
                if row['ground'] or lost >= 20:
                    result['outcome'] = 'support_lost'
                    break
            if result['outcome'] != 'time_limit':
                break
        result['events'] = controller.events
        write_json(folder/'result.json', result)
    finally:
        if sim is not None:
            sim.close()
    print(json.dumps(metrics(folder), indent=2), flush=True)


def metrics(folder, save=True):
    result, z, m, d, meta = context(folder)
    from segmented_pole_experiment import pipe_frame
    corner, source, target, radius = pipe_frame(m)
    time = z['time']-z['time'][0]
    use = time <= 4.+1e-8
    points = z['points_mm'][use]
    relative = points[:, 10:]-corner
    radial = relative-(relative@source)[:, :, None]*source
    distance = np.linalg.norm(radial, axis=2)
    unit = radial/distance[:, :, None]
    wrap = abs(np.arctan2(np.cross(unit[:, :-1], unit[:, 1:])@source,
                         np.sum(unit[:, :-1]*unit[:, 1:], axis=2)).sum(axis=1)/(2*np.pi))
    axis_error = []
    for frame in points:
        tangents = np.diff(frame[10:], axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        axis, flatness = window_axis(tangents)
        axis_error.append(np.degrees(np.arccos(np.clip(abs(axis@source), 0, 1))))
    normal = z['normal_force'][use]
    initial = np.load(folder/'initial_state.npz')['state']
    expected = np.load(BASELINE/'initial_state.npz')['state']
    assert np.array_equal(initial, expected)
    commands = json.loads((folder/'commands.json').read_text(encoding='utf8'))
    assert all(c['applied'] for c in commands)
    for c in commands:
        assert np.allclose(z['ctrl'][z['applied_id']==c['id']], c['target'], atol=1e-12)
    rows = []
    for seconds in (0., .2, .6, 1., 2., 3., 4.):
        i = np.argmin(abs(time[use]-seconds))
        rows.append(dict(seconds=float(time[i]), rear_mean_axis_distance_mm=float(distance[i].mean()),
            rear_axis_error_deg=float(axis_error[i]), rear_net_wrap_turns=float(wrap[i]),
            source_modules=np.flatnonzero(normal[i, 0] > .1).tolist(),
            target_modules=np.flatnonzero(normal[i, 1] > .1).tolist(),
            source_force_N=float(normal[i, 0].sum()), target_force_N=float(normal[i, 1].sum())))
    data = dict(mode=folder.name, exact_initial_state=True, commands_delivered=True,
                samples=rows, events=result.get('events', []), no_floor=not bool(np.any(z['ground'][use])))
    if save:
        write_json(folder/'diagnosis.json', data)
    return data


def analyze():
    """Use saved states for geometry attribution, never step MuJoCo here."""
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from snake_control.mujoco_checkpoint import load_checkpoint
    from snake_control.pole_compliance import ComplianceParameters
    from segmented_pole_experiment import pipe_frame
    m, d, meta, original = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, source, target, radius = pipe_frame(m)
    phase = meta['settings']['phase']
    geometry = meta['geometry_at_capture']
    measured = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    p = np.asarray(geometry['points_mm'])
    tangents = np.diff(p[10:], axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    zero = ComplianceParameters(0., original.parameters.pitch_angle, False)
    local = LocalSection(geometry, zero, 10, 20, source, corner)
    offset = local.align(measured[10:], phase)
    predicted = local.angles(phase+offset)
    context_errors = []
    for lo in (0, 8, 9, 10):
        section = LocalSection(geometry, zero, lo, 20, source, corner)
        basis = np.column_stack((section.angles(0)[10-lo:], section.angles(math.pi/2)[10-lo:]))
        w = np.linalg.lstsq(basis, measured[10:], rcond=None)[0]
        residual = basis@(w/np.linalg.norm(w))-measured[10:]
        context_errors.append(dict(start=lo, error_deg=np.degrees(residual),
                                   rms_deg=float(np.degrees(np.sqrt(np.mean(residual**2))))))
    actual = LocalSection(geometry, original.parameters, 10, 20, source, corner)
    aligned = actual.angles(phase+actual.align(d.ctrl[10:], phase))
    geometry_result = dict(design_pitch_deg=math.degrees(original.parameters.pitch_angle),
        measured_link_pitch_deg=np.degrees(np.arcsin(-(tangents@source))),
        zero_gain_zero_roll_aligned_error_deg=np.degrees(predicted-measured[10:]),
        zero_gain_zero_roll_max_error_deg=float(np.degrees(abs(predicted-measured[10:]).max())),
        nominal_target_change_deg=np.degrees(aligned-d.ctrl[10:]), context_errors=context_errors,
        interpretation='shape reconstruction is not an identity map; phase-only alignment cannot remove amplitude/distribution error')
    write_json(OUT/'geometry_attribution.json', geometry_result)
    folders = [BASELINE]+[OUT/name for name in ('held_middle', 'fixed_phase',
                                               'held_middle_fixed_phase', 'held_rear_targets')]
    all_metrics = [metrics(folder, save=folder != BASELINE) for folder in folders]
    write_json(OUT/'summary.json', all_metrics)
    joint_evidence = []
    for folder in (BASELINE, OUT/'fixed_phase', OUT/'held_rear_targets'):
        z = np.load(folder/'trace.npz')
        t = z['time']-z['time'][0]
        qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        for at in (0., .01, .2, .4, .6, .8, 1., 2., 4.):
            i = np.argmin(abs(t-at))
            joint_evidence.append(dict(mode=folder.name, time=float(t[i]), joint=17,
                measured_deg=float(np.degrees(z['qpos'][i, qadr[16]])),
                target_deg=float(np.degrees(z['ctrl'][i, 16])),
                actuator_torque_Nm=float(z['torque'][i, 16])))
    write_json(OUT/'joint17_evidence.json', joint_evidence)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for folder in folders:
        result, z, m, _, _ = context(folder)
        t = z['time']-z['time'][0]
        use = t <= 4.+1e-8
        pos = z['points_mm'][use, 10:]-corner
        radial = pos-(pos@source)[:, :, None]*source
        distance = np.linalg.norm(radial, axis=2)
        unit = radial/distance[:, :, None]
        winding = abs(np.arctan2(np.cross(unit[:, :-1], unit[:, 1:])@source,
                                np.sum(unit[:, :-1]*unit[:, 1:], axis=2)).sum(axis=1)/(2*np.pi))
        axes[0].plot(t[use], distance.mean(axis=1), label=folder.name)
        axes[1].plot(t[use], winding, label=folder.name)
        axes[2].plot(t[use], z['normal_force'][use, 0, 11:].sum(axis=1), label=folder.name)
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Rear mean distance to source axis (mm)')
    axes[1].set_ylabel('Rear net winding magnitude (turns)')
    axes[2].set_ylabel('Rear modules 11-20 source normal force (N)')
    axes[2].set_xlabel('Time since identical checkpoint (s)')
    fig.tight_layout()
    fig.savefig(OUT/'diagnosis.png', dpi=150)
    plt.close(fig)
    print(json.dumps(geometry_result, default=lambda a: a.tolist(), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['held_middle', 'fixed_phase', 'held_middle_fixed_phase',
                                         'held_rear_targets'])
    parser.add_argument('--baseline-metrics', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    if args.analyze:
        analyze()
    elif args.baseline_metrics:
        print(json.dumps(metrics(BASELINE, save=False), indent=2))
    else:
        if args.mode is None:
            parser.error('choose --mode, --analyze, or --baseline-metrics')
        run(args.mode)
