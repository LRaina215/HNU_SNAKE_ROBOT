"""Staged verification of the shape-preserving three-region repair."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from three_segment_experiment import ThreeTrial, DT
from three_segment_incremental import IncrementalThreeSegmentController
from three_segment_compliance import ThreeSegmentOptions
from spatial_takeover_experiment import points_for_angles
from segmented_pole_experiment import BASE, ROOT, HORIZONTAL, write_json, pipe_frame
from mujoco_checkpoint import load_checkpoint
from pole_compliance import ComplianceParameters
from three_segment_review import assess

OUT = BASE/'three_segment_repair_v1'


def checks():
    m, d, meta, original = load_checkpoint(HORIZONTAL, verify_hashes=False)
    corner, source, target, radius = pipe_frame(m)
    geometry = meta['geometry_at_capture']
    measured = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]].copy()
    fk = lambda q: points_for_angles(m, d, q)
    zero = ComplianceParameters(0., original.parameters.pitch_angle, False)
    # Middle reference must be held for the full-vector zero-increment check;
    # yielding under an existing position error is intentional, not drift.
    c = IncrementalThreeSegmentController(geometry, zero, d.ctrl, meta['settings']['phase'],
        corner, source, target, radius, measured=measured, forward_points=fk)
    result, info = c.compute(geometry, measured, c.phase, DT, [], migrate=False)
    outside = c.weights[:, 1] == 0
    error = float(abs(result[outside]-d.ctrl[outside]).max())
    assert error < 1e-12 and np.max(abs(info['compliance_delta'])) == 0
    assert np.allclose(info['weights'].sum(axis=1), 1.)
    # A genuine phase step must still move front/rear with K=0. It must not be
    # canceled by the per-frame phase fit used for the section difference.
    moved, _ = c.compute(geometry, measured, c.phase-.075, DT, [], migrate=False)
    rolling_change = float(abs(moved[outside]-result[outside]).max())
    assert rolling_change > 1e-3
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'checks.json', dict(zero_increment_outer_error_rad=error,
        zero_gain_correction_exact=True, nonzero_roll_changes_angles_rad=rolling_change,
        partition_weights_sum_one=True, no_SHA=True))
    print((OUT/'checks.json').read_text())


def run(mode):
    steps = dict(fixed_phase=20, rolling=64, extended=192)[mode]
    if mode == 'rolling':
        gate = json.loads((OUT/'fixed_phase/repair_gate.json').read_text())
        if not gate['passed']:
            raise RuntimeError('fixed-phase support gate failed; do not start rolling')
    if mode == 'extended':
        gate = json.loads((OUT/'rolling/assessment.json').read_text())
        if not gate['primary_gate_passed']:
            raise RuntimeError('short rolling transfer gate failed; do not extend')
    folder = OUT/mode
    sim = None
    try:
        sim = ThreeTrial(HORIZONTAL, folder)
        for name in ('codes/three_segment_incremental.py', 'codes/spatial_bishop_backbone.py',
                     'validation/three_segment_repair.py', 'validation/spatial_takeover_experiment.py'):
            dst = folder/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dst)
        g = sim.fresh()
        controller = IncrementalThreeSegmentController(g, sim.original.parameters, sim.d.ctrl, sim.phase,
            sim.corner, sim.source, sim.target, sim.radius, measured=sim.d.qpos[sim.qadr].copy(),
            forward_points=lambda q: points_for_angles(sim.m, sim.d, q))
        sim.partition[:] = [controller.front, controller.rear]
        sim.records[0]['partition'] = sim.partition.copy()
        phase0 = sim.phase
        result = dict(method=mode, checkpoint=str(HORIZONTAL), configuration=controller.configuration(),
            steps=0, settings=sim.meta['settings'], outcome='time_limit')
        write_json(folder/'plan.json', dict(**result, maximum_steps=steps, seconds_per_step=DT,
            phase_increment=0. if mode == 'fixed_phase' else -.075,
            fixed_phase_gate=dict(max_rear_mean_distance_growth_mm=10.,
                minimum_rear_wrap_fraction=.9, both_pipes_contact_throughout=True, no_floor=True),
            rolling_gate='existing physical transfer gate: two consecutive new modules, one link advance, both supports, persistent at end',
            stop='floor or whole-body support lost for 0.2 s', no_parameter_search=True, no_SHA=True))
        lost = 0
        for k in range(steps):
            g = sim.fresh()
            measured = np.radians(g['angles_deg'])*sim.bridge.signs
            phase = phase0 if mode == 'fixed_phase' else phase0-(k+1)*.075
            targets, info = controller.compute(g, measured, phase, DT,
                                               [r['normal_force'] for r in sim.records[-60:]])
            sim.partition[:] = [controller.front, controller.rear]
            sim.phase, sim.stage = phase, mode
            info.update(target=targets.copy(), measured=measured.copy(), phase=phase, step=k+1)
            sim.controls.append(info)
            sim.geometry_log.append(g)
            first = len(sim.records)
            sim.command(targets, g)
            sim.advance(DT)
            result['steps'] = k+1
            for row in sim.records[first:]:
                lost = 0 if np.any(row['normal_force'] > .1) else lost+1
                if row['ground'] or lost >= 20:
                    result['outcome'] = 'support_lost'
                    break
            if result['outcome'] != 'time_limit':
                break
            if (k+1) % 16 == 0:
                print(f'{mode}: step {k+1}, partition={sim.partition.tolist()}', flush=True)
        result['events'] = controller.events
        write_json(folder/'result.json', result)
    finally:
        if sim is not None:
            sim.close()
    summary = assess(folder)
    if mode == 'fixed_phase':
        z = np.load(folder/'trace.npz')
        p = z['points_mm'][:, 10:]-sim.corner
        radial = p-(p@sim.source)[:, :, None]*sim.source
        distance = np.linalg.norm(radial, axis=2)
        unit = radial/distance[:, :, None]
        wrap = abs(np.arctan2(np.cross(unit[:, :-1], unit[:, 1:])@sim.source,
                             np.sum(unit[:, :-1]*unit[:, 1:], axis=2)).sum(axis=1)/(2*np.pi))
        growth = float(np.max(distance.mean(axis=1)-distance[0].mean()))
        fraction = float(wrap.min()/wrap[0])
        contacts = bool(np.all(np.any(z['normal_force'][:, 0, 11:]>.1, axis=1)) and
                        np.all(np.any(z['normal_force'][:, 1, :6]>.1, axis=1)))
        write_json(folder/'repair_gate.json', dict(max_rear_mean_distance_growth_mm=growth,
            minimum_wrap_fraction=fraction, contacts=contacts,
            passed=bool(growth <= 10 and fraction >= .9 and contacts and not summary['floor_contact'])))
        print((folder/'repair_gate.json').read_text())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--mode', choices=['fixed_phase', 'rolling', 'extended'])
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    if args.checks:
        checks()
    elif args.mode:
        run(args.mode)
    else:
        parser.error('choose --checks or --mode')
