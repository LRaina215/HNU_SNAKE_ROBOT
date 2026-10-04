"""Checkpoint-first three-segment compliance validation through the EXE relay.

No real servo access, parameter search, or hash checks. Both modes start with
the exact saved physical state; geometry/contact truth is explicit test input.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import mujoco

from segmented_pole_experiment import ROOT, BASE, HORIZONTAL, VERTICAL, Trial, AxisReplay, write_json
from three_segment_compliance import ThreeSegmentController

OUT = BASE/'three_segment_v1'
SOURCES = ['codes/three_segment_compliance.py', 'validation/three_segment_experiment.py',
           'validation/three_segment_review.py', 'validation/three_segment_checks.py']
DT = .2


class ThreeTrial(Trial):
    def __init__(self, checkpoint, folder):
        self.partition = np.array([-1, -1])
        super().__init__(checkpoint, folder, verify_hashes=False)
        for name in SOURCES:
            dst = self.folder/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dst)

    def snapshot(self):
        row = super().snapshot()
        # The old base logger did not identify the named tail body as module 20.
        for k, contact in enumerate(self.scratch.contact):
            names = [self.m.geom(int(g)).name for g in contact.geom]
            bodies = [self.m.body(int(self.m.geom_bodyid[g])).name for g in contact.geom]
            if 'tail' not in bodies:
                continue
            side = 1-bodies.index('tail')
            row['ground'] |= names[side] == 'floor'
            pipes = ['pipe_straight_01', 'pipe_straight_03', 'pipe_unused_branch_02']
            if names[side] in pipes:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.scratch, k, force)
                row['normal_force'][pipes.index(names[side]), 20] += max(0., force[0])
        row['partition'] = self.partition.copy()
        return row


def run(root, mode, steps=64, scene='horizontal'):
    checkpoint = HORIZONTAL if scene == 'horizontal' else VERTICAL
    folder = root/f'{scene}_{mode}'
    sim = replay = controller = None
    result = dict(method=mode, checkpoint=str(checkpoint), scene=scene,
                  steps=0, outcome='engineering_failure', mujoco=mujoco.__version__,
                  step_seconds=DT, phase_step_rad=-.075, no_SHA_checks=True)
    try:
        sim = ThreeTrial(checkpoint, folder)
        phase0 = sim.phase
        g0 = sim.fresh()
        if mode == 'three':
            controller = ThreeSegmentController(g0, sim.original.parameters, sim.d.ctrl,
                sim.phase, sim.corner, sim.source, sim.target, sim.radius)
            sim.partition[:] = [controller.front, controller.rear]
            sim.records[0]['partition'] = sim.partition.copy()
            result['configuration'] = controller.configuration()
        else:
            replay = AxisReplay()
        result.update(settings=sim.meta['settings'], requested_steps=steps)
        # Freeze plan before running, not after inspecting the resulting movie.
        write_json(folder/'plan.json', dict(**result,
            primary_gate='two initially non-target modules persist in target region for 0.6 s; front advances one link; both pipes retain contact; no floor',
            extension='only extend three-segment run if the primary gate passes',
            stop='floor contact or whole-body pipe support absent 0.2 s',
            assessment='rolling, no fixed-target hold inserted'))
        lost = 0
        outcome = 'time_limit'
        for step in range(steps):
            g = sim.fresh()
            measured = np.radians(g['angles_deg'])*sim.bridge.signs
            phase = phase0-(step+1)*.075
            if controller is not None:
                window = [r['normal_force'] for r in sim.records[-60:]]
                target, info = controller.compute(g, measured, phase, DT, window)
                sim.partition[:] = [controller.front, controller.rear]
            else:
                control_g = g
                if step == 0:
                    control_g = copy.deepcopy(sim.meta['geometry_at_capture'])
                    for key in ('run_id', 'config_id', 'sequence', 'sample_unix_ms', 'age_ms'):
                        control_g[key] = g[key]
                replayed = replay.geometry(control_g, sim.meta['geometry_at_capture']['projection_axis'])
                if step:
                    control_g = replayed
                else:
                    sim.original.key = (control_g['run_id'], control_g['config_id'], -1)
                target = sim.original.compute(control_g, phase, sim.d.ctrl.copy(), reference_phase=sim.phase)
                if target is None:
                    outcome = 'baseline_feedback_invalid'
                    result['error'] = sim.original.status
                    break
                info = dict(raw=sim.original.backbone.joint_angles(phase+sim.original.phase_offset))
            sim.stage = 'rolling'
            sim.phase = phase
            info.update(step=step+1, target=target.copy(), measured=measured.copy(), phase=phase)
            sim.controls.append(info)
            sim.geometry_log.append(g)
            first = len(sim.records)
            sim.command(target, g)
            sim.advance(DT)
            result['steps'] = step+1
            for row in sim.records[first:]:
                lost = lost+1 if not np.any(row['normal_force'] > .1) else 0
                if row['ground']:
                    outcome = 'floor_contact'
                    break
                if lost >= 20:
                    outcome = 'whole_body_support_lost'
                    break
            if outcome != 'time_limit':
                break
            if (step+1) % 16 == 0:
                print(f'{folder.name}: {step+1}/{steps}, partition={sim.partition.tolist()}, '
                      f'target_modules={int(sim.records[-1]["region"].sum())}', flush=True)
        result['outcome'] = outcome
        if controller is not None:
            result['events'] = controller.events
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if replay is not None:
            replay.close()
        if sim is not None:
            write_json(folder/'result.json', result)
            sim.close()
    from three_segment_review import assess
    print(json.dumps(assess(folder), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['original', 'three'], required=True)
    parser.add_argument('--scene', choices=['horizontal', 'vertical_down'], default='horizontal')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output_root.resolve(), args.mode, args.steps, args.scene)
