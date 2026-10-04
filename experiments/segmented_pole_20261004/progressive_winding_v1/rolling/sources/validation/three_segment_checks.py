"""Focused parity, reference-yielding and contact-gate checks; no physics sweep."""
import json
import math

import numpy as np
from scipy.integrate import quad

from segmented_pole_experiment import HORIZONTAL, pipe_frame, true_axis_geometry
from mujoco_checkpoint import load_checkpoint
from pole_compliance import build_backbone
from three_segment_compliance import LocalSection, ThreeSegmentController


def main():
    m, d, meta, original = load_checkpoint(HORIZONTAL, verify_hashes=False)
    g = meta['geometry_at_capture']
    corner, source, target, radius = pipe_frame(m)
    p = original.parameters
    all_body = LocalSection(g, p, 0, 20, source, corner)
    baseline = build_backbone(true_axis_geometry(g, source, corner), p)
    same_axis_error = max(float(abs(all_body.angles(a)-baseline.joint_angles(a)).max()) for a in (0., -.7, 1.2))
    assert same_axis_error < 1e-12
    # Independent numeric integration with global yaw/pitch parity, including
    # odd-index segment starts and torsion accumulation inside each arc.
    parity_error = 0.
    for lo, hi in ((0, 6), (9, 20), (10, 20)):
        section = LocalSection(g, p, lo, hi, target if lo == 0 else source, corner)
        b = section.bone
        nodes = np.r_[0., np.cumsum(b.links)]
        phase = -.31
        values = []
        for j, (left, right) in enumerate(zip(nodes[:-2], nodes[2:])):
            value = 0.
            for k, seg in enumerate(b.segments):
                a, z = max(left, b.bounds[k]), min(right, b.bounds[k+1])
                if z <= a:
                    continue
                trig = math.sin if (lo+j) % 2 == 0 else math.cos
                value += quad(lambda s: -seg.kappa*trig(phase+b.twists[k]+seg.tau*(s-b.bounds[k])), a, z)[0]
            values.append(value)
        parity_error = max(parity_error, float(abs(section.angles(phase)-values).max()))
    assert parity_error < 1e-11
    ctl = ThreeSegmentController(g, p, d.ctrl, meta['settings']['phase'], corner, source, target, radius)
    q = d.ctrl.copy()
    q[ctl.front:ctl.rear] -= .1
    before = ctl.middle_reference.copy()
    _, info = ctl.compute(g, q, ctl.phase, .2, [], migrate=False)
    middle = slice(ctl.front, ctl.rear)
    ratio = (ctl.middle_reference[middle]-q[middle])/(before[middle]-q[middle])
    assert np.allclose(ratio, math.exp(-.2/ctl.options.follow_seconds))
    assert np.allclose(info['weights'].sum(axis=1), 1.)
    # Physical approach may admit a new front joint without contact. The same
    # movement MUST NOT release the rear without sustained target contact.
    ctl = ThreeSegmentController(g, p, d.ctrl, meta['settings']['phase'], corner, source, target, radius)
    points = np.asarray(g['points_mm']).copy()
    points[ctl.front+1] = corner+target*140+source*85
    points[ctl.rear+1] = corner-source*100+target*85
    rear = ctl.rear
    for _ in range(3):
        ctl.update_boundaries(points, np.zeros((60, 3, 21)), .2)
    assert ctl.front == ctl.initial_front+1 and ctl.rear == rear
    contact = np.zeros((60, 3, 21))
    contact[:, 1, 0] = 1.
    contact[:, 0, rear+2] = 1.
    for _ in range(3):
        ctl.update_boundaries(points, contact, .2)
    assert ctl.rear == rear+1
    ctl.middle_max = ctl.rear-ctl.front
    points[ctl.rear+1] = corner-source*100+target*85
    for _ in range(5):
        ctl.update_boundaries(points, contact, .2)
    assert ctl.rear == rear+1
    print(json.dumps(dict(same_axis_max_rad=same_axis_error, parity_integral_max_rad=parity_error,
        continuous_middle_follow=True, precontact_admission=True, release_requires_contact=True,
        middle_upper_bound=True, initial_partition=[ctl.initial_front, ctl.initial_rear]), indent=2))


if __name__ == '__main__':
    main()
