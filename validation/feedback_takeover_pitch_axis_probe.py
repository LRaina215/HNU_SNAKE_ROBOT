"""Test a known commanded pitch prior on local encoder tangents.

Inferred previous acquisition chooses the cohort. Simulator orientation is
used only to score completed estimates; this is not a closed-loop result.
"""
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from segmented_pole_experiment import ROOT, pipe_frame, write_json
from three_segment_review import context


def cone_axis(points, prior, stride, fixed_pitch):
    links = points[stride:] - points[:-stride]
    tangents = links / np.linalg.norm(links, axis=1)[:, None]
    _, vectors = np.linalg.eigh(np.cov(tangents.T, bias=True))
    seeds = [np.asarray(prior), *vectors.T]
    rows = []
    for seed in seeds:
        if seed @ prior < 0:
            seed = -seed
        h = np.eye(3)[np.argmin(abs(seed))]
        h -= (h @ seed) * seed
        h /= np.linalg.norm(h)
        v = np.cross(seed, h)
        def decode(x):
            axis = seed + x[0] * h + x[1] * v
            return axis / np.linalg.norm(axis)
        def residual(x):
            projection = tangents @ decode(x)
            return projection + np.sin(.4057779073) if fixed_pitch else projection - projection.mean()
        fit = least_squares(residual, [0., 0.], bounds=([-5, -5], [5, 5]), max_nfev=60)
        axis = decode(fit.x)
        if axis @ prior < 0:
            axis = -axis
        rows.append((float(fit.cost), axis))
    cost, axis = min(rows, key=lambda x: x[0])
    return axis, cost


def run():
    root = ROOT / 'experiments/segmented_pole_20261005/feedback_takeover'
    result = []
    for name in ('horizontal_encoder_live_whole_coil', 'vertical_encoder_live_whole_coil', 'horizontal_encoder_distinct'):
        folder = root / name
        _, z, m, _, _ = context(folder)
        controls = json.loads((folder / 'controls.json').read_text(encoding='utf8'))
        qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        own = mujoco.MjData(m)
        own.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        times = z['time'] - z['time'][0]
        _, _, true_axis, _ = pipe_frame(m)
        rows = []
        for ci in range(5, len(controls), 10):
            previous = controls[ci - 1]
            at = controls[ci]['time'] - .2
            if at > 100:
                break
            i = int(np.argmin(abs(times - at)))
            q = z['qpos'][i]
            own.qpos[qadr] = q[qadr]
            mujoco.mj_kinematics(m, own)
            points = 1000 * np.vstack([own.site('s_head').xpos, *[own.joint('joint' + str(j)).xanchor for j in range(1, 21)], own.site('s_tail').xpos])
            prior = previous['geometry_observer']['estimate']['target']
            estimates = []
            for window, count in [('fixed', 8), ('acquired', max(8, min(22, previous['actual_prefix'] + 2)))]:
                for stride in (1, 2, 3):
                    for fixed in (False, True):
                        axis, cost = cone_axis(points[:count], prior, stride, fixed)
                        estimates.append(dict(window=window, count=count, stride=stride, fixed_pitch=fixed, axis=axis, cost=cost))
            # Ground truth enters only the scorer below.
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, q[3:7])
            rot = rot.reshape(3, 3)
            error = lambda a: float(np.degrees(np.arccos(np.clip((rot @ a) @ true_axis, -1, 1))))
            for est in estimates:
                est['target_error_deg'] = error(est['axis'])
            rows.append(dict(time=float(times[i]), prior_error_deg=error(prior), estimates=estimates))
        summary = []
        for window in ('fixed', 'acquired'):
            for stride in (1, 2, 3):
                for fixed in (False, True):
                    errors = [e['target_error_deg'] for r in rows for e in r['estimates'] if e['window'] == window and e['stride'] == stride and e['fixed_pitch'] == fixed]
                    summary.append(dict(window=window, stride=stride, fixed_pitch=fixed, median_deg=float(np.median(errors)), maximum_deg=float(max(errors))))
        result.append(dict(run=name, rows=rows, summary=summary))
        print(name, summary, flush=True)
    write_json(root / 'pitch_axis_probe.json', dict(runs=result, limitation='Same-record stateless diagnostic; known commanded pitch is a prior, not a measured pipe property.'))


if __name__ == '__main__':
    run()
