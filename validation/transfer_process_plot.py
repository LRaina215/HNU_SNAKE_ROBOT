"""Plot recorded stage-1 quality metrics; no physics or repeated acceptance run."""
import argparse
import json
from pathlib import Path
import numpy as np
from three_segment_review import context


def plot(root, names, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for name in names:
        folder = root/name
        _, z, model, _, _ = context(folder)
        report = json.loads((folder/'process_metrics.json').read_text(encoding='utf8'))
        t = z['time']-z['time'][0]
        keep = (t >= 1)&(t <= report['metrics_end_seconds']+1e-6)
        centers = (z['points_mm'][:, :-1]+z['points_mm'][:, 1:])/2000
        speed = np.r_[np.zeros((1, 21)), np.linalg.norm(np.diff(centers, axis=0), axis=2)/np.diff(t)[:, None]]
        dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        joint = np.max(abs(np.degrees(z['qvel'][:, dofs])), axis=1)
        saturated = np.mean(abs(z['torque'][keep]) >= .99*model.actuator_forcerange[:, 1], axis=1)
        values = [speed[keep].max(axis=1), joint[keep], abs(z['error_deg'][keep]).max(axis=1),
                  100*np.cumsum(saturated)/np.arange(1, len(saturated)+1)]
        label = name.replace('radial_', '')
        for ax, value in zip(axes, values):
            ax.plot(t[keep], value, lw=.8, label=label)
    for ax, limit, label in zip(axes, [.5, 90, 15, 29.7],
            ['Max module speed (m/s)', 'Max joint speed (deg/s)', 'Max tracking error (deg)',
             'Cumulative saturated samples (%)']):
        ax.axhline(limit, color='black', ls='--', lw=1, label='Predeclared engineering filter')
        ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Physical time since identical checkpoint (s); initial 1 s excluded')
    fig.suptitle('Post-drape transfer: measured motion, not just commanded speed\n'
                 'Each curve ends at its own accepted rolling cycle, or failure. Physics unchanged.')
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('names', nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plot(a.root, a.names, a.output)
