"""Plot the two saved C0 trials. No physics, model sweep or hash checks."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1] / 'experiments/curve_guided_transfer_20261006'


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for version in ('V1', 'V2'):
        with np.load(BASE / (version + '_C0') / 'trace.npz') as r:
            t = r['sim_time']
            speed = np.linalg.norm(np.diff(r['centers_mm'], axis=0) / np.diff(t)[:, None, None], axis=2).max(1)
            axes[0, 0].plot(t[1:], speed, label=version)
            axes[0, 1].plot(t, np.linalg.norm((r['qpos'][:, :3] - r['qpos'][0, :3]) * 1000, axis=1), label=version)
            axes[1, 0].plot(t, np.degrees(abs(r['q'] - r['ctrl'])).max(1), label=version)
            axes[1, 1].plot(t, np.count_nonzero(r['contact_force_N'][:, 0] > .1, axis=1), label=version + ' source')
            axes[1, 1].plot(t, np.count_nonzero(r['contact_force_N'][:, 1] > .1, axis=1), '--', label=version + ' target')
    for ax, ylabel in zip(axes.ravel(), ['Maximum module speed (mm/s)', 'Free-root displacement from initial (mm)',
                                       'Maximum encoder tracking error (deg)', 'Contact modules above 0.1 N']):
        ax.set_xlabel('Physical time (s)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('C0: commanded feed = 0. Different manual initial states; not a B/F paired comparison.')
    fig.savefig(BASE / 'holding_diagnostics.png', dpi=160)
    plt.close(fig)
    print('Holding diagnostics saved; no physics advanced.')


if __name__ == '__main__':
    main()
