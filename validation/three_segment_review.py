"""Assess physical transfer separately from labels; render colored replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from segmented_pole_experiment import BASE, pipe_frame, write_json
from mujoco_checkpoint import load_checkpoint


def context(folder):
    result = json.loads((folder/'result.json').read_text(encoding='utf8'))
    z = np.load(folder/'trace.npz')
    checkpoint=Path(result['checkpoint'])
    if not checkpoint.exists():checkpoint=BASE/'inputs'/checkpoint.name
    m, d, meta, _ = load_checkpoint(checkpoint, verify_hashes=False)
    return result, z, m, d, meta


def assess(folder):
    result, z, m, d, meta = context(folder)
    corner, source, target, radius = pipe_frame(m)
    commands = json.loads((folder/'commands.json').read_text(encoding='utf8'))
    assert np.array_equal(z['qpos'][0], d.qpos) and np.array_equal(z['ctrl'][0], d.ctrl)
    assert all(command['applied'] for command in commands)
    applied_ids, delivered_targets = z['applied_id'], z['ctrl']
    for command in commands:
        # NPZ indexing decompresses the array; cache once for this read-only
        # audit instead of decompressing it for every delivered command.
        frames = applied_ids == command['id']
        assert np.any(frames) and np.allclose(delivered_targets[frames], command['target'], atol=1e-12)
    points = z['points_mm']
    centers = (points[:, :-1]+points[:, 1:])/2
    relative = centers-corner
    ds, dt = np.linalg.norm(np.cross(relative, source), axis=2), np.linalg.norm(np.cross(relative, target), axis=2)
    region = (dt+10 < ds) & (relative@target > radius) & (dt < radius+29+15)
    assert np.array_equal(region, z['region'])
    # The archived horizontal paired baseline uses the same fixed cohort as
    # the new controller, even though baseline itself has no partition labels.
    prefix = result.get('configuration', {}).get('initial_front', 5)
    eligible = (np.arange(21) >= prefix) & ~region[0]
    runs = np.zeros(21, int)
    persistent = np.zeros_like(region)
    for i, mask in enumerate(region):
        runs = np.where(mask & eligible, runs+1, 0)
        persistent[i] = runs >= 60
    count = persistent.sum(axis=1)
    consecutive_pair = np.any(persistent[:, :-1] & persistent[:, 1:], axis=1)
    normal = z['normal_force']
    front = np.any(normal[:, 1, :prefix+1] > .1, axis=1)
    rear = np.any(normal[:, 0, prefix+1:] > .1, axis=1)
    front_progress = (centers[:, :prefix].mean(axis=1)-centers[0, :prefix].mean(axis=0))@target
    rolling_gate = consecutive_pair & (front_progress >= 64.5) & front & rear & ~z['ground']
    sustained_gate = bool(len(rolling_gate) >= 60 and np.all(rolling_gate[-60:]))
    full = bool(len(region) >= 60 and np.all(region[-60:]) and
                np.all(np.any(normal[-60:, 1] > .1, axis=1)) and not np.any(normal[-60:, 0] > .1) and
                not np.any(z['ground']))
    report = dict(initial_pose_exact=True, EXE_delivery_checked=True,
        physical_seconds=float(z['time'][-1]-z['time'][0]),
        front_progress_mm=float(front_progress[-1]), maximum_front_progress_mm=float(front_progress.max()),
        maximum_sustained_new_modules=int(count.max()), final_sustained_new_modules=int(count[-1]),
        final_new_module_ids=np.flatnonzero(persistent[-1]).tolist(),
        front_target_contact_fraction=float(front.mean()), rear_source_contact_fraction=float(rear.mean()),
        floor_contact=bool(np.any(z['ground'])), maximum_tracking_error_deg=float(abs(z['error_deg']).max()),
        torque_saturated_fraction=float(np.mean(abs(z['torque']) >= 2.355*.99)),
        first_front_contact_loss_seconds=next((float(z['time'][i]-z['time'][0]) for i in range(20,len(front))
            if not np.any(front[i-19:i+1])), None),
        boundary_events=result.get('events', []), final_partition=z['partition'][-1].tolist(),
        primary_gate_passed=sustained_gate, full_crossing=full, outcome=result['outcome'])
    write_json(folder/'assessment.json', report)
    return report


def render(folder):
    import cv2
    from PIL import Image, ImageDraw
    result, z, m, d, meta = context(folder)
    # NPZ access decompresses a whole array. Load replay fields once instead
    # of repeatedly decompressing the full trajectory for each video frame.
    with z:
        z = {key: z[key] for key in ('time', 'qpos', 'qvel', 'ctrl', 'partition')}
    corner, source, target, _ = pipe_frame(m)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = corner/1000-source*.12+target*.12
    camera.distance, camera.azimuth, camera.elevation = 1.6, 125, -55
    colors = np.array([[.15,.7,.3,1.], [1.,.63,.1,1.], [.2,.5,1.,1.]])
    body_modules = {}
    for i in range(m.nbody):
        name = m.body(i).name
        module = 0 if name == 'head' else 20 if name == 'tail' else int(name[4:]) if name.startswith('link') else None
        if module is not None:
            body_modules[i] = module
    renderer = mujoco.Renderer(m, height=640, width=960)
    writer = cv2.VideoWriter(str(folder/'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (960,640))
    if not writer.isOpened():
        raise RuntimeError('video encoder unavailable')
    try:
        for i in range(0, len(z['time']), 5):
            d.qpos[:] = z['qpos'][i]
            d.qvel[:] = z['qvel'][i]
            d.ctrl[:] = z['ctrl'][i]
            f, r = z['partition'][i]
            if f < 0 and result['method'] == 'three':
                f = result['configuration']['initial_front']
                r = result['configuration']['initial_rear']
            if f >= 0:
                for geom in range(m.ngeom):
                    module = body_modules.get(int(m.geom_bodyid[geom]))
                    if module is not None:
                        m.geom_rgba[geom] = colors[0 if module <= f else 1 if module <= r else 2]
                        m.geom_matid[geom] = -1
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,960,46), fill='black')
            label = f'front/middle/rear joints: {f}/{r-f}/{20-r}' if f >= 0 else 'Original single-axis compliance'
            draw.text((8,5), f'{folder.name} +{z["time"][i]-z["time"][0]:.2f}s | {label}', fill='white')
            if f >= 0:
                draw.text((8,25), 'Green: front | Orange: yielding middle | Blue: rear | colors = membership, NOT contact', fill='white')
            writer.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
            if i == 0:
                im.save(folder/'initial.png')
            if i >= len(z['time'])-5:
                im.save(folder/'final.png')
            for at in (1., 3., 4.):
                if abs(z['time'][i]-z['time'][0]-at) < .001:
                    im.save(folder/f'at_{at:g}s.png')
    finally:
        writer.release()
        renderer.close()


def plot_comparison(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for mode in ('three', 'original'):
        result, z, m, _, _ = context(root/f'horizontal_{mode}')
        corner, source, target, _ = pipe_frame(m)
        t = z['time']-z['time'][0]
        points = z['points_mm']
        centers = (points[:, :-1]+points[:, 1:])/2
        # Fixed cohorts in both modes, so boundary-label changes cannot improve
        # these curves. Rear J11-J20 spans points[10:] throughout this trial.
        progress = (centers[:, :5].mean(axis=1)-centers[0, :5].mean(axis=0))@target
        relative = points[:, 10:]-corner
        radial = relative-np.einsum('tij,j->ti', relative, source)[:, :, None]*source
        unit = radial/np.linalg.norm(radial, axis=2)[:, :, None]
        winding = np.arctan2(np.cross(unit[:, :-1], unit[:, 1:])@source,
                            np.sum(unit[:, :-1]*unit[:, 1:], axis=2)).sum(axis=1)/(2*np.pi)
        axes[0].plot(t, progress, label=mode)
        axes[1].plot(t, abs(winding), label=mode)
        axes[2].plot(t, z['normal_force'][:, 0].sum(axis=1), label=mode+' source')
        axes[2].plot(t, z['normal_force'][:, 1].sum(axis=1), ls='--', label=mode+' target')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc='best')
    axes[0].set_ylabel('Fixed front 5 modules\nadvance (mm)')
    axes[1].set_ylabel('Rear signed net winding\nabsolute value (turns)')
    axes[2].set_ylabel('Total pipe normal force (N)')
    axes[2].set_xlabel('Time since same checkpoint (s)')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--comparison', action='store_true')
    args = parser.parse_args()
    if args.comparison:
        plot_comparison(args.folder)
    else:
        print(json.dumps(assess(args.folder), ensure_ascii=False, indent=2))
        if args.render:
            render(args.folder)
