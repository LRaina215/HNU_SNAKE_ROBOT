"""P1a only: geometry, command continuity and material identities. No mj_step."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from snake_control.curve_guided_transfer import FiveSegmentCurve, feed_schedule

BASE = ROOT / 'experiments/curve_guided_transfer_20261006'
MODEL = ROOT / 'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
CORNER = np.array([1100., 0., 1000.])


def points(model, data):
    return 1000 * np.vstack([data.site('s_head').xpos,
           *[data.joint('joint' + str(i)).xanchor for i in range(1, 21)], data.site('s_tail').xpos])


def placement(model, data, curve, feed, fit=False):
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[qadr] = curve.joint_angles(feed)
    position, rotation = curve.nominal_root(feed)
    data.qpos[:3] = (position + CORNER) / 1000
    q = Rotation.from_matrix(rotation).as_quat()
    data.qpos[3:7] = q[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    if fit:
        # A one-time least-squares rigid placement is permitted artificial initial registration.
        # No per-joint IK correction, no traction, no online placement.
        desired = curve.sample(curve.lambda0 + feed - np.arange(22) * curve.p.link_mm) + CORNER
        actual = points(model, data)
        a, b = actual - actual.mean(0), desired - desired.mean(0)
        u, _, v = np.linalg.svd(a.T @ b)
        rotate = v.T @ np.diag([1, 1, np.linalg.det(v.T @ u.T)]) @ u.T
        data.qpos[:3] = (rotate @ (data.qpos[:3] * 1000 - actual.mean(0)) + desired.mean(0)) / 1000
        q = Rotation.from_matrix(rotate @ rotation).as_quat()
        data.qpos[3:7] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
    return qadr


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    curve = FiveSegmentCurve()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nu == 20
    data = mujoco.MjData(model)
    head = model.body('head').id
    phases = [0., 32.25, 64.5, 80.625, 129., 161.25]
    rows = []
    fig = plt.figure(figsize=(12, 7))
    for index, feed in enumerate(phases):
        placement(model, data, curve, feed, fit=True)
        p = points(model, data)
        desired = curve.sample(curve.lambda0 + feed - np.arange(22) * 64.5) + CORNER
        pipe_penetration = self_penetration = 0.
        pipe_contacts = self_contacts = 0
        for contact in data.contact:
            b1, b2 = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if b1 >= head and b2 >= head and abs(b1 - b2) > 1:
                self_contacts += 1
                self_penetration = max(self_penetration, -1000 * contact.dist)
            if (b1 >= head) != (b2 >= head) and any((model.geom(g).name or '').startswith('pipe') for g in (contact.geom1, contact.geom2)):
                pipe_contacts += 1
                pipe_penetration = max(pipe_penetration, -1000 * contact.dist)
        centers = (p[:-1] + p[1:]) / 2 - CORNER
        # Geometry occupancy follows the chosen reference radius; it is not a
        # contact or dynamic success criterion. Dynamic scoring uses actual pipe contacts.
        envelope = curve.p.radius_mm + 25
        source_ids = np.flatnonzero((centers[:, 0] < -50) & (np.linalg.norm(centers[:, 1:], axis=1) < envelope)).tolist()
        target_ids = np.flatnonzero((centers[:, 1] > 50) & (np.linalg.norm(centers[:, [0, 2]], axis=1) < envelope)).tolist()
        row = dict(feed_mm=feed, maximum_angle_deg=float(np.degrees(abs(curve.joint_angles(feed))).max()),
            fk_reference_rms_mm=float(np.sqrt(np.mean((p - desired) ** 2))),
            fk_reference_max_point_mm=float(np.linalg.norm(p - desired, axis=1).max()),
            pipe_contacts=pipe_contacts, pipe_penetration_mm=pipe_penetration,
            nonadjacent_self_contacts=self_contacts, self_penetration_mm=self_penetration,
            source_region_modules=source_ids, target_region_modules=target_ids,
            material_ell_mm=curve.material_positions(feed).tolist())
        rows.append(row)
        ax = fig.add_subplot(2, 3, index + 1, projection='3d')
        ax.plot(*desired.T, '--', color='grey', label='continuous curve')
        ax.plot(*p.T, 'o-', markersize=3, label='20-joint FK')
        for module in (7, 8):
            midpoint = (p[module] + p[module + 1]) / 2
            ax.text(*midpoint, 'M' + str(module), color='red')
        ax.plot([500, 1100], [0, 0], [1000, 1000], color='black', linewidth=5)
        ax.plot([1100, 1100], [0, 600], [1000, 1000], color='black', linewidth=5)
        ax.set_title(f'feed={feed:g} mm; artificial rigid placement')
        ax.set_xlim(650, 1300); ax.set_ylim(-130, 500); ax.set_zlim(870, 1170)
        ax.view_init(25, -65)
    fig.suptitle('P1a reference/FK only. The root is manually placed in every panel; no dynamics or support claim.')
    fig.tight_layout()
    fig.savefig(BASE / 'key_postures.png', dpi=150)
    plt.close(fig)
    times = np.arange(0, 45.25 + .001, .05)
    commands = np.array([curve.joint_angles(feed_schedule(t, 'B')[0]) for t in times])
    continuity = float(np.degrees(abs(np.diff(commands, axis=0))).max())
    joints = [curve.frame(x + eps) for x in curve.bounds for eps in (-1e-7, 1e-7)]
    frame_error = max(float(abs(a - b).max()) for a, b in zip(joints[::2], joints[1::2]))
    ready = all(r['maximum_angle_deg'] < 90 and r['pipe_penetration_mm'] < 1 and
                r['self_penetration_mm'] < 1 and len(r['source_region_modules']) >= 2 and
                len(r['target_region_modules']) >= 2 for r in rows)
    result = dict(design_status='READY_FOR_MINIMAL_SIM' if ready else 'NOT_READY', physics_steps=0,
        model=MODEL.relative_to(ROOT).as_posix(), mujoco=mujoco.__version__,
        parameters=vars(curve.p), alpha_deg=float(np.degrees(curve.alpha)),
        arc_length_mm=curve.arc, transition_length_mm=float(curve.bounds[-1]),
        spatial_join_points_mm=np.vstack([curve.A, curve.B, curve.C, curve.E]).tolist(),
        body_length_mm=curve.length, head_ell_initial_mm=curve.lambda0,
        phase='rho=0; opposite helix phase changes emerge from material feed',
        maximum_50ms_command_step_deg=continuity, join_frame_error=frame_error,
        material_tracking='ell(Mi)=lambda0+feed-(i+.5)*64.5; M7/M8 were not initially in target helix',
        planned_total_feed_mm=161.25, rows=rows,
        limit='Artificial root placement at each offline posture; not physical trajectory or contact support.')
    (BASE / 'geometry_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    placement(model, data, curve, 0, fit=True)
    np.savez_compressed(BASE / 'initial_pose.npz', qpos=data.qpos, qvel=np.zeros(model.nv),
                        target=curve.joint_angles(0))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
