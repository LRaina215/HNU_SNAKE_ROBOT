"""Static checks for lab pipe scenes. No controller, serial port or climbing run.

Run with MuJoCo 3.3.0: python validation/lab_pipe_scenes.py [--render]
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'codes')]
from build_pipe_network_scenes import LAB_SCENES
from snake_control.adaptive_debug import centerline_nearest


def numeric(model, name):
    return model.numeric(name).data.copy()


def check(name, joints):
    path = ROOT / 'mujoco_model' / f'Snake_Robot_o{joints}_Pipe_{name}.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert model.nu == joints and np.isfinite(data.qpos).all(), name
    pipe = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pipe_')]
    robot = [i for i in range(model.ngeom) if model.geom_bodyid[i] and i not in pipe]
    gap = min(mujoco.mj_geomDistance(model, data, a, b, .1, None) for a in robot for b in pipe)
    assert gap >= .001 - 1e-8, (name, gap)
    floor = model.geom('floor').id
    floor_gap = min(mujoco.mj_geomDistance(model, data, a, floor, 10., None) for a in robot)
    assert floor_gap > 0, (name, floor_gap)
    points = numeric(model, 'pipe_centerline').reshape(-1, 4)
    tangent = np.diff(points[:, :3], axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    assert np.all(lengths > 1e-8), name
    tangent /= lengths[:, None]
    angles = np.degrees(np.arccos(np.clip(np.sum(tangent[1:] * tangent[:-1], axis=1), -1, 1)))
    spec = LAB_SCENES[name]
    if name.startswith(('elbow', 'tee')):
        assert np.count_nonzero(angles > 89.9) == 1, (name, angles)
    if name.startswith('u_'):
        assert np.dot(tangent[0], tangent[-1]) < -.999, name
    # Vertical/horizontal configurations must have their stated world directions.
    expected = {'upright': ([0, 0, 1], [1, 0, 0]), 'horizontal': ([1, 0, 0], [0, 1, 0]),
                'up': ([1, 0, 0], [0, 0, 1]), 'down': ([1, 0, 0], [0, 0, -1])}
    incoming, outgoing = expected[spec['layout']]
    assert np.allclose(tangent[0], incoming), name
    if name.startswith(('elbow', 'tee')):
        assert np.allclose(tangent[-1], outgoing), name
    if name.startswith('tee'):
        branch = numeric(model, 'pipe_unused_branch').reshape(-1, 4)
        goal = numeric(model, 'feature_span')[1]
        # Traveling on the unused arm must not count as passing the selected turn.
        unused = branch[0, :3] + np.linspace(.1, 1., 10)[:, None] * (branch[1, :3] - branch[0, :3])
        assert np.max(centerline_nearest(model, unused)[2]) < goal, name
        assert centerline_nearest(model, points[-1:, :3])[2][0] > goal, name
    if name.startswith('elbow'):
        # Both convex pieces must share all 64 vertices of the same miter face.
        root = ET.parse(path).getroot()
        rings = []
        for prefix in ('pipe_sharp_in', 'pipe_sharp_out'):
            geom = next(g for g in root.findall('.//geom') if g.get('name', '').startswith(prefix))
            mesh = root.find(f"asset/mesh[@name='{geom.get('mesh')}']")
            vertices = np.fromstring(mesh.get('vertex'), sep=' ').reshape(-1, 3)
            rings.append(vertices + np.fromstring(geom.get('pos'), sep=' '))
        shared = (np.linalg.norm(rings[0][:, None] - rings[1][None, :], axis=2).min(axis=1) < 1e-9).sum()
        assert shared == 64, (name, shared)
    return model, data, dict(scene=name, joints=joints, pipe_clearance_mm=gap*1000,
                            floor_clearance_mm=floor_gap*1000, max_axis_turn_deg=float(angles.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'validation/results/lab-pipe-scenes'
    output.mkdir(parents=True, exist_ok=True)
    report, panels = [], []
    for joints in (20, 26):
        for name in LAB_SCENES:
            model, data, row = check(name, joints)
            report.append(row)
            if args.render and joints == 20:
                from PIL import Image, ImageDraw
                camera = mujoco.MjvCamera()
                camera.lookat[:] = model.stat.center
                camera.distance = max(3., model.stat.extent * 1.7)
                camera.azimuth, camera.elevation = 135, -28
                with mujoco.Renderer(model, height=480, width=640) as renderer:
                    renderer.update_scene(data, camera=camera)
                    picture = Image.fromarray(renderer.render())
                ImageDraw.Draw(picture).text((12, 12), name, fill='white', stroke_width=1, stroke_fill='black')
                picture.save(output / f'{name}.png')
                panels.append(picture.resize((480, 360)))
    (output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if panels:
        sheet = Image.new('RGB', (480*4, 360*4), '#222222')
        for i, panel in enumerate(panels):
            sheet.paste(panel, ((i % 4)*480, (i // 4)*360))
        sheet.save(output / 'overview.png')
    print(f'PASS: {len(report)} models; joint counts, initial clearances, axis directions, miter seams and T routes.')


if __name__ == '__main__':
    main()
