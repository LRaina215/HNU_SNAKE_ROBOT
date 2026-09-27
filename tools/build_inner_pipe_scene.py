"""论文实验 3（Fig. 15）：管内攀爬，内径 340 mm 圆管连续过渡到 350×500 mm 矩形管道。单位 m。

运行：python tools/build_inner_pipe_scene.py（同时生成两个比例）
MuJoCo 网格按凸包碰撞，空心管用薄壁块拼成：圆管段和矩形段是长方体，
过渡段每层每条边是一个由上下两圈边界点构成的凸薄壁块。
论文机器人 28 节×89.5 mm，本模型 20 节×68.5 mm；原尺寸矩形段中心线一圈约 1.5 m，
身体绕不满一圈，因此另生成 ×0.8 版本（矩形段约 1.2 圈）。
管内姿态比管壁小一圈间隙，Ctrl+U 预设比管壁中心线大，依靠向外撑开夹紧。
沿用变径场景的 kp 与摩擦；模型内 experiment_inside=1 使运行程序切换为管内顺应性。
"""
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_variable_pipe_scene as base

PITCH_ANGLE = .30  # 论文实验 3 的设计螺距角 α
VARIANTS = {  # 名称: (缩放, 初始缠绕 R, Ctrl+U 预设 R)，R 为机器人中心线半径 mm
    'Snake_Robot_o20_Inner_Pipe.xml': (.8, 96., 125.),
    'Snake_Robot_o20_Inner_Pipe_Full.xml': (1., 128., 160.),
}
Z_START, Z_END, HEIGHT = .75, 1.15, 1.80
WALL = .010
SIDES, LAYERS = 48, 20
START_CENTER_Z = .50


def dimensions(scale):
    """返回 (圆管内半径, 矩形内半宽, 矩形内半高)，单位 m。"""
    return .170 * scale, .175 * scale, .250 * scale


def boundary(u, scale, sides=SIDES):
    """过渡段 u∈[0,1] 的内壁边界：按极角在圆与矩形之间线性插值半径。"""
    radius, half_x, half_y = dimensions(scale)
    theta = (np.arange(sides) + .5) * 2 * np.pi / sides
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    rect = np.minimum(half_x / np.maximum(c, 1e-9), half_y / np.maximum(s, 1e-9))
    r = (1 - u) * radius + u * rect
    return np.c_[r * np.cos(theta), r * np.sin(theta)]


def smooth(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def build(filename, scale, initial_radius_mm, apply_radius_mm):
    radius, half_x, half_y = dimensions(scale)
    output = ROOT / 'mujoco_model' / filename
    pitch_mm = 2 * math.pi * apply_radius_mm * math.tan(PITCH_ANGLE)
    tree = ET.parse(ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml')
    root = tree.getroot()
    root.set('model', 'adaptive-inner-pipe')
    world, asset = root.find('worldbody'), root.find('asset')
    for servo in root.iter('position'):
        servo.set('kp', f'{base.ACTUATOR_KP:g}')
    for geom in root.iter('geom'):
        if geom.get('friction') == '1 0.15 0.015':
            geom.set('friction', f'1 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}')
    for node in root.iter():
        node.text = node.tail = None
    world.remove(world.find("body[@name='pipe_v']"))
    asset.remove(asset.find("hfield[@name='terrain']"))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960', azimuth='135', elevation='-15')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    ET.SubElement(root, 'statistic', center='0 0 .8', extent='1.8')
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='experiment_helix_mm', data=base.numbers([apply_radius_mm, pitch_mm]))
    ET.SubElement(custom, 'numeric', name='experiment_inside', data='1')
    ET.SubElement(custom, 'numeric', name='inner_pipe_profile',
                  data=base.numbers([radius, half_x, half_y, Z_START, Z_END, HEIGHT]))
    pipe = ET.SubElement(world, 'body', name='inner_pipe')
    # 半透明，便于看到管内机器人；group 1 可在 viewer 中单独隐藏。
    common = dict(friction=f'2 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}', condim='6',
                  group='1', rgba='.28 .52 .72 .3')
    # 圆管段：SIDES 块长方体，内表面相切于内半径，宽度略放大以消除缝隙。
    width = 2 * (radius + WALL) * math.tan(math.pi / SIDES) * 1.02
    for k in range(SIDES):
        angle = (k + .5) * 2 * math.pi / SIDES
        center = (radius + WALL / 2) * np.array([math.cos(angle), math.sin(angle)])
        ET.SubElement(pipe, 'geom', name=f'pipe_round_{k:02d}', type='box',
                      size=base.numbers([WALL / 2, width / 2, Z_START / 2]),
                      pos=base.numbers([*center, Z_START / 2]), euler=f'0 0 {angle:.9g}', **common)
    # 过渡段：每层每条边一个凸薄壁块（上下两圈内壁点及其外移点）。
    zs = np.linspace(Z_START, Z_END, LAYERS + 1)
    for layer, (lo, hi) in enumerate(zip(zs[:-1], zs[1:])):
        rings = [boundary(smooth((z - Z_START) / (Z_END - Z_START)), scale) for z in (lo, hi)]
        mid = (lo + hi) / 2
        for k in range(SIDES):
            j = (k + 1) % SIDES
            points = []
            for ring, z in zip(rings, (lo, hi)):
                for p in (ring[k], ring[j]):
                    outward = p / np.linalg.norm(p)
                    points += [[*p, z - mid], [*(p + WALL * outward), z - mid]]
            name = f'pipe_transition_{layer:02d}_{k:02d}'
            ET.SubElement(asset, 'mesh', name=name, vertex=base.numbers(points))
            ET.SubElement(pipe, 'geom', name=name, type='mesh', mesh=name, pos=base.numbers([0, 0, mid]), **common)
    # 矩形段：四块板，角部互相搭接。
    top = HEIGHT - Z_END
    for name, size, pos in (
            ('pipe_rect_px', [WALL / 2, half_y + WALL, top / 2], [half_x + WALL / 2, 0]),
            ('pipe_rect_nx', [WALL / 2, half_y + WALL, top / 2], [-half_x - WALL / 2, 0]),
            ('pipe_rect_py', [half_x + WALL, WALL / 2, top / 2], [0, half_y + WALL / 2]),
            ('pipe_rect_ny', [half_x + WALL, WALL / 2, top / 2], [0, -half_y - WALL / 2])):
        ET.SubElement(pipe, 'geom', name=name, type='box', size=base.numbers(size),
                      pos=base.numbers([*pos, (HEIGHT + Z_END) / 2]), **common)
    for z, name in ((Z_START, 'transition_start'), (Z_END, 'transition_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.008',
                      pos=base.numbers([0, -(radius + WALL + .012), z]), rgba='1 .65 .1 1')
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    qpos = base.initial_pose(model, initial_radius_mm, pitch_mm)
    qpos[2] += START_CENTER_Z - base.START_CENTER_Z
    keyframes = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframes, 'key', name='experiment_start', qpos=base.numbers(qpos))
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    print(output)


if __name__ == '__main__':
    for filename, variant in VARIANTS.items():
        build(filename, *variant)
