"""论文实验 3（Fig. 15）：管内攀爬，内径 340 mm 圆管连续过渡到 350×500 mm 矩形管道。单位 m。

运行：python tools/build_inner_pipe_scene.py（同时生成两个比例）
MuJoCo 网格按凸包碰撞，空心管不能用一个网格表示。沿用 Pipe_Crawling 的做法，
用一圈光滑圆棒围成管壁：ROD_COUNT 根棒按内壁周长等分排列，棒心在内壁外法向偏移棒半径处；
圆管段、矩形段各是一根竖直胶囊，过渡段每层一段首尾相接的胶囊。整根管没有棱边。
（最初用平板/薄壁块拼接，块间棱边会卡住连杆，管内顺应性很快因轴估计失效而停止。）
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
    'Snake_Robot_o20_Inner_Pipe.xml': (.8, 102., 125.),
    'Snake_Robot_o20_Inner_Pipe_Full.xml': (1., 134., 160.),
}
Z_START, Z_END, HEIGHT = .95, 1.35, 2.20  # 初始姿态顶端约 0.73 m，下方留足等径段
# 棒数与棒径取 Pipe_Crawling 的做法：18 根，r ≈ 0.43 × 圆管内半径（原文 30 mm / 内切 70 mm）。
# 直管对照（内半径 136 mm，管内顺应性，100 步）：18 根粗棒（胶囊或圆柱）均跑满并上爬；
# 24 根粗棒、48 根 r=30 mm 细棒、平板管壁都在第 5 步左右因轴估计失效停止。
ROD_COUNT, ROD_RATIO = 18, .43
LAYERS = 20
START_CENTER_Z = .50


def dimensions(scale):
    """返回 (圆管内半径, 矩形内半宽, 矩形内半高)，单位 m。"""
    return .170 * scale, .175 * scale, .250 * scale


def rods(u, scale, count=None, dense=2048):
    """过渡段 u∈[0,1] 的棒心位置：内壁按周长等分，沿外法向偏移棒半径。

    内壁在圆与矩形之间按极角线性插值半径；等分周长使各处棒距一致，
    同一根棒在各层取相同的周长比例，因此沿高度连续移动。
    """
    count = ROD_COUNT if count is None else count
    radius, half_x, half_y = dimensions(scale)
    theta = np.arange(dense) * 2 * np.pi / dense
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    rect = np.minimum(half_x / np.maximum(c, 1e-9), half_y / np.maximum(s, 1e-9))
    r = (1 - u) * radius + u * rect
    wall = np.c_[r * np.cos(theta), r * np.sin(theta)]
    edge = np.roll(wall, -1, axis=0) - wall
    arc = np.r_[0., np.cumsum(np.linalg.norm(edge, axis=1))]
    # 法向取前后各约一个棒半径范围的平均切向，矩形角处得到角平分方向。
    rod_radius = ROD_RATIO * radius
    window = max(1, int(rod_radius / (arc[-1] / dense)))
    tangent = np.roll(wall, -window, axis=0) - np.roll(wall, window, axis=0)
    normal = np.c_[tangent[:, 1], -tangent[:, 0]]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    index = np.searchsorted(arc, np.arange(count) * arc[-1] / count) % dense
    return wall[index] + rod_radius * normal[index]


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
    zs = np.linspace(Z_START, Z_END, LAYERS + 1)
    centers = [rods(smooth((z - Z_START) / (Z_END - Z_START)), scale) for z in zs]
    spacing = max(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1).max() for c in centers)
    rod_radius = ROD_RATIO * radius
    if spacing >= 2 * rod_radius:
        raise ValueError(f"相邻棒心距 {spacing*1000:.1f} mm 不小于棒径，管壁会有缝隙；请增加 ROD_COUNT")
    for k in range(ROD_COUNT):
        # 圆管段与矩形段各一根竖直胶囊；过渡段逐层连接相邻两圈棒心。
        path = [np.r_[centers[0][k], 0.]] + [np.r_[c[k], z] for c, z in zip(centers, zs)] +                [np.r_[centers[-1][k], HEIGHT]]
        for n, (a, b) in enumerate(zip(path[:-1], path[1:])):
            ET.SubElement(pipe, 'geom', name=f'pipe_rod_{k:02d}_{n:02d}', type='capsule',
                          size=f'{rod_radius:g}', fromto=base.numbers([*a, *b]), **common)
    for z, name in ((Z_START, 'transition_start'), (Z_END, 'transition_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.008',
                      pos=base.numbers([0, -(radius + 2 * rod_radius + .012), z]), rgba='1 .65 .1 1')
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
