"""论文实验 2（Fig. 14）的等比缩小场景：单圆管 → 连续过渡 → 两根分离的圆管。单位 m。

运行：python tools/build_twin_pipe_scene.py（同时生成两个比例）
原尺寸：下段外径 200 mm；上段两管外径 100/150 mm，圆心各距原管轴 77.5 mm。
- ×0.9：下段与变径场景（外径 180 mm）相同，沿用其初始缠绕姿态（R=130）与 Ctrl+U 预设（R=110）。
- ×1.0：论文原尺寸；下段半径大 10 mm，初始姿态与预设各加 10 mm（R=140 / 120）。
两者均沿用变径场景的 kp 与摩擦。
过渡段截面取圆与双圆凸包的 Minkowski 线性插值，各切片均为凸形；
上段是两根独立圆柱，中间保留缝隙（非凸截面），不用凸包代替。
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_variable_pipe_scene as base

VARIANTS = {  # 名称: (缩放, 初始缠绕 R, Ctrl+U 预设 R)，R 单位 mm
    'Snake_Robot_o20_Twin_Pipe.xml': (.9, 130., 110.),
    'Snake_Robot_o20_Twin_Pipe_Full.xml': (1., 140., 120.),
}
Z_START, Z_END, HEIGHT = .70, 1.00, 1.80
AXIAL_SEGMENTS, SIDES = 30, 96


def dimensions(scale):
    """返回 (下段半径, ((圆心 x, 半径), (圆心 x, 半径)))，单位 m。"""
    return .100 * scale, ((-.0775 * scale, .050 * scale), (.0775 * scale, .075 * scale))


def section(u, scale, sides=SIDES):
    """过渡段 u∈[0,1] 的截面边界点：圆与双圆凸包的支撑点线性组合（保持凸性）。"""
    R_MAIN, TWIN = dimensions(scale)
    theta = np.arange(sides) * 2 * np.pi / sides
    normal = np.c_[np.cos(theta), np.sin(theta)]
    centers = np.array([[x, 0.] for x, _ in TWIN])
    radii = np.array([r for _, r in TWIN])
    # 凸包在方向 n 的支撑点：取 c·n + r 最大的那个圆上的 c + r·n。
    best = np.argmax(normal @ centers.T + radii, axis=1)
    twin = centers[best] + radii[best, None] * normal
    return (1 - u) * R_MAIN * normal + u * twin


def build(filename, scale, initial_radius_mm, apply_radius_mm):
    R_MAIN, TWIN = dimensions(scale)
    output = ROOT / 'mujoco_model' / filename
    tree = ET.parse(ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml')
    root = tree.getroot()
    root.set('model', 'adaptive-twin-pipe')
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
    ET.SubElement(custom, 'numeric', name='experiment_helix_mm',
                  data=base.numbers([apply_radius_mm, base.HELIX_PITCH_MM]))
    # 仿真场景自带管外/管内标记，优先于 COMPLIANCE_INSIDE；实机没有该标记，由开关手动选择。
    ET.SubElement(custom, 'numeric', name='experiment_inside', data='0')
    ET.SubElement(custom, 'numeric', name='twin_pipe_profile',
                  data=base.numbers([R_MAIN, *np.ravel(TWIN), Z_START, Z_END, HEIGHT]))
    pipe = ET.SubElement(world, 'body', name='twin_pipe')
    common = dict(friction=f'2 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}', condim='6',
                  group='1', rgba='.28 .52 .72 1')
    ET.SubElement(pipe, 'geom', name='pipe_lower', type='cylinder', size=base.numbers([R_MAIN, Z_START / 2]),
                  pos=base.numbers([0, 0, Z_START / 2]), **common)
    faces = []
    for i in range(SIDES):
        j = (i + 1) % SIDES
        faces.extend([(i, j, SIDES + j), (i, SIDES + j, SIDES + i)])
    for i in range(1, SIDES - 1):
        faces.extend([(0, i + 1, i), (SIDES, SIDES + i, SIDES + i + 1)])
    zs = np.linspace(Z_START, Z_END, AXIAL_SEGMENTS + 1)
    for i, (lo, hi) in enumerate(zip(zs[:-1], zs[1:])):
        vertices = np.vstack([np.c_[section((z - Z_START) / (Z_END - Z_START), scale), np.full(SIDES, z - (lo + hi) / 2)]
                              for z in (lo, hi)])
        name = f'pipe_transition_{i:02d}'
        ET.SubElement(asset, 'mesh', name=name, vertex=base.numbers(vertices),
                      face=' '.join(map(str, np.array(faces).ravel())))
        ET.SubElement(pipe, 'geom', name=name, type='mesh', mesh=name, pos=base.numbers([0, 0, (lo + hi) / 2]),
                      **common)
    for k, (x, r) in enumerate(TWIN):
        ET.SubElement(pipe, 'geom', name=f'pipe_twin_{k}', type='cylinder',
                      size=base.numbers([r, (HEIGHT - Z_END) / 2]),
                      pos=base.numbers([x, 0, (HEIGHT + Z_END) / 2]), **common)
    for z, name in ((Z_START, 'transition_start'), (Z_END, 'transition_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.008',
                      pos=base.numbers([0, -(R_MAIN + .018), z]), rgba='1 .65 .1 1')
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    keyframes = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframes, 'key', name='experiment_start', qpos=base.numbers(base.initial_pose(model, initial_radius_mm)))
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    print(output)


if __name__ == '__main__':
    for filename, variant in VARIANTS.items():
        build(filename, *variant)
