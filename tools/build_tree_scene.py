"""生成竖直、不规则截面、局部鼓包后收窄的树干近似场景（单位 m）。

运行：python tools/build_tree_scene.py
这是受论文实验 4 启发的可控几何，并非实测树木或论文尺寸复刻。
沿轴分段使用凸网格，避免 MuJoCo 把整根树干的收窄处用一个凸包填平。
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import build_variable_pipe_scene as base

OUTPUT = base.ROOT / 'mujoco_model/Snake_Robot_o20_Tree.xml'
HEIGHT, Z_START, Z_END = 2.0, .75, 1.75
SIDES, AXIAL_SEGMENTS = 64, 50
BASE_RADIUS = .090


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10 - 15*u + 6*u*u)


def section(z):
    """固定主轴的凸截面：圆 → 带小幅三瓣起伏的椭圆 → 鼓包 → 收窄。

    用椭圆支撑函数加三次谐波构造边界，起伏只有 1.5 mm，保持截面凸性。
    鼓包改变两条半轴的幅度不同；不是单纯把变径圆管涂成棕色。
    """
    shape = smooth((z - Z_START) / .25)
    narrowing = smooth((z - 1.35) / .40)
    bump = np.sin(np.pi * np.clip((z - .95) / .60, 0., 1.))**4
    a = BASE_RADIUS + .012*shape - .016*narrowing + .016*bump
    b = BASE_RADIUS - .008*shape - .006*narrowing + .008*bump
    theta = np.arange(SIDES) * 2*np.pi / SIDES
    c, s = np.cos(theta), np.sin(theta)
    h = np.sqrt((a*c)**2 + (b*s)**2)
    ripple = .0015 * shape
    dh = -3*ripple*np.sin(3*theta)
    # p = h*n + h'*t；椭圆部分直接用其解析支撑点。
    x = a*a*c/h + ripple*np.cos(3*theta)*c - dh*s
    y = b*b*s/h + ripple*np.cos(3*theta)*s + dh*c
    return np.c_[x, y]


def build():
    tree = ET.parse(base.ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml')
    root = tree.getroot()
    root.set('model', 'adaptive-tree-trunk')
    world, asset = root.find('worldbody'), root.find('asset')
    for node in root.iter():
        node.text = node.tail = None
    for servo in root.iter('position'):
        servo.set('kp', f'{base.ACTUATOR_KP:g}')
    for geom in root.iter('geom'):
        if geom.get('friction') == '1 0.15 0.015':
            geom.set('friction', f'1 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}')
    world.remove(world.find("body[@name='pipe_v']"))
    asset.remove(asset.find("hfield[@name='terrain']"))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960', azimuth='135', elevation='-15')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    ET.SubElement(root, 'statistic', center='0 0 1', extent='2')
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='experiment_helix_mm',
                  data=base.numbers([base.APPLY_RADIUS_MM, base.HELIX_PITCH_MM]))
    ET.SubElement(custom, 'numeric', name='experiment_inside', data='0')
    # 不写 pipe_profile：树干没有单一截面半径，不能套用圆管的调试读数公式。
    trunk = ET.SubElement(world, 'body', name='tree_trunk')
    common = dict(friction=f'2 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}',
                  condim='6', group='1', rgba='.40 .24 .12 1')
    faces = []
    for i in range(SIDES):
        j = (i + 1) % SIDES
        faces.extend([(i, j, SIDES+j), (i, SIDES+j, SIDES+i)])
    for i in range(1, SIDES-1):
        faces.extend([(0, i+1, i), (SIDES, SIDES+i, SIDES+i+1)])
    # 下段和上段也是同一多边形边界，接缝完全共用端环；不引入碰撞台阶。
    zs = np.r_[0., np.linspace(Z_START, Z_END, AXIAL_SEGMENTS+1), HEIGHT]
    rings = [section(z) for z in zs]
    for i, (lo, hi) in enumerate(zip(zs[:-1], zs[1:])):
        middle = (lo + hi) / 2
        vertices = np.vstack([np.c_[rings[k], np.full(SIDES, z-middle)]
                              for k, z in ((i, lo), (i+1, hi))])
        # pipe 前缀沿用现有 Ctrl+K 接触统计；树干同样是管外接触对象。
        name = f'pipe_tree_{i:02d}'
        ET.SubElement(asset, 'mesh', name=name, vertex=base.numbers(vertices),
                      face=' '.join(map(str, np.array(faces).ravel())))
        ET.SubElement(trunk, 'geom', name=name, type='mesh', mesh=name,
                      pos=base.numbers([0, 0, middle]), **common)
    # 三个点仅标示形状变化起点、鼓包中心和收窄终点，不参与碰撞。
    for z, name in ((Z_START, 'tree_shape_start'), (1.25, 'tree_bulge'), (Z_END, 'tree_narrow_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.006',
                      pos=base.numbers([section(z)[:, 0].max()+.018, 0, z]), rgba='1 .65 .1 1')
    ET.indent(tree, space='  ')
    tree.write(OUTPUT, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    keyframe = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframe, 'key', name='experiment_start', qpos=base.numbers(base.initial_pose(model)))
    ET.indent(tree, space='  ')
    tree.write(OUTPUT, encoding='utf-8', xml_declaration=True)
    print(OUTPUT)


if __name__ == '__main__':
    build()
