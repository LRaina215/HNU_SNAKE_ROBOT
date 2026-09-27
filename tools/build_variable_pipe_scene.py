"""生成单根连续变径外管及可重复的缠管初始姿态。单位 m。

运行：python tools/build_variable_pipe_scene.py
圆台均为凸体；相邻端环完全相同，没有圆柱台阶或轴向缝隙。
修改常量后重建即可；只改本场景的位置伺服 kp 与扭转/滚动摩擦，不改原 Pole 场景。
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from Adaptive_Pole_Climbing import HelixBackbone

OUTPUT = ROOT / 'mujoco_model/Snake_Robot_o20_Variable_Pipe.xml'
R_BOTTOM, R_TOP = .090, .115
Z_START, Z_END, HEIGHT = .70, 1.30, 1.80
AXIAL_SEGMENTS, SIDES = 40, 96
HELIX_RADIUS_MM, HELIX_PITCH_MM = 130., 180.  # 预缠绕初始姿态
APPLY_RADIUS_MM = 110.  # Ctrl+U 默认应用的 R；比初始姿态紧，才有夹紧力
# 原模型 kp=4 过软：顺应性目标只比实际紧 K，力矩约 kp*K*关节角，抱不住管。
# kp=20 时论文增益 K=0.04/0.08 可爬过变径区；kp=40 两端会收紧到关节极限。
ACTUATOR_KP = 20.
# 扭转/滚动摩擦（单位 m，MuJoCo 取接触双方较大值）。原值 0.15/0.1 远大于橡胶-管面的
# 接触斑尺度，身体自滚阻力矩与舵机力矩同量级，变径段约 1.08 m 处打滑停滞；
# 改为接近 MuJoCo 默认量级后可爬过变径区到达上段。滑动摩擦保持机器人 1、管 2。
TORSIONAL_FRICTION, ROLLING_FRICTION = .005, .002
START_CENTER_Z = .50


def pipe_radius(z):
    """两端一、二阶导数为零的五次平滑过渡；返回设计半径。"""
    u = np.clip((np.asarray(z) - Z_START) / (Z_END - Z_START), 0., 1.)
    return R_BOTTOM + (R_TOP - R_BOTTOM) * (10*u**3 - 15*u**4 + 6*u**5)


def numbers(values):
    return ' '.join(f'{float(x):.12g}' for x in np.asarray(values).ravel())


def initial_pose(model, radius_mm=HELIX_RADIUS_MM):
    """只设实验初始条件；不会在执行循环中直接覆写关节状态。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    def points():
        return np.array([data.site('s_head').xpos.copy()] +
                        [data.joint(f'joint{i}').xanchor.copy() for i in range(1, model.nu+1)] +
                        [data.site('s_tail').xpos.copy()])
    links = np.linalg.norm(np.diff(points(), axis=0), axis=1) * 1000
    data.qpos[7:] = HelixBackbone.from_radius_pitch(radius_mm, HELIX_PITCH_MM).joint_angles(model.nu, links, 0.)
    mujoco.mj_forward(model, data)
    points = points()[1:-1]
    center = points.mean(axis=0)
    tangents = np.diff(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    _, _, vt = np.linalg.svd(tangents - tangents.mean(axis=0))
    axis = vt[-1]
    if axis @ (points[0] - points[-1]) < 0:
        axis = -axis
    rotation = Rotation.align_vectors([[0, 0, 1]], [axis])[0]
    aligned = rotation.apply(points - center)
    circle = least_squares(lambda x: np.linalg.norm(aligned[:, :2] - x[:2], axis=1) - x[2],
                           [0., 0., radius_mm / 1000]).x
    data.qpos[:3] = rotation.apply(data.qpos[:3] - center) - np.r_[circle[:2], 0.] + [0., 0., START_CENTER_Z]
    quaternion = (rotation * Rotation.from_quat(data.qpos[[4, 5, 6, 3]])).as_quat()
    data.qpos[3:7] = quaternion[[3, 0, 1, 2]]
    return data.qpos.copy()


def build():
    tree = ET.parse(ROOT / 'mujoco_model/Snake_Robot_o20_Pole_Climbing.xml')
    root = tree.getroot()
    root.set('model', 'adaptive-variable-pipe')
    world, asset = root.find('worldbody'), root.find('asset')
    for servo in root.iter('position'):
        servo.set('kp', f'{ACTUATOR_KP:g}')
    for geom in root.iter('geom'):
        if geom.get('friction') == '1 0.15 0.015':  # 机器人各连杆
            geom.set('friction', f'1 {TORSIONAL_FRICTION:g} {ROLLING_FRICTION:g}')
    # 源文件的注释外说明文字不是 MJCF 内容；生成时只保留元素。
    for node in root.iter():
        node.text = node.tail = None
    world.remove(world.find("body[@name='pipe_v']"))
    asset.remove(asset.find("hfield[@name='terrain']"))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960', azimuth='135', elevation='-15')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    ET.SubElement(root, 'statistic', center='0 0 .8', extent='1.8')
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='experiment_helix_mm', data=numbers([APPLY_RADIUS_MM, HELIX_PITCH_MM]))
    ET.SubElement(custom, 'numeric', name='pipe_profile', data=numbers([R_BOTTOM, R_TOP, Z_START, Z_END, HEIGHT]))
    pipe = ET.SubElement(world, 'body', name='variable_pipe')
    common = dict(friction=f'2 {TORSIONAL_FRICTION:g} {ROLLING_FRICTION:g}', condim='6', group='1', rgba='.28 .52 .72 1')
    ET.SubElement(pipe, 'geom', name='pipe_lower', type='cylinder', size=numbers([R_BOTTOM, Z_START/2]),
                  pos=numbers([0, 0, Z_START/2]), **common)
    zs = np.linspace(Z_START, Z_END, AXIAL_SEGMENTS+1)
    theta = np.arange(SIDES) * 2*np.pi/SIDES
    rings = np.c_[np.cos(theta), np.sin(theta)]
    # 一圈的面均指向外；圆台底/顶封口存在于固体内部，不形成外部台阶。
    faces = []
    for i in range(SIDES):
        j = (i+1) % SIDES
        faces.extend([(i,j,SIDES+j),(i,SIDES+j,SIDES+i)])
    for i in range(1,SIDES-1):
        faces.extend([(0,i+1,i),(SIDES,SIDES+i,SIDES+i+1)])
    for i, (lo, hi) in enumerate(zip(zs[:-1], zs[1:])):
        vertices = np.vstack([np.c_[rings*pipe_radius(z), np.full(SIDES, z-(lo+hi)/2)] for z in (lo,hi)])
        name = f'pipe_taper_{i:02d}'
        ET.SubElement(asset, 'mesh', name=name, vertex=numbers(vertices), face=' '.join(map(str,np.array(faces).ravel())))
        ET.SubElement(pipe, 'geom', name=name, type='mesh', mesh=name, pos=numbers([0,0,(lo+hi)/2]), **common)
    ET.SubElement(pipe, 'geom', name='pipe_upper', type='cylinder', size=numbers([R_TOP, (HEIGHT-Z_END)/2]),
                  pos=numbers([0,0,(HEIGHT+Z_END)/2]), **common)
    # 刻度只作 site 标记，没有碰撞，不会成为机器人借力的凸缘。
    for z, r, name in ((Z_START,R_BOTTOM,'taper_start'),(Z_END,R_TOP,'taper_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.008', pos=numbers([r+.018,0,z]), rgba='1 .65 .1 1')
    ET.indent(tree, space='  ')
    tree.write(OUTPUT, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    keyframes = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframes, 'key', name='experiment_start', qpos=numbers(initial_pose(model)))
    # ctrl 默认零。U/apply 的控制目标仍必须先经 EXE 返回，才推进物理仿真。
    ET.indent(tree, space='  ')
    tree.write(OUTPUT, encoding='utf-8', xml_declaration=True)
    print(OUTPUT)


if __name__ == '__main__':
    build()
