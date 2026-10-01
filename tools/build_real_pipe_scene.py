"""按实物生成 20/26 关节蛇形机器人与 UPVC 变径管场景。单位 m（常量注明 mm 的除外）。

运行：python tools/build_real_pipe_scene.py --joints 20（默认 26，另存 XML，不覆盖另一种关节数）
一次生成两个方向：小管在下爬向大管（real_s2l，实验室布置），大管在下爬向小管（real_l2s，备用）。
机器人参数来自实机（2026-09-30）：26 关节、头长与连杆 64.5 mm、外壳半径 29 mm、单节 135.85 g；
舵机 SM-80BL-C001 悬臂实测等效刚度约 127 N·m/rad，力矩上限按手册堵转扭矩 × EXE 转矩限制。
关节轴、正负方向与原 20 关节模型相同，桥接与 EXE 不需要改动。
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_variable_pipe_scene as base

JOINTS = 26
LINK_M = .0645  # 相邻关节间距；头部尖端到关节 1、关节 26 到尾端也是 64.5 mm
BODY_RADIUS_M = .029  # 外壳 CAD（零件 1.1/1.2）外圆柱面半径
LINK_MASS = HEAD_MASS = .136  # 实测单节 135.85 g（舵机 + 外壳）；头部模块同样带舵机
STALL_TORQUE_NM = 7.85  # 手册：80 kg·cm @12 V
TORQUE_LIMIT_PERMILLE = 300  # EXE 使能时写入地址 48 的默认转矩限制
FORCE_LIMIT_NM = STALL_TORQUE_NM * TORQUE_LIMIT_PERMILLE / 1000
ACTUATOR_KP = 127.  # 悬臂实测：3.90 N·m 使关节偏 1.76°
# 原模型 kp=20、阻尼 0.6，阻尼比约 0.67；kp 提高后按同一阻尼比取阻尼，实机阻尼未测。
JOINT_DAMPING, ARMATURE = 1.5, .01
# 原网格：连杆半径 27.9 mm、关节间距 68.5 mm；按实物缩放，网格文件坐标即连杆 body 坐标。
MESH_SCALE = (BODY_RADIUS_M / .0279, BODY_RADIUS_M / .0279, LINK_M / .0685)
FRICTION = f'1 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}'  # 未实测，沿用管外场景

R_SMALL, R_LARGE = .055, .080  # GB/T 5836.2 外径 110 / 160 mm
TAPER_SLANT = .030  # 过渡段斜边
TAPER_LENGTH = float(np.sqrt(TAPER_SLANT**2 - (R_LARGE - R_SMALL)**2))  # 5√11 ≈ 16.6 mm
Z_TAPER, HEIGHT, SIDES = .95, 1.90, 96
PITCH_MM = 220.
APPLY_TIGHTEN_MM = 4.  # Ctrl+U 预设比贴管中心线紧的量
# 64.5 mm 连杆绕 84 mm 半径时，Yamada 离散关节点径向交替偏离约 ±17 mm；
# 初始半径取不穿透管面的最小值（离管面至少 1 mm），避免初始姿态离管太远而下坠。
START_CLEARANCE_M = .001
# 顺应性增益（首尾各 4 关节, 中部）。kp 从 20 升到 127 后，同样的 K 使 73% 的时间顶在力矩上限，
# 并在过渡段因轴估计失效停止；按刚度比缩小到 0.01/0.02 后负载降到 41%，可爬过过渡段。
COMPLIANCE_GAIN = (.01, .02)
VARIANTS = {'Snake_Robot_o26_Real_Pipe_S2L.xml': (R_SMALL, R_LARGE),
            'Snake_Robot_o26_Real_Pipe_L2S.xml': (R_LARGE, R_SMALL)}


def numbers(values):
    return base.numbers(values)


def robot(world, joints=JOINTS):
    """头部 → joints 个关节 → 尾端；各 body 绕 z 翻转 180°，关节轴交替 x / −y。"""
    head = ET.SubElement(world, 'body', name='head', pos='0 0 .1', euler='0 90 -90')
    ET.SubElement(head, 'freejoint', name='root')
    ET.SubElement(head, 'site', name='s_head', pos=numbers([0, 0, LINK_M]))
    # 头部没有专用网格：用外壳半径的圆柱覆盖关节 1 到尖端。
    ET.SubElement(head, 'geom', name='head_shell', type='cylinder', size=numbers([BODY_RADIUS_M, LINK_M/2]),
                  pos=numbers([0, 0, LINK_M/2]), mass=f'{HEAD_MASS:g}', condim='6', friction=FRICTION,
                  rgba='.15 .15 .15 1')
    parent = head
    for j in range(1, joints+1):
        name = 'tail' if j == joints else f'link{j}'
        body = ET.SubElement(parent, 'body', name=name, pos=numbers([0, 0, -LINK_M]), euler='0 0 -180')
        ET.SubElement(body, 'joint', name=f'joint{j}', type='hinge', pos=numbers([0, 0, LINK_M]),
                      axis='1 0 0' if j % 2 else '0 -1 0', limited='true', range='-90 90',
                      damping=f'{JOINT_DAMPING:g}', armature=f'{ARMATURE:g}')
        ET.SubElement(body, 'geom', type='mesh', mesh='h_link' if j % 2 else 'v_link', mass=f'{LINK_MASS:g}',
                      condim='6', friction=FRICTION, rgba='.2 .2 .2 1')
        if j == joints:
            ET.SubElement(body, 'site', name='s_tail', pos='0 0 0')
        parent = body


def new_scene(model_name='real-snake26-pipe'):
    """场景公共部分：求解器、地面、灯光；机器人由 finalize 按 joints 加入。"""
    root = ET.Element('mujoco', model=model_name)
    ET.SubElement(root, 'option', timestep='0.001', iterations='100', tolerance='1e-8', solver='Newton', cone='elliptic')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='texplane', type='2d', builtin='checker', rgb1='.8 .8 .8', rgb2='.1 .15 .2',
                  width='512', height='512', mark='cross', markrgb='.8 .8 .8')
    ET.SubElement(asset, 'material', name='matplane', reflectance='0.3', texture='texplane', texrepeat='2 2', texuniform='true')
    for mesh, stl in (('h_link', 'c_link_H.stl'), ('v_link', 'c_link_V.stl')):
        ET.SubElement(asset, 'mesh', name=mesh, file=f'./asset/{stl}', scale=numbers(MESH_SCALE))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .25', material='matplane', condim='6',
                  friction='0.25 0.1 0.01')
    ET.SubElement(world, 'light', directional='false', diffuse='1 1 1', specular='0 0 0', pos='0 0 15', dir='0 0 -1',
                  castshadow='false')
    return root, asset, world


PIPE_GEOM = dict(friction=f'2 {base.TORSIONAL_FRICTION:g} {base.ROLLING_FRICTION:g}', condim='6', group='1',
                 rgba='.92 .92 .9 1')


def finalize(root, world, filename, r_start, extra_numeric=(), rotation=None, center='0 0 .8', extent='1.8', joints=JOINTS):
    """加入机器人、执行器、显示与实验标记，按起始管半径求初始姿态并写出 XML。

    管道起始段必须沿 rotation·z 方向（默认竖直）且过原点；初始螺旋中心在该轴上 0.5 m 处。
    """
    robot(world, joints)
    actuator = ET.SubElement(root, 'actuator')
    for j in range(1, joints+1):
        ET.SubElement(actuator, 'position', name=f'servo_{j}', joint=f'joint{j}', kp=f'{ACTUATOR_KP:g}', gear='1',
                      ctrllimited='true', ctrlrange='-1.5708 1.5708', forcelimited='true',
                      forcerange=numbers([-FORCE_LIMIT_NM, FORCE_LIMIT_NM]))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960', azimuth='135', elevation='-15')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    ET.SubElement(root, 'statistic', center=center, extent=extent)
    contact_mm = (r_start + BODY_RADIUS_M) * 1000  # 贴管时机器人中心线半径
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='experiment_helix_mm', data=numbers([contact_mm - APPLY_TIGHTEN_MM, PITCH_MM]))
    ET.SubElement(custom, 'numeric', name='experiment_inside', data='0')
    ET.SubElement(custom, 'numeric', name='experiment_compliance_gain', data=numbers(COMPLIANCE_GAIN))
    for name, data in extra_numeric:
        ET.SubElement(custom, 'numeric', name=name, data=numbers(data))
    output = ROOT / 'mujoco_model' / filename
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    keyframes = ET.SubElement(root, 'keyframe')
    radius_mm, pitch_mm, qpos = start_pose(model, contact_mm, rotation)
    ET.SubElement(keyframes, 'key', name='experiment_start', qpos=numbers(qpos))
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    print(output, f'初始 R={radius_mm:g} P={pitch_mm:.0f} mm，Ctrl+U 预设 R={contact_mm - APPLY_TIGHTEN_MM:g} P={PITCH_MM:g} mm')
    return output


def build(filename, r_bottom, r_top, joints=JOINTS):
    root, asset, world = new_scene(f'real-snake{joints}-pipe')
    pipe = ET.SubElement(world, 'body', name='real_pipe')
    common = PIPE_GEOM
    z_end = Z_TAPER + TAPER_LENGTH
    ET.SubElement(pipe, 'geom', name='pipe_lower', type='cylinder', size=numbers([r_bottom, Z_TAPER/2]),
                  pos=numbers([0, 0, Z_TAPER/2]), **common)
    theta = np.arange(SIDES) * 2*np.pi/SIDES
    ring = np.c_[np.cos(theta), np.sin(theta)]
    vertices = np.vstack([np.c_[ring*r, np.full(SIDES, z)] for r, z in ((r_bottom, -TAPER_LENGTH/2), (r_top, TAPER_LENGTH/2))])
    ET.SubElement(asset, 'mesh', name='pipe_taper', vertex=numbers(vertices))  # 圆台为凸体，直接取凸包
    ET.SubElement(pipe, 'geom', name='pipe_taper', type='mesh', mesh='pipe_taper',
                  pos=numbers([0, 0, Z_TAPER + TAPER_LENGTH/2]), **common)
    ET.SubElement(pipe, 'geom', name='pipe_upper', type='cylinder', size=numbers([r_top, (HEIGHT-z_end)/2]),
                  pos=numbers([0, 0, (HEIGHT+z_end)/2]), **common)
    for z, r, name in ((Z_TAPER, r_bottom, 'taper_start'), (z_end, r_top, 'taper_end')):
        ET.SubElement(world, 'site', name=name, type='sphere', size='.008', pos=numbers([r+.018, 0, z]), rgba='1 .65 .1 1')
    finalize(root, world, filename, r_bottom,
             extra_numeric=[('pipe_profile', [r_bottom, r_top, Z_TAPER, z_end, HEIGHT])], joints=joints)


def matched_pitch_mm(radius_mm, apply_radius_mm, apply_pitch_mm=PITCH_MM):
    """与 Ctrl+U 预设螺旋挠率相同、半径为 radius_mm 的螺距；无解时返回 None。

    Yamada 关节角的相位沿身体按 τ·s 累积。初始姿态若只放大半径、螺距不变，挠率不同，
    26 关节尾部相位差可达 127°，Ctrl+U 时尾部会剧烈反拧。挠率相同则只差曲率（幅值）。
    """
    b1 = apply_pitch_mm / (2 * np.pi)
    tau = b1 / (apply_radius_mm ** 2 + b1 ** 2)
    disc = 1 - 4 * tau ** 2 * radius_mm ** 2
    if disc < 0:
        return None
    return 2 * np.pi * (1 - np.sqrt(disc)) / (2 * tau)  # 取较小的螺距（更紧的缠绕）


def start_pose(model, contact_mm, rotation=None):
    """从贴管半径起逐毫米增大，返回第一个整条机器人离管面不小于 START_CLEARANCE_M 的预缠绕姿态。

    挠率与 Ctrl+U 预设（贴管半径 − APPLY_TIGHTEN_MM，PITCH_MM）相同，见 matched_pitch_mm。
    rotation（scipy Rotation）把绕竖直轴的姿态整体转到倾斜起始管上。返回 (半径, 螺距, qpos)。
    """
    data = mujoco.MjData(model)
    pipe = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pipe')]
    body = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0 and i not in pipe]
    for radius_mm in np.arange(np.ceil(contact_mm), contact_mm + 60):
        pitch_mm = matched_pitch_mm(radius_mm, contact_mm - APPLY_TIGHTEN_MM)
        if pitch_mm is None:
            break
        data.qpos[:] = base.initial_pose(model, radius_mm=radius_mm, pitch_mm=pitch_mm)
        if rotation is not None:
            data.qpos[:3] = rotation.apply(data.qpos[:3])
            quaternion = (rotation * Rotation.from_quat(data.qpos[[4, 5, 6, 3]])).as_quat()
            data.qpos[3:7] = quaternion[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        gap = min(mujoco.mj_geomDistance(model, data, a, b, .05, None) for a in body for b in pipe)
        if gap >= START_CLEARANCE_M:
            return float(radius_mm), float(pitch_mm), data.qpos.copy()
    raise RuntimeError('找不到不穿透管面的初始姿态')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joints', type=int, choices=(20, 26), default=26)
    args = parser.parse_args()
    for name, (bottom, top) in VARIANTS.items():
        build(name.replace('o26_', f'o{args.joints}_'), bottom, top, args.joints)
