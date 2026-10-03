"""弯管、变径 + 弯管、法兰、偏心异径、三通、倾斜管、低摩擦带等管外攀爬场景（26 关节实物参数）。单位 m。

运行：python tools/build_pipe_network_scenes.py [场景名 ...] --joints 20（默认 26；无场景名时生成全部）
管道由中心线上依次排列的段描述：直管（可变径）、圆弧弯头、偏心异径、法兰、侧向支管。
等径直管用圆柱；其余按中心线细分，每片取相邻两个截面圆的凸包（MuJoCo 网格碰撞本身就是凸包），
相邻片共用截面圆，表面连续，没有台阶或缝隙。
机器人、执行器、初始姿态与 build_real_pipe_scene.py 相同；所有管道从竖直（或倾斜）起始段开始，
机器人预缠在起始段 0.5 m 处。中心线采样写入 XML 的 pipe_centerline，供 Ctrl+K 显示到管轴距离与沿管进度。
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_real_pipe_scene as real

numbers = real.numbers
R_SMALL, R_LARGE = real.R_SMALL, real.R_LARGE  # 裸 UPVC 外径 110 / 160 mm + 径向橡胶厚度
TAPER = real.TAPER_LENGTH  # 实验室异径接头：斜边 30 mm，轴向 16.6 mm
SIDES = 64
BEND_STEP_DEG = 5.
SAMPLE_M = .05  # 中心线导出间隔（MuJoCo numeric 长度有限；Ctrl+K 按折线段求最近点）
FLANGE_T = .022  # PVC 法兰厚度
BOLT_R, BOLT_H, BOLT_COUNT = .012, .010, 8  # M16 螺栓头（六角对边约 24 mm）
BELOW_FLOOR = .2  # 起始段伸入地面以下，避免管底悬空


# 起始段从地下 0.2 m 开始，长 1.15–1.2 m：所有特征都在 z≥0.95 m，高于初始螺旋（z 约 0.24–0.77 m）。
# 段描述：('straight', 长度, r0, r1[, 选项])、('bend', 弯曲半径, 角度°, 弯向世界方向)、
# ('eccentric', 长度, r0, r1, 平齐侧世界方向)、('flange', 凸起高度, 是否带螺栓)、('branch', 支管半径, 伸出长度, 世界方向)
SCENES = {
    # 弯管：小管竖直 1.0 m 后弯曲
    'bend90': dict(title='90° 弯头（弯曲半径 300 mm）',
                   segments=[('straight', 1.2, R_SMALL, R_SMALL), ('bend', .30, 90, 'x'), ('straight', 1.0, R_SMALL, R_SMALL)]),
    'bend90_tight': dict(title='90° 短弯头（弯曲半径 165 mm = 1.5D）',
                         segments=[('straight', 1.2, R_SMALL, R_SMALL), ('bend', .165, 90, 'x'), ('straight', 1.0, R_SMALL, R_SMALL)]),
    'bend45': dict(title='45° 弯头（弯曲半径 300 mm）',
                   segments=[('straight', 1.2, R_SMALL, R_SMALL), ('bend', .30, 45, 'x'), ('straight', 1.2, R_SMALL, R_SMALL)]),
    'offset': dict(title='S 形错位管（两个反向 45° 弯头，出口仍竖直）',
                   segments=[('straight', 1.2, R_SMALL, R_SMALL), ('bend', .30, 45, 'x'), ('straight', .25, R_SMALL, R_SMALL),
                             ('bend', .30, 45, '-x'), ('straight', 1.0, R_SMALL, R_SMALL)]),
    # 组合：先变径再弯 / 先弯再变径（实验室异径接头，小管在下）
    'taper_bend': dict(title='先变径（110→160）再 45° 弯',
                       segments=[('straight', 1.15, R_SMALL, R_SMALL), ('straight', TAPER, R_SMALL, R_LARGE),
                                 ('straight', .35, R_LARGE, R_LARGE), ('bend', .30, 45, 'x'), ('straight', 1.0, R_LARGE, R_LARGE)]),
    'bend_taper': dict(title='先 45° 弯再变径（110→160）',
                       segments=[('straight', 1.2, R_SMALL, R_SMALL), ('bend', .30, 45, 'x'), ('straight', .30, R_SMALL, R_SMALL),
                                 ('straight', TAPER, R_SMALL, R_LARGE), ('straight', 1.0, R_LARGE, R_LARGE)]),
    # 非连续：法兰凸起高度（法兰外半径 − 管外半径）。DN100 PVC 法兰外径约 220 mm，凸起约 55 mm。
    **{f'flange{h}': dict(title=f'法兰（凸起 {h} mm，厚 22 mm）',
                         segments=[('straight', 1.2, R_SMALL, R_SMALL), ('flange', h / 1000, False),
                                   ('straight', 1.2, R_SMALL, R_SMALL)])
       for h in (10, 25, 40, 55)},
    'flange_pair': dict(title='一对带螺栓法兰（凸起 55 mm，间距 300 mm，8×M16）',
                        segments=[('straight', 1.2, R_SMALL, R_SMALL), ('flange', .055, True), ('straight', .30, R_SMALL, R_SMALL),
                                  ('flange', .055, True), ('straight', 1.0, R_SMALL, R_SMALL)]),
    # 研究扩展
    'eccentric': dict(title='偏心异径（110→160，−x 侧母线平齐）',
                      segments=[('straight', 1.15, R_SMALL, R_SMALL), ('eccentric', .06, R_SMALL, R_LARGE, '-x'),
                                ('straight', 1.0, R_LARGE, R_LARGE)]),
    'tee': dict(title='三通：侧向支管（外径 75 mm，伸出 250 mm）',
                segments=[('straight', 1.2, R_SMALL, R_SMALL), ('branch', .0375 + real.RUBBER_THICKNESS_M, .25, 'x'), ('straight', 1.2, R_SMALL, R_SMALL)]),
    'incline30': dict(title='倾斜直管（偏离竖直 30°）', tilt_deg=30,
                      segments=[('straight', 2.4, R_SMALL, R_SMALL)]),
    'slip_band': dict(title='低摩擦带（z 0.95–1.25 m 滑动摩擦 0.3，模拟潮湿 / 油污）',
                      segments=[('straight', 1.15, R_SMALL, R_SMALL), ('straight', .30, R_SMALL, R_SMALL, dict(friction=.3)),
                                ('straight', 1.0, R_SMALL, R_SMALL)]),
}
WORLD = {'x': (1., 0, 0), '-x': (-1., 0, 0), 'y': (0, 1., 0), '-y': (0, -1., 0)}


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def ring(center, direction, radius):
    e1 = unit(np.cross(direction, [0, 1., 0] if abs(direction[1]) < .9 else [1., 0, 0]))
    e2 = np.cross(direction, e1)
    theta = np.arange(SIDES) * 2 * np.pi / SIDES
    return center + radius * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))


class PipeBuilder:
    def __init__(self, world, asset, tilt_deg=0.):
        self.world, self.asset = world, asset
        self.direction = Rotation.from_euler('y', tilt_deg, degrees=True).apply([0, 0, 1.])
        self.point = -BELOW_FLOOR * self.direction
        self.body = ET.SubElement(world, 'body', name='pipe_network')
        self.count = 0
        self.samples = []  # (x, y, z, r)，沿中心线
        self.markers = []
        self.s = -BELOW_FLOOR  # 沿中心线弧长，从地面起算
        self.features = []  # 各特征的 (起点, 终点) 弧长

    def name(self, kind):
        self.count += 1
        return f'pipe_{kind}_{self.count:02d}'

    def hull(self, kind, rings, **extra):
        """若干截面圆的凸包，网格坐标相对其中心。"""
        vertices = np.vstack(rings)
        center = vertices.mean(axis=0)
        name = self.name(kind)
        ET.SubElement(self.asset, 'mesh', name=name, vertex=numbers(vertices - center))
        ET.SubElement(self.body, 'geom', name=name, type='mesh', mesh=name, pos=numbers(center), **{**real.PIPE_GEOM, **extra})

    def sample(self, start, end, r0, r1):
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / SAMPLE_M)))
        for u in np.linspace(0, 1, n, endpoint=False):
            self.samples.append([*(start + u * (end - start)), r0 + u * (r1 - r0)])

    def straight(self, length, r0, r1, options=None):
        options = options or {}
        extra = {}
        if 'friction' in options:
            # MuJoCo 默认取接触双方摩擦的最大值；提高优先级才能让这一段的低摩擦生效。
            extra = dict(friction=f"{options['friction']:g} {real.base.TORSIONAL_FRICTION:g} {real.base.ROLLING_FRICTION:g}",
                         priority='1', rgba='.55 .75 .95 1')
        end = self.point + length * self.direction
        if abs(r0 - r1) < 1e-12:
            ET.SubElement(self.body, 'geom', name=self.name('straight'), type='cylinder', size=f'{r0:g}',
                          fromto=numbers([*self.point, *end]), **{**real.PIPE_GEOM, **extra})
        else:
            self.hull('taper', [ring(self.point, self.direction, r0), ring(end, self.direction, r1)], **extra)
            self.markers += [(self.point, r0), (end, r1)]
        self.sample(self.point, end, r0, r1)
        if abs(r0 - r1) > 1e-12 or options:
            self.features.append((self.s, self.s + length))
        self.s += length
        self.point, self.radius = end, r1

    def bend(self, bend_radius, angle_deg, toward):
        toward = unit(np.asarray(WORLD[toward]) - np.dot(WORLD[toward], self.direction) * self.direction)
        axis = np.cross(self.direction, toward)
        center = self.point + bend_radius * toward
        steps = int(np.ceil(angle_deg / BEND_STEP_DEG))
        self.markers.append((self.point, self.radius))
        self.features.append((self.s, self.s + bend_radius * np.radians(angle_deg)))
        self.s = self.features[-1][1]
        for i in range(steps):
            turn = Rotation.from_rotvec(axis * np.radians(angle_deg) / steps)
            point = center + turn.apply(self.point - center)
            direction = turn.apply(self.direction)
            self.hull('bend', [ring(self.point, self.direction, self.radius), ring(point, direction, self.radius)])
            self.sample(self.point, point, self.radius, self.radius)
            self.point, self.direction = point, direction
        self.markers.append((self.point, self.radius))

    def eccentric(self, length, r0, r1, flush):
        """偏心异径：截面始终垂直于管轴，圆心向平齐侧的反方向平移 r1−r0，使平齐侧母线为直线。"""
        side = unit(np.asarray(WORLD[flush]) - np.dot(WORLD[flush], self.direction) * self.direction)
        end = self.point + length * self.direction - (r1 - r0) * side
        self.hull('eccentric', [ring(self.point, self.direction, r0), ring(end, self.direction, r1)])
        self.markers += [(self.point, r0), (end, r1)]
        self.sample(self.point, end, r0, r1)
        self.features.append((self.s, self.s + np.linalg.norm(end - self.point)))
        self.s = self.features[-1][1]
        self.point, self.radius = end, r1

    def flange(self, height, bolts):
        half = FLANGE_T / 2 * self.direction
        outer = self.radius + height
        ET.SubElement(self.body, 'geom', name=self.name('flange'), type='cylinder', size=f'{outer:g}',
                      fromto=numbers([*(self.point - half), *(self.point + half)]), **{**real.PIPE_GEOM, 'rgba': '.85 .55 .25 1'})
        if bolts:
            e1 = unit(np.cross(self.direction, [0, 1., 0] if abs(self.direction[1]) < .9 else [1., 0, 0]))
            e2 = np.cross(self.direction, e1)
            circle = self.radius + .6 * height  # 螺栓孔中心圆
            for k in range(BOLT_COUNT):
                a = 2 * np.pi * (k + .5) / BOLT_COUNT
                p = self.point + circle * (np.cos(a) * e1 + np.sin(a) * e2)
                for sign in (-1, 1):  # 两面各一个螺栓头 / 螺母
                    face = p + sign * (FLANGE_T / 2) * self.direction
                    ET.SubElement(self.body, 'geom', name=self.name('bolt'), type='cylinder', size=f'{BOLT_R:g}',
                                  fromto=numbers([*face, *(face + sign * BOLT_H * self.direction)]),
                                  **{**real.PIPE_GEOM, 'rgba': '.6 .6 .65 1'})
        self.markers.append((self.point, outer))
        self.features.append((self.s - FLANGE_T / 2 - (BOLT_H if bolts else 0), self.s + FLANGE_T / 2 + (BOLT_H if bolts else 0)))

    def branch(self, radius, length, toward):
        side = unit(np.asarray(WORLD[toward]) - np.dot(WORLD[toward], self.direction) * self.direction)
        ET.SubElement(self.body, 'geom', name=self.name('branch'), type='cylinder', size=f'{radius:g}',
                      fromto=numbers([*self.point, *(self.point + length * side)]), **real.PIPE_GEOM)
        self.markers.append((self.point, self.radius))
        self.features.append((self.s - radius, self.s + radius))

    def build(self, segments):
        self.radius = segments[0][2]
        for kind, *args in segments:
            getattr(self, kind)(*args)
        self.samples.append([*self.point, self.radius])
        for i, (point, radius) in enumerate(self.markers):
            # 只作显示的特征标记（无碰撞）：贴在管面 −y 外侧（所有管道都在 x-z 平面内弯曲）
            ET.SubElement(self.world, 'site', name=f'feature_{i:02d}', type='sphere', size='.008',
                          pos=numbers(point + (radius + .018) * np.array([0, -1., 0])), rgba='1 .65 .1 1')
        return np.asarray(self.samples)


def build(name, joints=26):
    spec = SCENES[name]
    root, asset, world = real.new_scene(f'real-snake{joints}-{name}')
    builder = PipeBuilder(world, asset, spec.get('tilt_deg', 0.))
    samples = builder.build(spec['segments'])
    # 管轴采样只保留地面以上的部分，避免 Ctrl+K 最近点落到地下段
    above = samples[samples[:, 2] > -1e-6]
    low, high = above[:, :3].min(axis=0), above[:, :3].max(axis=0)
    rotation = Rotation.from_euler('y', spec.get('tilt_deg', 0.), degrees=True) if spec.get('tilt_deg') else None
    return real.finalize(root, world, f'Snake_Robot_o{joints}_Pipe_{name}.xml', spec['segments'][0][2],
                         extra_numeric=[('pipe_centerline', above.ravel()),
                                        # 特征所在沿管弧长 [起点, 终点]（m，从地面起）；无特征的倾斜直管取 1.0 m 处一条参考线，
                                        # 与其他场景特征位置相当，用于统一判定“整条身体越过”
                                        ('feature_span', [min(a for a, _ in builder.features), max(b for _, b in builder.features)]
                                         if builder.features else [1., 1.])],
                         rotation=rotation, joints=joints,
                         center=numbers((low + high) / 2), extent=f'{max(1.2, float(np.max(high - low))):g}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenes', nargs='*')
    parser.add_argument('--joints', type=int, choices=(20, 26), default=26)
    args = parser.parse_args()
    for scene in args.scenes or SCENES:
        print(SCENES[scene]['title'], end='：')
        build(scene, args.joints)
