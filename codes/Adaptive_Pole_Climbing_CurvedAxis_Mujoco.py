"""弯曲轴顺应性（2023 论文顺应性的扩展，算法见 curved_axis_compliance.py）的 MuJoCo 入口。

键位、场景（--scene）、EXE 转发、单步调度和轨迹记录都沿用原入口 Adaptive_Pole_Climbing_Mujoco.py，且不修改它：
运行时只把原入口使用的顺应性控制器换成 CurvedAxisComplianceController。Ctrl+E 关掉顺应性后仍是标准 Yamada 螺旋。
  --variant curved（默认）局部轴 + 局部截面 + 螺距角调制（第 1–3 层）
  --variant local          只用局部轴与局部截面（第 1、2 层，目标仍是直轴螺旋），用于消融对比
Ctrl+K 额外打印一行 CURVED：估计的轴弯曲角、螺距角范围，以及（仿真）局部轴相对真实管轴的角度误差。
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import math
import os
import queue
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Adaptive_Pole_Climbing_Mujoco as gait
from snake_control import pole_compliance
from snake_control.curved_axis_compliance import CurvedAxisComplianceController, CurvedAxisOptions

VARIANTS = {'curved': CurvedAxisOptions(modulation=True),
            'local': CurvedAxisOptions(modulation=False)}
MODELS = Path(__file__).resolve().parents[1] / 'mujoco_model'
SCENES = {'pole': 'Snake_Robot_o20_Pole_Climbing.xml', 'variable': 'Snake_Robot_o20_Variable_Pipe.xml',
          'twin': 'Snake_Robot_o20_Twin_Pipe.xml', 'twin_full': 'Snake_Robot_o20_Twin_Pipe_Full.xml',
          'inner': 'Snake_Robot_o20_Inner_Pipe.xml', 'inner_full': 'Snake_Robot_o20_Inner_Pipe_Full.xml',
          'tree': 'Snake_Robot_o20_Tree.xml', 'real_s2l': 'Snake_Robot_o26_Real_Pipe_S2L.xml',
          'real_l2s': 'Snake_Robot_o26_Real_Pipe_L2S.xml',
          **{f'pipe_{p.stem.removeprefix("Snake_Robot_o26_Pipe_")}': p.name
             for p in MODELS.glob('Snake_Robot_o26_Pipe_*.xml')}}

controllers = []  # 当前运行中创建的控制器（原入口每次 Ctrl+U / Ctrl+E 新建一个）


def true_centerline():
    """仿真场景 XML 中的管中心线（mm）；没有该标记的场景按世界 z 轴。"""
    path = os.environ.get('SNAKE_MUJOCO_MODEL')
    if not path:
        return None
    import mujoco
    model = mujoco.MjModel.from_xml_path(path)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, 'pipe_centerline')
    if key < 0:
        return None
    start, size = model.numeric_adr[key], model.numeric_size[key]
    return model.numeric_data[start:start + size].reshape(-1, 4)[:, :3] * 1000


def axis_error_deg(axes, points, centerline):
    """各连杆局部轴与最近中心线段切向的夹角（度）；无中心线时与世界 z 轴比较。"""
    middles = (points[:-1] + points[1:]) / 2
    if centerline is None:
        truth = np.tile([0, 0, 1.], (len(middles), 1))
    else:
        a, b = centerline[:-1], centerline[1:]
        ab = b - a
        u = np.clip(np.einsum('pij,ij->pi', middles[:, None] - a, ab) / np.einsum('ij,ij->i', ab, ab), 0, 1)
        nearest = np.argmin(np.linalg.norm(middles[:, None] - (a + u[..., None] * ab), axis=2), axis=1)
        truth = ab[nearest] / np.linalg.norm(ab[nearest], axis=1)[:, None]
    cosine = np.abs(np.einsum('ij,ij->i', axes, truth))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


class StatusQueue(queue.Queue):
    """原入口的命令队列；读到 status 时先打印弯曲轴诊断，再照常交给原入口处理。"""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.centerline = true_centerline()

    def get_nowait(self):
        command = super().get_nowait()
        if command.strip().lower() == 'status':
            self.report()
        return command

    def report(self):
        controller = next((c for c in reversed(controllers) if c.started), None)
        if controller is None:
            print(f'CURVED 变体={self.variant} 尚未计算（Ctrl+U 后第一次滚动开始）', flush=True)
            return
        info = controller.info
        text = (f'CURVED 变体={self.variant} 估计轴弯曲={info["axis_bend_deg"]:.1f}° '
                f'β={math.degrees(info["beta_min"]):.1f}–{math.degrees(info["beta_max"]):.1f}° '
                f'窗口平整度≤{info["flatness_max"]:.2f} 模式={"弯曲轴" if info.get("curved_mode", True) else "直管(原方法)"} '
                f'调制={"是" if info.get("curved_mode", True) and info.get("reliable") and controller.options.modulation else "否"}')
        points = info.get('points')
        if points is not None and os.environ.get('SNAKE_MUJOCO_MODEL'):
            error = axis_error_deg(info['axes'], points, self.centerline)
            text += f' 局部轴误差(仿真) 平均/最大={error.mean():.1f}/{error.max():.1f}°'
        print(text, flush=True)


def install(variant):
    options = VARIANTS[variant]
    if os.environ.get('SNAKE_CURVED_AXIS_GAIN'):  # 调参用：螺距角调制复现实测轴弯曲的比例
        from dataclasses import replace
        options = replace(options, axis_gain=float(os.environ['SNAKE_CURVED_AXIS_GAIN']))

    def factory(parameters, links):
        controller = CurvedAxisComplianceController(parameters, links, options)
        controllers.append(controller)
        del controllers[:-2]
        return controller

    pole_compliance.ComplianceController = factory  # 原入口在 main() 内 from pole_compliance import，运行时生效
    # 真类保留在 pole_compliance.OriginalComplianceController（local_pitch_runtime 继承用）
    gait.command_queue = StatusQueue(variant)


def main():
    parser = argparse.ArgumentParser(description='弯曲轴顺应性 Yamada 管道实验；键位与原入口相同')
    parser.add_argument('--scene', choices=sorted(SCENES), default=None)
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='curved')
    parser.add_argument("--joints", type=int, choices=(20, 26), default=26,
                        help="Joint count for real_* / pipe_* scenes (default: 26)")
    args = parser.parse_args()
    if args.scene is not None:
        filename = SCENES[args.scene].replace('o26_', f'o{args.joints}_')
        os.environ['SNAKE_MUJOCO_MODEL'] = str(MODELS / filename)
    gain = os.environ.get('SNAKE_COMPLIANCE_GAIN')  # 与 validation/adaptive_pole_single_step.py 相同：标量或 "端部:中部"
    if gain:
        gait.COMPLIANCE_GAIN = tuple(map(float, gain.split(':'))) if ':' in gain else float(gain)
    install(args.variant)
    headless = os.environ.get('SNAKE_MUJOCO_HEADLESS') == '1'
    if headless:
        gait.DEBUG = False
        threading.Thread(target=gait.uart_listener, daemon=True).start()  # 无界面时原入口不读控制台
    print(f'CURVED_AXIS 弯曲轴顺应性 变体={args.variant}（{"局部轴+局部截面+螺距角调制" if args.variant == "curved" else "局部轴+局部截面"}）', flush=True)
    gait.main()


if __name__ == '__main__':
    main()
