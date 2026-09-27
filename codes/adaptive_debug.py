"""MuJoCo 窗口内的实验快捷键及简短状态读数，不参与步态计算。"""
import numpy as np

# 使用 Ctrl 修饰，避开 MuJoCo 的 U/H/J 等内置显示快捷键。
HOTKEYS = {'U': 'apply', 'J': 'step', 'H': 'back', 'N': 'hold',
           'R': 'reset', 'G': 'run', 'E': 'compliance', 'K': 'status',
           '-': 'delta-', '=': 'delta+'}
HELP = """机器人调试（先点击 MuJoCo 窗口；Ctrl+字母，不用 F 键）：
  Ctrl+U 应用 R/P    Ctrl+J 相位+一步       Ctrl+H 相位-一步
  （变径场景 P=180 时 Ctrl+H 上行、Ctrl+J 下行；改变螺距符号方向随之反转）
  Ctrl+N 保持目标再走一段物理时间（相位不变）
  空格 立即暂停     Ctrl+G 连续运行/暂停     Ctrl+R 重置实验并暂停
  Ctrl+E 顺应性开/关 Ctrl+K 查看状态         Ctrl+-/= 减半/加倍相位步长
  Ctrl+上下 修改半径；Ctrl+左右 修改螺距；修改后 Ctrl+U 应用
控制台：apply / step / back / hold / pause / run / reset / compliance / status / help
参数：r=130 p=180 / delta=0.05 / gravity=0（几何调试）或 gravity=1（重力实验）
"""


def viewer_key_callback(queue):
    """MuJoCo 3.2.7 回调只接收首次按下，忽略 GLFW_REPEAT。

    回调在拥有 GLFW 上下文的 viewer 线程执行；输入只入队，不访问 mjData。
    因而终端输入不会触发这些热键，也无需系统全局键盘监听。
    """
    import glfw
    def on_key(key):
        if key == glfw.KEY_SPACE:
            queue.put('pause')
            return
        window = glfw.get_current_context()
        if not window or not (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                              glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS):
            return
        arrows = {glfw.KEY_UP: 'r+', glfw.KEY_DOWN: 'r-', glfw.KEY_RIGHT: 'p+', glfw.KEY_LEFT: 'p-'}
        command = arrows.get(key) or HOTKEYS.get(chr(key) if 0 <= key < 128 else '')
        if command:
            queue.put(command)
    return on_key


def measurements(model, data):
    """观察真实执行误差与管道接触；不把这些指标作为已实现的力控制器。"""
    pipe = {i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pipe')}
    contacts = sum(c.geom1 in pipe or c.geom2 in pipe for c in data.contact)
    actual = np.array([data.joint(f'joint{i}').qpos[0] for i in range(1, model.nu+1)])
    heights = [data.joint(f'joint{i}').xanchor[2] for i in range(1, model.nu+1)]
    return float(np.mean(heights)), contacts, float(np.max(np.abs(actual-data.ctrl)))
