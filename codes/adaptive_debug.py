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
控制台：apply / step / back / hold / pause / run / reset / compliance / status / help / quit
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


def global_key_listener(queue):
    """实机版用：系统全局热键，键位与 MuJoCo 窗口相同（Ctrl+字母/方向键、空格暂停）。

    在任何窗口按下都会触发；只入队命令。按住不放的自动重复被忽略。
    Windows 下按住 Ctrl 时 key.char 是控制字符，因此按虚拟键码识别字母。
    """
    from pynput import keyboard
    held, ctrl = set(), {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
    arrows = {keyboard.Key.up: 'r+', keyboard.Key.down: 'r-', keyboard.Key.right: 'p+', keyboard.Key.left: 'p-'}
    symbols = {189: '-', 187: '='}  # VK_OEM_MINUS / VK_OEM_PLUS

    def on_press(key):
        if key in held:
            return
        held.add(key)
        if key == keyboard.Key.space:
            queue.put('pause')
            return
        if not held & ctrl:
            return
        vk = getattr(key, 'vk', None)
        name = chr(vk) if vk is not None and 65 <= vk <= 90 else symbols.get(vk)
        command = arrows.get(key) or HOTKEYS.get(name or '')
        if command:
            queue.put(command)

    def on_release(key):
        held.discard(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


def measurements(model, data):
    """观察真实执行误差与管道接触；不把这些指标作为已实现的力控制器。"""
    pipe = {i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pipe')}
    contacts = sum(c.geom1 in pipe or c.geom2 in pipe for c in data.contact)
    actual = np.array([data.joint(f'joint{i}').qpos[0] for i in range(1, model.nu+1)])
    heights = [data.joint(f'joint{i}').xanchor[2] for i in range(1, model.nu+1)]
    return float(np.mean(heights)), contacts, float(np.max(np.abs(actual-data.ctrl)))


def circle_radius_mm(xy):
    """截面点的最小二乘圆半径；点数不足或近似共线时返回 None。"""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return None
    a = np.c_[2 * xy, np.ones(len(xy))]
    (cx, cy, c), *_ = np.linalg.lstsq(a, (xy ** 2).sum(axis=1), rcond=None)
    value = c + cx * cx + cy * cy
    return float(np.sqrt(value)) if value > 0 else None


def centerline_nearest(model, points):
    """弯管等场景：各点到 pipe_centerline 折线的距离、该处管半径与沿管弧长（m）；无该标记时返回 None。"""
    key = mujoco_numeric(model, 'pipe_centerline')
    if key is None:
        return None
    samples = key.reshape(-1, 4)
    a, b = samples[:-1, :3], samples[1:, :3]
    ab = b - a
    lengths = np.linalg.norm(ab, axis=1)
    start = np.r_[0., np.cumsum(lengths)][:-1]
    u = np.clip(np.einsum('pij,ij->pi', points[:, None] - a, ab) / np.maximum(lengths**2, 1e-12), 0, 1)
    nearest = a + u[..., None] * ab
    distance = np.linalg.norm(points[:, None] - nearest, axis=2)
    k = np.argmin(distance, axis=1)
    i = np.arange(len(points))
    radius = samples[k, 3] + u[i, k] * (samples[k + 1, 3] - samples[k, 3])
    return distance[i, k], radius, start[k] + u[i, k] * lengths[k]


def mujoco_numeric(model, name):
    import mujoco
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    return None if key < 0 else model.numeric_data[model.numeric_adr[key]:model.numeric_adr[key] + model.numeric_size[key]]


def pipe_progress(model, data):
    """各关节沿管进度的 (平均, 最小, 最大)（m，从地面起）。带 pipe_centerline 的场景沿中心线弧长，
    其余竖直管场景取关节高度；最小值即最靠后（尾部一侧）的关节。"""
    points = np.array([data.joint(f'joint{i}').xanchor for i in range(1, model.nu+1)])
    nearest = centerline_nearest(model, points)
    progress = points[:, 2] if nearest is None else nearest[2]
    return float(progress.mean()), float(progress.min()), float(progress.max())


def pipe_geometry(model, data):
    """仿真专用：关节到管轴的平均距离，及同位置的平均管半径（mm）。

    带 pipe_centerline 的场景（弯管、法兰等）按中心线折线计算；否则管轴为世界 z 轴。
    """
    points = np.array([data.joint(f'joint{i}').xanchor for i in range(1, model.nu+1)])
    nearest = centerline_nearest(model, points)
    if nearest is not None:
        return float(nearest[0].mean() * 1000), float(nearest[1].mean() * 1000)
    radial = float(np.linalg.norm(points[:, :2], axis=1).mean() * 1000)
    try:
        bottom, top, z0, z1, _ = model.numeric('pipe_profile').data
    except KeyError:
        return radial, None
    u = np.clip((points[:, 2] - z0) / (z1 - z0), 0, 1)
    return radial, float(np.mean(bottom + (top - bottom) * (10*u**3 - 15*u**4 + 6*u**5)) * 1000)
