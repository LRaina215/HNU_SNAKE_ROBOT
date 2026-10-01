"""实机入口的离线命令回归：改 R/P 不重绕、不清相位，reset 不保留旧目标。

只替换通信与输入，运行真实 main 分支；不启动 EXE、不访问串口、不驱动机器人。
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import queue
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]
import Adaptive_Pole_Climbing_Real as real
import python_client


class Exchange:
    def __init__(self):
        self.angles = np.zeros(20)
        self.frames = []
        self.command = ''
        self.last_command_id = 0
        self.sent = self.received = self.failures = 0
        self.reject = False
        self.last_error = None

    def latest_geometry(self):
        return dict(angles_deg=self.angles.tolist(), online=[True]*20, feedback_usable=False,
                    config=dict(joint_count=20, head_length_mm=64.5,
                                joints=[dict(link_length_mm=64.5, minimum_deg=-90, maximum_deg=90)]*20),
                    target_state=dict(dispatched_command_id=self.last_command_id))

    def publish_targets(self, degrees, source_geometry):
        if self.reject:
            return False
        self.last_command_id += 1
        self.sent += 1
        self.angles = np.array(degrees)
        self.frames.append((self.command, real.controller_angles(degrees)))
        return True

    def close(self):
        pass


def main():
    exchange = Exchange()
    commands = ['wind=2', 'step', 'wind=0.25'] + ['step']*4
    commands += ['r=104', 'r=105'] + ['step']*76
    commands += ['r=104', 'r+', 'r-', 'r=1', 'status', 'apply', 'r=103', 'compliance']
    commands += ['back']*3 + ['r=103', 'apply', 'status', 'back', 'r=102', 'status']
    commands += ['reset', 'back', 'step', 'status', 'quit']

    class Commands:
        def get(self, timeout=None):
            command = commands.pop(0)
            exchange.command = command
            exchange.reject = command == 'r=102'  # 发送失败时参数也不能提交。
            if command == 'reset':
                exchange.angles[:] = 0  # 模拟失能后姿态改变，旧目标仍是卷曲状态。
            return command

        def get_nowait(self):
            raise queue.Empty

    output = StringIO()
    with patch.object(python_client, 'GeometryExchange', return_value=exchange), \
         patch.object(real, 'command_queue', Commands()), \
         patch.object(real.threading, 'Thread'), \
         patch.object(real, 'DEBUG', False), patch.object(real, 'TELEMETRY_LOG', False), \
         patch.object(real, 'STEP_SETTLE_S', 0), redirect_stdout(output):
        real.main(hotkeys=False)

    text = output.getvalue()
    assert '单步最大关节变化' in text
    assert '顺应性滚动中不能直接应用标准螺旋' in text
    assert 'EXE 未接受目标' in text
    assert '已退回到起始姿态' in text
    assert '已缠绕=20/20节 R=104' in text
    assert 'R=103 P=220 phase=-0.4500' in text  # 重复 apply 未清零相位。
    assert 'R=103 P=220 phase=-0.6000' in text  # r=102 发送失败，保留 R=103。
    assert '已缠绕=0.25/20节' in text
    assert text.count('完成第 1 步') == 2  # reset 连步数也重置。
    links = np.full(21, 64.5)
    helix = lambda r, phase=0: real.HelixBackbone.from_radius_pitch(r, 220).joint_angles(20, links, phase)
    edits = [(c, q) for c, q in exchange.frames if c.startswith('r=')]
    assert [c for c, _ in edits] == ['r=104', 'r=105', 'r=104', 'r=103']
    np.testing.assert_allclose(edits[0][1], real.winding_targets(np.zeros(20), helix(104), 1))
    np.testing.assert_allclose(edits[2][1], helix(104))  # 缠满时修改几何不解缠。
    np.testing.assert_allclose(next(q for c, q in exchange.frames if c == 'r+'), helix(105))
    np.testing.assert_allclose(next(q for c, q in exchange.frames if c == 'r-'), helix(104))
    np.testing.assert_allclose(edits[3][1], helix(103, -.45))
    np.testing.assert_allclose(exchange.frames[-1][1], real.winding_targets(np.zeros(20), helix(103), .25))
    print('PASS: partial/full winding edits, nonzero phase, rejected edits, reset after pose change; no hardware')


if __name__ == '__main__':
    main()
