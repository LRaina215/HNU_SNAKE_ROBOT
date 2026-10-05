"""Focused switching checks using saved geometry; no network, hardware or climbing trials."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import io
import json
import math
import os
import queue
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]
from snake_control.pole_compliance import ComplianceController, ComplianceParameters, build_backbone
from snake_control.local_pitch_runtime import (InteractivePitchController, prepare_pitch_step, toggle_local_pitch,
                                 stop_local_pitch, resume_compliance, local_pitch_status)
from snake_control.mujoco_checkpoint import controller_state, restore_controller, json_value


def fixture(alpha=.4):
    geometry = json.loads((ROOT / 'validation/fixtures/geometry_example.json').read_text(encoding='utf8'))
    geometry.update(age_ms=0, sample_unix_ms=time.time() * 1000, feedback_usable=True)
    links = [geometry['config']['head_length_mm']] + [j['link_length_mm'] for j in geometry['config']['joints']]
    base = ComplianceController(ComplianceParameters((.01, .02), alpha), links)
    reference = build_backbone(geometry, base.parameters).joint_angles(0.)
    reference = base.compute(geometry, 0., reference, reference_phase=0.)
    return geometry, base, reference


class SwitchingChecks(unittest.TestCase):
    def step(self, controller, geometry, ref, phase, standard=None):
        geometry = copy.deepcopy(geometry)
        geometry['sequence'] += 1
        geometry['sample_unix_ms'] = time.time() * 1000
        target, proposed = prepare_pitch_step(controller, geometry, phase - .03, ref, phase, .2,
                                              ref if standard is None else standard)
        self.assertIsNotNone(target, proposed.status)
        return target, proposed, phase - .03

    def test_default_off_exact_and_enable_guards(self):
        g, base, ref = fixture()
        direct = copy.copy(base)
        a, proposed = prepare_pitch_step(base, g, -.1, ref, 0., .2, None)
        b = direct.compute(g, -.1, ref, reference_phase=0.)
        np.testing.assert_array_equal(a, b)
        self.assertIs(proposed, base)
        self.assertEqual(local_pitch_status(base), '关')
        with self.assertRaises(ValueError):
            toggle_local_pitch(base, False, ref, 0.)
        base.parameters = ComplianceParameters((.01, .02), .4, True)
        with self.assertRaises(ValueError):
            toggle_local_pitch(base, True, ref, 0.)

    def test_rate_cancel_return_and_standard_exit(self):
        g, base, ref = fixture()
        c = toggle_local_pitch(base, True, ref, 0.)
        initial = c.alpha_state.copy()
        target, proposed, _ = self.step(c, g, ref, 0.)
        np.testing.assert_array_equal(c.alpha_state, initial)  # Not dispatched.
        self.assertLessEqual(np.degrees(abs(proposed.alpha_state - initial)).max(), .2 + 1e-10)
        # An arbitrarily long idle interval cannot create a large pitch step.
        g['sample_unix_ms'] = time.time() * 1000
        _, idle = prepare_pitch_step(c, g, -.03, ref, 0., 999., ref)
        np.testing.assert_allclose(idle.alpha_state, proposed.alpha_state)
        phase = 0.
        for _ in range(4):
            ref, c, phase = self.step(c, g, ref, phase)
        g['sample_unix_ms'] = time.time() * 1000 - 1000
        bad, failed = prepare_pitch_step(c, g, phase-.03, ref, phase, .2, ref)
        self.assertIsNone(bad)
        np.testing.assert_array_equal(c.alpha_state, failed.alpha_state)
        c = toggle_local_pitch(c, True, ref, phase)
        for _ in range(14):
            ref, p, phase = self.step(c, g, ref, phase)
            c = p.settled()
            if not isinstance(c, InteractivePitchController):
                break
        self.assertIs(type(c), ComplianceController)
        np.testing.assert_allclose(c.backbone.pitch, .4)
        c = toggle_local_pitch(c, True, ref, phase)
        ref, c, phase = self.step(c, g, ref, phase)
        c = stop_local_pitch(c)
        self.assertFalse(c.requested)
        self.assertTrue(c.standard_return)
        self.assertFalse(resume_compliance(c).standard_return)
        standard = ref.copy()
        for _ in range(20):
            ref, p, phase = self.step(c, g, ref, phase, standard)
            c = p.settled()
            if c is None:
                break
        self.assertIsNone(c)
        np.testing.assert_allclose(ref, standard, atol=1e-12)

    def test_checkpoint_and_nonuniform_geometry(self):
        g, base, ref = fixture()
        c = toggle_local_pitch(base, True, ref, 0.)
        state = json.loads(json.dumps(controller_state(c), default=json_value, allow_nan=False))
        restored = restore_controller(state)
        np.testing.assert_array_equal(restored.alpha_state, c.alpha_state)
        ref, c, phase = self.step(c, g, ref, 0.)
        # Cover the variable-alpha path, including its saved geometry.
        c.alpha_state += np.linspace(-.002, .002, len(c.alpha_state))
        ref, c, phase = self.step(c, g, ref, phase)
        c = stop_local_pitch(c)
        restored = restore_controller(json.loads(json.dumps(controller_state(c), default=json_value, allow_nan=False)))
        self.assertTrue(restored.standard_return)
        a, ca, _ = self.step(c, g, ref, phase)
        b, cb, _ = self.step(restored, g, ref, phase)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(ca.alpha_state, cb.alpha_state, atol=1e-12)

    def test_real_entry_command_routing_without_hardware(self):
        import Adaptive_Pole_Climbing_Real as real
        import python_client
        from snake_control.adaptive_debug import HOTKEYS
        self.assertEqual(HOTKEYS['L'], 'localpitch')
        alpha = math.atan2(real.HELIX_PITCH_MM, 2 * math.pi * real.HELIX_RADIUS_MM)
        g, base, ref = fixture(alpha)
        g['angles_deg'] = real.protocol_angles(ref).tolist()
        commands = iter(['localpitch', 'apply', 'step', 'localpitch', 'status', 'step',
                         'localpitch', 'status', 'compliance', 'status', 'step', 'reset', 'status', 'quit'])

        class Commands:
            def get(self, **kwargs):
                return next(commands)
            def get_nowait(self):
                raise queue.Empty

        class Exchange:
            received = sent = failures = 0
            last_error = None
            last_command_id = 0
            last_ack = None
            def __init__(self, **kwargs):
                self.g = copy.deepcopy(g)
            def latest_geometry(self):
                self.g.update(sample_unix_ms=time.time()*1000, age_ms=0)
                return copy.deepcopy(self.g)
            def publish_targets(self, targets, **kwargs):
                self.sent += 1
                self.last_command_id += 1
                self.g['angles_deg'] = targets
                self.g['target_state'] = {'dispatched_command_id': self.last_command_id}
                return True
            def close(self):
                pass

        output = io.StringIO()
        with patch.object(python_client, 'GeometryExchange', Exchange), \
             patch.object(real, 'command_queue', Commands()), \
             patch.object(real, 'console_listener', lambda: None), \
             patch.object(real, 'DEBUG', False), patch.object(real, 'TELEMETRY_LOG', False), \
             patch.object(real, 'STEP_SETTLE_S', .02), redirect_stdout(output):
            real.main(hotkeys=False)
        text = output.getvalue()
        self.assertIn('请先开启顺应性并完成一次有效滚动', text)
        self.assertIn('局部螺距=开', text)
        self.assertIn('局部螺距=关（恢复设计角中）', text)
        self.assertIn('局部螺距=关（返回标准步态中）', text)
        self.assertIn('阶段=缠绕', text)
        self.assertIn('局部螺距=关 ', text)
        self.assertNotIn('局部螺距目标变化超过', text)

    def test_mujoco_entry_routing_without_physics_or_network(self):
        import mujoco
        import Adaptive_Pole_Climbing_Mujoco as gait
        from snake_control import mujoco_execution_bridge
        import python_client
        g, _, _ = fixture()
        commands = iter(['localpitch', 'apply', 'step', 'localpitch', 'step', 'status',
                         'localpitch', 'status', 'compliance', 'status', 'step', 'reset', 'status', 'quit'])
        busy = [False]

        class Commands:
            def get_nowait(self):
                if busy[0]:
                    raise queue.Empty
                command = next(commands)
                busy[0] = command in ('apply', 'step', 'back', 'hold')
                return command
            def empty(self):
                return True

        class Output(io.StringIO):
            def write(self, text):
                if '完成第 ' in text or '未执行：' in text or '本步取消' in text:
                    busy[0] = False
                return super().write(text)

        class Bridge:
            port = 0
            executor_ready = True
            sent = rejected = expired = applied = 0
            last_error = None
            def __init__(self, model, data):
                points = [data.site('s_head').xpos.copy()]
                points += [data.joint(f'joint{i}').xanchor.copy() for i in range(1, model.nu+1)]
                points += [data.site('s_tail').xpos.copy()]
                self.lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)*1000
                self.signs = np.where(np.arange(model.nu) % 2, -1, 1)
            def publish(self, data):
                pass
            def apply_received(self, data, expected_targets):
                data.ctrl[:] = expected_targets
                self.applied += 1
                return True
            def reset_execution(self):
                pass
            def close(self):
                pass

        class Exchange:
            received = sent = failures = 0
            last_error = None
            def __init__(self, expected_lengths, **kwargs):
                self.g = copy.deepcopy(g)
                self.g['config']['head_length_mm'] = expected_lengths[0]
                for j, length in zip(self.g['config']['joints'], expected_lengths[1:]):
                    j['link_length_mm'] = length
            def latest_geometry(self):
                self.g.update(sample_unix_ms=time.time()*1000, age_ms=0)
                return copy.deepcopy(self.g)
            def publish_targets(self, *args, **kwargs):
                self.sent += 1
                return True
            def close(self):
                pass

        output = Output()
        env = dict(SNAKE_MUJOCO_MODEL=str(ROOT/'validation/fixtures/pole28.xml'),
                   SNAKE_MUJOCO_HEADLESS='1', SNAKE_CONTROL_VIA_VISUALIZER='1',
                   SNAKE_GEOMETRY_EXCHANGE='1', SNAKE_VISUALIZER_ENABLED='1',
                   SNAKE_SINGLE_STEP='1', SNAKE_MUJOCO_RESTORE='', SNAKE_MUJOCO_TRACE='',
                   SNAKE_MUJOCO_MAX_SECONDS='15')
        def clock_only(model, data):
            data.time += model.opt.timestep
        with patch.dict(os.environ, env), patch.object(gait, 'DEBUG', False), \
             patch.object(gait, 'FRAME_SLEEP', 0.), patch.object(gait, 'command_queue', Commands()), \
             patch.object(mujoco_execution_bridge, 'MujocoExecutionBridge', Bridge), \
             patch.object(python_client, 'GeometryExchange', Exchange), \
             patch.object(mujoco, 'mj_step', clock_only), redirect_stdout(output):
            gait.main()
        text = output.getvalue()
        self.assertIn('局部螺距=开', text)
        self.assertIn('局部螺距=关（恢复设计角中）', text)
        self.assertIn('局部螺距=关（返回标准步态中）', text)
        self.assertIn('实验已重置并暂停', text)
        self.assertNotIn('局部螺距目标变化超过', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
