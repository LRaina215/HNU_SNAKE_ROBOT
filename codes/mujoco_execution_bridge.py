"""MuJoCo 执行端：实测上传与 EXE 目标回传共用一条持久连接。

网络线程只维护最新命令槽；主仿真线程在 mj_step 前调用 apply_received。
命令失效时保留上一组 ctrl，不以本地候选目标兜底；是否推进物理时间由主循环决定。
"""
import queue
import threading
import time

import numpy as np

from mujoco_visualizer_bridge import VisualizerBridge
from python_client import SnakeControlClient, ProtocolError


class MujocoExecutionBridge(VisualizerBridge):
    def __init__(self, model, data):
        self._command_lock = threading.Lock()
        self._command = None
        self._run_id = ""
        self._ready = False
        self._response_at = 0.
        self.applied_run_id = ""
        self.applied_delivery_id = self.applied_command_id = 0
        self.applied = self.expired = 0
        self.received_commands = []
        self.record_commands = False  # 验证时启用；日常运行不积累历史。
        super().__init__(model, data)

    @property
    def executor_ready(self):
        with self._command_lock:
            return self._ready and time.monotonic() - self._response_at <= .5

    def _packet_extra(self):
        # 与父类 publish 中的实测快照同时采集确认号，不能在网络发送时补写。
        with self._command_lock:
            return {"relay_enabled": True, "applied_run_id": self.applied_run_id,
                    "applied_delivery_id": self.applied_delivery_id}

    def _send_loop(self):
        client = SnakeControlClient(port=self.port, timeout=.5)
        try:
            while not self.stopping:
                try:
                    packet = self.latest.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    result = client.request(packet)
                    if "relay_ready" not in result:
                        raise ProtocolError("请使用 v1.4.0 EXE，当前服务不支持 MuJoCo 目标转发")
                    command = result.get("simulation_target")
                    with self._command_lock:
                        run_id = result.get("run_id", "")
                        if run_id != self._run_id or not result["relay_ready"]:
                            self._command = None
                        self._run_id = run_id
                        self._ready = result["relay_ready"]
                        self._response_at = time.monotonic()
                        if command is not None:
                            if command["run_id"] != run_id or command["config_id"] != run_id:
                                raise ValueError("回传目标的配置标识不一致")
                            self._command = command
                            if self.record_commands:
                                self.received_commands.append(dict(command))
                    self.sent += 1
                    self.last_error = ""
                except (OSError, ValueError, KeyError, ProtocolError) as error:
                    self.rejected += 1
                    self.last_error = str(error)
                    client.close()
                    with self._command_lock:
                        self._ready = False
                        self._command = None
        finally:
            client.close()

    def apply_received(self, data, expected_targets=None):
        """只由仿真线程在物理步进前调用；网络线程不访问 mjData。"""
        with self._command_lock:
            # 领取后即清空；同一目标不会因多次物理步进被反复执行。
            command, self._command = self._command, None
            if command is None:
                return False
            try:
                if (not self._ready or time.monotonic() - self._response_at > .5 or
                        command["run_id"] != self._run_id or
                        time.time() * 1000 >= command["expires_unix_ms"]):
                    self.expired += 1
                    return False
                if (self.applied_run_id == command["run_id"] and
                        command["delivery_id"] <= self.applied_delivery_id):
                    return False
                targets = np.asarray(command["targets_deg"], dtype=float)
                if targets.shape != (self.joint_count,) or not np.all(np.isfinite(targets)):
                    raise ValueError("回传目标角数量或数值无效")
                targets = np.deg2rad(targets * self.signs)
                # 单步等待只能领取本次候选，不能执行上一次在网络中迟到的目标。
                if expected_targets is not None and not np.allclose(targets, expected_targets, rtol=0, atol=1e-12):
                    return False
                joint_ids = self.model.actuator_trnid[:, 0]
                limited = self.model.jnt_limited[joint_ids].astype(bool)
                limits = self.model.jnt_range[joint_ids]
                if (np.any(targets[limited] < limits[limited, 0]) or
                        np.any(targets[limited] > limits[limited, 1])):
                    raise ValueError("回传目标超出 MuJoCo 关节范围")
                data.ctrl[:] = targets
                self.applied_run_id = command["run_id"]
                self.applied_delivery_id = command["delivery_id"]
                self.applied_command_id = command["command_id"]
                self.applied += 1
                return True
            except (KeyError, TypeError, ValueError) as error:
                self.rejected += 1
                self.last_error = str(error)
                return False

    def reset_execution(self):
        """实验重置时清空已领取目标与确认状态；只能由暂停的主线程调用。"""
        with self._command_lock:
            self._command = None
            self.applied_run_id = ""
            self.applied_delivery_id = self.applied_command_id = 0

    def close(self):
        self.stopping = True
        self.worker.join(timeout=1.5)
