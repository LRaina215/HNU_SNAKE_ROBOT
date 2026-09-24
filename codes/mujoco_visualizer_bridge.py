"""Latest-only MuJoCo actual-angle sender for the official visualizer.

Networking runs on a worker thread and never waits on the gait loop.
"""

import json
import math
import os
import queue
import socket
import threading
import time
import numpy as np


class VisualizerBridge:
    # Columns are visualizer X/Y/Z expressed in the MuJoCo head body frame.
    HEAD_BASIS = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])

    def __init__(self, model, data, rate_hz=15):
        self.model = model
        self.joint_count = model.nu
        self.qpos_adrs = np.array([int(np.asarray(model.joint(f"joint{i}").qposadr).ravel()[0])
                                   for i in range(1, model.nu + 1)], dtype=int)
        self.signs = np.array([1 if i % 2 == 0 else -1 for i in range(model.nu)])
        self.port = int(os.environ.get("SNAKE_VISUALIZER_PORT", "8765"))
        self.period = 1.0 / rate_hz
        self.last_publish = -math.inf
        self.latest = queue.Queue(maxsize=1)
        self.stopping = False
        self.sent = 0
        self.rejected = 0
        self.last_error = ""
        self.enqueue_ms = []
        self.lengths = self._measure_lengths(data)
        self.worker = threading.Thread(target=self._send_loop, name="MuJoCo visualizer sender", daemon=True)
        self.worker.start()

    def _measure_lengths(self, data):
        import mujoco
        mujoco.mj_forward(self.model, data)
        points = self.world_points(data)
        return [float(sum((a - b) ** 2) ** 0.5 * 1000) for a, b in zip(points[1:], points[:-1])]

    def world_points(self, data):
        points = [data.site("s_head").xpos.copy()]
        points.extend(data.joint(f"joint{i}").xanchor.copy() for i in range(1, self.joint_count + 1))
        points.append(data.site("s_tail").xpos.copy())
        return np.asarray(points)

    def publish(self, data):
        now = time.monotonic()
        if now - self.last_publish < self.period:
            return
        import mujoco
        # mj_step integrates qpos after its last kinematics stage. Refresh the
        # cached root/site pose so it belongs to the same instant as qpos.
        mujoco.mj_kinematics(self.model, data)
        started = time.perf_counter()
        self.last_publish = now
        angles = np.rad2deg(data.qpos[self.qpos_adrs] * self.signs).tolist()
        targets = np.rad2deg(data.ctrl[:self.joint_count] * self.signs).tolist()
        # qpos alone determines the shape only in the moving head frame. The
        # free root pose must also reach the viewer to reproduce the scene.
        head_origin_world_mm = (data.site("s_head").xpos * 1000).tolist()
        head_basis_world = (data.body("head").xmat.reshape(3, 3) @ self.HEAD_BASIS).reshape(-1).tolist()
        packet = {"cmd": "mujoco_frame", "angles_deg": angles,
                  "targets_deg": targets, "link_lengths_mm": self.lengths,
                  "head_origin_world_mm": head_origin_world_mm,
                  "head_basis_world": head_basis_world,
                  "sim_time": float(data.time), "captured_unix": time.time()}
        try:
            self.latest.put_nowait(packet)
        except queue.Full:
            try:
                self.latest.get_nowait()
            except queue.Empty:
                pass
            self.latest.put_nowait(packet)
        self.enqueue_ms.append((time.perf_counter() - started) * 1000)

    def _send_loop(self):
        while not self.stopping:
            try:
                packet = self.latest.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2) as sock:
                    sock.settimeout(0.2)
                    sock.sendall((json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8"))
                    response = b""
                    while not response.endswith(b"\n") and len(response) < 4096:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response += chunk
                    result = json.loads(response.decode("utf-8"))
                    if result.get("ok"):
                        self.sent += 1
                    else:
                        self.rejected += 1
                        self.last_error = result.get("error", "unknown receiver error")
            except (OSError, ValueError, json.JSONDecodeError) as ex:
                self.rejected += 1
                self.last_error = str(ex)

    def close(self):
        self.stopping = True
        self.worker.join(timeout=0.3)
