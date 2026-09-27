"""蛇形机器人桌面程序的可选本机控制客户端。

桌面程序始终是串口的唯一所有者。Python 只通过 127.0.0.1:8765
交换逐行 JSON，不直接打开 COM 口，也不能远程使能扭矩。
"""

from __future__ import annotations

import argparse
import json
import socket
import copy
import threading
import time
from typing import Any, Dict, Optional


class ProtocolError(RuntimeError):
    """服务器明确拒绝请求；连接仍可用。"""


class SnakeControlClient:
    """连接正在运行的 Windows 可视化主程序。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 1.5) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = self._reader = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._socket.settimeout(self.timeout)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._reader = self._socket.makefile("rb")

    def close(self) -> None:
        with self._lock:
            if self._reader is not None:
                self._reader.close()
            if self._socket is not None:
                self._socket.close()
            self._reader = self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 对目标写请求不自动重试：断线时无法确定服务器是否已接收。
        data = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > 1048576:
            raise ValueError("请求超过 1 MiB")
        with self._lock:
            try:
                self.connect()
                self._socket.sendall(data)
                response = self._reader.readline(1048577)
                if not response or len(response) > 1048576 or not response.endswith(b"\n"):
                    raise OSError("响应中断或超过 1 MiB")
                result = json.loads(response)
                if not isinstance(result, dict):
                    raise ValueError("响应不是 JSON 对象")
            except (OSError, ValueError):
                self.close()
                raise
            if not result.get("ok"):
                raise ProtocolError(result.get("error", "控制命令被拒绝"))
            return result

    def get_geometry(self) -> Dict[str, Any]:
        result = self.request({"cmd": "geometry"})
        if result.get("protocol_version") != 2:
            raise ProtocolError("不支持的几何协议版本")
        return result

    def submit_targets(self, geometry, targets_deg, command_id, *, execution="preview", valid_for_ms=500):
        return self.request({"cmd": "target_frame", "execution": execution,
            "run_id": geometry["run_id"], "config_id": geometry["config_id"],
            "feedback_sequence": geometry["sequence"], "command_id": command_id,
            "valid_for_ms": valid_for_ms, "targets_deg": list(targets_deg)})

    def cancel_targets(self):
        return self.request({"cmd": "cancel_targets"})

    def get_state(self) -> Dict[str, Any]:
        """取得全部关节角、目标角、在线状态和当前运行模式。"""
        return self.request({"cmd": "state"})

    def set_joint_target(self, joint_index: int, degrees: float) -> Dict[str, Any]:
        """给一个关节发送目标；joint_index 从 1 开始。"""
        return self.request(
            {
                "cmd": "target",
                "joint_index": joint_index,
                "degrees": degrees,
            }
        )

    def set_two_joint_target(
        self,
        joint1_deg: float,
        joint2_deg: float,
    ) -> Dict[str, Any]:
        """保留给原两关节脚本的兼容接口。"""
        return self.request(
            {
                "cmd": "target",
                "j1_deg": joint1_deg,
                "j2_deg": joint2_deg,
            }
        )

    def set_joint_limits(
        self,
        joint_index: int,
        torque_limit: Optional[int] = None,
        acceleration: Optional[int] = None,
    ) -> Dict[str, Any]:
        command: Dict[str, Any] = {
            "cmd": "limits",
            "joint_index": joint_index,
        }
        if torque_limit is not None:
            command["torque_limit"] = torque_limit
        if acceleration is not None:
            command["acceleration"] = acceleration
        return self.request(command)

    def release_torque(self) -> Dict[str, Any]:
        """请求桌面程序紧急释放全部扭矩。"""
        return self.request({"cmd": "release"})


class GeometryExchange:
    """后台读取几何并提交最新目标；主步态线程不等待网络。

    支持 preview、mujoco 与 live（实机，EXE 需处于实机模式并已手动使能扭矩）。
    反馈缓存只用于本次连接；断连或配置变化清空，旧目标不重试。
    """
    def __init__(self, host="127.0.0.1", port=8765, rate_hz=10, expected_lengths=None, execution="preview"):
        if rate_hz <= 0:
            raise ValueError("rate_hz 必须为正")
        if execution not in ("preview", "mujoco", "live"):
            raise ValueError("后台交换只支持 preview、mujoco 或 live")
        self.execution = execution
        self.client = SnakeControlClient(host, port, timeout=0.5)
        self.period = 1.0 / rate_hz
        self.expected_lengths = list(expected_lengths) if expected_lengths is not None else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = self._pending = None
        self._received_at = 0.0
        self.received = self.sent = self.dropped = self.failures = 0
        self.last_error = ""
        self.last_ack = None
        self.last_command_id = 0
        self.roundtrip_ms = 0.0
        self._worker = threading.Thread(target=self._run, name="Geometry feedback exchange", daemon=True)
        self._worker.start()

    def latest_geometry(self):
        with self._lock:
            if time.monotonic() - self._received_at > 0.5:
                return None
            return copy.deepcopy(self._latest)

    def publish_targets(self, targets_deg, source_geometry=None):
        """提交候选整组角度（度）；True 仅表示进入发送槽，不表示已执行。

        source_geometry 绑定算法实际使用的反馈，避免将旧计算结果标成新帧结果。
        """
        # 网络线程负责发送；缓存复制后才释放锁，调用者可继续修改自己的数组。
        with self._lock:
            geometry = self._latest
            # 实机只接受 Live 反馈；preview 与 mujoco 保持原来的 MuJoCo 数据源要求。
            source_mode = "Live" if self.execution == "live" else "Mujoco"
            if (geometry is None or time.monotonic() - self._received_at > 0.5 or
                    geometry["age_ms"] > 500 or geometry["mode"] != source_mode):
                return False
            if self.execution == "mujoco" and not geometry.get("target_state", {}).get("mujoco_executor_ready"):
                return False
            if source_geometry is not None:
                if ((source_geometry["run_id"], source_geometry["config_id"]) !=
                        (geometry["run_id"], geometry["config_id"]) or
                        source_geometry["sequence"] > geometry["sequence"] or
                        time.time() * 1000 - source_geometry["sample_unix_ms"] > 500):
                    return False
                geometry = source_geometry
            if len(targets_deg) != geometry["config"]["joint_count"]:
                return False
            if self.expected_lengths is not None:
                config = geometry["config"]
                lengths = [config["head_length_mm"]] + [j["link_length_mm"] for j in config["joints"]]
                if len(lengths) != len(self.expected_lengths) or any(abs(a-b) > 1e-6 for a,b in zip(lengths, self.expected_lengths)):
                    return False
            if self._pending is not None:
                self.dropped += 1
            self._pending = (geometry, list(targets_deg))
            return True

    def _run(self):
        identity = None
        command_id = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                geometry = self.client.get_geometry()
                current = (geometry["run_id"], geometry["config_id"])
                with self._lock:
                    if current != identity:
                        self._pending = None
                    identity = current
                    self._latest = geometry
                    self._received_at = time.monotonic()
                    self.received += 1
                    pending, self._pending = self._pending, None
                if pending is not None and not self._stop.is_set():
                    source, targets = pending
                    if (source["run_id"], source["config_id"]) == current:
                        command_id += 1
                        self.last_ack = self.client.submit_targets(source, targets, command_id, execution=self.execution)
                        self.last_command_id = command_id
                        self.sent += 1
                self.last_error = ""
            except (OSError, ValueError, ProtocolError) as error:
                self.failures += 1
                self.last_error = str(error)
                self.client.close()
                with self._lock:
                    self._latest = self._pending = None
                identity = None
            self.roundtrip_ms = (time.monotonic() - started) * 1000
            self._stop.wait(max(0.005, self.period - (time.monotonic() - started)))
        try:
            if self.client._socket is not None:
                self.client.cancel_targets()
        except (OSError, ValueError, ProtocolError):
            pass
        self.client.close()

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="通过桌面程序控制蛇形机器人")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state", help="读取全部实时状态")
    subparsers.add_parser("geometry", help="读取完整几何反馈")

    target = subparsers.add_parser("target", help="发送一个关节的目标角")
    target.add_argument("joint_index", type=int, help="从 1 开始的关节序号")
    target.add_argument("degrees", type=float, help="目标角（度）")

    target_two = subparsers.add_parser("target-two", help="兼容方式发送 J1/J2 目标角")
    target_two.add_argument("joint1_deg", type=float)
    target_two.add_argument("joint2_deg", type=float)

    limits = subparsers.add_parser("limits", help="设置一个关节的运行时限制")
    limits.add_argument("joint_index", type=int)
    limits.add_argument("--torque", type=int, help="0–1000")
    limits.add_argument("--acceleration", type=int, help="0–254")
    subparsers.add_parser("release", help="紧急释放全部扭矩")

    args = parser.parse_args()
    client = SnakeControlClient()
    if args.command == "geometry":
        result = client.get_geometry()
    elif args.command == "state":
        result = client.get_state()
    elif args.command == "target":
        result = client.set_joint_target(args.joint_index, args.degrees)
    elif args.command == "target-two":
        result = client.set_two_joint_target(args.joint1_deg, args.joint2_deg)
    elif args.command == "limits":
        result = client.set_joint_limits(
            args.joint_index,
            args.torque,
            args.acceleration,
        )
    else:
        result = client.release_torque()
    client.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
