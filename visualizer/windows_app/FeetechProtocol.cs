using System;
using System.Collections.Generic;
using System.IO.Ports;

namespace SnakeRobotVisualizer
{
    internal sealed class FeetechStatusPacket
    {
        public int Id;
        public int Error;
        public byte[] Parameters;
    }

    internal sealed class ServoReadResult
    {
        public int Id;
        public bool Success;
        public int Position;
        public string Error;
        // 仅状态块读取成功时非空；位置读取成功而状态块失败时 TelemetryFailed 为真。
        public ServoTelemetry Telemetry;
        public bool TelemetryFailed;
    }

    // SMS/STS 内存表地址 56–70 一次读出；负载与电流只作相对指标，不换算为力矩。
    internal sealed class ServoTelemetry
    {
        public int LoadPermille;     // 地址 60，bit10 为符号，0.1% 最大输出
        public double VoltageVolts;  // 地址 62，0.1 V
        public int TemperatureC;     // 地址 63
        public int ServoStatus;      // 地址 65，飞特状态位（电压、角度、过热、过流、过载等）
        public bool Moving;          // 地址 66
        public int CurrentRaw;       // 地址 69，bit15 为符号，原始单位（SM 系列手册未给换算）
    }

    // 可选能力：一次读出位置与状态；伪总线可不实现，运行时退回只读位置。
    internal interface IServoTelemetryBus
    {
        ServoReadResult[] ReadStatesTolerant(int[] servoIds);
    }

    internal interface IServoBus : IDisposable
    {
        bool IsOpen { get; }
        void Open();
        void Close();
        ServoReadResult[] ReadPositionsTolerant(int[] servoIds);
        void SetTorqueEnabled(int[] servoIds, bool enabled);
        void SetTorqueLimit(int[] servoIds, int limit);
        void SetAcceleration(int[] servoIds, int acceleration);
        void SetTargetPositions(int[] servoIds, int[] targets, int speedStepsPerSecond);
        void EnableSafely(
            int[] servoIds,
            int[] currentPositions,
            int torqueLimit,
            int speedStepsPerSecond,
            int acceleration);
    }

    internal sealed class FeetechBus : IServoBus, IServoTelemetryBus
    {
        private const int TorqueEnableAddress = 0x28;
        private const int AccelerationAddress = 0x29;
        private const int TargetPositionAddress = 0x2A;
        private const int RunningSpeedAddress = 0x2E;
        private const int TorqueLimitAddress = 0x30;
        private const int PresentPositionAddress = 0x38;
        private const int OperatingModeAddress = 0x21;
        internal const int StateBlockLength = 15;  // 地址 56–70
        private readonly SerialPort port;

        public FeetechBus(string portName, int baudRate, int readTimeoutMilliseconds = 60)
        {
            port = new SerialPort(portName, baudRate, Parity.None, 8, StopBits.One);
            port.ReadTimeout = Math.Max(5, Math.Min(1000, readTimeoutMilliseconds));
            port.WriteTimeout = 100;
            port.DtrEnable = false;
            port.RtsEnable = false;
        }

        public bool IsOpen { get { return port.IsOpen; } }

        public void Open()
        {
            port.Open();
        }

        public void Close()
        {
            if (port.IsOpen)
            {
                port.Close();
            }
        }

        public int ReadPosition(int servoId)
        {
            if (servoId < 0 || servoId > 253)
            {
                throw new ArgumentOutOfRangeException("servoId");
            }

            port.DiscardInBuffer();
            byte[] request = BuildReadPacket(servoId, PresentPositionAddress, 2);
            port.Write(request, 0, request.Length);
            FeetechStatusPacket status = ReadStatusPacket();

            return ValidatePositionStatus(status, servoId);
        }

        public bool Ping(int servoId)
        {
            if (servoId < 0 || servoId > 253)
            {
                return false;
            }
            try
            {
                port.DiscardInBuffer();
                byte[] request = BuildInstructionPacket(servoId, 0x01, new byte[0]);
                port.Write(request, 0, request.Length);
                FeetechStatusPacket status = ReadStatusPacket();
                return status.Id == servoId && status.Error == 0;
            }
            catch
            {
                return false;
            }
        }

        public ServoReadResult TryReadPosition(int servoId)
        {
            try
            {
                return new ServoReadResult
                {
                    Id = servoId,
                    Success = true,
                    Position = ReadPosition(servoId),
                    Error = String.Empty
                };
            }
            catch (Exception ex)
            {
                try { port.DiscardInBuffer(); } catch { }
                return new ServoReadResult
                {
                    Id = servoId,
                    Success = false,
                    Error = ex.Message
                };
            }
        }

        public ServoReadResult[] ReadStatesTolerant(int[] servoIds)
        {
            ServoReadResult[] result = new ServoReadResult[servoIds.Length];
            for (int i = 0; i < servoIds.Length; i++)
            {
                try
                {
                    port.DiscardInBuffer();
                    byte[] request = BuildReadPacket(servoIds[i], PresentPositionAddress, StateBlockLength);
                    port.Write(request, 0, request.Length);
                    result[i] = ParseStateStatus(ReadStatusPacket(), servoIds[i]);
                }
                catch (Exception ex)
                {
                    try { port.DiscardInBuffer(); } catch { }
                    // 状态块失败时仍保证本周期的位置反馈。
                    result[i] = TryReadPosition(servoIds[i]);
                    if (result[i].Success)
                    {
                        result[i].TelemetryFailed = true;
                        result[i].Error = "状态块读取失败：" + ex.Message;
                    }
                }
            }
            return result;
        }

        internal static ServoReadResult ParseStateStatus(FeetechStatusPacket status, int expectedId)
        {
            if (status == null)
            {
                throw new TimeoutException("没有收到舵机应答");
            }
            if (status.Id != expectedId)
            {
                throw new InvalidOperationException(
                    String.Format("请求 ID {0}，但收到 ID {1}", expectedId, status.Id));
            }
            if (status.Error != 0)
            {
                throw new InvalidOperationException(
                    String.Format("舵机 {0} 返回错误 0x{1:X2}", expectedId, status.Error));
            }
            byte[] p = status.Parameters;
            if (p == null || p.Length != StateBlockLength)
            {
                throw new InvalidOperationException("状态块应答长度无效");
            }
            return new ServoReadResult
            {
                Id = expectedId,
                Success = true,
                Position = p[0] | (p[1] << 8),
                Error = String.Empty,
                Telemetry = new ServoTelemetry
                {
                    LoadPermille = SignMagnitude(p[4] | (p[5] << 8), 10),
                    VoltageVolts = p[6] / 10.0,
                    TemperatureC = p[7],
                    ServoStatus = p[9],
                    Moving = p[10] != 0,
                    CurrentRaw = SignMagnitude(p[13] | (p[14] << 8), 15)
                }
            };
        }

        private static int SignMagnitude(int raw, int signBit)
        {
            int magnitude = raw & ((1 << signBit) - 1);
            return (raw & (1 << signBit)) != 0 ? -magnitude : magnitude;
        }

        public int ReadByte(int servoId, int address)
        {
            port.DiscardInBuffer();
            byte[] request = BuildReadPacket(servoId, address, 1);
            port.Write(request, 0, request.Length);
            FeetechStatusPacket status = ReadStatusPacket();
            if (status.Id != servoId || status.Error != 0 || status.Parameters.Length != 1)
            {
                throw new InvalidOperationException(
                    String.Format("读取舵机 {0} 地址 0x{1:X2} 失败", servoId, address));
            }
            return status.Parameters[0];
        }

        public Dictionary<int, int> ReadPositions(int[] servoIds)
        {
            Dictionary<int, int> result = new Dictionary<int, int>();
            foreach (int id in servoIds)
            {
                result[id] = ReadPosition(id);
            }
            return result;
        }

        public ServoReadResult[] ReadPositionsTolerant(int[] servoIds)
        {
            ServoReadResult[] result = new ServoReadResult[servoIds.Length];
            for (int i = 0; i < servoIds.Length; i++)
            {
                result[i] = TryReadPosition(servoIds[i]);
            }
            return result;
        }

        public void VerifyPositionMode(int[] servoIds)
        {
            foreach (int id in servoIds)
            {
                int mode = ReadByte(id, OperatingModeAddress);
                if (mode != 0)
                {
                    throw new InvalidOperationException(
                        String.Format("舵机 {0} 当前运行模式为 {1}，必须在 FD 中设置为位置伺服模式 0", id, mode));
                }
            }
        }

        public void SetTorqueEnabled(int[] servoIds, bool enabled)
        {
            Dictionary<int, byte[]> values = new Dictionary<int, byte[]>();
            foreach (int id in servoIds)
            {
                values[id] = new byte[] { enabled ? (byte)1 : (byte)0 };
            }
            WriteSync(TorqueEnableAddress, 1, servoIds, values);
        }

        public void SetTorqueLimit(int[] servoIds, int limit)
        {
            if (limit < 0 || limit > 1000)
            {
                throw new ArgumentOutOfRangeException("limit", "转矩限制必须在 0–1000 之间");
            }
            Dictionary<int, byte[]> values = new Dictionary<int, byte[]>();
            foreach (int id in servoIds)
            {
                values[id] = LittleEndian16(limit);
            }
            WriteSync(TorqueLimitAddress, 2, servoIds, values);
        }

        public void SetAcceleration(int[] servoIds, int acceleration)
        {
            if (acceleration < 0 || acceleration > 254)
            {
                throw new ArgumentOutOfRangeException("acceleration");
            }
            Dictionary<int, byte[]> values = new Dictionary<int, byte[]>();
            foreach (int id in servoIds)
            {
                values[id] = new byte[] { (byte)acceleration };
            }
            WriteSync(AccelerationAddress, 1, servoIds, values);
        }

        public void SetTargetPositions(int[] servoIds, int[] targets, int speedStepsPerSecond)
        {
            if (servoIds.Length != targets.Length)
            {
                throw new ArgumentException("舵机 ID 与目标位置数量不一致");
            }
            if (speedStepsPerSecond < 1 || speedStepsPerSecond > 32766)
            {
                throw new ArgumentOutOfRangeException("speedStepsPerSecond");
            }

            Dictionary<int, byte[]> values = new Dictionary<int, byte[]>();
            for (int i = 0; i < servoIds.Length; i++)
            {
                if (targets[i] < 0 || targets[i] > 4095)
                {
                    throw new ArgumentOutOfRangeException("targets", "单圈目标位置必须在 0–4095 之间");
                }
                byte[] target = LittleEndian16(targets[i]);
                byte[] speed = LittleEndian16(speedStepsPerSecond);
                values[servoIds[i]] = new byte[]
                {
                    target[0], target[1],
                    0, 0,
                    speed[0], speed[1]
                };
            }
            WriteSync(TargetPositionAddress, RunningSpeedAddress - TargetPositionAddress + 2, servoIds, values);
        }

        public void EnableSafely(
            int[] servoIds,
            int[] currentPositions,
            int torqueLimit,
            int speedStepsPerSecond,
            int acceleration)
        {
            VerifyPositionMode(servoIds);
            SetTargetPositions(servoIds, currentPositions, speedStepsPerSecond);
            SetTorqueLimit(servoIds, torqueLimit);
            SetAcceleration(servoIds, acceleration);
            SetTorqueEnabled(servoIds, true);
        }

        private void WriteSync(
            int address,
            int dataLength,
            int[] servoIds,
            Dictionary<int, byte[]> values)
        {
            if (servoIds.Length == 0)
            {
                throw new ArgumentException("至少需要一个舵机 ID");
            }
            List<byte[]> orderedValues = new List<byte[]>();
            foreach (int id in servoIds)
            {
                byte[] data;
                if (!values.TryGetValue(id, out data) || data.Length != dataLength)
                {
                    throw new ArgumentException("同步写数据长度错误");
                }
                orderedValues.Add(data);
            }
            byte[] packet = BuildSyncWritePacket(
                address,
                dataLength,
                servoIds,
                orderedValues.ToArray());
            port.Write(packet, 0, packet.Length);
            port.BaseStream.Flush();
        }

        internal static byte[] BuildSyncWritePacket(
            int address,
            int dataLength,
            int[] servoIds,
            byte[][] values)
        {
            if (servoIds.Length == 0 || servoIds.Length != values.Length)
            {
                throw new ArgumentException("同步写 ID 与数据数量不一致");
            }
            List<byte> parameters = new List<byte>();
            parameters.Add((byte)address);
            parameters.Add((byte)dataLength);
            for (int i = 0; i < servoIds.Length; i++)
            {
                if (values[i] == null || values[i].Length != dataLength)
                {
                    throw new ArgumentException("同步写数据长度错误");
                }
                parameters.Add((byte)servoIds[i]);
                parameters.AddRange(values[i]);
            }
            return BuildInstructionPacket(0xFE, 0x83, parameters.ToArray());
        }

        private static byte[] LittleEndian16(int value)
        {
            return new byte[] { (byte)(value & 0xFF), (byte)((value >> 8) & 0xFF) };
        }

        private static byte[] BuildReadPacket(int servoId, int address, int size)
        {
            return BuildInstructionPacket(
                servoId,
                0x02,
                new byte[] { (byte)address, (byte)size });
        }

        internal static byte[] BuildInstructionPacket(int servoId, int instruction, byte[] parameters)
        {
            int length = parameters.Length + 2;
            byte[] body = new byte[3 + parameters.Length];
            body[0] = (byte)servoId;
            body[1] = (byte)length;
            body[2] = (byte)instruction;
            if (parameters.Length > 0)
            {
                Buffer.BlockCopy(parameters, 0, body, 3, parameters.Length);
            }
            byte[] packet = new byte[body.Length + 3];
            packet[0] = 0xFF;
            packet[1] = 0xFF;
            Buffer.BlockCopy(body, 0, packet, 2, body.Length);
            packet[packet.Length - 1] = Checksum(body, 0, body.Length);
            return packet;
        }

        internal static FeetechStatusPacket ParseStatusPacketBytes(byte[] packet)
        {
            if (packet == null || packet.Length == 0)
            {
                throw new TimeoutException("等待舵机数据超时");
            }
            if (packet.Length < 4)
            {
                throw new TimeoutException("舵机应答为短包");
            }
            if (packet[0] != 0xFF || packet[1] != 0xFF)
            {
                throw new InvalidOperationException("舵机应答包头无效");
            }
            int length = packet[3];
            if (length < 2 || length > 250)
            {
                throw new InvalidOperationException("舵机应答长度字段无效");
            }
            int expectedTotal = length + 4;
            if (packet.Length < expectedTotal)
            {
                throw new TimeoutException("舵机应答为短包");
            }
            if (packet.Length != expectedTotal)
            {
                throw new InvalidOperationException("舵机应答包含多余数据");
            }
            byte expectedChecksum = Checksum(packet, 2, packet.Length - 3);
            if (expectedChecksum != packet[packet.Length - 1])
            {
                throw new InvalidOperationException("舵机应答校验和错误");
            }
            byte[] parameters = new byte[length - 2];
            if (parameters.Length > 0)
            {
                Buffer.BlockCopy(packet, 5, parameters, 0, parameters.Length);
            }
            return new FeetechStatusPacket
            {
                Id = packet[2],
                Error = packet[4],
                Parameters = parameters
            };
        }

        internal static int ValidatePositionStatus(FeetechStatusPacket status, int expectedId)
        {
            if (status == null)
            {
                throw new TimeoutException("没有收到舵机应答");
            }
            if (status.Id != expectedId)
            {
                throw new InvalidOperationException(
                    String.Format("请求 ID {0}，但收到 ID {1}", expectedId, status.Id));
            }
            if (status.Error != 0)
            {
                throw new InvalidOperationException(
                    String.Format("舵机 {0} 返回错误 0x{1:X2}", expectedId, status.Error));
            }
            if (status.Parameters == null || status.Parameters.Length != 2)
            {
                throw new InvalidOperationException("当前位置应答长度无效");
            }
            return status.Parameters[0] | (status.Parameters[1] << 8);
        }

        private FeetechStatusPacket ReadStatusPacket()
        {
            int previous = -1;
            while (true)
            {
                int current = port.ReadByte();
                if (previous == 0xFF && current == 0xFF)
                {
                    int id = port.ReadByte();
                    int length = port.ReadByte();
                    if (length < 2 || length > 250)
                    {
                        previous = current;
                        continue;
                    }

                    byte[] remainder = new byte[length];
                    ReadExactly(remainder, 0, remainder.Length);
                    byte[] packet = new byte[length + 4];
                    packet[0] = 0xFF;
                    packet[1] = 0xFF;
                    packet[2] = (byte)id;
                    packet[3] = (byte)length;
                    Buffer.BlockCopy(remainder, 0, packet, 4, remainder.Length);
                    return ParseStatusPacketBytes(packet);
                }
                previous = current;
            }
        }

        private void ReadExactly(byte[] buffer, int offset, int count)
        {
            int total = 0;
            while (total < count)
            {
                int read = port.Read(buffer, offset + total, count - total);
                if (read <= 0)
                {
                    throw new TimeoutException("等待舵机数据超时");
                }
                total += read;
            }
        }

        private static byte Checksum(byte[] values, int offset, int count)
        {
            int sum = 0;
            for (int i = offset; i < offset + count; i++)
            {
                sum = (sum + values[i]) & 0xFF;
            }
            return (byte)(~sum & 0xFF);
        }

        public void Dispose()
        {
            Close();
            port.Dispose();
        }
    }
}
