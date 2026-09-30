using System;

namespace SnakeRobotVisualizer
{
    internal static class ProtocolPacketTests
    {
        private static int failures;

        private static void Expect(string name, byte[] actual, string expectedHex)
        {
            byte[] expected = ParseHex(expectedHex);
            if (BitConverter.ToString(actual) != BitConverter.ToString(expected))
            {
                failures++;
                Console.Error.WriteLine(name + " failed");
                Console.Error.WriteLine("actual:   " + BitConverter.ToString(actual));
                Console.Error.WriteLine("expected: " + BitConverter.ToString(expected));
            }
            else
            {
                Console.WriteLine(name + " passed");
            }
        }

        private static byte[] ParseHex(string value)
        {
            string[] parts = value.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            byte[] result = new byte[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                result[i] = Convert.ToByte(parts[i], 16);
            }
            return result;
        }

        private static int Main()
        {
            Expect(
                "manual read-position packet",
                FeetechBus.BuildInstructionPacket(1, 0x02, new byte[] { 0x38, 0x02 }),
                "FF FF 01 04 02 38 02 BE");

            Expect(
                "two-servo torque enable packet",
                FeetechBus.BuildSyncWritePacket(
                    0x28, 1,
                    new int[] { 3, 41 },
                    new byte[][] { new byte[] { 1 }, new byte[] { 1 } }),
                "FF FF FE 08 83 28 01 03 01 29 01 1F");

            Expect(
                "two-servo target packet",
                FeetechBus.BuildSyncWritePacket(
                    0x2A, 6,
                    new int[] { 3, 41 },
                    new byte[][]
                    {
                        new byte[] { 0xDA, 0x08, 0, 0, 0x55, 0x01 },
                        new byte[] { 0x0D, 0x08, 0, 0, 0x55, 0x01 }
                    }),
                "FF FF FE 12 83 2A 06 03 DA 08 00 00 55 01 29 0D 08 00 00 55 01 6D");

            Expect(
                "two-servo ten-percent torque packet",
                FeetechBus.BuildSyncWritePacket(
                    0x30, 2,
                    new int[] { 3, 41 },
                    new byte[][]
                    {
                        new byte[] { 0x64, 0 },
                        new byte[] { 0x64, 0 }
                    }),
                "FF FF FE 0A 83 30 02 03 64 00 29 64 00 4E");

            ExpectSuccess(
                "valid position status",
                delegate
                {
                    FeetechStatusPacket status = FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 04 00 DA 08 16"));
                    if (FeetechBus.ValidatePositionStatus(status, 3) != 2266)
                    {
                        throw new InvalidOperationException("位置解析结果错误");
                    }
                });
            ExpectFailure<InvalidOperationException>(
                "checksum error rejected",
                delegate
                {
                    FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 04 00 DA 08 17"));
                });
            ExpectFailure<TimeoutException>(
                "short packet rejected",
                delegate
                {
                    FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 04 00 DA"));
                });
            ExpectFailure<TimeoutException>(
                "empty response treated as timeout",
                delegate { FeetechBus.ParseStatusPacketBytes(new byte[0]); });
            ExpectFailure<InvalidOperationException>(
                "wrong servo ID rejected",
                delegate
                {
                    FeetechStatusPacket status = FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 04 00 DA 08 16"));
                    FeetechBus.ValidatePositionStatus(status, 41);
                });
            ExpectFailure<InvalidOperationException>(
                "wrong position payload length rejected",
                delegate
                {
                    FeetechStatusPacket status = FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 03 00 DA 1F"));
                    FeetechBus.ValidatePositionStatus(status, 3);
                });
            ExpectSuccess(
                "state block status (position, signed load, voltage, temperature, current)",
                delegate
                {
                    // 状态包与指令包同格式：FF FF ID LEN ERR 参数 CHK。负载 -168（bit10 符号），电流 -25（bit15 符号）。
                    byte[] packet = FeetechBus.BuildInstructionPacket(3, 0x00, new byte[]
                    {
                        0xDA, 0x08, 0x00, 0x00, 0xA8, 0x04, 0x7B, 0x24,
                        0x00, 0x20, 0x01, 0x00, 0x00, 0x19, 0x80
                    });
                    ServoReadResult result = FeetechBus.ParseStateStatus(
                        FeetechBus.ParseStatusPacketBytes(packet), 3);
                    ServoTelemetry t = result.Telemetry;
                    if (result.Position != 2266 || t.LoadPermille != -168 || Math.Abs(t.VoltageVolts - 12.3) > 1e-9 ||
                        t.TemperatureC != 36 || t.ServoStatus != 0x20 || !t.Moving || t.CurrentRaw != -25)
                    {
                        throw new InvalidOperationException("状态块解析结果错误");
                    }
                });
            Expect(
                "state block read request (56, 15 bytes)",
                FeetechBus.BuildInstructionPacket(1, 0x02, new byte[] { 0x38, (byte)FeetechBus.StateBlockLength }),
                "FF FF 01 04 02 38 0F B1");
            ExpectFailure<InvalidOperationException>(
                "short state block rejected",
                delegate
                {
                    FeetechStatusPacket status = FeetechBus.ParseStatusPacketBytes(
                        ParseHex("FF FF 03 04 00 DA 08 16"));
                    FeetechBus.ParseStateStatus(status, 3);
                });

            return failures == 0 ? 0 : 1;
        }

        private static void ExpectSuccess(string name, Action action)
        {
            try
            {
                action();
                Console.WriteLine(name + " passed");
            }
            catch (Exception ex)
            {
                failures++;
                Console.Error.WriteLine(name + " failed: " + ex.Message);
            }
        }

        private static void ExpectFailure<T>(string name, Action action)
            where T : Exception
        {
            try
            {
                action();
                failures++;
                Console.Error.WriteLine(name + " failed: no exception");
            }
            catch (T)
            {
                Console.WriteLine(name + " passed");
            }
            catch (Exception ex)
            {
                failures++;
                Console.Error.WriteLine(
                    name + " failed: expected " + typeof(T).Name +
                    ", got " + ex.GetType().Name);
            }
        }
    }
}
