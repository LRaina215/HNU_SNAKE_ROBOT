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
