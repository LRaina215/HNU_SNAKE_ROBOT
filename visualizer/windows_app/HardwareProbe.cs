using System;
using System.Collections.Generic;
using System.Threading;

namespace SnakeRobotVisualizer
{
    internal static class HardwareProbe
    {
        private static int Main(string[] args)
        {
            string port = args.Length > 0 ? args[0] : "COM9";
            int id1 = args.Length > 1 ? Int32.Parse(args[1]) : 3;
            int id2 = args.Length > 2 ? Int32.Parse(args[2]) : 41;
            try
            {
                using (FeetechBus bus = new FeetechBus(port, 115200))
                {
                    bus.Open();
                    for (int sample = 1; sample <= 5; sample++)
                    {
                        Dictionary<int, int> values = bus.ReadPositions(new int[] { id1, id2 });
                        Console.WriteLine(
                            "Sample {0}: ID {1}={2}, ID {3}={4}",
                            sample, id1, values[id1], id2, values[id2]);
                        Thread.Sleep(100);
                    }
                }
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.ToString());
                return 1;
            }
        }
    }
}
