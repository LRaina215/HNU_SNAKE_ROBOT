using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace SnakeRobotVisualizer
{
    internal static class SoakTests
    {
        private static int Main(string[] args)
        {
            int seconds = args.Length > 0 ? Int32.Parse(args[0]) : 15;
            string root = Path.Combine(
                Path.GetTempPath(),
                "SnakeRobotVisualizer_Soak_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            RobotProfile profile = RobotProfile.CreateAlternating(28, "28关节持续运行");
            try
            {
                using (RobotRuntime runtime = new RobotRuntime())
                using (ExperimentRecorder recorder = new ExperimentRecorder())
                {
                    runtime.Start(profile, RuntimeMode.SimulationAuto);
                    string folder = recorder.Start(root, "持续记录", profile);
                    Stopwatch watch = Stopwatch.StartNew();
                    long last = -1;
                    while (watch.Elapsed.TotalSeconds < seconds)
                    {
                        RobotFrame frame;
                        if (runtime.TryGetLatest(out frame) && frame.Sequence != last)
                        {
                            recorder.Append(frame, profile);
                            last = frame.Sequence;
                        }
                        Thread.Sleep(10);
                    }
                    RuntimeStatistics statistics = runtime.GetStatistics();
                    runtime.Stop();
                    recorder.Stop();
                    PlaybackData playback = ExperimentPlayback.Load(folder);
                    Console.WriteLine(
                        "duration={0}s frames={1} playback={2} rate={3:0.0}",
                        seconds,
                        recorder.FrameCount,
                        playback.Frames.Count,
                        statistics.SamplesPerSecond);
                    if (recorder.FrameCount < seconds * 8 ||
                        playback.Frames.Count != recorder.FrameCount ||
                        statistics.SamplesPerSecond < 8 ||
                        statistics.SamplesPerSecond > 13)
                    {
                        Console.Error.WriteLine("SOAK TEST FAILED");
                        return 1;
                    }
                }
                Console.WriteLine("SOAK TEST PASSED");
                return 0;
            }
            finally
            {
                if (Directory.Exists(root))
                {
                    Directory.Delete(root, true);
                }
            }
        }
    }
}
