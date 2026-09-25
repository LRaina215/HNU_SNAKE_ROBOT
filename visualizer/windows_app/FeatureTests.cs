using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal static class FeatureTests
    {
        private static int failures;

        [STAThread]
        private static int Main()
        {
            TestProfiles();
            TestOfflineAndRecovery();
            TestSerialOwnerSafety();
            TestTwentyEightJointLivePath();
            TestSimulationRuntime();
            TestMujocoWorldPose();
            TestRecordingAndPlayback();
            TestRenderers();
            TestTwentyEightJointPerformance();
            Console.WriteLine(failures == 0
                ? "ALL FEATURE TESTS PASSED"
                : failures + " FEATURE TEST(S) FAILED");
            return failures == 0 ? 0 : 1;
        }

        private static void TestSerialOwnerSafety()
        {
            RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
            FakeServoBus fake = new FakeServoBus();
            using (RobotRuntime runtime = new RobotRuntime(delegate { return fake; }))
            {
                runtime.Start(profile, RuntimeMode.Live);
                RobotFrame frame = WaitForFrame(runtime, 2, 1500);
                Check(frame != null && frame.Joints.All(x => x.Online),
                    "fake live bus feeds all joints through single owner");
                runtime.QueueEnableTorque(300, 800, 20);
                Check(WaitUntil(delegate { return runtime.TorqueActive; }, 1500),
                    "safe enable command processed by serial-owner queue");
                runtime.QueueTarget(0, 999, 800, 300, 20);
                Check(WaitUntil(delegate { return fake.TargetCalls > 0; }, 1500),
                    "target command processed by serial-owner queue");
                Check(fake.LastTargets != null &&
                    fake.LastTargets[0] == Kinematics.DegreesToRaw(90, profile.Joints[0]),
                    "queued target clamped to joint safety range");
                runtime.Stop();
                Check(fake.DisableCalls > 0,
                    "runtime stop releases torque before closing bus");
                Check(!runtime.TorqueActive && !fake.IsOpen,
                    "runtime stop clears torque state and closes bus");
            }
        }

        private static void TestTwentyEightJointLivePath()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(28, "28关节实机通道测试");
            FakeServoBus fake = new FakeServoBus();
            using (RobotRuntime runtime = new RobotRuntime(delegate { return fake; }))
            {
                runtime.Start(profile, RuntimeMode.Live);
                RobotFrame frame = WaitForFrame(runtime, 25, 4000);
                RuntimeStatistics statistics = runtime.GetStatistics();
                Check(frame != null && frame.Joints.Length == 28 &&
                    frame.Points.Length == 30 &&
                    frame.Joints.All(x => x.Online),
                    "28-ID live polling path produces a complete latest frame");
                Check(statistics.SamplesPerSecond >= 8.0 &&
                    statistics.SamplesPerSecond <= 12.0,
                    "28-ID live polling path maintains approximately 10 Hz");
                runtime.QueueEnableTorque(250, 700, 15);
                Check(WaitUntil(delegate { return runtime.TorqueActive; }, 1500),
                    "28-ID live path processes safe-enable queue");
                runtime.QueueTarget(27, 32.0, 700, 250, 15);
                Check(WaitUntil(delegate { return fake.TargetCalls > 0; }, 1500) &&
                    fake.LastServoIds != null &&
                    fake.LastServoIds.Length == 1 &&
                    fake.LastServoIds[0] == 28,
                    "28th joint target maps to configured servo ID 28");
                runtime.Stop();
                Check(fake.DisableCalls > 0,
                    "28-ID live path releases torque on stop");
            }
        }

        private static void TestOfflineAndRecovery()
        {
            RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
            JointRuntimeState[] states =
            {
                new JointRuntimeState(),
                new JointRuntimeState()
            };
            long errors = 0;
            int online = RobotRuntime.ApplyReadResults(
                states,
                profile,
                new ServoReadResult[]
                {
                    new ServoReadResult { Id = 3, Success = true, Position = 2100 },
                    new ServoReadResult { Id = 41, Success = true, Position = 2000 }
                },
                DateTime.UtcNow,
                ref errors);
            Check(online == 2 && states.All(x => x.Online),
                "all successful servo reads become online");
            online = RobotRuntime.ApplyReadResults(
                states,
                profile,
                new ServoReadResult[]
                {
                    new ServoReadResult { Id = 3, Success = false, Error = "timeout" },
                    new ServoReadResult { Id = 41, Success = true, Position = 2001 }
                },
                DateTime.UtcNow,
                ref errors);
            Check(online == 1 && !states[0].Online && states[1].Online,
                "one joint timeout does not block other joint");
            Check(errors == 1 && states[0].Status == "超时",
                "timeout status and error count recorded");
            RobotRuntime.ApplyReadResults(
                states,
                profile,
                new ServoReadResult[]
                {
                    new ServoReadResult { Id = 3, Success = true, Position = 2102 },
                    new ServoReadResult { Id = 41, Success = true, Position = 2002 }
                },
                DateTime.UtcNow,
                ref errors);
            Check(states[0].Online && states[0].Status == "已恢复",
                "offline joint recovery recorded");
        }

        private static void TestProfiles()
        {
            string folder = Path.Combine(
                Path.GetTempPath(),
                "SnakeRobotVisualizer_ProfileTest_" + Guid.NewGuid().ToString("N"));
            try
            {
                ProfileStore store = new ProfileStore(folder);
                store.EnsureDefaultProfiles();
                RobotProfile[] profiles = store.LoadProfiles().ToArray();
                Check(profiles.Any(x => x.Joints.Count == 2), "default two-joint profile persisted");
                Check(profiles.Any(x => x.Joints.Count == 28), "default 28-joint profile persisted");
                RobotProfile duplicate = store.Duplicate(profiles[0], "测试副本");
                Check(store.LoadProfile("测试副本") != null, "profile duplicate persisted");
                ApplicationState state = new ApplicationState
                {
                    LastProfile = duplicate.Name,
                    Display = new DisplaySettings
                    {
                        Theme = ThemeMode.Light,
                        DisplayMode = DisplayMode.Split,
                        ShowTarget = true
                    }
                };
                store.SaveApplicationState(state);
                ApplicationState loaded = store.LoadApplicationState();
                Check(loaded.LastProfile == "测试副本", "last profile persisted");
                Check(loaded.Display.DisplayMode == DisplayMode.Split, "display state persisted");
            }
            finally
            {
                if (Directory.Exists(folder))
                {
                    Directory.Delete(folder, true);
                }
            }
        }

        private static void TestSimulationRuntime()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(28, "运行时测试");
            using (RobotRuntime runtime = new RobotRuntime())
            {
                for (int pass = 0; pass < 3; pass++)
                {
                    runtime.Start(profile, RuntimeMode.SimulationAuto);
                    RobotFrame frame = WaitForFrame(runtime, 8, 2500);
                    Check(frame != null, "automatic simulation publishes latest frame pass " + pass);
                    Check(frame != null && frame.Joints.Length == 28,
                        "automatic simulation keeps 28 joints pass " + pass);
                    Check(frame != null && frame.Points.Length == 30,
                        "automatic simulation produces 30 skeleton points pass " + pass);
                    runtime.Stop();
                    Check(runtime.Mode == RuntimeMode.Idle,
                        "simulation stops cleanly pass " + pass);
                }

                runtime.Start(profile, RuntimeMode.SimulationManual);
                runtime.SetManualSimulationAngle(7, 23.5);
                RobotFrame manual = WaitForFrame(runtime, 2, 1200);
                Check(manual != null && Math.Abs(manual.Joints[7].AngleDegrees - 23.5) < 0.01,
                    "manual simulation updates selected joint");
                runtime.SetTargetPreview(0, 20);
                RobotFrame targetPreview = WaitForFrame(runtime, manual.Sequence + 2, 1200);
                Check(targetPreview != null &&
                    (targetPreview.TargetPoints[targetPreview.TargetPoints.Length - 1] -
                     targetPreview.Points[targetPreview.Points.Length - 1]).Length > 1,
                    "measured and target poses are computed separately");
                runtime.SetAutomaticSimulationJoint(7);
                runtime.SetAutomaticSimulation(true);
                RobotFrame selected = WaitForFrame(runtime, manual.Sequence + 4, 1200);
                Check(selected != null && Math.Abs(selected.Joints[0].AngleDegrees) < 0.001,
                    "selected-only automatic simulation holds other joints");
                runtime.Stop();
            }
        }

        private static void TestMujocoWorldPose()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(2, "MuJoCo 世界位姿测试");
            profile.Joints[0].AxisType = JointAxisType.Pitch;
            profile.Joints[1].AxisType = JointAxisType.Pitch;
            double[] angles = { 30, -15 };
            double[] targets = { 20, -10 };
            Point3D origin = new Point3D(120, -50, 30);
            Matrix3 basis = new Matrix3 { M11 = 1, M23 = -1, M32 = 1 };
            using (RobotRuntime runtime = new RobotRuntime())
            {
                runtime.Start(profile, RuntimeMode.Mujoco);
                runtime.UpdateMujocoPose(angles, targets, origin, basis);
                RobotFrame frame;
                bool available = runtime.TryGetLatest(out frame);
                KinematicResult local = Kinematics.Forward(profile, angles);
                KinematicResult localTarget = Kinematics.Forward(profile, targets);
                Check(available && frame != null &&
                    frame.Points.Select((point, i) =>
                        (point - origin - basis.Transform(local.Points[i])).Length).Max() < 1e-8,
                    "MuJoCo actual skeleton uses joint angles plus free-root world pose");
                Check(frame != null && frame.Points.All(point => Math.Abs(point.Z - 30) < 1e-8),
                    "MuJoCo planar posture remains planar in world coordinates");
                Check(frame != null &&
                    frame.TargetPoints.Select((point, i) =>
                        (point - origin - basis.Transform(localTarget.Points[i])).Length).Max() < 1e-8,
                    "MuJoCo target overlay shares the same free-root world pose");
            }
        }

        private static void TestRecordingAndPlayback()
        {
            string root = Path.Combine(
                Path.GetTempPath(),
                "SnakeRobotVisualizer_RecordTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
                string folder;
                using (ExperimentRecorder recorder = new ExperimentRecorder())
                {
                    DisplaySettings recordedDisplay = new DisplaySettings
                    {
                        Theme = ThemeMode.Light,
                        DisplayMode = DisplayMode.Split,
                        ShowTarget = true
                    };
                    folder = recorder.Start(root, "记录测试", profile, recordedDisplay);
                    for (int i = 0; i < 4; i++)
                    {
                        double[] angles = { i * 2.0, -i * 1.5 };
                        KinematicResult actual = Kinematics.Forward(profile, angles);
                        RobotFrame frame = new RobotFrame
                        {
                            TimestampUtc = DateTime.UtcNow.AddMilliseconds(i * 100),
                            Sequence = i + 1,
                            Source = "测试",
                            Joints = new JointRuntimeState[]
                            {
                                new JointRuntimeState
                                {
                                    Raw = Kinematics.DegreesToRaw(angles[0], profile.Joints[0]),
                                    AngleDegrees = angles[0],
                                    TargetDegrees = angles[0] + 1,
                                    Online = i != 1,
                                    Status = i == 1 ? "超时" : i == 2 ? "已恢复" : "在线"
                                },
                                new JointRuntimeState
                                {
                                    Raw = Kinematics.DegreesToRaw(angles[1], profile.Joints[1]),
                                    AngleDegrees = angles[1],
                                    TargetDegrees = angles[1] - 1,
                                    Online = true,
                                    Status = "在线"
                                }
                            },
                            Points = actual.Points,
                            TargetPoints = actual.Points,
                            JointOrientations = actual.JointOrientations,
                            RollingAxis = new RollingAxisState
                            {
                                Point = new Point3D(i, 2, 3),
                                Direction = new Point3D(1, 0, 0),
                                Confidence = 0.83,
                                Valid = true,
                                Locked = i == 2,
                                Manual = false
                            }
                        };
                        Check(recorder.Append(frame, profile), "record frame " + (i + 1));
                    }
                    recorder.Stop();
                }
                Check(File.Exists(Path.Combine(folder, "frames.csv")), "CSV written");
                Check(File.Exists(Path.Combine(folder, "metadata.json")), "JSON metadata written");
                PlaybackData playback = ExperimentPlayback.Load(folder);
                Check(playback.Frames.Count == 4, "playback loads all frames");
                Check(playback.Profile.Joints.Count == 2, "playback restores profile");
                Check(playback.Display.DisplayMode == DisplayMode.Split &&
                    playback.Display.Theme == ThemeMode.Light &&
                    playback.Display.ShowTarget,
                    "playback metadata restores recorded display settings");
                Check(Math.Abs(playback.Frames[3].Joints[0].AngleDegrees - 6.0) < 0.001,
                    "playback restores measured angle");
                Check(Math.Abs(playback.Frames[2].RollingAxis.Confidence - 0.83) < 0.001,
                    "playback restores rolling axis confidence");
                Check(playback.Frames[2].RollingAxis.Locked,
                    "playback restores rolling axis lock state");
                Check(!playback.Frames[1].Joints[0].Online &&
                    playback.Frames[1].Joints[0].Status == "超时" &&
                    playback.Frames[2].Joints[0].Status == "已恢复",
                    "recording preserves dropout and recovery states");
            }
            finally
            {
                if (Directory.Exists(root))
                {
                    Directory.Delete(root, true);
                }
            }
        }

        private static void TestRenderers()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(28, "渲染测试");
            double[] angles = Enumerable.Range(0, 28)
                .Select(i => 28 * Math.Sin(i * 0.42)).ToArray();
            KinematicResult kinematics = Kinematics.Forward(profile, angles);
            RobotFrame frame = new RobotFrame
            {
                Points = kinematics.Points,
                TargetPoints = Kinematics.Forward(
                    profile,
                    angles.Select(x => x * 0.85).ToArray()).Points,
                JointOrientations = kinematics.JointOrientations,
                Joints = angles.Select(x => new JointRuntimeState
                {
                    AngleDegrees = x,
                    TargetDegrees = x * 0.85,
                    Online = true,
                    Status = "模拟"
                }).ToArray(),
                RollingAxis = RollingAxisEstimator.Estimate(kinematics.Points)
            };
            DisplaySettings settings = new DisplaySettings
            {
                Theme = ThemeMode.Dark,
                ShowWorldAxes = true,
                ShowRollingAxis = true,
                ShowLocalAxes = true,
                ShowGrid = true,
                ShowNodes = true,
                ShowLabels = true,
                ShowTarget = true,
                ShowPipeOutline = true,
                PipeEllipse = true,
                PipeRadiusX = 170,
                PipeRadiusY = 130
            };
            using (SnakeCanvas canvas = new SnakeCanvas())
            using (ProjectionCanvas projection = new ProjectionCanvas())
            using (Bitmap image3D = new Bitmap(900, 650))
            using (Bitmap image2D = new Bitmap(650, 650))
            {
                canvas.Size = image3D.Size;
                projection.Size = image2D.Size;
                canvas.SetFrame(frame, profile, settings);
                projection.SetFrame(frame, settings);
                canvas.DrawToBitmap(image3D, new Rectangle(Point.Empty, image3D.Size));
                projection.DrawToBitmap(image2D, new Rectangle(Point.Empty, image2D.Size));
                Check(HasNonBackgroundPixels(image3D), "3D renderer draws 28-joint skeleton");
                Check(HasNonBackgroundPixels(image2D), "rolling projection renderer draws skeleton");
            }
        }

        private static void TestTwentyEightJointPerformance()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(28, "性能测试");
            Stopwatch watch = Stopwatch.StartNew();
            RollingAxisEstimator estimator = new RollingAxisEstimator();
            for (int frame = 0; frame < 1000; frame++)
            {
                double[] angles = Enumerable.Range(0, 28)
                    .Select(i => 35.0 * Math.Sin(frame * 0.02 + i * 0.35)).ToArray();
                KinematicResult kinematics = Kinematics.Forward(profile, angles);
                RollingAxisState axis = estimator.Update(kinematics.Points);
                RollingProjection.Project(kinematics.Points, axis);
            }
            watch.Stop();
            Check(watch.ElapsedMilliseconds < 5000,
                "1000 frames of 28-joint kinematics/axis/projection under 5 seconds");
            Console.WriteLine("28-joint core: 1000 frames in " + watch.ElapsedMilliseconds + " ms");
        }

        private static RobotFrame WaitForFrame(
            RobotRuntime runtime,
            long minimumSequence,
            int timeoutMilliseconds)
        {
            Stopwatch watch = Stopwatch.StartNew();
            RobotFrame frame;
            while (watch.ElapsedMilliseconds < timeoutMilliseconds)
            {
                if (runtime.TryGetLatest(out frame) && frame.Sequence >= minimumSequence)
                {
                    return frame;
                }
                Thread.Sleep(20);
            }
            return null;
        }

        private static bool WaitUntil(Func<bool> condition, int timeoutMilliseconds)
        {
            Stopwatch watch = Stopwatch.StartNew();
            while (watch.ElapsedMilliseconds < timeoutMilliseconds)
            {
                if (condition())
                {
                    return true;
                }
                Thread.Sleep(20);
            }
            return false;
        }

        private static bool HasNonBackgroundPixels(Bitmap bitmap)
        {
            Color corner = bitmap.GetPixel(0, 0);
            for (int y = 20; y < bitmap.Height; y += 31)
            {
                for (int x = 20; x < bitmap.Width; x += 31)
                {
                    if (ColorDistance(corner, bitmap.GetPixel(x, y)) > 25)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        private static int ColorDistance(Color a, Color b)
        {
            return Math.Abs(a.R - b.R) + Math.Abs(a.G - b.G) + Math.Abs(a.B - b.B);
        }

        private static void Check(bool condition, string name)
        {
            if (condition)
            {
                Console.WriteLine("PASS: " + name);
            }
            else
            {
                failures++;
                Console.WriteLine("FAIL: " + name);
            }
        }

        private sealed class FakeServoBus : IServoBus
        {
            public bool IsOpen { get; private set; }
            public int DisableCalls;
            public int TargetCalls;
            public int[] LastServoIds;
            public int[] LastTargets;

            public void Open()
            {
                IsOpen = true;
            }

            public void Close()
            {
                IsOpen = false;
            }

            public ServoReadResult[] ReadPositionsTolerant(int[] servoIds)
            {
                ServoReadResult[] results = new ServoReadResult[servoIds.Length];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = new ServoReadResult
                    {
                        Id = servoIds[i],
                        Success = true,
                        Position = 2048 + i * 12
                    };
                }
                return results;
            }

            public void SetTorqueEnabled(int[] servoIds, bool enabled)
            {
                if (!enabled)
                {
                    Interlocked.Increment(ref DisableCalls);
                }
            }

            public void SetTorqueLimit(int[] servoIds, int limit)
            {
            }

            public void SetAcceleration(int[] servoIds, int acceleration)
            {
            }

            public void SetTargetPositions(
                int[] servoIds,
                int[] targets,
                int speedStepsPerSecond)
            {
                LastServoIds = (int[])servoIds.Clone();
                LastTargets = (int[])targets.Clone();
                Interlocked.Increment(ref TargetCalls);
            }

            public void EnableSafely(
                int[] servoIds,
                int[] currentPositions,
                int torqueLimit,
                int speedStepsPerSecond,
                int acceleration)
            {
            }

            public void Dispose()
            {
                Close();
            }
        }
    }
}
