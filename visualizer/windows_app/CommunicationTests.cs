using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Diagnostics;

namespace SnakeRobotVisualizer
{
    internal static class CommunicationTests
    {
        static void Check(bool ok, string message) { if (!ok) throw new Exception(message); Console.WriteLine("PASS " + message); }
        static Dictionary<string, object> Request(Dictionary<string, object> g, long id, double[] q, string execution = "preview", long ttl = 2000)
        {
            return new Dictionary<string, object> { { "_connection_id", "test-owner" }, { "run_id", g["run_id"] },
                { "config_id", g["config_id"] }, { "feedback_sequence", g["sequence"] }, { "command_id", id },
                { "valid_for_ms", ttl }, { "execution", execution }, { "targets_deg", q } };
        }
        static void Reject(Action action, string label)
        {
            bool failed = false;
            try { action(); } catch (ArgumentException) { failed = true; } catch (InvalidOperationException) { failed = true; }
            Check(failed, label);
        }
        static RobotFrame Wait(RobotRuntime r)
        {
            RobotFrame f;
            Stopwatch w = Stopwatch.StartNew();
            while (w.ElapsedMilliseconds < 2000) { if (r.TryGetLatest(out f)) return f; Thread.Sleep(10); }
            throw new Exception("frame timeout");
        }
        static int Main()
        {
            try { TestGeometryPreview(); TestMujocoExecution(); TestLiveMock(); TestDisableTorqueCapturesCurrentAngle(); TestLiveTelemetry(); Console.WriteLine("ALL COMMUNICATION TESTS PASSED"); return 0; }
            catch (Exception e) { Console.WriteLine(e); return 1; }
        }
        static void TestGeometryPreview()
        {
            RobotProfile p = RobotProfile.CreateAlternating(28, "protocol-test");
            double[] q = Enumerable.Range(0, 28).Select(i => 45 * Math.Sin(i * 0.17)).ToArray();
            Point3D origin = new Point3D(100, 200, 300);
            using (RobotRuntime r = new RobotRuntime(delegate { throw new Exception("preview opened hardware"); }))
            {
                r.Start(p, RuntimeMode.Mujoco);
                r.SetManualAxis(true, origin, new Point3D(0, 0, 1));
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                RobotFrame f = Wait(r);
                Dictionary<string, object> g = r.GeometryForApi();
                Check((long)g["sequence"] == f.Sequence && (string)g["run_id"] == f.RunId, "geometry uses single frame/config");
                Dictionary<string, object> c = (Dictionary<string, object>)g["curve"];
                object[] segments = (object[])c["segments"];
                Check(f.Curve.Valid && segments.Length > 0, "valid nonempty curve fixture");
                Check(segments.Length == f.Curve.Segments.Count, "all curve segments exported");
                for (int i = 0; i < segments.Length; i++)
                {
                    Dictionary<string, object> s = (Dictionary<string, object>)segments[i];
                    CurveSegment actual = f.Curve.Segments[i];
                    if ((double)s["radius_mm"] != actual.Radius || (double)s["length_mm"] != actual.Length ||
                        (double)s["turn_rad"] != actual.Turn || ((double[])s["start_mm"])[0] != actual.Start.X ||
                        ((double[])s["tangent"])[1] != actual.Tangent.Y) throw new Exception("segment mismatch");
                }
                Check(true, "segment parameters match internal geometry exactly");
                double[] targets = Enumerable.Range(0, 28).Select(i => i - 13.5).ToArray();
                r.SubmitTargetFrame(Request(g, 1, targets), 800);
                double[] invalid = (double[])targets.Clone(); invalid[27] = 999;
                Reject(delegate { r.SubmitTargetFrame(Request(g, 2, invalid), 800); }, "out-of-range frame rejected atomically");
                Reject(delegate { r.SubmitTargetFrame(Request(g, 2, new double[2]), 800); }, "wrong count rejected");
                invalid[27] = Double.NaN;
                Reject(delegate { r.SubmitTargetFrame(Request(g, 2, invalid), 800); }, "NaN rejected");
                Reject(delegate { r.SubmitTargetFrame(Request(g, 2, targets, "live"), 800); }, "simulation cannot execute live command");
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                f = Wait(r);
                Check(f.Joints.Select(x => x.TargetDegrees).SequenceEqual(targets), "preview survives incoming source targets");
                Check(f.Joints.Select(x => x.AngleDegrees).SequenceEqual(q), "preview does not change measured angles");
                KinematicResult fk = Kinematics.Forward(p, targets);
                Check(f.TargetPoints.Select((point, i) => (point - (origin + fk.Points[i])).Length).Max() < 1e-10,
                    "preview FK has correct target mapping/world origin");
                g = r.GeometryForApi();
                r.SubmitTargetFrame(Request(g, 2, targets), 800);
                double[] latest = targets.Select(x => -x).ToArray();
                r.SubmitTargetFrame(Request(g, 3, latest), 800);
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                f = Wait(r);
                Check(f.Joints.Select(x => x.TargetDegrees).SequenceEqual(latest), "only latest pending target applied");
                Check((long)((Dictionary<string, object>)r.GeometryForApi()["target_state"])["replaced_pending_count"] == 1,
                    "pending replacements counted");
                Reject(delegate { r.SubmitTargetFrame(Request(g, 3, targets), 800); }, "duplicate command rejected");
                r.CancelTargetFrames("test-owner");
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                Check(Wait(r).Joints.Select(x => x.TargetDegrees).SequenceEqual(q), "disconnect/cancel restores source preview");
                g = r.GeometryForApi();
                r.SubmitTargetFrame(Request(g, 4, targets, "preview", 20), 800);
                Thread.Sleep(35);
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                Check(Wait(r).Joints.Select(x => x.TargetDegrees).SequenceEqual(q), "expired pending preview never applied");
                Reject(delegate { r.SubmitTargetFrame(Request(g, 5, targets, "preview", 20), 800); }, "stale feedback rejected");
                r.Start(p, RuntimeMode.Mujoco);
                r.UpdateMujocoPose(q, q, origin, Matrix3.Identity);
                Reject(delegate { r.SubmitTargetFrame(Request(g, 6, targets), 800); }, "old run/config rejected after restart");
            }
        }
        static void TestMujocoExecution()
        {
            RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
            double[] actual = new double[] { 5, -3 }, target = new double[] { 20, -15 };
            using (RobotRuntime r = new RobotRuntime(delegate { throw new Exception("MuJoCo opened a servo bus"); }))
            {
                r.Start(profile, RuntimeMode.Mujoco);
                r.UpdateMujocoPose(actual, actual, new Point3D(), Matrix3.Identity);
                Dictionary<string, object> g = r.GeometryForApi();
                Reject(delegate { r.SubmitTargetFrame(Request(g, 1, target, "mujoco"), 800); }, "MuJoCo target requires executor");
                r.RegisterMujocoExecutor("sim-1");
                Reject(delegate { r.ValidateMujocoSource("foreign"); }, "other source cannot overwrite active simulator feedback");
                r.SubmitTargetFrame(Request(g, 1, actual, "mujoco"), 800);
                r.SubmitTargetFrame(Request(g, 2, target, "mujoco"), 800);
                Dictionary<string, object> reply = r.MujocoReply("sim-1", "", 0);
                Dictionary<string, object> command = (Dictionary<string, object>)reply["simulation_target"];
                Check(((double[])command["targets_deg"]).SequenceEqual(target) && (long)command["command_id"] == 2,
                    "MuJoCo executor receives latest complete target");
                Check(r.MujocoReply("sim-1", "", 0)["simulation_target"] == null, "delivered target is not replayed");
                Check(((double[])r.GeometryForApi()["angles_deg"]).SequenceEqual(actual), "dispatch never substitutes desired angles for actual angles");
                r.UpdateMujocoPose(actual, target, new Point3D(), Matrix3.Identity);
                r.MujocoReply("sim-1", (string)g["run_id"], (long)command["delivery_id"]);
                Dictionary<string, object> status = (Dictionary<string, object>)r.GeometryForApi()["target_state"];
                Check((long)status["mujoco_applied_delivery_id"] == 1, "simulator application acknowledgement is distinct from dispatch");
                r.SubmitTargetFrame(Request(g, 3, actual), 800);
                Check(r.MujocoReply("sim-1", "", 0)["simulation_target"] == null, "preview cannot execute in MuJoCo");
                r.CancelTargetFrames("test-owner");
                r.SubmitTargetFrame(Request(g, 4, target, "mujoco"), 800);
                r.ControlDisconnected("test-owner");
                Check(r.MujocoReply("sim-1", "", 0)["simulation_target"] == null, "controller disconnect cancels pending simulation target");
                r.SubmitTargetFrame(Request(g, 5, target, "mujoco"), 800);
                r.ControlDisconnected("sim-1");
                Reject(delegate { r.MujocoReply("sim-1", "", 0); }, "disconnected executor cannot receive target");
                r.RegisterMujocoExecutor("sim-2");
                Check(r.MujocoReply("sim-2", "", 0)["simulation_target"] == null, "new executor never inherits pending old command");
                r.UpdateMujocoPose(actual, actual, new Point3D(), Matrix3.Identity);
                g = r.GeometryForApi();
                r.SubmitTargetFrame(Request(g, 6, target, "mujoco", 30), 800);
                Thread.Sleep(40);
                Check(r.MujocoReply("sim-2", "", 0)["simulation_target"] == null, "expired simulation target is never delivered");
                r.Start(profile, RuntimeMode.Mujoco);
                r.UpdateMujocoPose(actual, actual, new Point3D(), Matrix3.Identity);
                r.RegisterMujocoExecutor("sim-2");
                Reject(delegate { r.SubmitTargetFrame(Request(g, 7, target, "mujoco"), 800); }, "old configuration cannot execute in restarted simulator");
                r.Stop();
                Reject(delegate { r.MujocoReply("sim-2", "", 0); }, "stopped simulator cannot receive target");
            }
        }

        static void TestLiveMock()
        {
            RobotProfile p = RobotProfile.CreateDefaultTwoJoint();
            FakeBus b = new FakeBus();
            using (RobotRuntime r = new RobotRuntime(delegate { return b; }))
            {
                r.Start(p, RuntimeMode.Live); Wait(r);
                Dictionary<string, object> g = r.GeometryForApi();
                Reject(delegate { r.SubmitTargetFrame(Request(g, 1, new double[] { 12, -8 }, "live"), 800); }, "live requires manual torque enable");
                r.QueueEnableTorque(300, 800, 20);
                Stopwatch w = Stopwatch.StartNew();
                while (!r.TorqueActive && w.ElapsedMilliseconds < 1500) Thread.Sleep(10);
                Check(r.TorqueActive, "fake bus torque enabled through existing queue");
                g = r.GeometryForApi();
                r.SubmitTargetFrame(Request(g, 1, new double[] { 12, -8 }, "live"), 800);
                w.Restart(); while (b.Calls == 0 && w.ElapsedMilliseconds < 1500) Thread.Sleep(10);
                Check(b.Calls == 1 && b.Ids.SequenceEqual(new int[] { 3, 41 }), "one batch uses physical servo ID mapping");
                Check(b.Targets.SequenceEqual(new int[] { Kinematics.DegreesToRaw(12, p.Joints[0]), Kinematics.DegreesToRaw(-8, p.Joints[1]) }), "batch target calibration correct");
                Check((long)((Dictionary<string, object>)r.GeometryForApi()["target_state"])["dispatched_command_id"] == 1, "dispatch status distinct from acceptance");
                r.QueueDisableTorque();
                Thread.Sleep(150);
                Check(!r.TorqueActive, "release clears controller and disables fake bus");
                r.Stop();
                Reject(delegate { r.GeometryForApi(); }, "stopped runtime exports no active feedback");
            }
        }
        static void TestDisableTorqueCapturesCurrentAngle()
        {
            // 回归：紧急释放扭矩后，目标角曾冻结在释放前的最后一次目标值，
            // 导致下次使能时安全使能逻辑与界面显示的目标角不一致，必须断开重连才能恢复。
            RobotProfile p = RobotProfile.CreateDefaultTwoJoint();
            FakeBus b = new FakeBus();
            using (RobotRuntime r = new RobotRuntime(delegate { return b; }))
            {
                r.Start(p, RuntimeMode.Live);
                RobotFrame f = Wait(r);
                double[] angleBeforeRelease = f.Joints.Select(x => x.AngleDegrees).ToArray();

                r.QueueEnableTorque(300, 800, 20);
                Stopwatch w = Stopwatch.StartNew();
                while (!r.TorqueActive && w.ElapsedMilliseconds < 1500) Thread.Sleep(10);
                Check(r.TorqueActive, "torque enabled before release test");

                Dictionary<string, object> g = r.GeometryForApi();
                r.SubmitTargetFrame(Request(g, 1, new double[] { 12, -8 }, "live"), 800);
                w.Restart(); while (b.Calls == 0 && w.ElapsedMilliseconds < 1500) Thread.Sleep(10);
                f = Wait(r);
                Check(f.Joints.Select(x => x.TargetDegrees).SequenceEqual(new double[] { 12, -8 }),
                    "target set away from current angle before release");

                r.QueueDisableTorque();
                w.Restart(); while (r.TorqueActive && w.ElapsedMilliseconds < 1500) Thread.Sleep(10);
                Check(!r.TorqueActive, "torque disabled for release test");

                f = Wait(r);
                Check(f.Joints.Select(x => x.TargetDegrees).SequenceEqual(angleBeforeRelease),
                    "disable torque resets target to current angle instead of freezing stale target");
            }
        }
        static void TestLiveTelemetry()
        {
            RobotProfile p = RobotProfile.CreateDefaultTwoJoint();
            using (RobotRuntime r = new RobotRuntime(delegate { return new TelemetryBus(); }))
            {
                r.Start(p, RuntimeMode.Live);
                Dictionary<string, object> telemetry = null;
                Stopwatch w = Stopwatch.StartNew();
                while (telemetry == null && w.ElapsedMilliseconds < 2000)
                {
                    try { telemetry = (Dictionary<string, object>)r.GeometryForApi()["telemetry"]; } catch (InvalidOperationException) { }
                    Thread.Sleep(10);
                }
                Check(telemetry != null, "live runtime reads servo state block");
                object[] load = (object[])telemetry["load_permille"];
                Check(Convert.ToInt32(load[0]) == -168 && Convert.ToInt32(load[1]) == 90, "state block load keeps sign and joint order");
                Check(Convert.ToInt32(((object[])telemetry["temperature_c"])[1]) == 41, "state block temperature exported");
                Check(((double[])telemetry["sample_unix_ms"]).All(x => x > 0), "state block sample time exported per joint");
                r.Stop();
            }
            using (RobotRuntime r = new RobotRuntime(delegate { return new FakeBus(); }))
            {
                r.Start(p, RuntimeMode.Live); Wait(r);
                Check(r.GeometryForApi()["telemetry"] == null, "bus without state block exports no telemetry");
                r.Stop();
            }
        }
        sealed class TelemetryBus : IServoBus, IServoTelemetryBus
        {
            public bool IsOpen { get; private set; }
            public void Open() { IsOpen = true; }
            public void Close() { IsOpen = false; }
            public void Dispose() { Close(); }
            public ServoReadResult[] ReadPositionsTolerant(int[] ids) { return ids.Select(id => new ServoReadResult { Id = id, Success = true, Position = 2048 }).ToArray(); }
            public ServoReadResult[] ReadStatesTolerant(int[] ids)
            {
                return ids.Select((id, i) => new ServoReadResult { Id = id, Success = true, Position = 2048,
                    Telemetry = new ServoTelemetry { LoadPermille = i == 0 ? -168 : 90, TemperatureC = 36 + 5 * i, VoltageVolts = 12.3 } }).ToArray();
            }
            public void SetTargetPositions(int[] ids, int[] targets, int speed) { }
            public void SetTorqueEnabled(int[] ids, bool enabled) { }
            public void SetTorqueLimit(int[] ids, int limit) { }
            public void SetAcceleration(int[] ids, int acceleration) { }
            public void EnableSafely(int[] ids, int[] positions, int torque, int speed, int acceleration) { }
        }
        sealed class FakeBus : IServoBus
        {
            public bool IsOpen { get; private set; }
            public int Calls;
            public int[] Ids, Targets;
            public void Open() { IsOpen = true; }
            public void Close() { IsOpen = false; }
            public void Dispose() { Close(); }
            public ServoReadResult[] ReadPositionsTolerant(int[] ids) { return ids.Select(id => new ServoReadResult { Id = id, Success = true, Position = 2048 }).ToArray(); }
            public void SetTargetPositions(int[] ids, int[] targets, int speed) { Ids = ids; Targets = targets; Interlocked.Increment(ref Calls); }
            public void SetTorqueEnabled(int[] ids, bool enabled) { }
            public void SetTorqueLimit(int[] ids, int limit) { }
            public void SetAcceleration(int[] ids, int acceleration) { }
            public void EnableSafely(int[] ids, int[] positions, int torque, int speed, int acceleration) { }
        }
    }
}
