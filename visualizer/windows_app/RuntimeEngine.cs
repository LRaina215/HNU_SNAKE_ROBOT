using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace SnakeRobotVisualizer
{
    internal enum RuntimeControlType
    {
        EnableTorque,
        DisableTorque,
        SetTarget,
        SetLimits
    }

    internal sealed class RuntimeControlCommand
    {
        public RuntimeControlType Type;
        public int JointIndex = -1;
        public double TargetDegrees;
        public int TorqueLimit;
        public int SpeedStepsPerSecond;
        public int Acceleration;
    }

    internal sealed class RuntimeStatistics
    {
        public double SamplesPerSecond;
        public long TotalSamples;
        public long CommunicationErrors;
        public string Status = "空闲";
        public bool Error;
    }

    internal sealed partial class RobotRuntime : IDisposable
    {
        private readonly object sync = new object();
        private readonly object axisSync = new object();
        private readonly object externalUpdateSync = new object();
        private readonly ConcurrentQueue<RuntimeControlCommand> commands =
            new ConcurrentQueue<RuntimeControlCommand>();
        private readonly RollingAxisEstimator axisEstimator = new RollingAxisEstimator();
        private readonly Func<RobotProfile, IServoBus> busFactory;
        private RobotProfile profile;
        private JointRuntimeState[] states = new JointRuntimeState[0];
        private Thread worker;
        private volatile bool stopping;
        private volatile RuntimeMode mode = RuntimeMode.Idle;
        private IServoBus bus;
        private RobotFrame latestFrame;
        private long sequence;
        private long communicationErrors;
        private double samplesPerSecond;
        private DateTime mujocoRateStartUtc;
        private int mujocoRateSamples;
        private string status = "空闲";
        private bool statusError;
        private bool torqueActive;
        private double simulationPhase;
        private int automaticSimulationJoint = -1;
        private RollingAxisState previousProjectionAxis;
        private double curveGamma = Math.PI * 0.01;

        public void SetCurveGamma(double radians)
        {
            if (Double.IsNaN(radians) || Double.IsInfinity(radians) || radians <= 0 || radians >= Math.PI / 2)
                throw new ArgumentOutOfRangeException("radians");
            lock (axisSync) { curveGamma = radians; }
        }

        public RuntimeMode Mode { get { return mode; } }
        public bool TorqueActive { get { return torqueActive; } }

        public RobotRuntime()
            : this(delegate(RobotProfile p)
            {
                return new FeetechBus(p.Port, p.BaudRate);
            })
        {
        }

        internal RobotRuntime(Func<RobotProfile, IServoBus> factory)
        {
            if (factory == null)
            {
                throw new ArgumentNullException("factory");
            }
            busFactory = factory;
        }

        public RobotProfile Profile
        {
            get
            {
                lock (sync)
                {
                    return profile == null ? null : profile.Clone();
                }
            }
        }

        public void Start(RobotProfile sourceProfile, RuntimeMode requestedMode)
        {
            lock (externalUpdateSync) { StartCore(sourceProfile, requestedMode); }
        }

        private void StartCore(RobotProfile sourceProfile, RuntimeMode requestedMode)
        {
            if (requestedMode != RuntimeMode.Live &&
                requestedMode != RuntimeMode.SimulationManual &&
                requestedMode != RuntimeMode.SimulationAuto &&
                requestedMode != RuntimeMode.Mujoco)
            {
                throw new ArgumentException("运行模式无效");
            }
            Stop();

            lock (axisSync)
            {
                axisEstimator.ResetTracking();
                previousProjectionAxis = null;
            }

            List<string> errors = sourceProfile.Validate();
            if (errors.Count > 0)
            {
                throw new InvalidOperationException(String.Join("\r\n", errors.ToArray()));
            }

            lock (sync)
            {
                profile = sourceProfile.Clone();
                ResetCommunication();
                states = new JointRuntimeState[profile.Joints.Count];
                for (int i = 0; i < states.Length; i++)
                {
                    states[i] = new JointRuntimeState
                    {
                        Raw = profile.Joints[i].Zero,
                        AngleDegrees = 0,
                        TargetDegrees = 0,
                        Online = requestedMode != RuntimeMode.Live,
                        Status = requestedMode == RuntimeMode.Live ? "等待读取" : requestedMode == RuntimeMode.Mujoco ? "等待 MuJoCo" : "模拟"
                    };
                }
                latestFrame = null;
                sequence = 0;
                communicationErrors = 0;
                samplesPerSecond = 0;
                mujocoRateStartUtc = DateTime.UtcNow;
                mujocoRateSamples = 0;
                simulationPhase = 0;
                torqueActive = false;
                statusError = false;
                status = requestedMode == RuntimeMode.Live
                    ? "正在连接 " + profile.Port
                    : requestedMode == RuntimeMode.Mujoco ? "等待 MuJoCo 实测关节角" : "模拟模式正在运行";
            }

            stopping = false;
            mode = requestedMode;
            if (requestedMode == RuntimeMode.Mujoco)
            {
                worker = null;
                return;
            }
            worker = new Thread(WorkerLoop);
            worker.Name = requestedMode == RuntimeMode.Live
                ? "Multi-servo reader"
                : "Robot simulation";
            worker.IsBackground = true;
            worker.Start();
        }

        public void Stop()
        {
            lock (externalUpdateSync) { StopCore(); }
        }

        private void StopCore()
        {
            stopping = true;
            CancelTargetFrames(null);
            Thread thread = worker;
            if (thread != null && thread.IsAlive && Thread.CurrentThread != thread)
            {
                // Let the serial-owner thread finish its current poll and release
                // torque in RunLive.finally before the port is closed.
                if (!thread.Join(3500))
                {
                    IServoBus current = bus;
                    if (current != null)
                    {
                        try { current.Close(); } catch { }
                    }
                    thread.Join(500);
                }
            }
            worker = null;
            bus = null;
            torqueActive = false;
            mode = RuntimeMode.Idle;
            SetStatus("已停止", false);
            RuntimeControlCommand ignored;
            while (commands.TryDequeue(out ignored))
            {
            }
        }

        public bool TryGetLatest(out RobotFrame frame)
        {
            lock (sync)
            {
                frame = latestFrame;
                return frame != null;
            }
        }

        public void UpdateMujocoPose(double[] angles, double[] targets,
            Point3D headOriginWorld, Matrix3 headBasisWorld, double capturedUnix = 0, double simTime = 0)
        {
            lock (externalUpdateSync)
            {
            if (mode != RuntimeMode.Mujoco)
                throw new InvalidOperationException("未处于 MuJoCo 可视化模式");
            if (angles == null || targets == null || angles.Length != states.Length || targets.Length != states.Length)
                throw new ArgumentException("MuJoCo 关节数量与档案不符");
            DateTime now = DateTime.UtcNow;
            lock (sync)
            {
                for (int i = 0; i < states.Length; i++)
                {
                    if (Double.IsNaN(angles[i]) || Double.IsInfinity(angles[i]) ||
                        Double.IsNaN(targets[i]) || Double.IsInfinity(targets[i]))
                        throw new ArgumentException("MuJoCo 关节角包含无效数值");
                    states[i].AngleDegrees = angles[i];
                    states[i].TargetDegrees = targets[i];
                    states[i].Raw = Kinematics.DegreesToRaw(angles[i], profile.Joints[i]);
                    states[i].Online = true;
                    states[i].Status = "MuJoCo 实测";
                    states[i].LastSeenUtc = now;
                }
                status = "MuJoCo 实测：" + states.Length + " 个关节";
                statusError = false;
            }
            PublishFrame("MuJoCo 实测", headOriginWorld, headBasisWorld, true, capturedUnix, simTime);
            lock (sync)
            {
                mujocoRateSamples++;
                double elapsed = (DateTime.UtcNow - mujocoRateStartUtc).TotalSeconds;
                if (elapsed >= 1.0)
                {
                    samplesPerSecond = mujocoRateSamples / elapsed;
                    mujocoRateSamples = 0;
                    mujocoRateStartUtc = DateTime.UtcNow;
                }
            }
            }
        }

        public RuntimeStatistics GetStatistics()
        {
            lock (sync)
            {
                return new RuntimeStatistics
                {
                    SamplesPerSecond = samplesPerSecond,
                    TotalSamples = sequence,
                    CommunicationErrors = communicationErrors,
                    Status = status,
                    Error = statusError
                };
            }
        }

        public void SetManualSimulationAngle(int jointIndex, double degrees)
        {
            lock (sync)
            {
                if (jointIndex < 0 || jointIndex >= states.Length || profile == null)
                {
                    return;
                }
                JointConfig joint = profile.Joints[jointIndex];
                double clamped = Math.Max(joint.MinimumAngle, Math.Min(joint.MaximumAngle, degrees));
                states[jointIndex].AngleDegrees = clamped;
                states[jointIndex].TargetDegrees = clamped;
                states[jointIndex].Raw = Kinematics.DegreesToRaw(clamped, joint);
                states[jointIndex].Online = true;
                states[jointIndex].Status = "手动模拟";
            }
        }

        public void SetTargetPreview(int jointIndex, double degrees)
        {
            lock (sync)
            {
                if (jointIndex < 0 || jointIndex >= states.Length || profile == null)
                {
                    return;
                }
                JointConfig joint = profile.Joints[jointIndex];
                states[jointIndex].TargetDegrees = Math.Max(
                    joint.MinimumAngle,
                    Math.Min(joint.MaximumAngle, degrees));
            }
        }

        public void SetAutomaticSimulation(bool automatic)
        {
            if (mode == RuntimeMode.SimulationManual || mode == RuntimeMode.SimulationAuto)
            {
                mode = automatic ? RuntimeMode.SimulationAuto : RuntimeMode.SimulationManual;
                SetStatus(automatic ? "自动模拟正在运行" : "手动模拟正在运行", false);
            }
        }

        public void SetAutomaticSimulationJoint(int jointIndex)
        {
            lock (sync)
            {
                automaticSimulationJoint = jointIndex >= 0 && jointIndex < states.Length
                    ? jointIndex
                    : -1;
            }
        }

        public void SetAxisLocked(bool locked)
        {
            lock (axisSync)
            {
                axisEstimator.Locked = locked;
            }
        }

        public void SetManualAxis(bool manual, Point3D point, Point3D direction)
        {
            lock (axisSync)
            {
                axisEstimator.SetManual(point, direction);
                axisEstimator.UseManual = manual;
            }
        }

        public void UseCurrentAxisAsManual()
        {
            lock (axisSync)
            {
                axisEstimator.UseCurrentAsManual();
            }
        }

        public RollingAxisState GetAxisState()
        {
            lock (axisSync)
            {
                return axisEstimator.Current;
            }
        }

        public void QueueEnableTorque(int torqueLimit, int speed, int acceleration)
        {
            commands.Enqueue(new RuntimeControlCommand
            {
                Type = RuntimeControlType.EnableTorque,
                TorqueLimit = torqueLimit,
                SpeedStepsPerSecond = speed,
                Acceleration = acceleration
            });
        }

        public void QueueDisableTorque()
        {
            CancelTargetFrames(null);
            RuntimeControlCommand ignored;
            while (commands.TryDequeue(out ignored)) { }
            commands.Enqueue(new RuntimeControlCommand { Type = RuntimeControlType.DisableTorque });
        }

        public void QueueTarget(
            int jointIndex,
            double targetDegrees,
            int speed,
            int torqueLimit,
            int acceleration)
        {
            lock (sync)
            {
                ExpireTargetFrames();
                if (targetOwner != "") throw new InvalidOperationException("整帧控制连接正在输出，请先停止该控制器");
            commands.Enqueue(new RuntimeControlCommand
            {
                Type = RuntimeControlType.SetTarget,
                JointIndex = jointIndex,
                TargetDegrees = targetDegrees,
                SpeedStepsPerSecond = speed,
                TorqueLimit = torqueLimit,
                Acceleration = acceleration
            });
            }
        }

        public void QueueLimits(int jointIndex, int torqueLimit, int acceleration)
        {
            commands.Enqueue(new RuntimeControlCommand
            {
                Type = RuntimeControlType.SetLimits,
                JointIndex = jointIndex,
                TorqueLimit = torqueLimit,
                Acceleration = acceleration
            });
        }

        private void WorkerLoop()
        {
            try
            {
                if (mode == RuntimeMode.Live)
                {
                    RunLive();
                }
                else
                {
                    RunSimulation();
                }
            }
            catch (Exception ex)
            {
                if (!stopping)
                {
                    SetStatus(FriendlyError(ex), true);
                }
            }
            finally
            {
                bus = null;
                torqueActive = false;
                if (!stopping)
                {
                    mode = RuntimeMode.Idle;
                }
            }
        }

        private void RunLive()
        {
            RobotProfile localProfile = Profile;
            int[] ids = localProfile.Joints.Select(x => x.ServoId).ToArray();
            using (IServoBus localBus = busFactory(localProfile))
            {
                bus = localBus;
                localBus.Open();
                SetStatus("已连接 " + localProfile.Port + "，正在读取 " + ids.Length + " 个关节", false);
                Stopwatch rateWatch = Stopwatch.StartNew();
                long rateSamples = 0;
                try
                {
                    while (!stopping)
                    {
                        Stopwatch loop = Stopwatch.StartNew();
                        ServoReadResult[] reads = localBus.ReadPositionsTolerant(ids);
                        DateTime now = DateTime.UtcNow;
                        int onlineCount;
                        lock (sync)
                        {
                            onlineCount = ApplyReadResults(
                                states,
                                localProfile,
                                reads,
                                now,
                                ref communicationErrors);
                        }

                        ProcessCommands(localBus, localProfile);
                        DispatchTargetFrame(localBus, localProfile);
                        PublishFrame("实机");
                        rateSamples++;
                        if (rateWatch.ElapsedMilliseconds >= 2000)
                        {
                            lock (sync)
                            {
                                samplesPerSecond = rateSamples / rateWatch.Elapsed.TotalSeconds;
                                status = onlineCount == ids.Length
                                    ? "全部关节在线"
                                    : String.Format("{0}/{1} 个关节在线", onlineCount, ids.Length);
                                statusError = onlineCount == 0;
                            }
                            rateWatch.Restart();
                            rateSamples = 0;
                        }

                        int remaining = 100 - (int)loop.ElapsedMilliseconds;
                        if (remaining > 0)
                        {
                            Thread.Sleep(remaining);
                        }
                    }
                }
                finally
                {
                    if (torqueActive && localBus.IsOpen)
                    {
                        try { localBus.SetTorqueEnabled(ids, false); } catch { }
                    }
                    torqueActive = false;
                }
            }
        }

        private void RunSimulation()
        {
            Stopwatch rateWatch = Stopwatch.StartNew();
            long rateSamples = 0;
            while (!stopping)
            {
                Stopwatch loop = Stopwatch.StartNew();
                lock (sync)
                {
                    if (mode == RuntimeMode.SimulationAuto)
                    {
                        for (int i = 0; i < states.Length; i++)
                        {
                            if (automaticSimulationJoint >= 0 && i != automaticSimulationJoint)
                            {
                                states[i].Status = "自动模拟（保持）";
                                continue;
                            }
                            JointConfig joint = profile.Joints[i];
                            double amplitude = Math.Min(
                                35.0,
                                Math.Min(Math.Abs(joint.MinimumAngle), Math.Abs(joint.MaximumAngle)) * 0.72);
                            if (amplitude < 5.0)
                            {
                                amplitude = (joint.MaximumAngle - joint.MinimumAngle) * 0.35;
                            }
                            double angle = amplitude * Math.Sin(simulationPhase + i * 0.43);
                            angle = Math.Max(joint.MinimumAngle, Math.Min(joint.MaximumAngle, angle));
                            states[i].AngleDegrees = angle;
                            states[i].TargetDegrees = angle;
                            states[i].Raw = Kinematics.DegreesToRaw(angle, joint);
                            states[i].Online = true;
                            states[i].Status = "自动模拟";
                            states[i].LastSeenUtc = DateTime.UtcNow;
                        }
                        simulationPhase += 0.075;
                    }
                }

                PublishFrame(mode == RuntimeMode.SimulationAuto ? "自动模拟" : "手动模拟");
                rateSamples++;
                if (rateWatch.ElapsedMilliseconds >= 2000)
                {
                    lock (sync)
                    {
                        samplesPerSecond = rateSamples / rateWatch.Elapsed.TotalSeconds;
                        status = mode == RuntimeMode.SimulationAuto
                            ? "自动模拟正常"
                            : "手动模拟正常";
                        statusError = false;
                    }
                    rateWatch.Restart();
                    rateSamples = 0;
                }
                int remaining = 100 - (int)loop.ElapsedMilliseconds;
                // Windows Sleep can overshoot a 100 ms sample period by one
                // timer quantum. Reserve the final quantum for a short wait
                // so the 10 Hz simulation is not systematically slower.
                if (remaining > 16)
                    Thread.Sleep(remaining - 16);
                while (!stopping && loop.ElapsedMilliseconds < 100)
                    Thread.SpinWait(40);
            }
        }

        private void PublishFrame(string source)
        {
            PublishFrame(source, new Point3D(), Matrix3.Identity, false);
        }

        private void PublishFrame(string source, Point3D headOriginWorld,
            Matrix3 headBasisWorld)
        {
            PublishFrame(source, headOriginWorld, headBasisWorld, true);
        }

        private void PublishFrame(string source, Point3D headOriginWorld,
            Matrix3 headBasisWorld, bool hasExternalRootPose, double capturedUnix = 0, double simTime = 0)
        {
            RobotProfile localProfile;
            JointRuntimeState[] stateCopies;
            string localRunId, localMode;
            DateTime sampleTime = capturedUnix > 0
                ? new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(capturedUnix)
                : DateTime.UtcNow;
            lock (sync)
            {
                localProfile = profile.Clone();
                localRunId = runId;
                localMode = mode.ToString();
                stateCopies = new JointRuntimeState[states.Length];
                for (int i = 0; i < states.Length; i++)
                {
                    stateCopies[i] = states[i].Clone();
                }
                ApplyPreview(stateCopies);
            }

            double[] angles = stateCopies.Select(x => x.AngleDegrees).ToArray();
            double[] targets = stateCopies.Select(x => x.TargetDegrees).ToArray();
            KinematicResult actual = Kinematics.Forward(localProfile, angles);
            KinematicResult target = Kinematics.Forward(localProfile, targets);
            if (hasExternalRootPose)
            {
                for (int i = 0; i < actual.Points.Length; i++)
                    actual.Points[i] = headOriginWorld + headBasisWorld.Transform(actual.Points[i]);
                for (int i = 0; i < target.Points.Length; i++)
                    target.Points[i] = headOriginWorld + headBasisWorld.Transform(target.Points[i]);
                for (int i = 0; i < actual.JointOrientations.Length; i++)
                    actual.JointOrientations[i] = headBasisWorld * actual.JointOrientations[i];
            }
            RollingAxisState axis;
            RollingAxisState raw;
            double gamma;
            lock (axisSync)
            {
                axis = axisEstimator.Update(actual.Points, stateCopies.All(x => x.Online));
                raw = axisEstimator.RawEstimate.Clone();
                gamma = curveGamma;
                RollingProjection.SetBasis(axis, previousProjectionAxis, false);
                previousProjectionAxis = axis.Clone();
            }
            RobotFrame frame = new RobotFrame
            {
                TimestampUtc = DateTime.UtcNow,
                SampleTimestampUtc = sampleTime, RunId = localRunId, RuntimeModeName = localMode,
                ConfigSnapshot = localProfile, SimTime = simTime,
                Joints = stateCopies,
                Points = actual.Points,
                TargetPoints = target.Points,
                JointOrientations = actual.JointOrientations,
                RollingAxis = axis,
                RawAxis = raw,
                Sequence = Interlocked.Increment(ref sequence),
                Source = source
            };
            FrameGeometry.Prepare(frame, localProfile, null, gamma);
            lock (sync)
            {
                if (!stopping && localRunId == runId)
                {
                    latestFrame = frame;
                    feedbackTimes[frame.Sequence] = sampleTime;
                    foreach (long key in feedbackTimes.Keys.Where(x => x < frame.Sequence - 63).ToArray())
                        feedbackTimes.Remove(key);
                }
            }
        }

        private void ProcessCommands(IServoBus localBus, RobotProfile localProfile)
        {
            RuntimeControlCommand command;
            while (commands.TryDequeue(out command))
            {
                try
                {
                    if (command.Type == RuntimeControlType.EnableTorque)
                    {
                        List<int> ids = new List<int>();
                        List<int> positions = new List<int>();
                        lock (sync)
                        {
                            for (int i = 0; i < states.Length; i++)
                            {
                                if (states[i].Online)
                                {
                                    ids.Add(localProfile.Joints[i].ServoId);
                                    positions.Add(states[i].Raw);
                                }
                            }
                        }
                        if (ids.Count == 0)
                        {
                            throw new InvalidOperationException("没有在线舵机，不能使能扭矩");
                        }
                        localBus.EnableSafely(
                            ids.ToArray(),
                            positions.ToArray(),
                            command.TorqueLimit,
                            command.SpeedStepsPerSecond,
                            command.Acceleration);
                        torqueActive = true;
                        SetStatus("扭矩已安全使能", false);
                    }
                    else if (command.Type == RuntimeControlType.DisableTorque)
                    {
                        int[] ids = localProfile.Joints.Select(x => x.ServoId).ToArray();
                        localBus.SetTorqueEnabled(ids, false);
                        torqueActive = false;
                        SetStatus("扭矩已释放", false);
                    }
                    else if (command.Type == RuntimeControlType.SetTarget)
                    {
                        lock (sync)
                        {
                        ExpireTargetFrames();
                        if (targetOwner != "") continue;
                        if (!torqueActive)
                        {
                            throw new InvalidOperationException("请先在桌面程序中安全使能扭矩");
                        }
                        if (command.JointIndex < 0 || command.JointIndex >= localProfile.Joints.Count)
                        {
                            throw new InvalidOperationException("目标关节无效");
                        }
                        JointConfig joint = localProfile.Joints[command.JointIndex];
                        double target = Math.Max(
                            joint.MinimumAngle,
                            Math.Min(joint.MaximumAngle, command.TargetDegrees));
                        int rawTarget = Kinematics.DegreesToRaw(target, joint);
                        localBus.SetTargetPositions(
                            new int[] { joint.ServoId },
                            new int[] { rawTarget },
                            command.SpeedStepsPerSecond);
                        lock (sync)
                        {
                            states[command.JointIndex].TargetDegrees = target;
                        }
                        }
                    }
                    else if (command.Type == RuntimeControlType.SetLimits)
                    {
                        if (command.JointIndex < 0 || command.JointIndex >= localProfile.Joints.Count)
                        {
                            continue;
                        }
                        int id = localProfile.Joints[command.JointIndex].ServoId;
                        localBus.SetTorqueLimit(new int[] { id }, command.TorqueLimit);
                        localBus.SetAcceleration(new int[] { id }, command.Acceleration);
                    }
                }
                catch (Exception ex)
                {
                    try
                    {
                        localBus.SetTorqueEnabled(
                            localProfile.Joints.Select(x => x.ServoId).ToArray(),
                            false);
                    }
                    catch
                    {
                    }
                    torqueActive = false;
                    SetStatus("控制命令失败，已尝试释放扭矩：" + ex.Message, true);
                }
            }
        }

        private void SetStatus(string message, bool error)
        {
            lock (sync)
            {
                status = message;
                statusError = error;
            }
        }

        internal static int ApplyReadResults(
            JointRuntimeState[] targetStates,
            RobotProfile targetProfile,
            ServoReadResult[] reads,
            DateTime timestampUtc,
            ref long errorCount)
        {
            if (targetStates == null || targetProfile == null || reads == null ||
                targetStates.Length != targetProfile.Joints.Count ||
                reads.Length != targetStates.Length)
            {
                throw new ArgumentException("关节状态、档案与读取结果数量必须一致");
            }
            int onlineCount = 0;
            for (int i = 0; i < reads.Length; i++)
            {
                ServoReadResult read = reads[i];
                JointRuntimeState state = targetStates[i];
                if (read.Success)
                {
                    bool recovered = !state.Online && state.LastSeenUtc != DateTime.MinValue;
                    state.Raw = read.Position;
                    state.AngleDegrees = Kinematics.CalibrateDegrees(
                        read.Position,
                        targetProfile.Joints[i]);
                    state.Online = true;
                    state.LastSeenUtc = timestampUtc;
                    state.Status = recovered ? "已恢复" : "在线";
                    onlineCount++;
                }
                else
                {
                    if (state.Online || state.LastSeenUtc == DateTime.MinValue)
                    {
                        errorCount++;
                    }
                    state.Online = false;
                    state.Status = "超时";
                }
            }
            return onlineCount;
        }

        private static string FriendlyError(Exception ex)
        {
            string message = ex.Message;
            if (message.IndexOf("access", StringComparison.OrdinalIgnoreCase) >= 0 ||
                message.IndexOf("denied", StringComparison.OrdinalIgnoreCase) >= 0 ||
                message.IndexOf("拒绝", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return "串口被占用：请先关闭其他串口程序";
            }
            return "运行失败：" + message;
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
