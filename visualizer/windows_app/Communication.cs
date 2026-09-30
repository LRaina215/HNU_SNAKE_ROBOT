using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SnakeRobotVisualizer
{
    internal sealed class TargetFrameCommand
    {
        public string Owner, Execution;
        public long CommandId, FeedbackSequence;
        public DateTime ExpiresUtc;
        public double[] Targets;
        public int Speed;
    }

    // 通信只做快照导出、目标校验和转发；曲率/挠率与 Yamada 算法在 Python。
    // 状态与 RuntimeEngine 共用 sync 锁；不在网络回调中直接写真实串口。
    internal sealed partial class RobotRuntime
    {
        private string runId = "";
        private readonly Dictionary<long, DateTime> feedbackTimes = new Dictionary<long, DateTime>();
        private TargetFrameCommand pendingTargetFrame, activePreview;
        private string targetOwner = "", targetStatus = "none";
        private readonly Dictionary<string, long> commandWatermarks = new Dictionary<string, long>();
        private long acceptedCommand, appliedCommand, dispatchedCommand, replacedTargets;
        private DateTime ownerExpiresUtc;
        private string mujocoExecutor = "";
        private DateTime mujocoExecutorSeen;
        // delivery 表示已交给执行端；applied 表示执行端确认写入 ctrl，均非到位确认。
        private long mujocoDelivery, mujocoAppliedDelivery;

        private void ResetCommunication()
        {
            runId = Guid.NewGuid().ToString("N");
            feedbackTimes.Clear();
            commandWatermarks.Clear();
            pendingTargetFrame = activePreview = null;
            targetOwner = "";
            targetStatus = "none";
            acceptedCommand = appliedCommand = dispatchedCommand = replacedTargets = 0;
            mujocoExecutor = "";
            mujocoDelivery = mujocoAppliedDelivery = 0;
        }

        // 同一持久连接既上传实测角又领取执行目标，不允许另一个模型混入反馈。
        public void ValidateMujocoSource(string owner)
        {
            lock (sync)
            {
                ExpireMujocoExecutor();
                if (mujocoExecutor != "" && mujocoExecutor != owner)
                    throw new InvalidOperationException("executor_busy: 另一 MuJoCo 连接持有执行端");
            }
        }

        public void RegisterMujocoExecutor(string owner)
        {
            lock (sync)
            {
                ValidateMujocoSource(owner);
                if (stopping || mode != RuntimeMode.Mujoco)
                    throw new InvalidOperationException("not_mujoco: 当前不是 MuJoCo 模式");
                mujocoExecutor = owner;
                mujocoExecutorSeen = DateTime.UtcNow;
            }
        }

        private void ExpireMujocoExecutor()
        {
            if (mujocoExecutor != "" && (DateTime.UtcNow - mujocoExecutorSeen).TotalSeconds > 1)
            {
                mujocoExecutor = "";
                CancelTargetFrames(null);
            }
        }

        public void ControlDisconnected(string owner)
        {
            lock (sync)
            {
                if (owner == mujocoExecutor)
                {
                    mujocoExecutor = "";
                    CancelTargetFrames(null);
                }
                else CancelTargetFrames(owner);
            }
        }

        public Dictionary<string, object> MujocoReply(string owner, string appliedRun, long appliedDelivery)
        {
            lock (sync)
            {
                ExpireMujocoExecutor();
                ExpireTargetFrames();
                if (mode != RuntimeMode.Mujoco || stopping || owner != mujocoExecutor)
                    throw new InvalidOperationException("executor_missing: 执行连接已失效");
                if (appliedRun == runId && appliedDelivery > mujocoAppliedDelivery && appliedDelivery <= mujocoDelivery)
                    mujocoAppliedDelivery = appliedDelivery;
                object target = null;
                if (pendingTargetFrame != null && pendingTargetFrame.Execution == "mujoco")
                {
                    TargetFrameCommand command = pendingTargetFrame;
                    pendingTargetFrame = null;
                    mujocoDelivery++;
                    dispatchedCommand = command.CommandId;
                    targetStatus = "dispatched_mujoco";
                    target = new Dictionary<string, object> {
                        { "run_id", runId }, { "config_id", runId }, { "delivery_id", mujocoDelivery },
                        { "command_id", command.CommandId }, { "feedback_sequence", command.FeedbackSequence },
                        { "targets_deg", command.Targets }, { "expires_unix_ms", UnixMs(command.ExpiresUtc) }
                    };
                }
                return new Dictionary<string, object> { { "ok", true }, { "relay_ready", true },
                    { "run_id", runId }, { "simulation_target", target } };
            }
        }

        public void CancelTargetFrames(string owner)
        {
            lock (sync)
            {
                if (owner != null && owner != targetOwner) return;
                pendingTargetFrame = activePreview = null;
                targetOwner = "";
                targetStatus = "cancelled";
            }
        }

        private void ExpireTargetFrames()
        {
            if (targetOwner != "" && DateTime.UtcNow >= ownerExpiresUtc)
            {
                pendingTargetFrame = activePreview = null;
                targetOwner = "";
                targetStatus = "expired";
            }
        }

        private Dictionary<string, object> TargetStatus()
        {
            ExpireMujocoExecutor();
            ExpireTargetFrames();
            return new Dictionary<string, object> {
                { "status", targetStatus }, { "accepted_command_id", acceptedCommand },
                { "applied_preview_command_id", appliedCommand },
                { "dispatched_command_id", dispatchedCommand },
                { "pending_command_id", pendingTargetFrame == null ? 0 : pendingTargetFrame.CommandId },
                { "replaced_pending_count", replacedTargets },
                { "mujoco_executor_ready", mujocoExecutor != "" && mode == RuntimeMode.Mujoco && !stopping },
                { "mujoco_delivery_id", mujocoDelivery }, { "mujoco_applied_delivery_id", mujocoAppliedDelivery }
            };
        }

        public Dictionary<string, object> GeometryForApi()
        {
            RobotFrame f;
            Dictionary<string, object> commandStatus;
            lock (sync)
            {
                if (stopping || mode == RuntimeMode.Idle || latestFrame == null || latestFrame.RunId != runId)
                    throw new InvalidOperationException("no_frame: 当前没有运行中的反馈帧");
                f = latestFrame;
                commandStatus = TargetStatus();
            }
            double age = Math.Max(0, (DateTime.UtcNow - f.SampleTimestampUtc).TotalMilliseconds);
            bool axisReady = f.RollingAxis.Valid && !f.RollingAxis.Fallback &&
                (f.RawAxis.Valid || f.RollingAxis.Manual || f.RollingAxis.Locked);
            return new Dictionary<string, object> {
                { "ok", true }, { "protocol_version", 2 }, { "run_id", f.RunId },
                { "config_id", f.RunId }, { "sequence", f.Sequence }, { "mode", f.RuntimeModeName },
                { "source", f.Source }, { "sample_unix_ms", UnixMs(f.SampleTimestampUtc) },
                { "published_unix_ms", UnixMs(f.TimestampUtc) }, { "age_ms", age },
                { "sim_time", f.SimTime }, { "config", ConfigForApi(f.ConfigSnapshot) },
                { "angles_deg", f.Joints.Select(x => x.AngleDegrees).ToArray() },
                { "targets_deg", f.Joints.Select(x => x.TargetDegrees).ToArray() },
                { "online", f.Joints.Select(x => x.Online).ToArray() },
                { "joint_sample_unix_ms", f.Joints.Select(x => x.LastSeenUtc == DateTime.MinValue ? 0 : UnixMs(x.LastSeenUtc)).ToArray() },
                { "points_mm", f.Points.Select(PointArray).ToArray() },
                { "target_points_mm", f.TargetPoints.Select(PointArray).ToArray() },
                { "projected_mm", f.Projected.Select(ProjectionArray).ToArray() },
                { "projection_axis", FullAxisForApi(f.RollingAxis) },
                { "raw_axis", FullAxisForApi(f.RawAxis) },
                { "curve", CurveForApi(f.Curve, f.CurveGammaRadians) },
                { "feedback_usable", age <= 500 && f.Joints.All(x => x.Online) && axisReady && f.Curve != null && f.Curve.Valid },
                { "target_state", commandStatus },
                { "telemetry", TelemetryForApi(f.Joints) }
            };
        }

        // 实机状态块；从未读到时为 null。某关节缺值时对应 sample_unix_ms 为 0。
        internal static Dictionary<string, object> TelemetryForApi(JointRuntimeState[] joints)
        {
            if (joints == null || !joints.Any(x => x.Telemetry != null)) return null;
            Func<Func<ServoTelemetry, object>, object[]> column = read =>
                joints.Select(x => x.Telemetry == null ? (object)0 : read(x.Telemetry)).ToArray();
            return new Dictionary<string, object> {
                { "load_permille", column(x => x.LoadPermille) },
                { "current_raw", column(x => x.CurrentRaw) },
                { "temperature_c", column(x => x.TemperatureC) },
                { "voltage_v", column(x => x.VoltageVolts) },
                { "servo_status", column(x => x.ServoStatus) },
                { "moving", joints.Select(x => x.Telemetry != null && x.Telemetry.Moving).ToArray() },
                { "sample_unix_ms", joints.Select(x => x.Telemetry == null ? 0.0 : UnixMs(x.TelemetryUtc)).ToArray() }
            };
        }

        public Dictionary<string, object> SubmitTargetFrame(Dictionary<string, object> request, int speed)
        {
            string owner = RequiredText(request, "_connection_id");
            string execution = RequiredText(request, "execution");
            if (execution != "preview" && execution != "live" && execution != "mujoco")
                throw new ArgumentException("execution 必须为 preview、mujoco 或 live");
            long id = RequiredInteger(request, "command_id");
            long sequenceId = RequiredInteger(request, "feedback_sequence");
            long validMs = RequiredInteger(request, "valid_for_ms");
            if (id <= 0 || validMs < 20 || validMs > 2000)
                throw new ArgumentException("command_id 必须为正，valid_for_ms 范围 20..2000");
            double[] targets = FiniteArray(request, "targets_deg");
            lock (sync)
            {
                ExpireMujocoExecutor();
                ExpireTargetFrames();
                if (stopping || mode == RuntimeMode.Idle || latestFrame == null ||
                    RequiredText(request, "run_id") != runId || RequiredText(request, "config_id") != runId)
                    throw new InvalidOperationException("configuration_changed: 请重新读取运行配置和反馈");
                DateTime sample;
                if (!feedbackTimes.TryGetValue(sequenceId, out sample))
                    throw new ArgumentException("unknown_feedback: 反馈帧不存在或已退出缓存");
                DateTime expires = sample.AddMilliseconds(validMs);
                if (DateTime.UtcNow >= expires)
                    throw new InvalidOperationException("expired: 目标所依据的反馈已过期");
                if (targets.Length != profile.Joints.Count)
                    throw new ArgumentException("joint_count: 目标角数量与配置不符");
                for (int i = 0; i < targets.Length; i++)
                    if (targets[i] < profile.Joints[i].MinimumAngle || targets[i] > profile.Joints[i].MaximumAngle)
                        throw new ArgumentException("joint_range: 关节 " + (i + 1) + " 超出范围，整帧拒绝");
                if (execution == "live" && (mode != RuntimeMode.Live || !torqueActive))
                    throw new InvalidOperationException("not_enabled: live 目标要求手动使能的实机模式");
                if (execution == "live" && !latestFrame.Joints.All(x => x.Online))
                    throw new InvalidOperationException("offline: 存在离线关节");
                if (execution == "mujoco" && (mode != RuntimeMode.Mujoco || mujocoExecutor == ""))
                    throw new InvalidOperationException("executor_missing: 没有已连接的 MuJoCo 执行端");
                if (targetOwner != "" && targetOwner != owner)
                    throw new InvalidOperationException("controller_busy: 另一连接持有目标输出");
                long watermark;
                if (commandWatermarks.TryGetValue(owner, out watermark) && id <= watermark)
                    throw new ArgumentException("command_order: 命令号必须递增");
                commandWatermarks[owner] = id;
                if (pendingTargetFrame != null) replacedTargets++;
                targetOwner = owner;
                ownerExpiresUtc = expires;
                acceptedCommand = id;
                targetStatus = "accepted";
                pendingTargetFrame = new TargetFrameCommand {
                    Owner = owner, Execution = execution, CommandId = id,
                    FeedbackSequence = sequenceId, ExpiresUtc = expires, Targets = targets, Speed = speed
                };
                // 不同时保留旧的预览或旧版单关节目标。
                activePreview = null;
                RuntimeControlCommand old;
                List<RuntimeControlCommand> retained = new List<RuntimeControlCommand>();
                while (commands.TryDequeue(out old))
                    if (old.Type != RuntimeControlType.SetTarget) retained.Add(old);
                foreach (RuntimeControlCommand command in retained) commands.Enqueue(command);
                return new Dictionary<string, object> { { "ok", true }, { "command_id", id },
                    { "status", "accepted" }, { "execution", execution }, { "expires_unix_ms", UnixMs(expires) } };
            }
        }

        // 仅在采集后台生成下一帧时应用预览；它永远不调用舵机总线。
        private void ApplyPreview(JointRuntimeState[] copies)
        {
            ExpireTargetFrames();
            if (pendingTargetFrame != null && pendingTargetFrame.Execution == "preview")
            {
                activePreview = pendingTargetFrame;
                pendingTargetFrame = null;
                appliedCommand = activePreview.CommandId;
                targetStatus = "applied_preview";
            }
            if (activePreview != null)
                for (int i = 0; i < copies.Length; i++) copies[i].TargetDegrees = activePreview.Targets[i];
        }

        private void DispatchTargetFrame(IServoBus localBus, RobotProfile localProfile)
        {
            lock (sync)
            {
                ExpireTargetFrames();
                if (pendingTargetFrame == null || pendingTargetFrame.Execution != "live") return;
                TargetFrameCommand command = pendingTargetFrame;
                pendingTargetFrame = null;
                if (stopping || mode != RuntimeMode.Live || !torqueActive) { targetStatus = "cancelled"; return; }
                try
                {
                    int[] positions = command.Targets.Select((value, i) => Kinematics.DegreesToRaw(value, localProfile.Joints[i])).ToArray();
                    localBus.SetTargetPositions(localProfile.Joints.Select(x => x.ServoId).ToArray(), positions, command.Speed);
                    for (int i = 0; i < states.Length; i++) states[i].TargetDegrees = command.Targets[i];
                    dispatchedCommand = command.CommandId;
                    targetStatus = "dispatched";
                }
                catch
                {
                    targetOwner = "";
                    targetStatus = "failed";
                    throw;
                }
            }
        }

        internal static double UnixMs(DateTime value)
        { return (value - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds; }
        private static double[] PointArray(Point3D p) { return new double[] { p.X, p.Y, p.Z }; }
        private static double[] ProjectionArray(ProjectionPoint p) { return new double[] { p.X, p.Y }; }
        internal static Dictionary<string, object> FullAxisForApi(RollingAxisState a)
        {
            return new Dictionary<string, object> { { "point_mm", PointArray(a.Point) },
                { "direction", PointArray(a.Direction) }, { "horizontal", PointArray(a.Horizontal) },
                { "vertical", PointArray(a.Vertical) }, { "valid", a.Valid }, { "has_basis", a.HasBasis },
                { "confidence", a.Confidence }, { "fallback", a.Fallback }, { "locked", a.Locked },
                { "manual", a.Manual }, { "status", a.Status } };
        }
        internal static Dictionary<string, object> ConfigForApi(RobotProfile p)
        {
            return new Dictionary<string, object> { { "name", p.Name }, { "length_unit", "mm" },
                { "joint_angle_unit", "deg" }, { "curve_angle_unit", "rad" },
                { "head_length_mm", p.HeadLength }, { "joint_count", p.Joints.Count },
                { "joints", p.Joints.Select((j, i) => new Dictionary<string, object> {
                    { "joint_index", i + 1 }, { "servo_id", j.ServoId }, { "axis", j.AxisType.ToString() },
                    { "link_length_mm", j.LinkLength }, { "zero_raw", j.Zero }, { "direction", j.Direction },
                    { "minimum_deg", j.MinimumAngle }, { "maximum_deg", j.MaximumAngle }
                }).ToArray() } };
        }
        internal static Dictionary<string, object> CurveForApi(CrossSectionCurve c, double gamma)
        {
            return new Dictionary<string, object> { { "algorithm", CrossSectionCurve.AlgorithmVersion },
                { "valid", c != null && c.Valid }, { "status", c == null ? "" : c.Status },
                { "gamma_min_rad", gamma }, { "modified_count", c == null ? 0 : c.ModifiedCount },
                { "max_midpoint_error_mm", c == null ? 0 : c.MaximumMidpointError },
                { "segments", c == null ? new object[0] : c.Segments.Select((s, i) => (object)new Dictionary<string, object> {
                    { "segment_index", i }, { "kind", s.Radius == 0 ? "line" : "arc" },
                    { "joint_index", s.Case == "端部" ? (object)null : s.JointIndex },
                    { "case", s.Case }, { "radius_mm", s.Radius }, { "turn_rad", s.Turn },
                    { "length_mm", s.Length }, { "start_mm", ProjectionArray(s.Start) },
                    { "tangent", ProjectionArray(s.Tangent) }
                }).ToArray() } };
        }
        private static string RequiredText(Dictionary<string, object> r, string key)
        {
            object value;
            if (!r.TryGetValue(key, out value) || !(value is string) || String.IsNullOrEmpty((string)value))
                throw new ArgumentException("缺少字符串 " + key);
            return (string)value;
        }
        private static long RequiredInteger(Dictionary<string, object> r, string key)
        {
            object value;
            if (!r.TryGetValue(key, out value) || !(value is int || value is long))
                throw new ArgumentException("缺少整数 " + key);
            return Convert.ToInt64(value);
        }
        internal static double[] FiniteArray(Dictionary<string, object> r, string key)
        {
            object data;
            if (!r.TryGetValue(key, out data) || !(data is IEnumerable) || data is string || data is IDictionary)
                throw new ArgumentException("缺少数组 " + key);
            List<double> values = new List<double>();
            foreach (object item in (IEnumerable)data)
            {
                if (!(item is int || item is long || item is double || item is decimal))
                    throw new ArgumentException(key + " 只能包含数值");
                double value = Convert.ToDouble(item);
                if (Double.IsNaN(value) || Double.IsInfinity(value)) throw new ArgumentException(key + " 包含非有限数值");
                values.Add(value);
            }
            return values.ToArray();
        }
    }
}
