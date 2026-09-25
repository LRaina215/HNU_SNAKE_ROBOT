using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Web.Script.Serialization;

namespace SnakeRobotVisualizer
{
    internal sealed class ExperimentMetadata
    {
        public string Format = "SnakeRobotVisualizer/2";
        public string GeometryAlgorithm = CrossSectionCurve.AlgorithmVersion;
        public string Name = "";
        public DateTime StartedUtc;
        public DateTime FinishedUtc;
        public long FrameCount;
        public RobotProfile Profile;
        public DisplaySettings Display;
        public string CsvFile = "frames.csv";
    }

    internal sealed class RecordedGeometry
    {
        public string Version = CrossSectionCurve.AlgorithmVersion;
        public RollingAxisState Axis;
        public RollingAxisState RawAxis;
        public double Gamma;
    }

    internal sealed class ExperimentRecorder : IDisposable
    {
        private readonly object sync = new object();
        private readonly JavaScriptSerializer json = new JavaScriptSerializer();
        private StreamWriter writer;
        private ExperimentMetadata metadata;
        private string folder;
        private DateTime startedUtc;
        private long lastSequence = -1;

        public bool IsRecording { get { lock (sync) { return writer != null; } } }
        public string Folder { get { lock (sync) { return folder; } } }
        public long FrameCount { get { lock (sync) { return metadata == null ? 0 : metadata.FrameCount; } } }
        public TimeSpan Duration
        {
            get
            {
                lock (sync)
                {
                    return writer == null ? TimeSpan.Zero : DateTime.UtcNow - startedUtc;
                }
            }
        }

        public string Start(
            string rootDirectory,
            string experimentName,
            RobotProfile profile,
            DisplaySettings display = null)
        {
            lock (sync)
            {
                StopInternal();
                string safeName = SafeFileName(String.IsNullOrWhiteSpace(experimentName)
                    ? profile.Name
                    : experimentName.Trim());
                string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                folder = Path.Combine(rootDirectory, stamp + "_" + safeName);
                int suffix = 2;
                while (Directory.Exists(folder))
                {
                    folder = Path.Combine(rootDirectory, stamp + "_" + safeName + "_" + suffix);
                    suffix++;
                }
                Directory.CreateDirectory(folder);
                startedUtc = DateTime.UtcNow;
                lastSequence = -1;
                metadata = new ExperimentMetadata
                {
                    Name = experimentName,
                    StartedUtc = startedUtc,
                    Profile = profile.Clone(),
                    Display = display ?? new DisplaySettings()
                };
                writer = new StreamWriter(
                    Path.Combine(folder, metadata.CsvFile),
                    false,
                    new UTF8Encoding(false));
                writer.WriteLine(
                    "timestamp_utc,elapsed_s,sequence,source,joint_index,servo_id,raw," +
                    "actual_deg,target_deg,online,status,axis_px,axis_py,axis_pz," +
                    "axis_dx,axis_dy,axis_dz,axis_confidence,axis_locked,axis_manual,geometry_json");
                WriteMetadata();
                return folder;
            }
        }

        public bool Append(RobotFrame frame, RobotProfile profile)
        {
            lock (sync)
            {
                if (writer == null || frame == null || frame.Sequence == lastSequence)
                {
                    return false;
                }
                lastSequence = frame.Sequence;
                double elapsed = (frame.TimestampUtc - startedUtc).TotalSeconds;
                RollingAxisState axis = frame.RollingAxis ?? new RollingAxisState();
                if (!axis.HasBasis)
                    FrameGeometry.Prepare(frame, profile, null, frame.CurveGammaRadians);
                string geometry = json.Serialize(new RecordedGeometry
                {
                    Axis = frame.RollingAxis, RawAxis = frame.RawAxis, Gamma = frame.CurveGammaRadians
                });
                for (int i = 0; i < frame.Joints.Length; i++)
                {
                    JointRuntimeState state = frame.Joints[i];
                    int id = i < profile.Joints.Count ? profile.Joints[i].ServoId : -1;
                    string[] values =
                    {
                        frame.TimestampUtc.ToString("o", CultureInfo.InvariantCulture),
                        D(elapsed),
                        frame.Sequence.ToString(CultureInfo.InvariantCulture),
                        Csv(frame.Source),
                        (i + 1).ToString(CultureInfo.InvariantCulture),
                        id.ToString(CultureInfo.InvariantCulture),
                        state.Raw.ToString(CultureInfo.InvariantCulture),
                        D(state.AngleDegrees),
                        D(state.TargetDegrees),
                        state.Online ? "1" : "0",
                        Csv(state.Status),
                        D(axis.Point.X), D(axis.Point.Y), D(axis.Point.Z),
                        D(axis.Direction.X), D(axis.Direction.Y), D(axis.Direction.Z),
                        D(axis.Confidence),
                        axis.Locked ? "1" : "0",
                        axis.Manual ? "1" : "0",
                        i == 0 ? Csv(geometry) : ""
                    };
                    writer.WriteLine(String.Join(",", values));
                }
                metadata.FrameCount++;
                if (metadata.FrameCount % 10 == 0)
                {
                    writer.Flush();
                }
                return true;
            }
        }

        public void Stop()
        {
            lock (sync)
            {
                StopInternal();
            }
        }

        private void StopInternal()
        {
            if (writer == null)
            {
                return;
            }
            metadata.FinishedUtc = DateTime.UtcNow;
            writer.Flush();
            writer.Dispose();
            writer = null;
            WriteMetadata();
        }

        private void WriteMetadata()
        {
            if (metadata != null && folder != null)
            {
                File.WriteAllText(
                    Path.Combine(folder, "metadata.json"),
                    json.Serialize(metadata),
                    new UTF8Encoding(false));
            }
        }

        private static string D(double value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        private static string Csv(string value)
        {
            value = value ?? "";
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static string SafeFileName(string name)
        {
            string value = String.IsNullOrWhiteSpace(name) ? "实验" : name;
            foreach (char invalid in Path.GetInvalidFileNameChars())
            {
                value = value.Replace(invalid, '_');
            }
            return value;
        }

        public void Dispose()
        {
            Stop();
        }
    }

    internal sealed class PlaybackData
    {
        public RobotProfile Profile;
        public DisplaySettings Display;
        public List<RobotFrame> Frames = new List<RobotFrame>();
        public double[] ElapsedSeconds = new double[0];
        public string Folder;
        public List<RobotFrame> RecomputedFrames;
    }

    internal static class ExperimentPlayback
    {
        private sealed class JointRow
        {
            public DateTime Timestamp;
            public double Elapsed;
            public long Sequence;
            public string Source;
            public int JointIndex;
            public int ServoId;
            public int Raw;
            public double Actual;
            public double Target;
            public bool Online;
            public string Status;
            public RollingAxisState Axis;
            public RecordedGeometry Geometry;
        }

        public static PlaybackData Load(string selectedPath)
        {
            string folder = Directory.Exists(selectedPath)
                ? selectedPath
                : Path.GetDirectoryName(selectedPath);
            string metadataPath = Path.Combine(folder, "metadata.json");
            JavaScriptSerializer json = new JavaScriptSerializer();
            ExperimentMetadata metadata = json.Deserialize<ExperimentMetadata>(
                File.ReadAllText(metadataPath, Encoding.UTF8));
            if (metadata == null || metadata.Profile == null)
            {
                throw new InvalidDataException("元数据中缺少机器人档案");
            }
            metadata.Profile.Reindex();
            string csvPath = Path.Combine(folder, metadata.CsvFile ?? "frames.csv");
            Dictionary<long, List<JointRow>> groups = new Dictionary<long, List<JointRow>>();
            List<long> order = new List<long>();
            using (StreamReader reader = new StreamReader(csvPath, Encoding.UTF8))
            {
                string header = reader.ReadLine();
                if (header == null)
                {
                    throw new InvalidDataException("CSV 文件为空");
                }
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (String.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }
                    string[] c = ParseCsv(line).ToArray();
                    if (c.Length < 20)
                    {
                        continue;
                    }
                    JointRow row = new JointRow
                    {
                        Timestamp = DateTime.Parse(c[0], CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind),
                        Elapsed = PD(c[1]),
                        Sequence = PL(c[2]),
                        Source = c[3],
                        JointIndex = PI(c[4]),
                        ServoId = PI(c[5]),
                        Raw = PI(c[6]),
                        Actual = PD(c[7]),
                        Target = PD(c[8]),
                        Online = c[9] == "1",
                        Status = c[10],
                        Geometry = c.Length > 20 && !String.IsNullOrWhiteSpace(c[20])
                            ? json.Deserialize<RecordedGeometry>(c[20]) : null,
                        Axis = new RollingAxisState
                        {
                            Point = new Point3D(PD(c[11]), PD(c[12]), PD(c[13])),
                            Direction = new Point3D(PD(c[14]), PD(c[15]), PD(c[16])),
                            Confidence = PD(c[17]),
                            Locked = c[18] == "1",
                            Manual = c[19] == "1",
                            Valid = true,
                            Status = "记录轴"
                        }
                    };
                    List<JointRow> rows;
                    if (!groups.TryGetValue(row.Sequence, out rows))
                    {
                        rows = new List<JointRow>();
                        groups.Add(row.Sequence, rows);
                        order.Add(row.Sequence);
                    }
                    rows.Add(row);
                }
            }

            PlaybackData result = new PlaybackData
            {
                Profile = metadata.Profile.Clone(),
                Display = metadata.Display ?? new DisplaySettings(),
                Folder = folder
            };
            List<double> elapsedList = new List<double>();
            foreach (long key in order)
            {
                List<JointRow> rows = groups[key].OrderBy(x => x.JointIndex).ToList();
                int count = result.Profile.Joints.Count;
                JointRuntimeState[] joints = new JointRuntimeState[count];
                for (int i = 0; i < count; i++)
                {
                    JointRow row = rows.FirstOrDefault(x => x.JointIndex == i + 1);
                    joints[i] = row == null
                        ? new JointRuntimeState { Status = "记录缺失" }
                        : new JointRuntimeState
                        {
                            Raw = row.Raw,
                            AngleDegrees = row.Actual,
                            TargetDegrees = row.Target,
                            Online = row.Online,
                            Status = row.Status,
                            LastSeenUtc = row.Timestamp
                        };
                }
                double[] actualAngles = joints.Select(x => x.AngleDegrees).ToArray();
                double[] targetAngles = joints.Select(x => x.TargetDegrees).ToArray();
                KinematicResult actualKinematics = Kinematics.Forward(result.Profile, actualAngles);
                KinematicResult targetKinematics = Kinematics.Forward(result.Profile, targetAngles);
                JointRow first = rows.First();
                RecordedGeometry recorded = rows.Where(x => x.Geometry != null).Select(x => x.Geometry).FirstOrDefault();
                RobotFrame frame = new RobotFrame
                {
                    TimestampUtc = first.Timestamp,
                    Sequence = key,
                    Source = "回放",
                    Joints = joints,
                    Points = actualKinematics.Points,
                    TargetPoints = targetKinematics.Points,
                    JointOrientations = actualKinematics.JointOrientations,
                    RollingAxis = recorded == null ? first.Axis : recorded.Axis,
                    RawAxis = recorded == null ? new RollingAxisState { Status = "旧记录未保存原始估计" } : recorded.RawAxis
                };
                double gamma = recorded == null ? Math.PI * 0.01 : recorded.Gamma;
                FrameGeometry.Prepare(frame, result.Profile, null, gamma, recorded == null);
                result.Frames.Add(frame);
                elapsedList.Add(first.Elapsed);
            }
            result.ElapsedSeconds = elapsedList.ToArray();
            if (result.Frames.Count == 0)
            {
                throw new InvalidDataException("记录中没有有效帧");
            }
            return result;
        }

        public static void EnsureRecomputed(PlaybackData data)
        {
            if (data.RecomputedFrames != null) return;
            List<RobotFrame> frames = new List<RobotFrame>();
            RollingAxisEstimator estimator = new RollingAxisEstimator();
            RollingAxisState previous = null;
            foreach (RobotFrame source in data.Frames)
            {
                RobotFrame frame = source.Clone();
                frame.RollingAxis = estimator.Update(frame.Points, frame.Joints.All(x => x.Online));
                frame.RawAxis = estimator.RawEstimate.Clone();
                FrameGeometry.Prepare(frame, data.Profile, previous, frame.CurveGammaRadians);
                previous = frame.RollingAxis.Clone();
                frames.Add(frame);
            }
            data.RecomputedFrames = frames;
        }

        private static IEnumerable<string> ParseCsv(string line)
        {
            List<string> values = new List<string>();
            StringBuilder current = new StringBuilder();
            bool quoted = false;
            for (int i = 0; i < line.Length; i++)
            {
                char ch = line[i];
                if (quoted)
                {
                    if (ch == '"' && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else if (ch == '"')
                    {
                        quoted = false;
                    }
                    else
                    {
                        current.Append(ch);
                    }
                }
                else if (ch == '"')
                {
                    quoted = true;
                }
                else if (ch == ',')
                {
                    values.Add(current.ToString());
                    current.Length = 0;
                }
                else
                {
                    current.Append(ch);
                }
            }
            values.Add(current.ToString());
            return values;
        }

        private static int PI(string value)
        {
            return Int32.Parse(value, CultureInfo.InvariantCulture);
        }

        private static long PL(string value)
        {
            return Int64.Parse(value, CultureInfo.InvariantCulture);
        }

        private static double PD(string value)
        {
            return Double.Parse(value, CultureInfo.InvariantCulture);
        }
    }
}
