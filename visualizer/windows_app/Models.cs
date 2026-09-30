using System;
using System.Collections.Generic;

namespace SnakeRobotVisualizer
{
    internal enum JointAxisType
    {
        Yaw,
        Pitch
    }

    internal enum ThemeMode
    {
        Dark,
        Light
    }

    internal enum DisplayMode
    {
        ThreeD,
        Projection,
        Split
    }

    internal enum RuntimeMode
    {
        Idle,
        Live,
        SimulationManual,
        SimulationAuto,
        Mujoco,
        Playback
    }

    internal enum CurveDisplayMode { Discrete, Continuous, Overlay }

    internal sealed class JointConfig
    {
        public int Index = 1;
        public int ServoId = 1;
        public JointAxisType AxisType = JointAxisType.Yaw;
        public int Zero = 2048;
        public int Direction = 1;
        public double LinkLength = 89.5;
        public double MinimumAngle = -90.0;
        public double MaximumAngle = 90.0;

        public JointConfig Clone()
        {
            return new JointConfig
            {
                Index = Index,
                ServoId = ServoId,
                AxisType = AxisType,
                Zero = Zero,
                Direction = Direction,
                LinkLength = LinkLength,
                MinimumAngle = MinimumAngle,
                MaximumAngle = MaximumAngle
            };
        }

        public void Normalize()
        {
            ServoId = Math.Max(0, Math.Min(253, ServoId));
            Zero = Math.Max(0, Math.Min(65535, Zero));
            Direction = Direction < 0 ? -1 : 1;
            LinkLength = Math.Max(1.0, Math.Min(2000.0, LinkLength));
            MinimumAngle = Math.Max(-180.0, Math.Min(180.0, MinimumAngle));
            MaximumAngle = Math.Max(-180.0, Math.Min(180.0, MaximumAngle));
            if (MinimumAngle > MaximumAngle)
            {
                double temporary = MinimumAngle;
                MinimumAngle = MaximumAngle;
                MaximumAngle = temporary;
            }
        }
    }

    internal sealed class RobotProfile
    {
        public string Name = "默认两关节";
        public string Port = "COM9";
        public int BaudRate = 115200;
        public double HeadLength = 89.5;
        public List<JointConfig> Joints = new List<JointConfig>();

        public RobotProfile Clone()
        {
            RobotProfile copy = new RobotProfile
            {
                Name = Name,
                Port = Port,
                BaudRate = BaudRate,
                HeadLength = HeadLength
            };
            foreach (JointConfig joint in Joints)
            {
                copy.Joints.Add(joint.Clone());
            }
            copy.Reindex();
            return copy;
        }

        public void Reindex()
        {
            for (int i = 0; i < Joints.Count; i++)
            {
                Joints[i].Index = i + 1;
                Joints[i].Normalize();
            }
        }

        public List<string> Validate()
        {
            List<string> errors = new List<string>();
            if (String.IsNullOrWhiteSpace(Name))
            {
                errors.Add("机器人档案名称不能为空");
            }
            if (String.IsNullOrWhiteSpace(Port))
            {
                errors.Add("串口不能为空");
            }
            if (BaudRate <= 0)
            {
                errors.Add("波特率必须大于零");
            }
            if (HeadLength < 1.0)
            {
                errors.Add("头部到第一关节的长度必须大于零");
            }
            if (Joints.Count == 0)
            {
                errors.Add("至少需要一个关节");
                return errors;
            }

            HashSet<int> servoIds = new HashSet<int>();
            for (int i = 0; i < Joints.Count; i++)
            {
                JointConfig joint = Joints[i];
                joint.Normalize();
                if (!servoIds.Add(joint.ServoId))
                {
                    errors.Add("舵机 ID 重复：" + joint.ServoId);
                }
                if (joint.MinimumAngle >= joint.MaximumAngle)
                {
                    errors.Add("关节 " + (i + 1) + " 的最小角必须小于最大角");
                }
            }
            return errors;
        }

        public static RobotProfile CreateDefaultTwoJoint()
        {
            RobotProfile profile = new RobotProfile
            {
                Name = "默认两关节",
                Port = "COM9",
                BaudRate = 115200,
                HeadLength = 89.5
            };
            profile.Joints.Add(new JointConfig
            {
                Index = 1,
                ServoId = 3,
                AxisType = JointAxisType.Yaw,
                Zero = 2048,
                Direction = 1,
                LinkLength = 89.5
            });
            profile.Joints.Add(new JointConfig
            {
                Index = 2,
                ServoId = 41,
                AxisType = JointAxisType.Pitch,
                Zero = 2048,
                Direction = -1,
                LinkLength = 89.5
            });
            return profile;
        }

        public static RobotProfile CreateAlternating(int count, string name)
        {
            RobotProfile profile = new RobotProfile
            {
                Name = name,
                Port = "COM9",
                BaudRate = 115200,
                HeadLength = 89.5
            };
            for (int i = 0; i < Math.Max(1, count); i++)
            {
                profile.Joints.Add(new JointConfig
                {
                    Index = i + 1,
                    ServoId = i + 1,
                    AxisType = i % 2 == 0 ? JointAxisType.Yaw : JointAxisType.Pitch,
                    Zero = 2048,
                    Direction = 1,
                    LinkLength = 89.5,
                    MinimumAngle = -90.0,
                    MaximumAngle = 90.0
                });
            }
            return profile;
        }
    }

    internal sealed class JointRuntimeState
    {
        public int Raw = 2048;
        public double AngleDegrees;
        public double TargetDegrees;
        public bool Online;
        public string Status = "未连接";
        public DateTime LastSeenUtc = DateTime.MinValue;
        // 实机状态块（负载、温度等）；只在实机模式按周期读取，未读到时为 null。
        public ServoTelemetry Telemetry;
        public DateTime TelemetryUtc = DateTime.MinValue;

        public JointRuntimeState Clone()
        {
            return new JointRuntimeState
            {
                Raw = Raw,
                AngleDegrees = AngleDegrees,
                TargetDegrees = TargetDegrees,
                Online = Online,
                Status = Status,
                LastSeenUtc = LastSeenUtc,
                Telemetry = Telemetry,
                TelemetryUtc = TelemetryUtc
            };
        }
    }

    internal sealed class RollingAxisState
    {
        public Point3D Point = new Point3D(0, 0, 0);
        public Point3D Direction = new Point3D(1, 0, 0);
        public double Confidence;
        public bool Valid;
        public bool Locked;
        public bool Manual;
        public string Status = "尚未估计";
        public bool HasBasis;
        public Point3D Horizontal;
        public Point3D Vertical;
        public double MeanPitchDegrees;
        public double PitchDeviationDegrees;
        public bool Fallback;

        public RollingAxisState Clone()
        {
            return new RollingAxisState
            {
                Point = Point,
                Direction = Direction,
                Confidence = Confidence,
                Valid = Valid,
                Locked = Locked,
                Manual = Manual,
                Status = Status,
                HasBasis = HasBasis,
                Horizontal = Horizontal,
                Vertical = Vertical,
                MeanPitchDegrees = MeanPitchDegrees,
                PitchDeviationDegrees = PitchDeviationDegrees,
                Fallback = Fallback
            };
        }
    }

    internal sealed class RobotFrame
    {
        public DateTime TimestampUtc = DateTime.UtcNow;
        public DateTime SampleTimestampUtc = DateTime.UtcNow;
        public string RunId = "", RuntimeModeName = "";
        public RobotProfile ConfigSnapshot;
        public double SimTime;
        public JointRuntimeState[] Joints = new JointRuntimeState[0];
        public Point3D[] Points = new Point3D[0];
        public Point3D[] TargetPoints = new Point3D[0];
        public Matrix3[] JointOrientations = new Matrix3[0];
        public RollingAxisState RollingAxis = new RollingAxisState();
        public RollingAxisState RawAxis = new RollingAxisState();
        public ProjectionPoint[] Projected = new ProjectionPoint[0];
        public ProjectionPoint[] TargetProjected = new ProjectionPoint[0];
        // Geometry is immutable after publication; frame clones may share these caches.
        public CrossSectionCurve Curve;
        public CrossSectionCurve TargetCurve;
        public double CurveGammaRadians = Math.PI * 0.01;
        public string GeometryNote = "";
        public long Sequence;
        public string Source = "数据";

        public RobotFrame Clone()
        {
            RobotFrame copy = new RobotFrame
            {
                TimestampUtc = TimestampUtc,
                SampleTimestampUtc = SampleTimestampUtc, RunId = RunId,
                RuntimeModeName = RuntimeModeName, ConfigSnapshot = ConfigSnapshot, SimTime = SimTime,
                Points = (Point3D[])Points.Clone(),
                TargetPoints = (Point3D[])TargetPoints.Clone(),
                JointOrientations = (Matrix3[])JointOrientations.Clone(),
                RollingAxis = RollingAxis == null ? new RollingAxisState() : RollingAxis.Clone(),
                RawAxis = RawAxis == null ? new RollingAxisState() : RawAxis.Clone(),
                Projected = Projected,
                TargetProjected = TargetProjected,
                Curve = Curve,
                TargetCurve = TargetCurve,
                CurveGammaRadians = CurveGammaRadians,
                GeometryNote = GeometryNote,
                Sequence = Sequence,
                Source = Source
            };
            copy.Joints = new JointRuntimeState[Joints.Length];
            for (int i = 0; i < Joints.Length; i++)
            {
                copy.Joints[i] = Joints[i].Clone();
            }
            return copy;
        }
    }

    internal sealed class DisplaySettings
    {
        public ThemeMode Theme = ThemeMode.Dark;
        public DisplayMode DisplayMode = DisplayMode.ThreeD;
        public bool ShowWorldAxes = true;
        public bool ShowRollingAxis = true;
        public bool ShowLocalAxes = false;
        public bool ShowGrid = true;
        public bool ShowNodes = true;
        public bool ShowLabels = true;
        public bool ShowTarget = false;
        public bool FollowRobot = false;
        public bool AlignRollingAxis = false;
        public bool ShowPipeOutline = false;
        public bool PipeEllipse = false;
        public double PipeRadiusX = 160.0;
        public double PipeRadiusY = 160.0;
        public double PipeCenterX = 0;
        public double PipeCenterY = 0;
        public CurveDisplayMode CurveMode = CurveDisplayMode.Overlay;
        public double CurveGammaDegrees = 1.8;
    }

    internal sealed class ApplicationState
    {
        public string LastProfile = "默认两关节";
        public DisplaySettings Display = new DisplaySettings();
    }
}
