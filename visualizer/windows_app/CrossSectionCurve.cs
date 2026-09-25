using System;
using System.Collections.Generic;

namespace SnakeRobotVisualizer
{
    // Paper III-B4, equations (11)-(22). D arcs are an inward approximation,
    // not a claim that every original midpoint or corner is interpolated.
    internal sealed class CurveSegment
    {
        public ProjectionPoint Start;
        public ProjectionPoint Tangent;
        public double Length;
        public double Radius;
        public double Turn;
        public int JointIndex;
        public string Case;
        public ProjectionPoint[] Samples;

        public ProjectionPoint At(double fraction)
        {
            if (Radius == 0) return Start + Tangent * (Length * fraction);
            double theta = Turn * fraction;
            double signedRadius = Radius * Math.Sign(Turn);
            ProjectionPoint normal = new ProjectionPoint(-Tangent.Y, Tangent.X);
            return Start + Tangent * (signedRadius * Math.Sin(theta)) +
                normal * (signedRadius * 2 * Math.Pow(Math.Sin(theta / 2), 2));
        }

        public ProjectionPoint TangentAt(double fraction)
        {
            double theta = Turn * fraction;
            return new ProjectionPoint(Tangent.X * Math.Cos(theta) - Tangent.Y * Math.Sin(theta),
                Tangent.X * Math.Sin(theta) + Tangent.Y * Math.Cos(theta));
        }
    }

    internal sealed class CrossSectionCurve
    {
        public const string AlgorithmVersion = "Paper2023-visualization/1.2.0";
        public readonly List<CurveSegment> Segments = new List<CurveSegment>();
        public bool Valid;
        public int ModifiedCount;
        public double MaximumMidpointError;
        public string Status = "连续近似尚未生成";

        public static CrossSectionCurve Build(ProjectionPoint[] points, double gammaMin, int winding = 0)
        {
            CrossSectionCurve result = new CrossSectionCurve();
            if (points == null || points.Length < 3)
                return Fail(result, "投影点不足，显示原始折线");
            if (Double.IsNaN(gammaMin) || Double.IsInfinity(gammaMin) || gammaMin <= 0 || gammaMin >= Math.PI / 2)
                return Fail(result, "圆弧阈值无效，显示原始折线");
            ProjectionPoint[] unit = new ProjectionPoint[points.Length - 1];
            double[] lengths = new double[unit.Length];
            for (int i = 0; i < unit.Length; i++)
            {
                ProjectionPoint delta = points[i + 1] - points[i];
                lengths[i] = delta.Length;
                if (Double.IsNaN(lengths[i]) || Double.IsInfinity(lengths[i]) || lengths[i] < 1e-8)
                    return Fail(result, "零长度或无效投影段，显示原始折线");
                unit[i] = delta * (1.0 / lengths[i]);
            }
            double[] turns = new double[unit.Length - 1];
            double totalTurn = 0;
            for (int i = 0; i < turns.Length; i++)
            {
                turns[i] = Math.Atan2(unit[i].X * unit[i + 1].Y - unit[i].Y * unit[i + 1].X,
                    unit[i].X * unit[i + 1].X + unit[i].Y * unit[i + 1].Y);
                if (Math.Abs(turns[i]) > Math.PI - 1e-4)
                    return Fail(result, "截面接近折返，显示原始折线");
                totalTurn += turns[i];
            }
            // The traversal can be clockwise or anticlockwise. A global winding
            // sign preserves negative (outward) LOCAL turns for paper case D.
            if (winding == 0)
            {
                if (Math.Abs(totalTurn) < 1e-5)
                    return Fail(result, "截面内侧方向不明确，显示原始折线");
                winding = Math.Sign(totalTurn);
            }
            winding = Math.Sign(winding);
            ProjectionPoint position = points[0];
            ProjectionPoint tangent = unit[0];
            Add(result, ref position, ref tangent, lengths[0] / 2, 0, 0, 0, "端部");
            for (int i = 1; i < points.Length - 1; i++)
            {
                double a = lengths[i - 1] / 2;
                double b = lengths[i] / 2;
                double gamma = turns[i - 1] * winding;
                if (gamma <= gammaMin)
                {
                    result.ModifiedCount++;
                    Add(result, ref position, ref tangent, a + b,
                        (a + b) / gammaMin, winding * gammaMin, i, "D");
                }
                else
                {
                    double radius = Math.Min(a, b) / Math.Tan(gamma / 2);
                    string kind = Math.Abs(a - b) <= 1e-10 * Math.Max(a, b) ? "C" : a > b ? "A" : "B";
                    if (a > b) Add(result, ref position, ref tangent, a - b, 0, 0, i, kind);
                    Add(result, ref position, ref tangent, radius * gamma, radius, winding * gamma, i, kind);
                    if (b > a) Add(result, ref position, ref tangent, b - a, 0, 0, i, kind);
                }
                ProjectionPoint midpoint = (points[i] + points[i + 1]) * 0.5;
                result.MaximumMidpointError = Math.Max(result.MaximumMidpointError, (position - midpoint).Length);
            }
            Add(result, ref position, ref tangent, lengths[lengths.Length - 1] / 2, 0, 0, points.Length - 2, "端部");
            result.Valid = true;
            result.Status = String.Format("论文连续近似 · D类 {0} 段 · 中点最大偏差 {1:0.###} mm",
                result.ModifiedCount, result.MaximumMidpointError);
            List<double> sortedLengths = new List<double>(lengths);
            sortedLengths.Sort();
            double typicalLength = sortedLengths[sortedLengths.Count / 2];
            if (result.MaximumMidpointError > Math.Max(5.0, typicalLength * 0.2))
                result.Status += " · D类近似偏离实测较大，建议叠加实测折线";
            return result;
        }

        private static CrossSectionCurve Fail(CrossSectionCurve curve, string status)
        { curve.Status = status; return curve; }

        private static void Add(CrossSectionCurve curve, ref ProjectionPoint position,
            ref ProjectionPoint tangent, double length, double radius, double turn, int joint, string kind)
        {
            if (length < 1e-12) return;
            CurveSegment segment = new CurveSegment
            {
                Start = position, Tangent = tangent, Length = length,
                Radius = radius, Turn = turn, JointIndex = joint, Case = kind
            };
            // Chord sagitta <= 0.15 mm for normal ranges; cap pathological output.
            double step = radius > 0 ? 2 * Math.Acos(Math.Max(-1, Math.Min(1, 1 - 0.15 / radius))) : 1;
            int count = radius == 0 ? 1 : Math.Max(2, Math.Min(512, (int)Math.Ceiling(Math.Abs(turn) / Math.Max(1e-5, step))));
            segment.Samples = new ProjectionPoint[count + 1];
            for (int i = 0; i <= count; i++) segment.Samples[i] = segment.At((double)i / count);
            curve.Segments.Add(segment);
            position = segment.At(1);
            tangent = segment.TangentAt(1);
        }
    }

    internal static class FrameGeometry
    {
        public static void Prepare(RobotFrame frame, RobotProfile profile,
            RollingAxisState previous, double gamma, bool legacy = false)
        {
            if (!frame.RollingAxis.HasBasis) RollingProjection.SetBasis(frame.RollingAxis, previous, legacy);
            frame.CurveGammaRadians = gamma;
            frame.Projected = RollingProjection.Project(frame.Points, frame.RollingAxis);
            frame.TargetProjected = RollingProjection.Project(frame.TargetPoints, frame.RollingAxis);
            if (frame.RollingAxis.Valid)
            {
                frame.Curve = CrossSectionCurve.Build(frame.Projected, gamma);
                frame.TargetCurve = CrossSectionCurve.Build(frame.TargetProjected, gamma);
            }
            else
            {
                frame.Curve = new CrossSectionCurve { Status = "无可靠轴，仅显示原始折线" };
                frame.TargetCurve = new CrossSectionCurve { Status = "无可靠轴" };
            }
            frame.GeometryNote = legacy ? "旧记录兼容投影基；连续近似由新版重建" : "";
            bool alternating = true;
            bool equalLength = true;
            for (int i = 0; i < profile.Joints.Count; i++)
            {
                alternating &= profile.Joints[i].AxisType == (i % 2 == 0 ? JointAxisType.Yaw : JointAxisType.Pitch);
                equalLength &= Math.Abs(profile.Joints[i].LinkLength - profile.HeadLength) < 1e-6;
            }
            if (!alternating || !equalLength)
                frame.GeometryNote += " 非标准交替等长结构：使用论文几何方法的扩展";
        }
    }
}
