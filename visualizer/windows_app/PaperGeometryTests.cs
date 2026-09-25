using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SnakeRobotVisualizer
{
    internal static class PaperGeometryTests
    {
        private static int failures;
        private static void Check(bool ok, string name)
        { Console.WriteLine((ok ? "PASS " : "FAIL ") + name); if (!ok) failures++; }
        private static void Near(double actual, double expected, double tolerance, string name)
        { Check(Math.Abs(actual - expected) <= tolerance, name + " error=" + Math.Abs(actual - expected).ToString("G4")); }
        private static double Angle(Point3D a, Point3D b)
        { return Math.Acos(Math.Max(-1, Math.Min(1, Math.Abs(Point3D.Dot(a.Normalized(), b.Normalized()))))) * 180 / Math.PI; }

        private static int Main()
        {
            TestAxis();
            TestProjection();
            TestCurves();
            TestRecording();
            TestPerformance();
            Console.WriteLine(failures == 0 ? "ALL PAPER GEOMETRY TESTS PASSED" : failures + " PAPER GEOMETRY TESTS FAILED");
            return failures == 0 ? 0 : 1;
        }

        private static Point3D[] Helix(double rise, double ry, double rz, bool irregular)
        {
            Point3D[] points = new Point3D[60];
            double x = 0;
            for (int i = 0; i < points.Length; i++)
            {
                double phase = i * 0.19;
                double r = irregular ? 1 + 0.12 * Math.Cos(3 * phase) : 1;
                points[i] = new Point3D(x, ry * r * Math.Cos(phase), rz * r * Math.Sin(phase));
                if (i > 0 && irregular)
                {
                    double dy = points[i].Y - points[i - 1].Y;
                    double dz = points[i].Z - points[i - 1].Z;
                    x += Math.Sqrt(dy * dy + dz * dz) * Math.Tan(0.25);
                    points[i].X = x;
                }
                else x += rise;
            }
            return points;
        }

        private static void TestAxis()
        {
            Matrix3 rotation = Matrix3.RotationY(0.62) * Matrix3.RotationZ(-1.07);
            foreach (double rise in new double[] { 0, 0.6, 12 })
            {
                Point3D[] p = Helix(rise, 70, 70, false);
                RollingAxisState axis = RollingAxisEstimator.Estimate(p);
                Check(Angle(axis.Direction, new Point3D(1, 0, 0)) <= 1, "A1 canonical axis <=1deg rise=" + rise);
                if (rise > 0) Check(Point3D.Dot(axis.Direction, p[0] - p[p.Length - 1]) > 0, "A1 headward sign");
                Point3D[] turned = p.Select(x => rotation.Transform(x) + new Point3D(57, 3, -8)).ToArray();
                RollingAxisState rotated = RollingAxisEstimator.Estimate(turned);
                Check(Angle(rotated.Direction, rotation.Transform(axis.Direction)) <= 0.2, "A1 rotation covariance <=0.2deg");
            }

            RobotProfile profile = RobotProfile.CreateAlternating(28, "axis reference");
            Point3D[] jointCircle = Kinematics.Forward(profile,
                Enumerable.Range(0, 28).Select(i => i % 2 == 0 ? 30.0 : 0.0).ToArray()).Points;
            Check(Angle(RollingAxisEstimator.Estimate(jointCircle).Direction, new Point3D(0, 0, 1)) <= 1,
                "A1 28-joint alternating FK planar coil <=1deg");
            Matrix3 pair = Matrix3.RotationZ(Math.PI / 9) * Matrix3.RotationY(Math.PI / 9);
            Point3D screw = new Point3D(pair.M32 - pair.M23, pair.M13 - pair.M31, pair.M21 - pair.M12).Normalized();
            KinematicResult jointHelix = Kinematics.Forward(profile, Enumerable.Repeat(20.0, 28).ToArray());
            RollingAxisState jointAxis = RollingAxisEstimator.Estimate(jointHelix.Points);
            Check(Angle(jointAxis.Direction, screw) <= 1, "A1 28-joint FK screw helix agrees with independent pair rotation <=1deg");
            RobotFrame jointFrame = new RobotFrame { Points = jointHelix.Points, TargetPoints = jointHelix.Points,
                RollingAxis = jointAxis, RawAxis = jointAxis.Clone() };
            FrameGeometry.Prepare(jointFrame, profile, null, Math.PI * 0.01);
            Check(jointFrame.Curve.Valid && jointFrame.Projected.Length == 30,
                "C1 known 28-joint helical angles complete FK/axis/projection/curve chain");
            List<Point3D[]> examples = new List<Point3D[]> { Helix(0, 65, 45, true) };
            Random noise = new Random(20260922);
            for (int test = 0; test < 6; test++)
            {
                double[] angles = Enumerable.Range(0, 28).Select(i =>
                    30 * Math.Sin(i * 0.39 + test * 0.6) + (noise.NextDouble() - 0.5) * 2).ToArray();
                examples.Add(Kinematics.Forward(profile, angles).Points);
            }
            for (int i = 0; i < examples.Count; i++)
            {
                RollingAxisState actual = RollingAxisEstimator.Estimate(examples[i]);
                double value = ReferenceVariance(examples[i], actual.Direction);
                double reference = ReferenceBest(examples[i]);
                Console.WriteLine("REFERENCE " + i + " actual=" + value.ToString("G12") + " reference=" + reference.ToString("G12"));
                Check(value <= reference + 1e-7, "A1 independent objective comparison " + i);
            }

            Point3D[] straight = Enumerable.Range(0, 30).Select(i => new Point3D(-i * 89.5, 0, 0)).ToArray();
            Check(RollingAxisEstimator.Estimate(straight).Confidence < 0.42, "A2 straight is unobservable");
            RollingAxisEstimator estimator = new RollingAxisEstimator();
            RollingAxisState first = estimator.Update(straight);
            Check(!first.Valid && first.Fallback, "A2 first ambiguous frame is labelled fallback");
            RollingAxisState good = estimator.Update(Helix(4, 50, 50, false));
            foreach (Point3D[] bad in new Point3D[][] { straight, new Point3D[30],
                new Point3D[] { new Point3D(Double.NaN, 0, 0), new Point3D(), new Point3D() },
                Kinematics.Forward(RobotProfile.CreateDefaultTwoJoint(), new double[] { 20, -15 }).Points })
            {
                RollingAxisState held = estimator.Update(bad);
                Check(held.Confidence < 0.42 && held.Fallback, "A2 low-confidence hold");
                Near((held.Direction - good.Direction).Length, 0, 1e-9, "A2 direction hold");
                Near((held.Point - good.Point).Length, 0, 1e-9, "A2 origin hold");
            }
            RollingAxisState missing = estimator.Update(Helix(6, 40, 40, false), false);
            Check(missing.Confidence == 0 && !estimator.RawEstimate.Valid, "A2 failed joint invalidates raw estimate");
            Near((missing.Direction - good.Direction).Length, 0, 1e-9, "A2 failed joint keeps display axis");
            Point3D[] changed = Helix(4, 50, 50, false).Select(rotation.Transform).ToArray();
            RollingAxisState smoothed = estimator.Update(changed);
            Check(Angle(smoothed.Direction, estimator.RawEstimate.Direction) > 1, "A2 raw and smoothed axes stored separately");
        }

        // Independent equation (7)-(9), using unnormalised chords and acos.
        private static double ReferenceVariance(Point3D[] points, Point3D axis)
        {
            double[] pitch = new double[points.Length - 2];
            for (int i = 1; i < points.Length - 1; i++)
            {
                Point3D w = points[i - 1] - points[i + 1];
                Point3D projected = w - axis * Point3D.Dot(w, axis);
                double ratio = projected.Length < 1e-12 ? 0 : Point3D.Dot(w, projected) / (w.Length * projected.Length);
                pitch[i - 1] = Math.Acos(Math.Max(-1, Math.Min(1, ratio)));
            }
            double mean = pitch.Average();
            return pitch.Select(x => (x - mean) * (x - mean)).Average();
        }

        private static Point3D Spherical(double azimuth, double elevation)
        { return new Point3D(Math.Cos(elevation) * Math.Cos(azimuth), Math.Cos(elevation) * Math.Sin(azimuth), Math.Sin(elevation)); }

        private static double ReferenceBest(Point3D[] points)
        {
            List<double[]> candidates = new List<double[]>();
            for (int i = 0; i < 4096; i++)
            {
                double az = i * 2.399963229728653;
                double el = Math.Asin(1 - 2 * (i + 0.5) / 4096);
                candidates.Add(new double[] { ReferenceVariance(points, Spherical(az, el)), az, el });
            }
            double best = Double.PositiveInfinity;
            foreach (double[] initial in candidates.OrderBy(x => x[0]).Take(12))
            {
                double value = initial[0], az = initial[1], el = initial[2], step = 0.2;
                for (int iteration = 0; iteration < 250 && step > 1e-8; iteration++)
                {
                    double next = value, nextAz = az, nextEl = el;
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            double v = ReferenceVariance(points, Spherical(az + dx * step, el + dy * step));
                            if (v < next) { next = v; nextAz = az + dx * step; nextEl = el + dy * step; }
                        }
                    if (next < value) { value = next; az = nextAz; el = nextEl; }
                    else step /= 2;
                }
                best = Math.Min(best, value);
            }
            return best;
        }

        private static void TestProjection()
        {
            RollingAxisState previous = null;
            foreach (double z in new double[] { 0.8499, 0.8501 })
            {
                RollingAxisState axis = new RollingAxisState { Valid = true, Direction = new Point3D(Math.Sqrt(1 - z * z), 0, z) };
                RollingProjection.SetBasis(axis, previous, false);
                Near(axis.Horizontal.Length, 1, 1e-10, "A3 horizontal unit");
                Near(axis.Vertical.Length, 1, 1e-10, "A3 vertical unit");
                Near(Point3D.Dot(axis.Horizontal, axis.Vertical), 0, 1e-10, "A3 orthogonal basis");
                Near(Point3D.Dot(axis.Horizontal, axis.Direction), 0, 1e-10, "A3 basis perpendicular to axis");
                if (previous != null) Check(Point3D.Dot(previous.Horizontal, axis.Horizontal) > Math.Cos(0.1 * Math.PI / 180), "A3 no threshold jump <0.1deg");
                previous = axis;
            }
        }

        private static void TestCurves()
        {
            double gamma = Math.PI * 0.01;
            foreach (double[] lengths in new double[][] { new double[] { 4, 2 }, new double[] { 2, 4 }, new double[] { 2, 2 } })
            {
                ProjectionPoint[] points = { new ProjectionPoint(0, 0), new ProjectionPoint(lengths[0], 0), new ProjectionPoint(lengths[0], lengths[1]) };
                CrossSectionCurve curve = CrossSectionCurve.Build(points, gamma);
                string kind = lengths[0] > lengths[1] ? "A" : lengths[0] < lengths[1] ? "B" : "C";
                Check(curve.Valid && curve.ModifiedCount == 0, "C1 case " + kind + " valid");
                CurveSegment arc = curve.Segments.First(x => x.Radius > 0);
                Check(arc.Case == kind, "C1 classification " + kind);
                Near(arc.Radius, Math.Min(lengths[0], lengths[1]) / 2, 1e-8, "C1 radius " + kind);
                Near(curve.Segments.Where(x => x.Radius == 0 && x.Case != "端部").Sum(x => x.Length),
                    Math.Abs(lengths[0] - lengths[1]) / 2, 1e-8, "C1 residual line length " + kind);
                Near(arc.Turn, Math.PI / 2, 1e-8, "C1 arc angle " + kind);
                Near(curve.MaximumMidpointError, 0, 1e-8, "C1 exact midpoints " + kind);
                Near((curve.Segments.Last().At(1) - points.Last()).Length, 0, 1e-8, "C1 tail position " + kind);
                Continuity(curve, kind);
            }
            foreach (ProjectionPoint last in new ProjectionPoint[] { new ProjectionPoint(200, 0), new ProjectionPoint(100, -100) })
            {
                CrossSectionCurve curve = CrossSectionCurve.Build(new ProjectionPoint[] { new ProjectionPoint(0, 0), new ProjectionPoint(100, 0), last }, gamma, 1);
                CurveSegment arc = curve.Segments.First(x => x.Radius > 0);
                Check(curve.Valid && curve.ModifiedCount == 1, "C1 D class marked");
                Near(arc.Radius * arc.Turn, 100, 1e-8, "C1 D preserves specified arc length");
                Near(arc.Turn, gamma, 1e-12, "C1 D prescribed gamma");
                Check(curve.MaximumMidpointError > 0, "C1 D deviation reported");
                Continuity(curve, "D");
            }
            ProjectionPoint[] circle = Enumerable.Range(0, 60).Select(i => new ProjectionPoint(70 * Math.Cos(i * 0.2), 70 * Math.Sin(i * 0.2))).ToArray();
            CrossSectionCurve multi = CrossSectionCurve.Build(circle, gamma);
            Check(multi.Valid && multi.ModifiedCount == 0, "C2 multicircle retains traversal order");
            Near((multi.Segments.First().Start - multi.Segments.Last().At(1)).Length, (circle.First() - circle.Last()).Length, 1e-8, "C2 open ends not closed");
            CrossSectionCurve clockwise = CrossSectionCurve.Build(circle.Select(x => new ProjectionPoint(x.X, -x.Y)).ToArray(), gamma);
            Check(clockwise.Valid && clockwise.ModifiedCount == 0 && clockwise.Segments.Any(x => x.Turn < 0), "C2 clockwise winding preserves inward arcs");
            Check(!CrossSectionCurve.Build(new ProjectionPoint[] { new ProjectionPoint(), new ProjectionPoint(), new ProjectionPoint(2, 0) }, gamma).Valid, "C2 zero projected link handled");
            Check(!CrossSectionCurve.Build(new ProjectionPoint[] { new ProjectionPoint(), new ProjectionPoint(1, 0), new ProjectionPoint(0, 0.000001) }, gamma).Valid, "C2 reversal handled");
        }

        private static void Continuity(CrossSectionCurve curve, string kind)
        {
            double positionError = 0, tangentError = 0;
            for (int i = 1; i < curve.Segments.Count; i++)
            {
                positionError = Math.Max(positionError, (curve.Segments[i - 1].At(1) - curve.Segments[i].Start).Length);
                tangentError = Math.Max(tangentError, (curve.Segments[i - 1].TangentAt(1) - curve.Segments[i].Tangent).Length);
            }
            Near(positionError, 0, 1e-8, "C1 position continuity " + kind);
            Near(tangentError, 0, 1e-8, "C1 tangent continuity " + kind);
        }

        private static RobotFrame MakeFrame(RobotProfile profile, int index, RollingAxisState previous)
        {
            double[] angles = Enumerable.Range(0, 28).Select(i => 25 * Math.Sin(i * 0.4 + index * 0.03)).ToArray();
            KinematicResult fk = Kinematics.Forward(profile, angles);
            RobotFrame frame = new RobotFrame
            {
                Sequence = index + 1, Points = fk.Points, TargetPoints = fk.Points, JointOrientations = fk.JointOrientations,
                Joints = angles.Select(x => new JointRuntimeState { AngleDegrees = x, TargetDegrees = x, Online = true, Status = "测试,含逗号" }).ToArray(),
                RollingAxis = RollingAxisEstimator.Estimate(fk.Points)
            };
            frame.RawAxis = frame.RollingAxis.Clone();
            FrameGeometry.Prepare(frame, profile, previous, 0.01 * Math.PI + index * 0.0001);
            return frame;
        }

        private static void TestRecording()
        {
            string root = Path.Combine(Path.GetTempPath(), "SnakePaperAcceptance-" + Guid.NewGuid().ToString("N"));
            RobotProfile profile = RobotProfile.CreateAlternating(28, "roundtrip");
            List<RobotFrame> originals = new List<RobotFrame>();
            try
            {
                string folder;
                using (ExperimentRecorder recorder = new ExperimentRecorder())
                {
                    folder = recorder.Start(root, "precision", profile);
                    RollingAxisState previous = null;
                    for (int i = 0; i < 6; i++)
                    {
                        RobotFrame frame = MakeFrame(profile, i, previous);
                        previous = frame.RollingAxis;
                        if (i == 5) frame.RollingAxis.Valid = false;
                        FrameGeometry.Prepare(frame, profile, null, frame.CurveGammaRadians);
                        originals.Add(frame);
                        recorder.Append(frame, profile);
                    }
                }
                PlaybackData playback = ExperimentPlayback.Load(folder);
                double maxError = 0;
                foreach (int i in new int[] { 4, 0, 3, 1, 2, 5 })
                {
                    RobotFrame actual = playback.Frames[i], original = originals[i];
                    for (int j = 0; j < original.Projected.Length; j++) maxError = Math.Max(maxError, (actual.Projected[j] - original.Projected[j]).Length);
                    if (original.Curve.Valid)
                        for (int j = 0; j < original.Curve.Segments.Count; j++)
                            maxError = Math.Max(maxError, (original.Curve.Segments[j].At(0.37) - actual.Curve.Segments[j].At(0.37)).Length);
                    Near(actual.CurveGammaRadians, original.CurveGammaRadians, 1e-12, "R1 per-frame gamma restored " + i);
                    Check(actual.RollingAxis.Valid == original.RollingAxis.Valid, "R1 valid flag restored " + i);
                }
                Near(maxError, 0, 1e-6, "R1 random seek geometry roundtrip mm");
                ExperimentPlayback.EnsureRecomputed(playback);
                Point3D cached = playback.RecomputedFrames[3].RollingAxis.Horizontal;
                ExperimentPlayback.EnsureRecomputed(playback);
                Near((cached - playback.RecomputedFrames[3].RollingAxis.Horizontal).Length, 0, 1e-10, "R1 recomputation cache independent of seek order");

                // Strip the added field using the original 20-column boundary.
                // Joint status in this fixture contains a comma, so respect CSV quotes.
                string csv = Path.Combine(folder, "frames.csv");
                File.WriteAllLines(csv, File.ReadAllLines(csv).Select(FirstTwenty).ToArray());
                PlaybackData legacy = ExperimentPlayback.Load(folder);
                Check(legacy.Frames.Count == 6 && legacy.Frames[0].GeometryNote.Contains("旧记录"), "R1 legacy 20-column CSV loads");
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        private static string FirstTwenty(string line)
        {
            bool quoted = false;
            int columns = 0;
            for (int i = 0; i < line.Length; i++)
            {
                if (line[i] == '"')
                {
                    if (quoted && i + 1 < line.Length && line[i + 1] == '"') i++;
                    else quoted = !quoted;
                }
                else if (line[i] == ',' && !quoted && ++columns == 20) return line.Substring(0, i);
            }
            return line;
        }

        private static void TestPerformance()
        {
            RobotProfile profile = RobotProfile.CreateAlternating(28, "complete pipeline");
            Stopwatch watch = Stopwatch.StartNew();
            RollingAxisState previous = null;
            for (int i = 0; i < 1000; i++) previous = MakeFrame(profile, i, previous).RollingAxis;
            watch.Stop();
            Console.WriteLine("P1 complete 1000-frame pipeline ms=" + watch.ElapsedMilliseconds);
            Check(watch.Elapsed.TotalMilliseconds / 1000 < 50, "P1 complete pipeline mean <50ms");
        }
    }
}
