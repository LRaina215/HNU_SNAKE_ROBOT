using System;
using System.Collections.Generic;

namespace SnakeRobotVisualizer
{
    internal sealed class RollingAxisEstimator
    {
        private static readonly Point3D[] CoarseDirections = BuildCoarseDirections();
        private RollingAxisState lastValid = new RollingAxisState();
        private RollingAxisState manual = new RollingAxisState
        {
            Point = new Point3D(0, 0, 0),
            Direction = new Point3D(1, 0, 0),
            Confidence = 1.0,
            Valid = true,
            Manual = true,
            Status = "手动轴"
        };
        private bool useManual;
        private bool locked;
        private double smoothing = 0.22;
        public RollingAxisState RawEstimate { get; private set; }

        public void ResetTracking()
        {
            lastValid = new RollingAxisState();
            RawEstimate = new RollingAxisState();
            locked = false;
        }

        public bool UseManual
        {
            get { return useManual; }
            set { useManual = value; }
        }

        public bool Locked
        {
            get { return locked; }
            set { locked = value; }
        }

        public double Smoothing
        {
            get { return smoothing; }
            set { smoothing = Math.Max(0.01, Math.Min(1.0, value)); }
        }

        public RollingAxisState Current
        {
            get
            {
                RollingAxisState state = useManual ? manual.Clone() : lastValid.Clone();
                state.Locked = locked;
                return state;
            }
        }

        public void SetManual(Point3D point, Point3D direction)
        {
            manual.Point = point;
            manual.Direction = direction.Normalized();
            manual.Confidence = 1.0;
            manual.Valid = true;
            manual.Manual = true;
            manual.Status = "手动轴";
        }

        public void UseCurrentAsManual()
        {
            if (lastValid.Valid)
            {
                SetManual(lastValid.Point, lastValid.Direction);
            }
            useManual = true;
        }

        public RollingAxisState Update(Point3D[] points)
        {
            return Update(points, true);
        }

        public RollingAxisState Update(Point3D[] points, bool reliable)
        {
            RollingAxisState estimate = Estimate(points);
            if (!reliable)
            {
                estimate.Confidence = 0;
                estimate.Valid = false;
                estimate.Status = "关节数据失效，无法更新自动轴";
            }
            RawEstimate = estimate.Clone();
            if (useManual)
            {
                RollingAxisState state = manual.Clone();
                state.Locked = locked;
                return state;
            }
            if (locked && lastValid.Valid)
            {
                RollingAxisState lockedState = lastValid.Clone();
                lockedState.Locked = true;
                lockedState.Status = "自动轴已锁定";
                return lockedState;
            }

            if (!estimate.Valid || estimate.Confidence < 0.42)
            {
                if (lastValid.Valid)
                {
                    RollingAxisState fallback = lastValid.Clone();
                    fallback.Confidence = estimate.Confidence;
                    fallback.Fallback = true;
                    fallback.Status = estimate.Status + "；保持上一条有效轴";
                    return fallback;
                }
                estimate.Valid = false;
                estimate.Fallback = true;
                estimate.Status += "；自动轴置信度不足，仅显示后备投影";
                return estimate;
            }

            if (lastValid.Valid)
            {
                if (Point3D.Dot(estimate.Direction, lastValid.Direction) < 0)
                {
                    estimate.Direction = estimate.Direction * -1.0;
                }
                estimate.Point = Point3D.Lerp(lastValid.Point, estimate.Point, smoothing);
                estimate.Direction = Point3D.Lerp(
                    lastValid.Direction,
                    estimate.Direction,
                    smoothing).Normalized();
            }
            estimate.Status = "自动轴正常（螺距角方差）";
            lastValid = estimate.Clone();
            return estimate;
        }

        internal static RollingAxisState Estimate(Point3D[] points)
        {
            RollingAxisState result = new RollingAxisState();
            if (points == null || points.Length < 3)
            {
                result.Status = "骨架点不足";
                return result;
            }

            Point3D centroid = new Point3D();
            for (int i = 0; i < points.Length; i++)
            {
                if (!Finite(points[i]))
                {
                    result.Status = "骨架包含无效数值";
                    return result;
                }
                centroid += points[i];
            }
            centroid /= points.Length;

            double[,] positionCovariance = new double[3, 3];
            for (int i = 0; i < points.Length; i++)
            {
                Point3D p = points[i] - centroid;
                AccumulateOuterProduct(positionCovariance, p);
            }
            double positionTrace = positionCovariance[0, 0] +
                positionCovariance[1, 1] + positionCovariance[2, 2];
            if (positionTrace < 1e-9)
            {
                result.Point = centroid;
                result.Status = "骨架退化";
                return result;
            }

            List<Point3D> tangents = new List<Point3D>();
            // The paper's two-link chord assumes a regular joint spacing.
            // MuJoCo's head/tail links are much shorter than its interior
            // links, so their boundary chords are not comparable pitch-angle
            // samples. Keep all skeletal points for projection and axis origin,
            // but exclude only those biased boundary chords from the fit.
            int firstChord = 1;
            int lastChord = points.Length - 2;
            if (points.Length >= 12)
            {
                List<double> interiorLengths = new List<double>();
                for (int i = 1; i < points.Length - 2; i++)
                    interiorLengths.Add((points[i + 1] - points[i]).Length);
                interiorLengths.Sort();
                double typicalLength = interiorLengths[interiorLengths.Count / 2];
                if ((points[1] - points[0]).Length < typicalLength * 0.8)
                    firstChord += 2;
                if ((points[points.Length - 1] - points[points.Length - 2]).Length < typicalLength * 0.8)
                    lastChord -= 2;
            }
            for (int i = firstChord; i <= lastChord; i++)
            {
                // Skip one adjacent point on either side, as in the paper's
                // pitch-angle estimate for alternating yaw/pitch joints.
                Point3D chord = points[i + 1] - points[i - 1];
                if (chord.Length > 1e-8)
                {
                    tangents.Add(chord.Normalized());
                }
            }

            double[] positionValues;
            Point3D[] positionVectors;
            SymmetricEigen(positionCovariance, out positionValues, out positionVectors);
            Point3D extentAxis = positionVectors[2];

            if (tangents.Count < 3)
            {
                result.Point = centroid;
                result.Direction = OrientAxis(extentAxis, points, tangents);
                result.Confidence = 0.25;
                result.Valid = true;
                result.Manual = false;
                result.Status = "骨架段数不足，使用伸展方向";
                return result;
            }

            Point3D meanTangent = new Point3D();
            for (int i = 0; i < tangents.Count; i++)
            {
                meanTangent += tangents[i];
            }
            meanTangent /= tangents.Count;

            double[,] tangentCovariance = new double[3, 3];
            for (int i = 0; i < tangents.Count; i++)
            {
                AccumulateOuterProduct(
                    tangentCovariance,
                    tangents[i] - meanTangent);
            }
            ScaleMatrix(tangentCovariance, 1.0 / tangents.Count);

            double[] tangentValues;
            Point3D[] tangentVectors;
            SymmetricEigen(tangentCovariance, out tangentValues, out tangentVectors);
            double tangentSpread = Math.Max(0,
                tangentValues[0] + tangentValues[1] + tangentValues[2]);

            Point3D direction;
            double pitchVariance;
            double confidence;
            string status;

            // A nearly straight robot has no unique pitch-variance minimum.
            // Preserve the intuitive longitudinal axis in that special case.
            if (tangentSpread < 0.015)
            {
                direction = meanTangent.Length > 1e-8
                    ? meanTangent.Normalized()
                    : extentAxis;
                pitchVariance = PitchVariance(tangents, direction);
                confidence = 0;
                status = "近直线，螺旋轴不可唯一辨识";
            }
            else
            {
                List<Point3D> seeds = new List<Point3D>
                {
                    tangentVectors[0],
                    tangentVectors[1],
                    tangentVectors[2],
                    extentAxis
                };
                Point3D endDirection = points[points.Length - 1] - points[0];
                if (endDirection.Length > 1e-8)
                {
                    seeds.Add(endDirection.Normalized());
                }

                // Cover the sphere, then refine the eight best coarse starts.
                // No target pitch angle enters this objective.
                double[] coarseValues = new double[CoarseDirections.Length];
                for (int i = 0; i < CoarseDirections.Length; i++)
                    coarseValues[i] = PitchVariance(tangents, CoarseDirections[i]);
                for (int j = 0; j < 4; j++)
                {
                    int best = 0;
                    for (int i = 1; i < CoarseDirections.Length; i++)
                        if (coarseValues[i] < coarseValues[best]) best = i;
                    seeds.Add(CoarseDirections[best]);
                    coarseValues[best] = Double.PositiveInfinity;
                }

                direction = tangentVectors[0];
                pitchVariance = Double.PositiveInfinity;
                for (int i = 0; i < seeds.Count; i++)
                {
                    double candidateVariance;
                    Point3D candidate = RefinePitchAxis(
                        tangents,
                        seeds[i],
                        out candidateVariance);
                    if (candidateVariance < pitchVariance)
                    {
                        direction = candidate;
                        pitchVariance = candidateVariance;
                    }
                }

                double second = Math.Max(0, tangentValues[1]);
                double smallest = Math.Max(0, tangentValues[0]);
                double eigenGap = second < 1e-9
                    ? 0
                    : Math.Max(0, Math.Min(1, (second - smallest) / second));
                double coverage = Math.Max(0,
                    Math.Min(1, tangentSpread / 0.08));
                double pitchStability = Math.Max(0,
                    Math.Min(1, 1.0 - Math.Sqrt(pitchVariance) / 0.35));
                confidence = eigenGap * coverage * pitchStability;
                status = "自动轴候选（螺距角方差）";

                // When the tangent changes span only one direction, the axis
                // is not uniquely observable. Keep a low-confidence fallback.
                if (second < Math.Max(1e-8, tangentSpread * 1e-4))
                {
                    direction = extentAxis;
                    confidence = Math.Min(confidence, 0.25);
                    status = "轴不可唯一辨识，使用伸展方向";
                }
            }

            result.Point = centroid;
            result.Direction = OrientAxis(direction, points, tangents);
            double meanPitch = 0;
            foreach (Point3D tangent in tangents)
                meanPitch += Math.Asin(Math.Min(1, Math.Abs(Point3D.Dot(tangent, result.Direction))));
            result.MeanPitchDegrees = meanPitch / tangents.Count * 180.0 / Math.PI;
            result.PitchDeviationDegrees = Math.Sqrt(PitchVariance(tangents, result.Direction)) * 180.0 / Math.PI;
            result.Confidence = Math.Max(0, Math.Min(1, confidence));
            result.Valid = true;
            result.Manual = false;
            result.Status = status;
            return result;
        }

        private static Point3D[] BuildCoarseDirections()
        {
            Point3D[] directions = new Point3D[256];
            for (int i = 0; i < directions.Length; i++)
            {
                double z = 1.0 - 2.0 * (i + 0.5) / directions.Length;
                double radius = Math.Sqrt(1.0 - z * z);
                double azimuth = i * 2.399963229728653;
                directions[i] = new Point3D(radius * Math.Cos(azimuth), radius * Math.Sin(azimuth), z);
            }
            return directions;
        }

        private static Point3D RefinePitchAxis(
            List<Point3D> tangents,
            Point3D seed,
            out double variance)
        {
            Point3D direction = seed.Normalized();
            variance = PitchVariance(tangents, direction);
            double step = 0.35;
            for (int iteration = 0; iteration < 100 && step > 1e-7 && variance > 1e-18; iteration++)
            {
                Point3D reference = Math.Abs(direction.Z) < 0.85
                    ? new Point3D(0, 0, 1)
                    : new Point3D(0, 1, 0);
                Point3D first = Point3D.Cross(reference, direction).Normalized();
                Point3D second = Point3D.Cross(direction, first).Normalized();
                Point3D[] candidates = new Point3D[]
                {
                    (direction + first * step).Normalized(),
                    (direction - first * step).Normalized(),
                    (direction + second * step).Normalized(),
                    (direction - second * step).Normalized(),
                    (direction + (first + second) * (step * 0.7071)).Normalized(),
                    (direction + (first - second) * (step * 0.7071)).Normalized(),
                    (direction + (first * -1.0 + second) * (step * 0.7071)).Normalized(),
                    (direction - (first + second) * (step * 0.7071)).Normalized()
                };
                bool improved = false;
                for (int i = 0; i < candidates.Length; i++)
                {
                    double candidateVariance = PitchVariance(tangents, candidates[i]);
                    if (candidateVariance + 1e-18 < variance)
                    {
                        direction = candidates[i];
                        variance = candidateVariance;
                        improved = true;
                    }
                }
                if (!improved)
                {
                    step *= 0.5;
                }
            }
            return direction;
        }

        private static double PitchVariance(
            List<Point3D> tangents,
            Point3D direction)
        {
            double mean = 0;
            double sum = 0;
            for (int i = 0; i < tangents.Count; i++)
            {
                double axial = Math.Abs(Point3D.Dot(tangents[i], direction));
                axial = Math.Max(0, Math.Min(1, axial));
                double pitch = Math.Asin(axial);
                double delta = pitch - mean;
                mean += delta / (i + 1);
                sum += delta * (pitch - mean);
            }
            return tangents.Count == 0 ? 0 : Math.Max(0, sum / tangents.Count);
        }

        internal static bool Finite(Point3D p)
        {
            return !(Double.IsNaN(p.X) || Double.IsNaN(p.Y) || Double.IsNaN(p.Z) ||
                Double.IsInfinity(p.X) || Double.IsInfinity(p.Y) || Double.IsInfinity(p.Z));
        }

        private static Point3D OrientAxis(
            Point3D direction,
            Point3D[] points,
            List<Point3D> tangents)
        {
            direction = direction.Normalized();
            Point3D endDirection = points[0] - points[points.Length - 1];
            double endAlignment = endDirection.Length > 1e-8
                ? Point3D.Dot(direction, endDirection.Normalized())
                : 0;
            if (Math.Abs(endAlignment) > 0.08)
            {
                return endAlignment < 0 ? direction * -1.0 : direction;
            }

            double tangentAlignment = 0;
            for (int i = 0; i < tangents.Count; i++)
            {
                tangentAlignment -= Point3D.Dot(direction, tangents[i]);
            }
            if (Math.Abs(tangentAlignment) > tangents.Count * 0.02)
            {
                return tangentAlignment < 0 ? direction * -1.0 : direction;
            }

            double x = Math.Abs(direction.X);
            double y = Math.Abs(direction.Y);
            double z = Math.Abs(direction.Z);
            bool negative = x >= y && x >= z
                ? direction.X < 0
                : y >= z ? direction.Y < 0 : direction.Z < 0;
            return negative ? direction * -1.0 : direction;
        }

        private static void AccumulateOuterProduct(double[,] matrix, Point3D value)
        {
            matrix[0, 0] += value.X * value.X;
            matrix[0, 1] += value.X * value.Y;
            matrix[0, 2] += value.X * value.Z;
            matrix[1, 0] += value.Y * value.X;
            matrix[1, 1] += value.Y * value.Y;
            matrix[1, 2] += value.Y * value.Z;
            matrix[2, 0] += value.Z * value.X;
            matrix[2, 1] += value.Z * value.Y;
            matrix[2, 2] += value.Z * value.Z;
        }

        private static void ScaleMatrix(double[,] matrix, double scale)
        {
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    matrix[row, column] *= scale;
                }
            }
        }

        private static void SymmetricEigen(
            double[,] source,
            out double[] values,
            out Point3D[] vectors)
        {
            double[,] matrix = (double[,])source.Clone();
            double[,] eigenvectors = new double[3, 3]
            {
                { 1, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 1 }
            };

            for (int iteration = 0; iteration < 32; iteration++)
            {
                int p = 0;
                int q = 1;
                double largest = Math.Abs(matrix[0, 1]);
                if (Math.Abs(matrix[0, 2]) > largest)
                {
                    p = 0;
                    q = 2;
                    largest = Math.Abs(matrix[0, 2]);
                }
                if (Math.Abs(matrix[1, 2]) > largest)
                {
                    p = 1;
                    q = 2;
                    largest = Math.Abs(matrix[1, 2]);
                }
                if (largest < 1e-12)
                {
                    break;
                }

                double angle = 0.5 * Math.Atan2(
                    2.0 * matrix[p, q],
                    matrix[q, q] - matrix[p, p]);
                double cosine = Math.Cos(angle);
                double sine = Math.Sin(angle);
                double pp = matrix[p, p];
                double qq = matrix[q, q];
                double pq = matrix[p, q];
                matrix[p, p] = cosine * cosine * pp -
                    2.0 * sine * cosine * pq + sine * sine * qq;
                matrix[q, q] = sine * sine * pp +
                    2.0 * sine * cosine * pq + cosine * cosine * qq;
                matrix[p, q] = 0;
                matrix[q, p] = 0;
                for (int index = 0; index < 3; index++)
                {
                    if (index == p || index == q)
                    {
                        continue;
                    }
                    double ip = matrix[index, p];
                    double iq = matrix[index, q];
                    matrix[index, p] = cosine * ip - sine * iq;
                    matrix[p, index] = matrix[index, p];
                    matrix[index, q] = sine * ip + cosine * iq;
                    matrix[q, index] = matrix[index, q];
                }
                for (int row = 0; row < 3; row++)
                {
                    double vp = eigenvectors[row, p];
                    double vq = eigenvectors[row, q];
                    eigenvectors[row, p] = cosine * vp - sine * vq;
                    eigenvectors[row, q] = sine * vp + cosine * vq;
                }
            }

            values = new double[]
            {
                matrix[0, 0], matrix[1, 1], matrix[2, 2]
            };
            vectors = new Point3D[]
            {
                new Point3D(eigenvectors[0, 0], eigenvectors[1, 0], eigenvectors[2, 0]).Normalized(),
                new Point3D(eigenvectors[0, 1], eigenvectors[1, 1], eigenvectors[2, 1]).Normalized(),
                new Point3D(eigenvectors[0, 2], eigenvectors[1, 2], eigenvectors[2, 2]).Normalized()
            };
            for (int i = 0; i < values.Length - 1; i++)
            {
                for (int j = i + 1; j < values.Length; j++)
                {
                    if (values[j] < values[i])
                    {
                        double value = values[i];
                        values[i] = values[j];
                        values[j] = value;
                        Point3D vector = vectors[i];
                        vectors[i] = vectors[j];
                        vectors[j] = vector;
                    }
                }
            }
        }
    }

    internal struct ProjectionPoint
    {
        public double X;
        public double Y;

        public ProjectionPoint(double x, double y)
        {
            X = x;
            Y = y;
        }
        public double Length { get { return Math.Sqrt(X * X + Y * Y); } }
        public static ProjectionPoint operator +(ProjectionPoint a, ProjectionPoint b)
        { return new ProjectionPoint(a.X + b.X, a.Y + b.Y); }
        public static ProjectionPoint operator -(ProjectionPoint a, ProjectionPoint b)
        { return new ProjectionPoint(a.X - b.X, a.Y - b.Y); }
        public static ProjectionPoint operator *(ProjectionPoint a, double b)
        { return new ProjectionPoint(a.X * b, a.Y * b); }
    }

    internal static class RollingProjection
    {
        public static void SetBasis(RollingAxisState axis, RollingAxisState previous, bool legacy)
        {
            Point3D direction = axis.Valid ? axis.Direction.Normalized() : new Point3D(1, 0, 0);
            Point3D horizontal = new Point3D();
            if (!legacy && previous != null && previous.HasBasis)
                horizontal = previous.Horizontal - direction * Point3D.Dot(previous.Horizontal, direction);
            if (horizontal.Length < 1e-8)
            {
                Point3D reference = Math.Abs(direction.Z) < 0.85
                    ? new Point3D(0, 0, 1) : new Point3D(0, 1, 0);
                horizontal = Point3D.Cross(reference, direction);
                if (!legacy && previous != null && previous.HasBasis &&
                    Point3D.Dot(horizontal, previous.Horizontal) < 0) horizontal = horizontal * -1;
            }
            axis.Horizontal = horizontal.Normalized();
            axis.Vertical = Point3D.Cross(direction, axis.Horizontal).Normalized();
            axis.HasBasis = true;
        }

        public static ProjectionPoint[] Project(Point3D[] points, RollingAxisState axis)
        {
            if (points == null)
            {
                return new ProjectionPoint[0];
            }
            Point3D origin = axis != null ? axis.Point : new Point3D();
            RollingAxisState basis = axis == null ? new RollingAxisState() : axis;
            if (!basis.HasBasis)
            {
                basis = basis.Clone();
                SetBasis(basis, null, true);
            }
            Point3D horizontal = basis.Horizontal;
            Point3D vertical = basis.Vertical;

            ProjectionPoint[] result = new ProjectionPoint[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                if (!RollingAxisEstimator.Finite(points[i])) return new ProjectionPoint[0];
                Point3D relative = points[i] - origin;
                result[i] = new ProjectionPoint(
                    Point3D.Dot(relative, horizontal),
                    Point3D.Dot(relative, vertical));
            }
            return result;
        }
    }
}
