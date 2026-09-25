using System;
using System.Collections.Generic;

namespace SnakeRobotVisualizer
{
    internal struct Point3D
    {
        public double X;
        public double Y;
        public double Z;

        public Point3D(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public double Length
        {
            get { return Math.Sqrt(X * X + Y * Y + Z * Z); }
        }

        public static Point3D operator +(Point3D a, Point3D b)
        {
            return new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        public static Point3D operator -(Point3D a, Point3D b)
        {
            return new Point3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public static Point3D operator *(Point3D a, double value)
        {
            return new Point3D(a.X * value, a.Y * value, a.Z * value);
        }

        public static Point3D operator /(Point3D a, double value)
        {
            return new Point3D(a.X / value, a.Y / value, a.Z / value);
        }

        public static double Dot(Point3D a, Point3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }

        public static Point3D Cross(Point3D a, Point3D b)
        {
            return new Point3D(
                a.Y * b.Z - a.Z * b.Y,
                a.Z * b.X - a.X * b.Z,
                a.X * b.Y - a.Y * b.X);
        }

        public Point3D Normalized()
        {
            double length = Length;
            return length < 1e-12 ? new Point3D(1, 0, 0) : this / length;
        }

        public static Point3D Lerp(Point3D a, Point3D b, double amount)
        {
            return a * (1.0 - amount) + b * amount;
        }
    }

    internal struct Matrix3
    {
        public double M11, M12, M13;
        public double M21, M22, M23;
        public double M31, M32, M33;

        public static Matrix3 Identity
        {
            get
            {
                return new Matrix3
                {
                    M11 = 1,
                    M22 = 1,
                    M33 = 1
                };
            }
        }

        public static Matrix3 RotationY(double angle)
        {
            double c = Math.Cos(angle);
            double s = Math.Sin(angle);
            return new Matrix3
            {
                M11 = c,
                M13 = s,
                M22 = 1,
                M31 = -s,
                M33 = c
            };
        }

        public static Matrix3 RotationZ(double angle)
        {
            double c = Math.Cos(angle);
            double s = Math.Sin(angle);
            return new Matrix3
            {
                M11 = c,
                M12 = -s,
                M21 = s,
                M22 = c,
                M33 = 1
            };
        }

        public static Matrix3 operator *(Matrix3 a, Matrix3 b)
        {
            return new Matrix3
            {
                M11 = a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31,
                M12 = a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32,
                M13 = a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33,
                M21 = a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31,
                M22 = a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32,
                M23 = a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33,
                M31 = a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31,
                M32 = a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32,
                M33 = a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33
            };
        }

        public Point3D Transform(Point3D point)
        {
            return new Point3D(
                M11 * point.X + M12 * point.Y + M13 * point.Z,
                M21 * point.X + M22 * point.Y + M23 * point.Z,
                M31 * point.X + M32 * point.Y + M33 * point.Z);
        }
    }

    internal sealed class KinematicResult
    {
        public Point3D[] Points = new Point3D[0];
        public Matrix3[] JointOrientations = new Matrix3[0];
    }

    internal static class Kinematics
    {
        public const int CountsPerRevolution = 4096;

        public static int ShortestCountDelta(int raw, int zero)
        {
            int half = CountsPerRevolution / 2;
            int value = (raw - zero + half) % CountsPerRevolution;
            if (value < 0)
            {
                value += CountsPerRevolution;
            }
            return value - half;
        }

        public static double CalibrateRadians(int raw, int zero, int direction)
        {
            return direction * ShortestCountDelta(raw, zero)
                * (2.0 * Math.PI / CountsPerRevolution);
        }

        public static double CalibrateDegrees(int raw, JointConfig joint)
        {
            return CalibrateRadians(raw, joint.Zero, joint.Direction) * 180.0 / Math.PI;
        }

        public static int DegreesToRaw(double degrees, JointConfig joint)
        {
            double clamped = Math.Max(joint.MinimumAngle, Math.Min(joint.MaximumAngle, degrees));
            int delta = (int)Math.Round(
                clamped / joint.Direction * CountsPerRevolution / 360.0);
            int raw = (joint.Zero + delta) % CountsPerRevolution;
            return raw < 0 ? raw + CountsPerRevolution : raw;
        }

        public static KinematicResult Forward(RobotProfile profile, IList<double> anglesDegrees)
        {
            if (profile == null)
            {
                throw new ArgumentNullException("profile");
            }
            if (anglesDegrees == null || anglesDegrees.Count != profile.Joints.Count)
            {
                throw new ArgumentException("角度数量必须与关节数量一致");
            }

            int count = profile.Joints.Count;
            Point3D[] points = new Point3D[count + 2];
            Matrix3[] orientations = new Matrix3[count];
            points[0] = new Point3D(0, 0, 0);
            Matrix3 orientation = Matrix3.Identity;
            points[1] = points[0] + orientation.Transform(
                new Point3D(-Math.Max(1.0, profile.HeadLength), 0, 0));

            for (int i = 0; i < count; i++)
            {
                JointConfig joint = profile.Joints[i];
                double radians = anglesDegrees[i] * Math.PI / 180.0;
                Matrix3 localRotation = joint.AxisType == JointAxisType.Yaw
                    ? Matrix3.RotationZ(radians)
                    : Matrix3.RotationY(radians);
                orientation = orientation * localRotation;
                orientations[i] = orientation;
                points[i + 2] = points[i + 1] + orientation.Transform(
                    new Point3D(-Math.Max(1.0, joint.LinkLength), 0, 0));
            }

            return new KinematicResult
            {
                Points = points,
                JointOrientations = orientations
            };
        }

        // Compatibility helper retained for the original two-joint tests.
        public static Point3D[] Forward(double yaw, double pitch, double length)
        {
            RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
            profile.HeadLength = length;
            profile.Joints[0].LinkLength = length;
            profile.Joints[1].LinkLength = length;
            return Forward(
                profile,
                new double[]
                {
                    yaw * 180.0 / Math.PI,
                    pitch * 180.0 / Math.PI
                }).Points;
        }
    }
}
