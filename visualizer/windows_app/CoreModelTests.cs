using System;
using System.Collections.Generic;

namespace SnakeRobotVisualizer
{
    internal static class CoreModelTests
    {
        private static int failures;

        private static void Expect(string name, bool condition)
        {
            if (!condition)
            {
                failures++;
                Console.Error.WriteLine(name + " failed");
            }
            else
            {
                Console.WriteLine(name + " passed");
            }
        }

        private static void ExpectNear(string name, double actual, double expected, double tolerance)
        {
            Expect(name, Math.Abs(actual - expected) <= tolerance);
        }

        private static int Main()
        {
            RobotProfile two = RobotProfile.CreateDefaultTwoJoint();
            Expect("default two-joint profile validates", two.Validate().Count == 0);
            Expect("default IDs preserved", two.Joints[0].ServoId == 3 && two.Joints[1].ServoId == 41);
            Expect("alternating axes preserved",
                two.Joints[0].AxisType == JointAxisType.Yaw &&
                two.Joints[1].AxisType == JointAxisType.Pitch);

            RobotProfile twentyEight = RobotProfile.CreateAlternating(28, "test");
            double[] angles = new double[28];
            for (int i = 0; i < angles.Length; i++)
            {
                angles[i] = 18.0 * Math.Sin(i * 0.43);
            }
            KinematicResult result = Kinematics.Forward(twentyEight, angles);
            Expect("28 joints produce 30 skeleton points", result.Points.Length == 30);
            Expect("28 joint orientations produced", result.JointOrientations.Length == 28);

            RobotProfile oneYaw = RobotProfile.CreateAlternating(1, "known yaw");
            oneYaw.HeadLength = 10;
            oneYaw.Joints[0].LinkLength = 20;
            oneYaw.Joints[0].AxisType = JointAxisType.Yaw;
            Point3D yawEnd = Kinematics.Forward(oneYaw, new double[] { 90 }).Points[2];
            ExpectNear("known yaw end X", yawEnd.X, -10, 1e-9);
            ExpectNear("known yaw end Y", yawEnd.Y, -20, 1e-9);
            ExpectNear("known yaw end Z", yawEnd.Z, 0, 1e-9);

            RobotProfile onePitch = oneYaw.Clone();
            onePitch.Joints[0].AxisType = JointAxisType.Pitch;
            Point3D pitchEnd = Kinematics.Forward(onePitch, new double[] { 90 }).Points[2];
            ExpectNear("known pitch end X", pitchEnd.X, -10, 1e-9);
            ExpectNear("known pitch end Y", pitchEnd.Y, 0, 1e-9);
            ExpectNear("known pitch end Z", pitchEnd.Z, 20, 1e-9);

            JointConfig wrapJoint = new JointConfig
            {
                Zero = 4090,
                Direction = 1,
                MinimumAngle = -90,
                MaximumAngle = 90
            };
            ExpectNear(
                "encoder wrap remains continuous",
                Kinematics.CalibrateDegrees(10, wrapJoint),
                16.0 * 360.0 / 4096.0,
                1e-9);
            wrapJoint.Direction = -1;
            ExpectNear(
                "joint direction reverses calibrated angle",
                Kinematics.CalibrateDegrees(10, wrapJoint),
                -16.0 * 360.0 / 4096.0,
                1e-9);

            Point3D[] helix = new Point3D[40];
            for (int i = 0; i < helix.Length; i++)
            {
                double x = i * 18.0;
                double phase = i * 0.48;
                helix[i] = new Point3D(x, 45.0 * Math.Cos(phase), 45.0 * Math.Sin(phase));
            }
            RollingAxisState axis = RollingAxisEstimator.Estimate(helix);
            Expect("helix rolling axis valid", axis.Valid);
            Expect("helix rolling axis follows longitudinal direction", Math.Abs(axis.Direction.X) > 0.95);
            Expect("helix rolling axis confidence useful", axis.Confidence > 0.7);

            Point3D[] shortHelix = new Point3D[30];
            for (int i = 0; i < shortHelix.Length; i++)
            {
                double phase = i * 0.35;
                shortHelix[i] = new Point3D(
                    i * 1.0,
                    50.0 * Math.Cos(phase),
                    50.0 * Math.Sin(phase));
            }
            RollingAxisState shortAxis = RollingAxisEstimator.Estimate(shortHelix);
            Expect("short thick helix axis follows pitch direction",
                shortAxis.Valid && Math.Abs(shortAxis.Direction.X) > 0.95);
            Expect("short thick helix axis confidence useful", shortAxis.Confidence > 0.7);

            Point3D[] planarCircle = new Point3D[48];
            for (int i = 0; i < planarCircle.Length; i++)
            {
                double phase = 2.0 * Math.PI * i / planarCircle.Length;
                planarCircle[i] = new Point3D(
                    70.0 * Math.Cos(phase),
                    70.0 * Math.Sin(phase),
                    15.0);
            }
            RollingAxisState circleAxis = RollingAxisEstimator.Estimate(planarCircle);
            Expect("planar circle rolling axis is normal to its plane",
                circleAxis.Valid && Math.Abs(circleAxis.Direction.Z) > 0.98);
            Expect("planar circle rolling axis confidence useful", circleAxis.Confidence > 0.8);

            ProjectionPoint[] circleProjection = RollingProjection.Project(
                planarCircle,
                circleAxis);
            double circleRadius = 0;
            for (int i = 0; i < circleProjection.Length; i++)
            {
                circleRadius += Math.Sqrt(
                    circleProjection[i].X * circleProjection[i].X +
                    circleProjection[i].Y * circleProjection[i].Y);
            }
            circleRadius /= circleProjection.Length;
            ExpectNear("planar circle projection preserves radius", circleRadius, 70.0, 2.0);

            Point3D tiltedNormal = new Point3D(1, 2, 3).Normalized();
            Point3D tiltedU = Point3D.Cross(
                new Point3D(0, 0, 1),
                tiltedNormal).Normalized();
            Point3D tiltedV = Point3D.Cross(tiltedNormal, tiltedU).Normalized();
            Point3D tiltedCenter = new Point3D(20, -30, 10);
            Point3D[] tiltedCircle = new Point3D[52];
            for (int i = 0; i < tiltedCircle.Length; i++)
            {
                double phase = 2.0 * Math.PI * i / tiltedCircle.Length;
                tiltedCircle[i] = tiltedCenter +
                    tiltedU * (62.0 * Math.Cos(phase)) +
                    tiltedV * (62.0 * Math.Sin(phase));
            }
            RollingAxisState tiltedAxis = RollingAxisEstimator.Estimate(tiltedCircle);
            Expect("tilted circle rolling axis follows plane normal",
                tiltedAxis.Valid &&
                Math.Abs(Point3D.Dot(tiltedAxis.Direction, tiltedNormal)) > 0.98);
            Expect("tilted circle rolling axis confidence useful", tiltedAxis.Confidence > 0.8);

            ProjectionPoint[] projected = RollingProjection.Project(helix, axis);
            Expect("rolling projection preserves point count", projected.Length == helix.Length);
            double averageRadius = 0;
            for (int i = 0; i < projected.Length; i++)
            {
                averageRadius += Math.Sqrt(
                    projected[i].X * projected[i].X +
                    projected[i].Y * projected[i].Y);
            }
            averageRadius /= projected.Length;
            ExpectNear("rolling projection radius", averageRadius, 45.0, 4.0);

            RollingAxisEstimator tracked = new RollingAxisEstimator();
            RollingAxisState firstAxis = tracked.Update(helix);
            Point3D[] reversed = new Point3D[helix.Length];
            for (int i = 0; i < helix.Length; i++)
            {
                reversed[i] = helix[helix.Length - 1 - i];
            }
            RollingAxisState continuous = tracked.Update(reversed);
            Expect("rolling axis sign remains continuous",
                Point3D.Dot(firstAxis.Direction, continuous.Direction) > 0.99);
            RollingAxisEstimator smoothedEstimator = new RollingAxisEstimator();
            RollingAxisState beforeTurn = smoothedEstimator.Update(helix);
            Point3D[] turnedHelix = new Point3D[helix.Length];
            for (int i = 0; i < helix.Length; i++)
            {
                turnedHelix[i] = new Point3D(
                    helix[i].Y,
                    helix[i].X,
                    helix[i].Z);
            }
            RollingAxisState afterTurn = smoothedEstimator.Update(turnedHelix);
            Expect("rolling axis update is smoothed",
                Math.Abs(Point3D.Dot(beforeTurn.Direction, afterTurn.Direction)) > 0.5 &&
                Math.Abs(afterTurn.Direction.Y) > 0.1);
            Point3D[] degenerate = new Point3D[helix.Length];
            for (int i = 0; i < degenerate.Length; i++)
            {
                degenerate[i] = new Point3D(1, 1, 1);
            }
            RollingAxisState fallback = tracked.Update(degenerate);
            Expect("low-confidence axis holds last valid axis",
                fallback.Valid &&
                fallback.Status.IndexOf("保持上一条有效轴") >= 0);
            tracked.Locked = true;
            RollingAxisState locked = tracked.Update(helix);
            Expect("rolling axis lock reported", locked.Locked);
            tracked.SetManual(new Point3D(3, 4, 5), new Point3D(0, 2, 0));
            tracked.UseManual = true;
            RollingAxisState manualAxis = tracked.Update(degenerate);
            Expect("manual rolling axis overrides automatic estimate",
                manualAxis.Manual && Math.Abs(manualAxis.Direction.Y - 1) < 1e-9);

            RobotProfile duplicate = RobotProfile.CreateDefaultTwoJoint();
            duplicate.Joints[1].ServoId = duplicate.Joints[0].ServoId;
            Expect("duplicate IDs rejected", duplicate.Validate().Count > 0);

            return failures == 0 ? 0 : 1;
        }
    }
}
