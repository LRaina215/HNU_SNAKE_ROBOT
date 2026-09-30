using System;

namespace SnakeRobotVisualizer
{
    internal static class CameraReferenceTests
    {
        private const double Tolerance = 1e-9;

        private static void ExpectNear(string name, Point3D point, double x, double y, double z)
        {
            if (Math.Abs(point.X - x) > Tolerance ||
                Math.Abs(point.Y - y) > Tolerance ||
                Math.Abs(point.Z - z) > Tolerance)
            {
                throw new Exception(name + " failed: " + point.X + ", " + point.Y + ", " + point.Z);
            }
            Console.WriteLine(name + " passed");
        }

        private static int Main()
        {
            Point3D[] straight = Kinematics.Forward(0.0, 0.0, 89.5);
            Point3D[] bent = Kinematics.Forward(0.7, -0.5, 89.5);

            ExpectNear(
                "world origin remains fixed for straight pose",
                SnakeCanvas.ComputeViewCenter(straight, ViewReference.WorldOrigin),
                0.0, 0.0, 0.0);
            ExpectNear(
                "world origin remains fixed for bent pose",
                SnakeCanvas.ComputeViewCenter(bent, ViewReference.WorldOrigin),
                0.0, 0.0, 0.0);

            Point3D followed = SnakeCanvas.ComputeViewCenter(bent, ViewReference.RobotCenter);
            if (Math.Abs(followed.X) < Tolerance &&
                Math.Abs(followed.Y) < Tolerance &&
                Math.Abs(followed.Z) < Tolerance)
            {
                throw new Exception("robot-center mode did not follow the bent pose");
            }
            Console.WriteLine("robot-center mode follows pose passed");

            // 轴竖直显示：X/Y 与二维投影一致，Z 为沿 rolling 轴的坐标，且不产生镜像。
            RollingAxisState axis = new RollingAxisState
            {
                Valid = true,
                Point = new Point3D(10, -20, 5),
                Direction = new Point3D(1, 2, 2).Normalized()
            };
            RollingProjection.SetBasis(axis, null, false);
            Point3D[] pose = Kinematics.Forward(0.7, -0.5, 89.5);
            RobotFrame aligned = SnakeCanvas.AlignToRollingAxis(new RobotFrame { Points = pose, RollingAxis = axis });
            ProjectionPoint[] projected = RollingProjection.Project(pose, axis);
            for (int i = 0; i < pose.Length; i++)
            {
                ExpectNear("aligned point " + i, aligned.Points[i], projected[i].X, projected[i].Y,
                    Point3D.Dot(pose[i] - axis.Point, axis.Direction));
            }
            Point3D a = pose[1] - pose[0], b = pose[2] - pose[1];
            Point3D alignedCross = Point3D.Cross(aligned.Points[1] - aligned.Points[0], aligned.Points[2] - aligned.Points[1]);
            Point3D expectedCross = Point3D.Cross(a, b);
            if (Math.Abs(alignedCross.Length - expectedCross.Length) > Tolerance ||
                Math.Abs(alignedCross.Z - Point3D.Dot(expectedCross, axis.Direction)) > Tolerance)
            {
                throw new Exception("aligned view is not a proper rotation");
            }
            Console.WriteLine("aligned view is a proper rotation passed");
            return 0;
        }
    }
}
