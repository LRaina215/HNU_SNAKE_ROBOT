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
            return 0;
        }
    }
}
