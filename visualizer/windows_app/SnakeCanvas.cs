using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal enum ViewReference
    {
        WorldOrigin,
        RobotCenter
    }

    internal static class VisualPalette
    {
        public static readonly Color[] Links =
        {
            Color.FromArgb(65, 152, 255),
            Color.FromArgb(255, 170, 55),
            Color.FromArgb(72, 205, 146),
            Color.FromArgb(202, 126, 255),
            Color.FromArgb(255, 105, 130),
            Color.FromArgb(52, 205, 220),
            Color.FromArgb(238, 216, 82),
            Color.FromArgb(138, 169, 255)
        };

        public static Color Background(ThemeMode theme)
        {
            return theme == ThemeMode.Dark
                ? Color.FromArgb(17, 22, 31)
                : Color.FromArgb(246, 249, 253);
        }

        public static Color Foreground(ThemeMode theme)
        {
            return theme == ThemeMode.Dark
                ? Color.FromArgb(220, 228, 239)
                : Color.FromArgb(42, 51, 66);
        }

        public static Color Muted(ThemeMode theme)
        {
            return theme == ThemeMode.Dark
                ? Color.FromArgb(134, 148, 168)
                : Color.FromArgb(105, 114, 128);
        }

        public static Color Grid(ThemeMode theme)
        {
            return theme == ThemeMode.Dark
                ? Color.FromArgb(38, 75, 92, 115)
                : Color.FromArgb(26, 55, 70, 90);
        }
    }

    internal sealed class SnakeCanvas : Control
    {
        private RobotFrame frame;
        private RobotProfile profile = RobotProfile.CreateDefaultTwoJoint();
        private DisplaySettings settings = new DisplaySettings();
        private double viewYaw = -0.65;
        private double viewPitch = 0.48;
        private double zoom = 1.0;
        private double panX;
        private double panY;
        private ViewReference viewReference = ViewReference.WorldOrigin;
        private Point lastMouse;
        private MouseButtons dragButton;

        private struct ScreenPoint
        {
            public float X;
            public float Y;
            public double Depth;

            public ScreenPoint(float x, float y, double depth)
            {
                X = x;
                Y = y;
                Depth = depth;
            }
        }

        public SnakeCanvas()
        {
            SetStyle(
                ControlStyles.AllPaintingInWmPaint |
                ControlStyles.UserPaint |
                ControlStyles.OptimizedDoubleBuffer |
                ControlStyles.ResizeRedraw,
                true);
            BackColor = VisualPalette.Background(ThemeMode.Dark);
            Cursor = Cursors.Hand;
            KinematicResult initial = Kinematics.Forward(profile, new double[] { 0, 0 });
            frame = new RobotFrame
            {
                Points = initial.Points,
                TargetPoints = initial.Points,
                JointOrientations = initial.JointOrientations,
                Joints = new JointRuntimeState[]
                {
                    new JointRuntimeState(),
                    new JointRuntimeState()
                }
            };
        }

        public void SetFrame(RobotFrame value, RobotProfile robotProfile, DisplaySettings display)
        {
            if (value != null)
            {
                frame = value;
            }
            if (robotProfile != null)
            {
                profile = robotProfile;
            }
            if (display != null)
            {
                settings = display;
                viewReference = display.FollowRobot
                    ? ViewReference.RobotCenter
                    : ViewReference.WorldOrigin;
                BackColor = VisualPalette.Background(display.Theme);
            }
            Invalidate();
        }

        // Compatibility API retained for the original two-joint camera tests.
        public void SetPose(Point3D[] newPoints, double newLinkLength)
        {
            frame.Points = newPoints ?? new Point3D[0];
            profile.HeadLength = Math.Max(1.0, newLinkLength);
            Invalidate();
        }

        public void SetViewReference(ViewReference reference)
        {
            viewReference = reference;
            settings.FollowRobot = reference == ViewReference.RobotCenter;
            Invalidate();
        }

        public void ResetView()
        {
            viewYaw = -0.65;
            viewPitch = 0.48;
            zoom = 1.0;
            panX = 0;
            panY = 0;
            Invalidate();
        }

        internal static Point3D ComputeViewCenter(Point3D[] pose, ViewReference reference)
        {
            Point3D center = new Point3D(0, 0, 0);
            if (reference != ViewReference.RobotCenter || pose == null || pose.Length == 0)
            {
                return center;
            }
            for (int i = 0; i < pose.Length; i++)
            {
                center += pose[i];
            }
            return center / pose.Length;
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left || e.Button == MouseButtons.Right)
            {
                dragButton = e.Button;
                lastMouse = e.Location;
                Capture = true;
            }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (dragButton == MouseButtons.None)
            {
                return;
            }
            if (dragButton == MouseButtons.Left)
            {
                viewYaw += (e.X - lastMouse.X) * 0.01;
                viewPitch += (e.Y - lastMouse.Y) * 0.01;
                viewPitch = Math.Max(-1.45, Math.Min(1.45, viewPitch));
            }
            else
            {
                panX += e.X - lastMouse.X;
                panY += e.Y - lastMouse.Y;
            }
            lastMouse = e.Location;
            Invalidate();
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            dragButton = MouseButtons.None;
            Capture = false;
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            base.OnMouseWheel(e);
            zoom *= e.Delta > 0 ? 1.12 : 0.89;
            zoom = Math.Max(0.08, Math.Min(8.0, zoom));
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.Clear(VisualPalette.Background(settings.Theme));
            if (frame == null || frame.Points == null || frame.Points.Length == 0)
            {
                DrawCenteredMessage(g, "等待姿态数据");
                return;
            }

            // 只改绘制坐标：实机没有头部世界位姿，按 rolling 轴与二维投影基重画，使轴竖直、画面不随头部转动。
            RobotFrame source = frame;
            bool aligned = settings.AlignRollingAxis && CanAlign(source);
            if (aligned)
            {
                frame = AlignToRollingAxis(source);
            }
            try
            {
                PaintFrame(g, aligned);
            }
            finally
            {
                frame = source;
            }
        }

        internal static bool CanAlign(RobotFrame value)
        {
            return value != null && value.RollingAxis != null && value.RollingAxis.Valid &&
                value.RollingAxis.HasBasis && value.Points != null;
        }

        // 新坐标：X/Y 为二维投影的水平/竖直基，Z 为 rolling 轴，原点为轴上一点；(H, V, D) 为右手系，不产生镜像。
        internal static RobotFrame AlignToRollingAxis(RobotFrame value)
        {
            RollingAxisState axis = value.RollingAxis;
            Matrix3 rotation = new Matrix3
            {
                M11 = axis.Horizontal.X, M12 = axis.Horizontal.Y, M13 = axis.Horizontal.Z,
                M21 = axis.Vertical.X, M22 = axis.Vertical.Y, M23 = axis.Vertical.Z,
                M31 = axis.Direction.X, M32 = axis.Direction.Y, M33 = axis.Direction.Z
            };
            Func<Point3D[], Point3D[]> map = points => points == null ? null :
                points.Select(x => rotation.Transform(x - axis.Point)).ToArray();
            RollingAxisState alignedAxis = axis.Clone();
            alignedAxis.Point = new Point3D(0, 0, 0);
            alignedAxis.Direction = new Point3D(0, 0, 1);
            alignedAxis.Horizontal = new Point3D(1, 0, 0);
            alignedAxis.Vertical = new Point3D(0, 1, 0);
            return new RobotFrame
            {
                Joints = value.Joints,
                Points = map(value.Points),
                TargetPoints = map(value.TargetPoints),
                JointOrientations = value.JointOrientations == null ? null :
                    value.JointOrientations.Select(x => rotation * x).ToArray(),
                RollingAxis = alignedAxis
            };
        }

        private void PaintFrame(Graphics g, bool aligned)
        {
            Point3D center = ComputeViewCenter(frame.Points, viewReference);
            double extent = ComputeExtent(frame.Points, center);
            double scale = Math.Max(0.02, Math.Min(ClientSize.Width, ClientSize.Height)
                / Math.Max(100.0, extent * 2.7) * zoom);

            if (settings.ShowGrid)
            {
                g.SmoothingMode = SmoothingMode.None;
                DrawGrid(g, center, scale, extent);
            }
            g.SmoothingMode = SmoothingMode.AntiAlias;
            if (settings.ShowWorldAxes)
            {
                DrawWorldAxes(g, center, scale, extent);
            }
            if (settings.ShowRollingAxis && frame.RollingAxis != null && frame.RollingAxis.Valid)
            {
                DrawRollingAxis(g, center, scale, extent);
            }
            if (settings.ShowTarget && frame.TargetPoints != null)
            {
                DrawSkeleton(g, frame.TargetPoints, center, scale, true);
            }
            DrawSkeleton(g, frame.Points, center, scale, false);
            if (settings.ShowLocalAxes)
            {
                DrawLocalAxes(g, center, scale, extent);
            }
            DrawOverlay(g, aligned);
        }

        private static double ComputeExtent(Point3D[] points, Point3D center)
        {
            double extent = 1;
            for (int i = 0; i < points.Length; i++)
            {
                extent = Math.Max(extent, (points[i] - center).Length);
            }
            return extent;
        }

        private void DrawGrid(Graphics g, Point3D center, double scale, double extent)
        {
            double step = NiceStep(extent / 5.0);
            double span = step * 7;
            using (Pen grid = new Pen(VisualPalette.Grid(settings.Theme), 1.0f))
            {
                for (double value = -span; value <= span + step * 0.1; value += step)
                {
                    DrawWorldLine(g, new Point3D(-span, value, 0), new Point3D(span, value, 0), center, scale, grid);
                    DrawWorldLine(g, new Point3D(value, -span, 0), new Point3D(value, span, 0), center, scale, grid);
                }
            }
        }

        private void DrawWorldAxes(Graphics g, Point3D center, double scale, double extent)
        {
            double length = Math.Max(profile.HeadLength, extent * 0.35);
            DrawAxis(g, new Point3D(), new Point3D(length, 0, 0), center, scale, Color.FromArgb(235, 80, 80), "X");
            DrawAxis(g, new Point3D(), new Point3D(0, length, 0), center, scale, Color.FromArgb(72, 207, 124), "Y");
            DrawAxis(g, new Point3D(), new Point3D(0, 0, length), center, scale, Color.FromArgb(70, 133, 245), "Z");
        }

        private void DrawRollingAxis(Graphics g, Point3D center, double scale, double extent)
        {
            RollingAxisState axis = frame.RollingAxis;
            Point3D a = axis.Point - axis.Direction * extent * 1.3;
            Point3D b = axis.Point + axis.Direction * extent * 1.3;
            using (Pen pen = new Pen(Color.FromArgb(245, 76, 220, 230), 2.2f))
            {
                pen.DashStyle = DashStyle.Dash;
                DrawWorldLine(g, a, b, center, scale, pen);
            }
            ScreenPoint label = Project(b, center, scale);
            using (Brush brush = new SolidBrush(Color.FromArgb(76, 220, 230)))
            {
                g.DrawString("rolling 轴", Font, brush, label.X + 4, label.Y);
            }
        }

        private void DrawSkeleton(Graphics g, Point3D[] pose, Point3D center, double scale, bool target)
        {
            if (pose == null || pose.Length < 2)
            {
                return;
            }
            for (int i = 0; i < pose.Length - 1; i++)
            {
                ScreenPoint a = Project(pose[i], center, scale);
                ScreenPoint b = Project(pose[i + 1], center, scale);
                Color color = VisualPalette.Links[i % VisualPalette.Links.Length];
                if (target)
                {
                    using (Pen pen = new Pen(Color.FromArgb(125, color), 3.0f))
                    {
                        pen.DashStyle = DashStyle.Dash;
                        pen.StartCap = pen.EndCap = LineCap.Round;
                        g.DrawLine(pen, a.X, a.Y, b.X, b.Y);
                    }
                }
                else
                {
                    using (Pen shadow = new Pen(
                        settings.Theme == ThemeMode.Dark
                            ? Color.FromArgb(95, 0, 0, 0)
                            : Color.FromArgb(35, 0, 0, 0),
                        9.0f))
                    using (Pen pen = new Pen(color, 5.5f))
                    {
                        shadow.StartCap = shadow.EndCap = LineCap.Round;
                        pen.StartCap = pen.EndCap = LineCap.Round;
                        g.DrawLine(shadow, a.X + 2, a.Y + 3, b.X + 2, b.Y + 3);
                        g.DrawLine(pen, a.X, a.Y, b.X, b.Y);
                    }
                }
            }
            if (!target && (settings.ShowNodes || settings.ShowLabels))
            {
                DrawNodes(g, pose, center, scale);
            }
        }

        private void DrawNodes(Graphics g, Point3D[] pose, Point3D center, double scale)
        {
            Color foreground = VisualPalette.Foreground(settings.Theme);
            Color fill = settings.Theme == ThemeMode.Dark
                ? Color.FromArgb(27, 34, 47)
                : Color.White;
            for (int i = 0; i < pose.Length; i++)
            {
                ScreenPoint p = Project(pose[i], center, scale);
                if (settings.ShowNodes)
                {
                    float radius = i == 0 || i == pose.Length - 1 ? 4.5f : 6.0f;
                    using (Brush brush = new SolidBrush(fill))
                    using (Pen pen = new Pen(foreground, 1.4f))
                    {
                        g.FillEllipse(brush, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                        g.DrawEllipse(pen, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                    }
                }
                if (settings.ShowLabels)
                {
                    string text = i == 0 ? "头部" :
                        i == pose.Length - 1 ? "尾部" : "J" + i;
                    using (Brush brush = new SolidBrush(foreground))
                    {
                        g.DrawString(text, Font, brush, p.X + 7, p.Y - 16);
                    }
                }
            }
        }

        private void DrawLocalAxes(Graphics g, Point3D center, double scale, double extent)
        {
            if (frame.JointOrientations == null)
            {
                return;
            }
            double length = Math.Max(12, extent * 0.08);
            for (int i = 0; i < frame.JointOrientations.Length && i + 1 < frame.Points.Length; i++)
            {
                Point3D origin = frame.Points[i + 1];
                Matrix3 orientation = frame.JointOrientations[i];
                DrawAxis(g, origin, origin + orientation.Transform(new Point3D(length, 0, 0)),
                    center, scale, Color.FromArgb(200, 235, 80, 80), "");
                DrawAxis(g, origin, origin + orientation.Transform(new Point3D(0, length, 0)),
                    center, scale, Color.FromArgb(200, 72, 207, 124), "");
                DrawAxis(g, origin, origin + orientation.Transform(new Point3D(0, 0, length)),
                    center, scale, Color.FromArgb(200, 70, 133, 245), "");
            }
        }

        private void DrawAxis(
            Graphics g,
            Point3D start,
            Point3D end,
            Point3D center,
            double scale,
            Color color,
            string label)
        {
            ScreenPoint a = Project(start, center, scale);
            ScreenPoint b = Project(end, center, scale);
            using (Pen pen = new Pen(color, 1.8f))
            using (Brush brush = new SolidBrush(color))
            {
                pen.EndCap = LineCap.ArrowAnchor;
                g.DrawLine(pen, a.X, a.Y, b.X, b.Y);
                if (label.Length > 0)
                {
                    g.DrawString(label, Font, brush, b.X + 3, b.Y + 2);
                }
            }
        }

        private void DrawOverlay(Graphics g, bool aligned)
        {
            string reference = viewReference == ViewReference.WorldOrigin
                ? "固定世界原点"
                : "跟随机器人中心";
            if (aligned)
            {
                reference += " · 螺旋轴竖直（Z=rolling 轴）";
            }
            else if (settings.AlignRollingAxis)
            {
                reference += " · 轴无效，暂按原坐标显示";
            }
            string axisText = frame.RollingAxis == null
                ? ""
                : String.Format(
                    " · rolling 置信度 {0:P0}{1} · {2}",
                    frame.RollingAxis.Confidence,
                    frame.RollingAxis.Locked ? "（已锁定）" : "",
                    frame.RollingAxis.Status);
            using (Brush brush = new SolidBrush(VisualPalette.Muted(settings.Theme)))
            {
                g.DrawString(
                    "左键旋转 · 右键平移 · 滚轮缩放 · " + reference + axisText,
                    Font,
                    brush,
                    14,
                    Math.Max(4, ClientSize.Height - 26));
            }
        }

        private void DrawWorldLine(
            Graphics g,
            Point3D a,
            Point3D b,
            Point3D center,
            double scale,
            Pen pen)
        {
            ScreenPoint pa = Project(a, center, scale);
            ScreenPoint pb = Project(b, center, scale);
            g.DrawLine(pen, pa.X, pa.Y, pb.X, pb.Y);
        }

        private ScreenPoint Project(Point3D point, Point3D center, double scale)
        {
            Point3D p = point - center;
            double cy = Math.Cos(viewYaw);
            double sy = Math.Sin(viewYaw);
            double cp = Math.Cos(viewPitch);
            double sp = Math.Sin(viewPitch);
            double x1 = cy * p.X - sy * p.Y;
            double y1 = sy * p.X + cy * p.Y;
            double y2 = cp * y1 - sp * p.Z;
            double depth = sp * y1 + cp * p.Z;
            return new ScreenPoint(
                (float)(ClientSize.Width * 0.5 + panX + x1 * scale),
                (float)(ClientSize.Height * 0.5 + panY - y2 * scale),
                depth);
        }

        private void DrawCenteredMessage(Graphics g, string message)
        {
            SizeF size = g.MeasureString(message, Font);
            using (Brush brush = new SolidBrush(VisualPalette.Muted(settings.Theme)))
            {
                g.DrawString(message, Font, brush,
                    (ClientSize.Width - size.Width) / 2,
                    (ClientSize.Height - size.Height) / 2);
            }
        }

        private static double NiceStep(double raw)
        {
            raw = Math.Max(1, raw);
            double power = Math.Pow(10, Math.Floor(Math.Log10(raw)));
            double normalized = raw / power;
            double nice = normalized < 2 ? 1 : normalized < 5 ? 2 : 5;
            return nice * power;
        }
    }

    internal sealed class ProjectionCanvas : Control
    {
        private RobotFrame frame;
        private DisplaySettings settings = new DisplaySettings();
        private double zoom = 1.0;
        private double panX;
        private double panY;
        private Point lastMouse;
        private bool dragging;

        public ProjectionCanvas()
        {
            SetStyle(
                ControlStyles.AllPaintingInWmPaint |
                ControlStyles.UserPaint |
                ControlStyles.OptimizedDoubleBuffer |
                ControlStyles.ResizeRedraw,
                true);
            BackColor = VisualPalette.Background(ThemeMode.Dark);
        }

        public void SetFrame(RobotFrame value, DisplaySettings display)
        {
            if (value != null)
            {
                frame = value;
                if (frame.Projected.Length != frame.Points.Length)
                {
                    // Compatibility for callers constructing a frame directly.
                    // Live/playback frames already contain the per-frame geometry.
                    frame.Projected = RollingProjection.Project(frame.Points, frame.RollingAxis);
                    frame.TargetProjected = RollingProjection.Project(frame.TargetPoints, frame.RollingAxis);
                    frame.Curve = CrossSectionCurve.Build(frame.Projected, frame.CurveGammaRadians);
                    frame.TargetCurve = CrossSectionCurve.Build(frame.TargetProjected, frame.CurveGammaRadians);
                }
            }
            if (display != null)
            {
                settings = display;
                BackColor = VisualPalette.Background(display.Theme);
            }
            Invalidate();
        }

        public void ResetView()
        {
            zoom = 1;
            panX = 0;
            panY = 0;
            Invalidate();
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            {
                dragging = true;
                lastMouse = e.Location;
                Capture = true;
            }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragging)
            {
                return;
            }
            panX += e.X - lastMouse.X;
            panY += e.Y - lastMouse.Y;
            lastMouse = e.Location;
            Invalidate();
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            dragging = false;
            Capture = false;
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            base.OnMouseWheel(e);
            zoom *= e.Delta > 0 ? 1.12 : 0.89;
            zoom = Math.Max(0.1, Math.Min(8, zoom));
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(VisualPalette.Background(settings.Theme));
            if (frame == null || frame.Points == null || frame.Points.Length == 0)
            {
                return;
            }

            ProjectionPoint[] pose = frame.Projected;
            ProjectionPoint[] target = frame.TargetProjected;
            double extent = 1;
            for (int i = 0; i < pose.Length; i++)
            {
                extent = Math.Max(extent, Math.Sqrt(pose[i].X * pose[i].X + pose[i].Y * pose[i].Y));
            }
            if (settings.CurveMode != CurveDisplayMode.Discrete)
            {
                extent = CurveExtent(frame.Curve, extent);
                if (settings.ShowTarget) extent = CurveExtent(frame.TargetCurve, extent);
            }
            if (settings.ShowPipeOutline)
            {
                extent = Math.Max(extent,
                    Math.Max(settings.PipeRadiusX, settings.PipeEllipse
                        ? settings.PipeRadiusY
                        : settings.PipeRadiusX));
            }
            double scale = Math.Min(ClientSize.Width, ClientSize.Height)
                / Math.Max(80, extent * 2.65) * zoom;
            PointF origin = new PointF(
                (float)(ClientSize.Width * 0.5 + panX),
                (float)(ClientSize.Height * 0.5 + panY));

            if (settings.ShowGrid)
            {
                DrawGrid(g, origin, scale, extent);
            }
            if (settings.ShowPipeOutline)
            {
                DrawPipe(g, origin, scale);
            }
            if (settings.ShowTarget)
            {
                DrawPose(g, target, origin, scale, true, ShowDiscrete(frame.TargetCurve));
                if (settings.CurveMode != CurveDisplayMode.Discrete) DrawCurve(g, frame.TargetCurve, origin, scale, true);
            }
            if (settings.CurveMode != CurveDisplayMode.Discrete) DrawCurve(g, frame.Curve, origin, scale, false);
            DrawPose(g, pose, origin, scale, false, ShowDiscrete(frame.Curve));
            DrawAxes(g, origin);
            DrawOverlay(g);
        }

        private void DrawGrid(Graphics g, PointF origin, double scale, double extent)
        {
            double step = SnakeCanvasNiceStep(Math.Max(extent / 4, 1));
            int lines = 7;
            using (Pen pen = new Pen(VisualPalette.Grid(settings.Theme), 1))
            {
                for (int i = -lines; i <= lines; i++)
                {
                    float offset = (float)(i * step * scale);
                    g.DrawLine(pen, origin.X - lines * (float)(step * scale), origin.Y + offset,
                        origin.X + lines * (float)(step * scale), origin.Y + offset);
                    g.DrawLine(pen, origin.X + offset, origin.Y - lines * (float)(step * scale),
                        origin.X + offset, origin.Y + lines * (float)(step * scale));
                }
            }
        }

        private void DrawPipe(Graphics g, PointF origin, double scale)
        {
            double radiusX = Math.Max(1, settings.PipeRadiusX);
            double radiusY = settings.PipeEllipse
                ? Math.Max(1, settings.PipeRadiusY)
                : radiusX;
            float x = (float)(origin.X + settings.PipeCenterX * scale - radiusX * scale);
            float y = (float)(origin.Y - settings.PipeCenterY * scale - radiusY * scale);
            using (Pen pen = new Pen(Color.FromArgb(165, 155, 165, 178), 2))
            {
                pen.DashStyle = DashStyle.Dash;
                g.DrawEllipse(pen, x, y, (float)(2 * radiusX * scale), (float)(2 * radiusY * scale));
            }
        }

        private void DrawPose(
            Graphics g,
            ProjectionPoint[] pose,
            PointF origin,
            double scale,
            bool target,
            bool drawLines)
        {
            if (pose == null || pose.Length < 2)
            {
                return;
            }
            for (int i = 0; drawLines && i < pose.Length - 1; i++)
            {
                PointF a = ToScreen(pose[i], origin, scale);
                PointF b = ToScreen(pose[i + 1], origin, scale);
                Color color = VisualPalette.Links[i % VisualPalette.Links.Length];
                using (Pen pen = new Pen(
                    target ? Color.FromArgb(125, color) : color,
                    target ? 2.5f : settings.CurveMode == CurveDisplayMode.Overlay ? 1.5f : 5.0f))
                {
                    if (target)
                    {
                        pen.DashStyle = DashStyle.Dash;
                    }
                    pen.StartCap = pen.EndCap = LineCap.Round;
                    g.DrawLine(pen, a, b);
                }
            }
            if (!target && settings.ShowNodes)
            {
                using (Brush fill = new SolidBrush(VisualPalette.Background(settings.Theme)))
                using (Pen outline = new Pen(VisualPalette.Foreground(settings.Theme), 1.4f))
                {
                    for (int i = 0; i < pose.Length; i++)
                    {
                        PointF p = ToScreen(pose[i], origin, scale);
                        g.FillEllipse(fill, p.X - 4, p.Y - 4, 8, 8);
                        g.DrawEllipse(outline, p.X - 4, p.Y - 4, 8, 8);
                    }
                }
            }
            if (!target && settings.ShowLabels)
            {
                using (Brush brush = new SolidBrush(VisualPalette.Foreground(settings.Theme)))
                {
                    for (int i = 0; i < pose.Length; i++)
                    {
                        PointF p = ToScreen(pose[i], origin, scale);
                        string label = i == 0 ? "头" : i == pose.Length - 1 ? "尾" : "J" + i;
                        g.DrawString(label, Font, brush, p.X + 5, p.Y - 15);
                    }
                }
            }
        }

        private void DrawAxes(Graphics g, PointF origin)
        {
            using (Pen x = new Pen(Color.FromArgb(200, 65, 152, 255), 1.6f))
            using (Pen y = new Pen(Color.FromArgb(200, 72, 205, 146), 1.6f))
            using (Brush brush = new SolidBrush(VisualPalette.Foreground(settings.Theme)))
            {
                x.EndCap = LineCap.ArrowAnchor;
                y.EndCap = LineCap.ArrowAnchor;
                g.DrawLine(x, origin.X, origin.Y, origin.X + 48, origin.Y);
                g.DrawLine(y, origin.X, origin.Y, origin.X, origin.Y - 48);
                g.DrawString("u", Font, brush, origin.X + 50, origin.Y - 8);
                g.DrawString("v", Font, brush, origin.X + 4, origin.Y - 58);
            }
        }

        private void DrawOverlay(Graphics g)
        {
            RollingAxisState axis = frame.RollingAxis ?? new RollingAxisState();
            string text = String.Format(
                "垂直 rolling 轴投影 · {0} · 置信度 {1:P0}{2} · {3}",
                axis.Manual ? "手动轴" : "自动轴",
                axis.Confidence,
                axis.Locked ? " · 已锁定" : "",
                axis.Status);
            if (frame.RawAxis != null && frame.RawAxis.Valid && frame.RawAxis.Confidence >= 0.42)
                text += String.Format("\n原始估计螺距角 {0:0.00}° ± {1:0.00}°（标准差）",
                    frame.RawAxis.MeanPitchDegrees, frame.RawAxis.PitchDeviationDegrees);
            else text += "\n当前数据不足以可靠估计螺距角";
            if (settings.CurveMode != CurveDisplayMode.Discrete && frame.Curve != null)
                text += "\n" + frame.Curve.Status;
            if (!String.IsNullOrWhiteSpace(frame.GeometryNote)) text += "\n" + frame.GeometryNote;
            if (!axis.Manual && axis.Confidence < 0.42)
            {
                text += " · 建议锁定当前轴或切换手动轴";
            }
            using (Brush brush = new SolidBrush(VisualPalette.Muted(settings.Theme)))
            {
                SizeF size = g.MeasureString(text, Font, Math.Max(50, ClientSize.Width - 24));
                g.DrawString(text, Font, brush,
                    new RectangleF(12, Math.Max(4, ClientSize.Height - size.Height - 8),
                        Math.Max(50, ClientSize.Width - 24), size.Height));
            }
        }

        private bool ShowDiscrete(CrossSectionCurve curve)
        {
            return settings.CurveMode != CurveDisplayMode.Continuous || curve == null || !curve.Valid;
        }

        private static double CurveExtent(CrossSectionCurve curve, double extent)
        {
            if (curve != null && curve.Valid)
                foreach (CurveSegment segment in curve.Segments)
                    foreach (ProjectionPoint p in segment.Samples) extent = Math.Max(extent, p.Length);
            return extent;
        }

        private void DrawCurve(Graphics g, CrossSectionCurve curve, PointF origin, double scale, bool target)
        {
            if (curve == null || !curve.Valid) return;
            foreach (CurveSegment segment in curve.Segments)
            {
                Color color = VisualPalette.Links[segment.JointIndex % VisualPalette.Links.Length];
                using (Pen pen = new Pen(target ? Color.FromArgb(125, color) : color, target ? 2.5f : 4f))
                {
                    if (target) pen.DashStyle = DashStyle.Dash;
                    pen.StartCap = pen.EndCap = LineCap.Round;
                    PointF[] samples = new PointF[segment.Samples.Length];
                    for (int i = 0; i < samples.Length; i++) samples[i] = ToScreen(segment.Samples[i], origin, scale);
                    g.DrawLines(pen, samples);
                }
                if (segment.Case == "D" && !target)
                {
                    PointF marker = ToScreen(segment.At(0.5), origin, scale);
                    using (Brush brush = new SolidBrush(VisualPalette.Foreground(settings.Theme)))
                        g.DrawString("D", Font, brush, marker.X + 4, marker.Y + 4);
                }
            }
        }

        private static PointF ToScreen(ProjectionPoint point, PointF origin, double scale)
        {
            return new PointF(
                (float)(origin.X + point.X * scale),
                (float)(origin.Y - point.Y * scale));
        }

        private static double SnakeCanvasNiceStep(double raw)
        {
            raw = Math.Max(1, raw);
            double power = Math.Pow(10, Math.Floor(Math.Log10(raw)));
            double normalized = raw / power;
            return (normalized < 2 ? 1 : normalized < 5 ? 2 : 5) * power;
        }
    }
}
