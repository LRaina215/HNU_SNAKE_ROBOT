using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal static class UiSmokeTests
    {
        private static int failures;
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        private static int Main()
        {
            SetProcessDPIAware();
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.ThrowException);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            string storagePath = Path.Combine(Path.GetTempPath(),
                "SnakeRobotUiSmoke-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(storagePath);
            try
            {
                using (MainFormV2 form = new MainFormV2(delegate(RobotProfile profile)
                {
                    Thread.Sleep(1200);
                    return new List<int>();
                }, storagePath))
                {
                    form.WindowState = FormWindowState.Minimized;
                    form.ShowInTaskbar = false;
                    // Run the same message loop as the application. DoEvents drains
                    // the whole queue and can starve the test when painting is continuous.
                    form.Shown += async delegate
                    {
                        try { await RunChecks(form); }
                        catch (Exception ex)
                        {
                            failures++;
                            Console.Error.WriteLine(ex);
                        }
                        finally { form.Close(); }
                    };
                    Application.Run(form);
                }
            }
            finally
            {
                try { Directory.Delete(storagePath, true); } catch { }
            }
            Console.WriteLine(failures == 0
                ? "ALL UI SMOKE TESTS PASSED"
                : failures + " UI SMOKE TEST(S) FAILED");
            return failures == 0 ? 0 : 1;
        }

        private static async Task RunChecks(MainFormV2 form)
        {
            await Task.Delay(300);

            Button start = Field<Button>(form, "startButton");
            Button stop = Field<Button>(form, "stopButton");
            DataGridView grid = Field<DataGridView>(form, "jointGrid");
            ComboBox input = Field<ComboBox>(form, "inputModeBox");
            ComboBox profiles = Field<ComboBox>(form, "profileBox");
            ComboBox display = Field<ComboBox>(form, "displayModeBox");
            ComboBox theme = Field<ComboBox>(form, "themeBox");
            Button debug = Field<Button>(form, "debugToggleButton");
            Button refreshPorts = Field<Button>(form, "refreshPortsButton");
            Button scanBus = Field<Button>(form, "scanBusButton");
            Panel debugPanel = Field<Panel>(form, "debugPanel");
            Button enableTorque = Field<Button>(form, "enableTorqueButton");
            Button emergencyRelease = Field<Button>(form, "emergencyReleaseButton");
            CheckBox safetyUnlock = Field<CheckBox>(form, "safetyUnlockBox");
            Label debugHint = Field<Label>(form, "debugHintLabel");

            Check(input.SelectedIndex == 1, "manual simulation is initial data source");
            Check(!refreshPorts.Enabled, "serial refresh disabled in simulation mode");
            profiles.SelectedItem = "默认两关节";
            await Task.Delay(100);
            Check(Field<Label>(form, "selectedJointLabel").Text.StartsWith("J1"),
                "first joint is the default debug selection");

            input.SelectedIndex = 0;
            scanBus.PerformClick();
            await Task.Delay(500);
            Check(!scanBus.Enabled && !start.Enabled,
                "background full-ID scan owns serial without blocking UI");
            debug.PerformClick();
            Check(Field<Panel>(form, "debugPanel").Visible,
                "UI remains interactive during background bus scan");
            await Task.Delay(9000);
            Check(scanBus.Enabled && start.Enabled,
                "background full-ID scan completes and releases controls");
            debug.PerformClick();
            input.SelectedIndex = 1;

            for (int pass = 0; pass < 3; pass++)
            {
                start.PerformClick();
                await Task.Delay(700);
                Check(stop.Enabled, "UI start remains responsive pass " + pass);
                Check(grid.Rows.Count == 2, "two-joint rows retained pass " + pass);
                Check(Convert.ToString(grid.Rows[0].Cells["Status"].Value).Contains("模拟"),
                    "latest simulation frame reaches UI pass " + pass);
                stop.PerformClick();
                await Task.Delay(250);
                Check(start.Enabled, "UI stop permits restart pass " + pass);
            }

            profiles.SelectedItem = "28关节模拟档案";
            form.WindowState = FormWindowState.Normal;
            // A covered window need not receive WM_PAINT; this test measures visible drawing.
            form.TopMost = true;
            form.Activate();
            form.Size = new Size(1500, 960);
            form.Location = Point.Empty;
            input.SelectedIndex = 2;
            start.PerformClick();
            await Task.Delay(1200);
            Check(grid.Rows.Count == 28, "28-joint profile dynamically populates main list");
            Check(Convert.ToString(grid.Rows[20].Cells["Status"].Value).Contains("模拟"),
                "28-joint automatic simulation updates late joint");
            display.SelectedIndex = 2;
            theme.SelectedIndex = 1;
            debug.PerformClick();
            await Task.Delay(300);
            Check(debugPanel.Visible,
                "debug panel expands on demand");
            Check(debugPanel.AutoScroll,
                "debug panel supports scrolling at constrained size and DPI");
            Check(enableTorque.Visible && emergencyRelease.Visible,
                "debug panel exposes torque enable and emergency release controls");
            Check(!safetyUnlock.Enabled && debugHint.Text.Contains("实机串口"),
                "simulation mode explains why servo output is unavailable");
            debug.PerformClick();
            ComboBox curveMode = Field<ComboBox>(form, "curveModeBox");
            NumericUpDown gammaBox = Field<NumericUpDown>(form, "curveGammaBox");
            for (int mode = 0; mode < 3; mode++)
            {
                curveMode.SelectedIndex = mode;
                await Task.Delay(150);
                Check(Field<ApplicationState>(form, "applicationState").Display.CurveMode == (CurveDisplayMode)mode,
                    "C2 curve display mode " + mode);
            }
            gammaBox.Value = 2.1m;
            // Geometry is produced asynchronously; wait for the matching snapshot.
            Stopwatch geometryWait = Stopwatch.StartNew();
            while (Math.Abs(Field<RobotFrame>(form, "displayedFrame").CurveGammaRadians - 2.1 * Math.PI / 180) >= 1e-10
                && geometryWait.ElapsedMilliseconds < 3000)
                await Task.Delay(20);
            Check(Math.Abs(Field<RobotFrame>(form, "displayedFrame").CurveGammaRadians - 2.1 * Math.PI / 180) < 1e-10,
                "C2 gamma control reaches live geometry");
            gammaBox.Value = 1.8m;
            theme.SelectedIndex = 0;
            await Task.Delay(500);
            int paints3D = 0, paints2D = 0;
            SnakeCanvas visible3D = Field<SnakeCanvas>(form, "threeDCanvas");
            ProjectionCanvas visible2D = Field<ProjectionCanvas>(form, "projectionCanvas");
            visible3D.Paint += delegate { paints3D++; };
            visible2D.Paint += delegate { paints2D++; };
            long initialSequence = Field<RobotFrame>(form, "displayedFrame").Sequence;
            Stopwatch rate = Stopwatch.StartNew();
            // No polling: let the native timer and WM_PAINT run undisturbed.
            await Task.Delay(11000);
            long finalSequence = Field<RobotFrame>(form, "displayedFrame").Sequence;
            double observedRate = (finalSequence - initialSequence) / rate.Elapsed.TotalSeconds;
            Console.WriteLine("P1 visible split-view UI rate=" + observedRate.ToString("0.000") + " Hz");
            Check(observedRate >= 9.5, "P1 28-joint visible split view >=9.5Hz for 11 seconds");
            Console.WriteLine("P1 paint counts 3D=" + paints3D + " 2D=" + paints2D);
            Check(paints3D / rate.Elapsed.TotalSeconds >= 9.5 && paints2D / rate.Elapsed.TotalSeconds >= 9.5,
                "P1 both visible canvases actually paint >=9.5Hz");
            string artifacts = Environment.GetEnvironmentVariable("SNAKE_ACCEPTANCE_ARTIFACTS");
            if (!String.IsNullOrEmpty(artifacts))
            {
                Directory.CreateDirectory(artifacts);
                foreach (int selectedTheme in new int[] { 0, 1 })
                {
                    theme.SelectedIndex = selectedTheme;
                    await Task.Delay(150);
                    using (Bitmap screenshot = new Bitmap(form.Width, form.Height))
                    {
                        using (Graphics capture = Graphics.FromImage(screenshot))
                            capture.CopyFromScreen(form.Location, Point.Empty, screenshot.Size);
                        screenshot.Save(Path.Combine(artifacts, selectedTheme == 0 ? "split-dark.png" : "split-light.png"));
                    }
                }
                Rectangle client = form.RectangleToScreen(form.ClientRectangle);
                Check(client.Contains(curveMode.RectangleToScreen(curveMode.ClientRectangle)) &&
                    client.Contains(gammaBox.RectangleToScreen(gammaBox.ClientRectangle)),
                    "C2 curve controls fit inside the window");
            }
            stop.PerformClick();
            await Task.Delay(250);

            // Restore shipping defaults before the form persists application state.
            display.SelectedIndex = 0;
            theme.SelectedIndex = 0;
            input.SelectedIndex = 1;
            profiles.SelectedItem = "默认两关节";
            await Task.Delay(200);
        }

        private static T Field<T>(object instance, string name) where T : class
        {
            FieldInfo field = instance.GetType().GetField(
                name,
                BindingFlags.Instance | BindingFlags.NonPublic);
            return (T)field.GetValue(instance);
        }

        private static void Check(bool condition, string name)
        {
            if (condition)
            {
                Console.WriteLine("PASS: " + name);
            }
            else
            {
                failures++;
                Console.WriteLine("FAIL: " + name);
            }
        }
    }
}
