using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        private static void Main()
        {
            try { SetProcessDPIAware(); } catch { }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
            Application.ThreadException += delegate(object sender, System.Threading.ThreadExceptionEventArgs e)
            {
                ReportFatal(e.Exception);
            };
            AppDomain.CurrentDomain.UnhandledException += delegate(object sender, UnhandledExceptionEventArgs e)
            {
                ReportFatal(e.ExceptionObject as Exception ??
                    new Exception(Convert.ToString(e.ExceptionObject)));
            };
            try
            {
                Application.Run(new MainFormV2());
            }
            catch (Exception ex)
            {
                ReportFatal(ex);
            }
        }

        private static void ReportFatal(Exception exception)
        {
            try
            {
                File.AppendAllText(
                    Path.Combine(Application.StartupPath, "crash.log"),
                    DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "\r\n" +
                    exception + "\r\n\r\n",
                    new UTF8Encoding(false));
            }
            catch
            {
            }
            try
            {
                MessageBox.Show(
                    "程序遇到未处理错误，详细信息已写入 crash.log：\r\n\r\n" +
                    exception.Message,
                    "蛇形机器人可视化程序",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
            catch
            {
            }
        }
    }
}
