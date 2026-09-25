using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Threading;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal sealed class MainFormV2 : Form
    {
        private readonly ProfileStore store;
        private readonly RobotRuntime runtime = new RobotRuntime();
        private readonly ExperimentRecorder recorder = new ExperimentRecorder();
        private readonly System.Windows.Forms.Timer refreshTimer =
            new System.Windows.Forms.Timer();
        private bool recomputingPlayback;
        private readonly ApplicationState applicationState;
        private readonly List<RobotProfile> profiles = new List<RobotProfile>();
        private RobotProfile activeProfile;
        private RobotProfile calibrationBackup;
        private RobotProfile displayProfile;
        private RobotFrame displayedFrame;
        private long displayedSequence = -1;
        private bool closing;
        private bool updatingTarget;
        private volatile bool serialOperation;
        private volatile int controlTorqueLimit = 300;
        private volatile int controlSpeed = 800;
        private volatile int controlAcceleration = 20;
        private DateTime lastRealtimeSend = DateTime.MinValue;
        private LocalControlServer localServer;
        private readonly object mujocoSync = new object();
        private MujocoFramePayload pendingMujocoFrame;
        private bool mujocoStoppedByUser;
        private volatile bool mujocoReady;
        private long displayedFrameCount;
        private long painted3DCount;
        private long painted2DCount;
        private DateTime lastStatisticsUiUtc = DateTime.MinValue;
        private double lastMujocoSimTime;
        private double lastMujocoCapturedUnix;
        private readonly Func<RobotProfile, List<int>> busScanOverride;

        private sealed class MujocoFramePayload
        {
            public double[] Angles;
            public double[] Targets;
            public double[] Lengths;
            public Point3D HeadOriginWorld;
            public Matrix3 HeadBasisWorld;
            public double SimTime;
            public double CapturedUnix;
        }

        private readonly ComboBox profileBox = new ComboBox();
        private readonly ComboBox inputModeBox = new ComboBox();
        private readonly Label portLabel = new Label();
        private readonly Button startButton = new Button();
        private readonly Button stopButton = new Button();
        private readonly Button refreshPortsButton = new Button();
        private readonly Button checkIdsButton = new Button();
        private readonly Button scanBusButton = new Button();
        private readonly Button editProfileButton = new Button();
        private readonly Button duplicateProfileButton = new Button();
        private readonly Button calibrateJointButton = new Button();
        private readonly Button calibrateAllButton = new Button();
        private readonly Button undoCalibrationButton = new Button();
        private readonly DataGridView jointGrid = new BufferedJointGrid();
        private readonly TableLayoutPanel leftLayout = new TableLayoutPanel();
        private readonly Panel debugPanel = new Panel();
        private readonly Button debugToggleButton = new Button();
        private bool debugExpanded;

        private readonly TrackBar targetSlider = new TrackBar();
        private readonly NumericUpDown targetNumber = new NumericUpDown();
        private readonly Button sendTargetButton = new Button();
        private readonly CheckBox realtimeSendBox = new CheckBox();
        private readonly CheckBox safetyUnlockBox = new CheckBox();
        private readonly Button enableTorqueButton = new Button();
        private readonly Button emergencyReleaseButton = new Button();
        private readonly NumericUpDown torqueLimitBox = new NumericUpDown();
        private readonly NumericUpDown speedBox = new NumericUpDown();
        private readonly NumericUpDown accelerationBox = new NumericUpDown();
        private readonly CheckBox simulateSelectedOnlyBox = new CheckBox();
        private readonly Label selectedJointLabel = new Label();
        private readonly Label debugHintLabel = new Label();

        private readonly SnakeCanvas threeDCanvas = new SnakeCanvas();
        private readonly ProjectionCanvas projectionCanvas = new ProjectionCanvas();
        private readonly SplitContainer mainSplit = new SplitContainer();
        private readonly SplitContainer viewSplit = new SplitContainer();
        private readonly ComboBox displayModeBox = new ComboBox();
        private readonly ComboBox themeBox = new ComboBox();
        private readonly ComboBox curveModeBox = new ComboBox();
        private readonly NumericUpDown curveGammaBox = new NumericUpDown();
        private readonly CheckBox axesBox = new CheckBox();
        private readonly CheckBox rollingAxisBox = new CheckBox();
        private readonly CheckBox localAxesBox = new CheckBox();
        private readonly CheckBox gridBox = new CheckBox();
        private readonly CheckBox nodesBox = new CheckBox();
        private readonly CheckBox labelsBox = new CheckBox();
        private readonly CheckBox targetOverlayBox = new CheckBox();
        private readonly CheckBox followBox = new CheckBox();
        private readonly Button resetViewButton = new Button();
        private readonly Button lockAxisButton = new Button();
        private readonly ComboBox axisModeBox = new ComboBox();
        private readonly Button axisEditButton = new Button();
        private readonly Button currentToManualButton = new Button();
        private readonly Button pipeButton = new Button();

        private readonly TextBox experimentNameBox = new TextBox();
        private readonly Button recordButton = new Button();
        private readonly Button stopRecordButton = new Button();
        private readonly Label recordStateLabel = new Label();
        private readonly Button loadPlaybackButton = new Button();
        private readonly Button playbackPlayButton = new Button();
        private readonly Button playbackStopButton = new Button();
        private readonly Button stepBackButton = new Button();
        private readonly Button stepForwardButton = new Button();
        private readonly TrackBar playbackTrack = new TrackBar();
        private readonly ComboBox playbackSpeedBox = new ComboBox();
        private readonly CheckBox recomputeAxisBox = new CheckBox();
        private PlaybackData playback;
        private int playbackIndex;
        private bool playbackPlaying;
        private bool updatingPlaybackTrack;
        private DateTime playbackTickUtc;
        private double playbackPositionSeconds;

        private readonly Label stateLabel = new Label();
        private readonly Label statisticsLabel = new Label();

        public MainFormV2()
            : this(null, null)
        {
        }

        internal MainFormV2(Func<RobotProfile, List<int>> busScanOverride)
            : this(busScanOverride, null)
        {
        }

        internal MainFormV2(
            Func<RobotProfile, List<int>> busScanOverride,
            string storagePathOverride)
        {
            this.busScanOverride = busScanOverride;
            Text = "蛇形机器人实时运动学重建 v1.4.1";
            StartPosition = FormStartPosition.CenterScreen;
            MinimumSize = new Size(1160, 720);
            Size = new Size(1520, 900);
            WindowState = FormWindowState.Maximized;
            Font = new Font("Microsoft YaHei UI", 9F);
            KeyPreview = true;

            store = new ProfileStore(String.IsNullOrWhiteSpace(storagePathOverride)
                ? Application.StartupPath
                : storagePathOverride);
            applicationState = store.LoadApplicationState();
            profiles.AddRange(store.LoadProfiles());
            activeProfile = profiles.FirstOrDefault(x => x.Name == applicationState.LastProfile)
                ?? profiles[0];
            displayProfile = activeProfile;

            BuildUi();
            threeDCanvas.Paint += delegate { Interlocked.Increment(ref painted3DCount); };
            projectionCanvas.Paint += delegate { Interlocked.Increment(ref painted2DCount); };
            LoadProfilesIntoCombo(activeProfile.Name);
            BindProfile(activeProfile);
            ApplyDisplaySettingsToControls();
            ApplyDisplayMode();
            UpdateControlStates();
            Shown += delegate
            {
                if (viewSplit.Width > 500)
                {
                    viewSplit.SplitterDistance = (int)(viewSplit.Width * 0.62);
                }
                if (mainSplit.Width > 750)
                {
                    mainSplit.SplitterDistance = 460;
                }
                ApplyDisplayMode();
            };

            refreshTimer.Interval = 30;
            refreshTimer.Tick += RefreshTick;
            refreshTimer.Start();

            int bridgePort;
            if (!Int32.TryParse(Environment.GetEnvironmentVariable("SNAKE_VISUALIZER_PORT"), out bridgePort) ||
                bridgePort < 1024 || bridgePort > 65535)
                bridgePort = 8765;
            localServer = new LocalControlServer(bridgePort, HandleLocalCommand, runtime.ControlDisconnected);
            if (!localServer.Start())
            {
                stateLabel.Text = "本机 Python 接口端口 " + bridgePort + " 启动失败";
            }
        }

        private void BuildUi()
        {
            TableLayoutPanel root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 4
            };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            Controls.Add(root);

            Control displayToolbar = BuildDisplayToolbar();
            root.Controls.Add(displayToolbar, 0, 0);
            root.SizeChanged += delegate
            {
                displayToolbar.MaximumSize = new Size(Math.Max(100, root.ClientSize.Width - 6), 0);
            };

            mainSplit.Dock = DockStyle.Fill;
            mainSplit.SplitterDistance = 460;
            mainSplit.FixedPanel = FixedPanel.Panel1;
            root.Controls.Add(mainSplit, 0, 1);
            mainSplit.Panel1.Controls.Add(BuildLeftPanel());
            mainSplit.Panel2.Controls.Add(BuildViews());

            root.Controls.Add(BuildPlaybackBar(), 0, 2);

            Panel status = new Panel { Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 2) };
            stateLabel.AutoSize = true;
            stateLabel.Text = "空闲";
            stateLabel.Dock = DockStyle.Left;
            statisticsLabel.AutoSize = true;
            statisticsLabel.Dock = DockStyle.Right;
            status.Controls.Add(statisticsLabel);
            status.Controls.Add(stateLabel);
            root.Controls.Add(status, 0, 3);
        }

        private Control BuildDisplayToolbar()
        {
            FlowLayoutPanel toolbar = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                WrapContents = true,
                Padding = new Padding(7, 6, 7, 4)
            };
            displayModeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            displayModeBox.Items.AddRange(new object[] { "三维", "二维", "三维 + 二维" });
            displayModeBox.SelectedIndexChanged += delegate
            {
                applicationState.Display.DisplayMode = displayModeBox.SelectedIndex == 1
                    ? DisplayMode.Projection
                    : displayModeBox.SelectedIndex == 2 ? DisplayMode.Split : DisplayMode.ThreeD;
                ApplyDisplayMode();
            };
            themeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            themeBox.Items.AddRange(new object[] { "深色论文风格", "浅色" });
            themeBox.SelectedIndexChanged += delegate
            {
                applicationState.Display.Theme = themeBox.SelectedIndex == 1
                    ? ThemeMode.Light
                    : ThemeMode.Dark;
                RefreshCanvases();
            };
            toolbar.Controls.Add(LabeledInline("视图", displayModeBox, 105));
            toolbar.Controls.Add(LabeledInline("主题", themeBox, 125));
            curveModeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            curveModeBox.Items.AddRange(new object[] { "原始折线", "论文连续近似", "折线 + 连续近似" });
            curveModeBox.SelectedIndexChanged += delegate
            {
                applicationState.Display.CurveMode = (CurveDisplayMode)Math.Max(0, curveModeBox.SelectedIndex);
                RefreshCanvases();
            };
            curveGammaBox.Minimum = 0.01m;
            curveGammaBox.Maximum = 30m;
            curveGammaBox.DecimalPlaces = 2;
            curveGammaBox.Increment = 0.1m;
            curveGammaBox.Value = 1.8m;
            curveGammaBox.ValueChanged += delegate
            {
                if (playback != null) return;
                applicationState.Display.CurveGammaDegrees = (double)curveGammaBox.Value;
                double gamma = (double)curveGammaBox.Value * Math.PI / 180;
                runtime.SetCurveGamma(gamma);
                if (displayedFrame != null && displayProfile != null)
                {
                    displayedFrame = displayedFrame.Clone();
                    FrameGeometry.Prepare(displayedFrame, displayProfile, null, gamma);
                    RefreshCanvases();
                }
            };
            toolbar.Controls.Add(LabeledInline("二维曲线", curveModeBox, 140));
            toolbar.Controls.Add(LabeledInline("圆弧阈值 γmin(°)", curveGammaBox, 70));

            ConfigureToggle(axesBox, "世界轴", delegate { applicationState.Display.ShowWorldAxes = axesBox.Checked; });
            ConfigureToggle(rollingAxisBox, "rolling 轴", delegate { applicationState.Display.ShowRollingAxis = rollingAxisBox.Checked; });
            ConfigureToggle(localAxesBox, "局部轴", delegate { applicationState.Display.ShowLocalAxes = localAxesBox.Checked; });
            ConfigureToggle(gridBox, "网格", delegate { applicationState.Display.ShowGrid = gridBox.Checked; });
            ConfigureToggle(nodesBox, "关节点", delegate { applicationState.Display.ShowNodes = nodesBox.Checked; });
            ConfigureToggle(labelsBox, "编号", delegate { applicationState.Display.ShowLabels = labelsBox.Checked; });
            ConfigureToggle(targetOverlayBox, "目标姿态", delegate { applicationState.Display.ShowTarget = targetOverlayBox.Checked; });
            ConfigureToggle(followBox, "跟随机器人", delegate
            {
                applicationState.Display.FollowRobot = followBox.Checked;
                threeDCanvas.SetViewReference(followBox.Checked
                    ? ViewReference.RobotCenter
                    : ViewReference.WorldOrigin);
            });
            toolbar.Controls.AddRange(new Control[]
            {
                axesBox, rollingAxisBox, localAxesBox, gridBox,
                nodesBox, labelsBox, targetOverlayBox, followBox
            });

            resetViewButton.Text = "重置视角";
            resetViewButton.AutoSize = true;
            resetViewButton.Click += delegate
            {
                threeDCanvas.ResetView();
                projectionCanvas.ResetView();
            };
            lockAxisButton.Text = "锁定当前轴";
            lockAxisButton.AutoSize = true;
            lockAxisButton.Click += ToggleAxisLock;
            axisModeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            axisModeBox.Items.AddRange(new object[] { "自动轴", "手动轴" });
            axisModeBox.SelectedIndex = 0;
            axisModeBox.SelectedIndexChanged += AxisModeChanged;
            axisEditButton.Text = "轴参数";
            axisEditButton.AutoSize = true;
            axisEditButton.Click += EditAxis;
            currentToManualButton.Text = "当前轴→手动";
            currentToManualButton.AutoSize = true;
            currentToManualButton.Click += CurrentAxisToManual;
            pipeButton.Text = "静态轮廓";
            pipeButton.AutoSize = true;
            pipeButton.Click += EditPipe;
            toolbar.Controls.AddRange(new Control[]
            {
                resetViewButton, lockAxisButton,
                LabeledInline("轴来源", axisModeBox, 85),
                axisEditButton, currentToManualButton, pipeButton
            });
            return toolbar;
        }

        private Control BuildLeftPanel()
        {
            leftLayout.Dock = DockStyle.Fill;
            leftLayout.Padding = new Padding(7);
            leftLayout.ColumnCount = 1;
            leftLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            leftLayout.RowCount = 4;
            leftLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 260));
            leftLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            leftLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 35));
            leftLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 0));

            leftLayout.Controls.Add(BuildConnectionPanel(), 0, 0);
            ConfigureJointGrid();
            leftLayout.Controls.Add(jointGrid, 0, 1);
            debugToggleButton.Dock = DockStyle.Fill;
            debugToggleButton.Text = "▶ 舵机调试（默认折叠）";
            debugToggleButton.TextAlign = ContentAlignment.MiddleLeft;
            debugToggleButton.Click += ToggleDebugPanel;
            leftLayout.Controls.Add(debugToggleButton, 0, 2);
            BuildDebugPanel();
            leftLayout.Controls.Add(debugPanel, 0, 3);
            return leftLayout;
        }

        private Control BuildConnectionPanel()
        {
            GroupBox box = new GroupBox
            {
                Text = "机器人档案与数据来源",
                Dock = DockStyle.Fill
            };
            FlowLayoutPanel flow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                WrapContents = true,
                AutoScroll = true,
                Padding = new Padding(5)
            };
            profileBox.DropDownStyle = ComboBoxStyle.DropDownList;
            profileBox.Width = 210;
            profileBox.SelectedIndexChanged += ProfileChanged;
            inputModeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            inputModeBox.Items.AddRange(new object[] { "实机串口", "手动模拟", "自动模拟", "MuJoCo 实测" });
            inputModeBox.SelectedIndex = 1;
            inputModeBox.Width = 115;
            inputModeBox.SelectedIndexChanged += delegate { UpdateControlStates(); };
            portLabel.AutoSize = true;
            portLabel.Width = 330;

            ConfigureButton(startButton, "开始读取", StartRuntime);
            ConfigureButton(stopButton, "停止", StopRuntimeClicked);
            ConfigureButton(refreshPortsButton, "刷新串口", RefreshSerialPorts);
            ConfigureButton(checkIdsButton, "检查档案 ID", CheckProfileIds);
            ConfigureButton(scanBusButton, "扫描总线 ID", ScanAllIds);
            ConfigureButton(editProfileButton, "编辑档案", EditProfile);
            ConfigureButton(duplicateProfileButton, "复制档案", DuplicateProfile);
            ConfigureButton(calibrateJointButton, "标定选中零位", CalibrateSelected);
            ConfigureButton(calibrateAllButton, "整机零位标定", CalibrateAll);
            ConfigureButton(undoCalibrationButton, "撤销标定", UndoCalibration);
            undoCalibrationButton.Enabled = false;

            flow.Controls.Add(new Label { Text = "档案", AutoSize = true, Margin = new Padding(3, 8, 2, 0) });
            flow.Controls.Add(profileBox);
            flow.Controls.Add(editProfileButton);
            flow.Controls.Add(duplicateProfileButton);
            flow.Controls.Add(new Label { Text = "来源", AutoSize = true, Margin = new Padding(3, 8, 2, 0) });
            flow.Controls.Add(inputModeBox);
            flow.Controls.Add(startButton);
            flow.Controls.Add(stopButton);
            flow.SetFlowBreak(stopButton, true);
            flow.Controls.Add(portLabel);
            flow.SetFlowBreak(portLabel, true);
            flow.Controls.Add(refreshPortsButton);
            flow.Controls.Add(checkIdsButton);
            flow.Controls.Add(scanBusButton);
            flow.SetFlowBreak(scanBusButton, true);
            flow.Controls.Add(calibrateJointButton);
            flow.Controls.Add(calibrateAllButton);
            flow.Controls.Add(undoCalibrationButton);
            box.Controls.Add(flow);
            return box;
        }

        private sealed class BufferedJointGrid : DataGridView
        {
            public BufferedJointGrid()
            {
                // 28 关节每帧更新多个单元格；先画到缓冲区再提交，避免逐格 GDI
                // 重绘阻塞 UI 消息循环，影响三维/二维画布的最新帧刷新。
                DoubleBuffered = true;
            }
        }

        private void ConfigureJointGrid()
        {
            jointGrid.Dock = DockStyle.Fill;
            jointGrid.ReadOnly = true;
            jointGrid.AllowUserToAddRows = false;
            jointGrid.AllowUserToDeleteRows = false;
            jointGrid.RowHeadersVisible = false;
            jointGrid.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            jointGrid.MultiSelect = false;
            jointGrid.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            jointGrid.DefaultCellStyle.Font = Font;
            jointGrid.ColumnHeadersDefaultCellStyle.Font = Font;
            jointGrid.Columns.Add("Joint", "关节");
            jointGrid.Columns.Add("Id", "ID");
            jointGrid.Columns.Add("Axis", "轴");
            jointGrid.Columns.Add("Actual", "实测角");
            jointGrid.Columns.Add("Target", "目标角");
            jointGrid.Columns.Add("Status", "状态");
            jointGrid.Columns.Add("Raw", "原始值");
            jointGrid.Columns["Joint"].FillWeight = 48;
            jointGrid.Columns["Id"].FillWeight = 42;
            jointGrid.Columns["Axis"].FillWeight = 48;
            jointGrid.Columns["Raw"].FillWeight = 62;
            jointGrid.SelectionChanged += SelectedJointChanged;
        }

        private void BuildDebugPanel()
        {
            debugPanel.Dock = DockStyle.Fill;
            debugPanel.Visible = false;
            debugPanel.AutoScroll = true;
            TableLayoutPanel table = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                ColumnCount = 4,
                RowCount = 8,
                Padding = new Padding(5)
            };
            for (int row = 0; row < table.RowCount; row++)
            {
                table.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            }
            selectedJointLabel.Text = "未选择关节";
            selectedJointLabel.AutoSize = true;
            selectedJointLabel.Font = new Font(Font, FontStyle.Bold);
            table.SetColumnSpan(selectedJointLabel, 4);
            table.Controls.Add(selectedJointLabel, 0, 0);
            targetSlider.Minimum = -90;
            targetSlider.Maximum = 90;
            targetSlider.TickFrequency = 15;
            targetSlider.Dock = DockStyle.Fill;
            targetSlider.ValueChanged += TargetSliderChanged;
            table.SetColumnSpan(targetSlider, 3);
            table.Controls.Add(targetSlider, 0, 1);
            targetNumber.DecimalPlaces = 1;
            targetNumber.Minimum = -180;
            targetNumber.Maximum = 180;
            targetNumber.Dock = DockStyle.Fill;
            targetNumber.ValueChanged += TargetNumberChanged;
            table.Controls.Add(targetNumber, 3, 1);
            realtimeSendBox.Text = "滑动实时发送（显式启用）";
            realtimeSendBox.AutoSize = true;
            table.SetColumnSpan(realtimeSendBox, 2);
            table.Controls.Add(realtimeSendBox, 0, 2);
            simulateSelectedOnlyBox.Text = "自动模拟仅选中关节";
            simulateSelectedOnlyBox.AutoSize = true;
            simulateSelectedOnlyBox.CheckedChanged += delegate
            {
                runtime.SetAutomaticSimulationJoint(
                    simulateSelectedOnlyBox.Checked ? SelectedJointIndex : -1);
            };
            table.SetColumnSpan(simulateSelectedOnlyBox, 2);
            table.Controls.Add(simulateSelectedOnlyBox, 2, 2);

            torqueLimitBox.Minimum = 0;
            torqueLimitBox.Maximum = 1000;
            torqueLimitBox.Value = 300;
            torqueLimitBox.ValueChanged += delegate { controlTorqueLimit = (int)torqueLimitBox.Value; };
            speedBox.Minimum = 1;
            speedBox.Maximum = 32766;
            speedBox.Value = 800;
            speedBox.ValueChanged += delegate { controlSpeed = (int)speedBox.Value; };
            accelerationBox.Minimum = 0;
            accelerationBox.Maximum = 254;
            accelerationBox.Value = 20;
            accelerationBox.ValueChanged += delegate { controlAcceleration = (int)accelerationBox.Value; };
            AddDebugField(table, 3, 0, "转矩上限", torqueLimitBox);
            AddDebugField(table, 3, 2, "速度", speedBox);
            AddDebugField(table, 4, 0, "加速度", accelerationBox);
            sendTargetButton.Text = "发送目标";
            sendTargetButton.Dock = DockStyle.Fill;
            sendTargetButton.Click += SendSelectedTarget;
            table.SetColumnSpan(sendTargetButton, 2);
            table.Controls.Add(sendTargetButton, 2, 4);

            debugHintLabel.Text = "控制顺序：①安全确认  ②安全使能扭矩  ③发送目标或启用滑动实时发送";
            debugHintLabel.AutoSize = true;
            debugHintLabel.MaximumSize = new Size(420, 0);
            debugHintLabel.ForeColor = Color.FromArgb(174, 112, 20);
            debugHintLabel.Margin = new Padding(3, 6, 3, 6);
            table.SetColumnSpan(debugHintLabel, 4);
            table.Controls.Add(debugHintLabel, 0, 5);

            safetyUnlockBox.Text = "我已确认周围安全，解锁舵机输出";
            safetyUnlockBox.AutoSize = true;
            safetyUnlockBox.CheckedChanged += delegate
            {
                if (!safetyUnlockBox.Checked)
                {
                    realtimeSendBox.Checked = false;
                }
                UpdateControlStates();
            };
            table.SetColumnSpan(safetyUnlockBox, 4);
            table.Controls.Add(safetyUnlockBox, 0, 6);
            enableTorqueButton.Text = "安全使能扭矩";
            enableTorqueButton.Dock = DockStyle.Fill;
            enableTorqueButton.Click += EnableTorque;
            emergencyReleaseButton.Text = "紧急释放全部扭矩";
            emergencyReleaseButton.BackColor = Color.FromArgb(180, 58, 58);
            emergencyReleaseButton.ForeColor = Color.White;
            emergencyReleaseButton.Dock = DockStyle.Fill;
            emergencyReleaseButton.Click += EmergencyRelease;
            table.SetColumnSpan(enableTorqueButton, 2);
            table.SetColumnSpan(emergencyReleaseButton, 2);
            table.Controls.Add(enableTorqueButton, 0, 7);
            table.Controls.Add(emergencyReleaseButton, 2, 7);
            debugPanel.Controls.Add(table);
        }

        private Control BuildViews()
        {
            viewSplit.Dock = DockStyle.Fill;
            viewSplit.Orientation = Orientation.Vertical;
            viewSplit.SplitterDistance = 650;
            viewSplit.Panel1.Controls.Add(WithCaption("三维运动学骨架", threeDCanvas));
            viewSplit.Panel2.Controls.Add(WithCaption("沿 rolling 轴的二维投影", projectionCanvas));
            return viewSplit;
        }

        private Control BuildPlaybackBar()
        {
            FlowLayoutPanel bar = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                WrapContents = true,
                Padding = new Padding(7, 4, 7, 4)
            };
            experimentNameBox.Width = 130;
            experimentNameBox.Text = activeProfile.Name;
            ConfigureButton(recordButton, "开始记录", StartRecording);
            ConfigureButton(stopRecordButton, "停止记录", StopRecording);
            stopRecordButton.Enabled = false;
            recordStateLabel.Text = "未记录";
            recordStateLabel.AutoSize = true;
            recordStateLabel.Margin = new Padding(4, 8, 12, 0);
            ConfigureButton(loadPlaybackButton, "加载回放", LoadPlayback);
            ConfigureButton(playbackPlayButton, "播放", TogglePlayback);
            ConfigureButton(playbackStopButton, "停止回放", StopPlayback);
            ConfigureButton(stepBackButton, "◀ 单帧", delegate { StepPlayback(-1); });
            ConfigureButton(stepForwardButton, "单帧 ▶", delegate { StepPlayback(1); });
            playbackTrack.Minimum = 0;
            playbackTrack.Maximum = 0;
            playbackTrack.Width = 260;
            playbackTrack.TickStyle = TickStyle.None;
            playbackTrack.Scroll += PlaybackTrackScrolled;
            playbackSpeedBox.DropDownStyle = ComboBoxStyle.DropDownList;
            playbackSpeedBox.Items.AddRange(new object[] { "0.25×", "0.5×", "1×", "2×" });
            playbackSpeedBox.SelectedIndex = 2;
            recomputeAxisBox.Text = "回放时重新计算轴";
            recomputeAxisBox.AutoSize = true;
            recomputeAxisBox.CheckedChanged += delegate
            {
                if (playback != null)
                {
                    ShowPlaybackFrame(playbackIndex);
                }
            };
            bar.Controls.Add(new Label { Text = "实验名", AutoSize = true, Margin = new Padding(3, 8, 2, 0) });
            bar.Controls.Add(experimentNameBox);
            bar.Controls.Add(recordButton);
            bar.Controls.Add(stopRecordButton);
            bar.Controls.Add(recordStateLabel);
            bar.Controls.Add(loadPlaybackButton);
            bar.Controls.Add(playbackPlayButton);
            bar.Controls.Add(playbackStopButton);
            bar.Controls.Add(stepBackButton);
            bar.Controls.Add(playbackTrack);
            bar.Controls.Add(stepForwardButton);
            bar.Controls.Add(playbackSpeedBox);
            bar.Controls.Add(recomputeAxisBox);
            return bar;
        }

        private void LoadProfilesIntoCombo(string selectName)
        {
            profileBox.BeginUpdate();
            profileBox.Items.Clear();
            foreach (RobotProfile profile in profiles.OrderBy(x => x.Name))
            {
                profileBox.Items.Add(profile.Name);
            }
            profileBox.SelectedItem = selectName;
            if (profileBox.SelectedIndex < 0 && profileBox.Items.Count > 0)
            {
                profileBox.SelectedIndex = 0;
            }
            profileBox.EndUpdate();
        }

        private void BindProfile(RobotProfile profile)
        {
            activeProfile = profile;
            displayProfile = profile;
            portLabel.Text = String.Format(
                "{0} · {1} baud · {2} 个关节（启动时不会自动连接）",
                profile.Port,
                profile.BaudRate,
                profile.Joints.Count);
            jointGrid.Rows.Clear();
            foreach (JointConfig joint in profile.Joints)
            {
                jointGrid.Rows.Add(
                    "J" + joint.Index,
                    joint.ServoId,
                    joint.AxisType == JointAxisType.Yaw ? "yaw" : "pitch",
                    "—",
                    "0.0°",
                    "未运行",
                    "—");
            }
            if (jointGrid.Rows.Count > 0)
            {
                jointGrid.Rows[0].Selected = true;
                jointGrid.CurrentCell = jointGrid.Rows[0].Cells[0];
            }
            experimentNameBox.Text = profile.Name;
            KinematicResult initial = Kinematics.Forward(
                profile,
                Enumerable.Repeat(0.0, profile.Joints.Count).ToArray());
            displayedFrame = new RobotFrame
            {
                Joints = profile.Joints.Select(x => new JointRuntimeState
                {
                    Raw = x.Zero,
                    Status = "未运行"
                }).ToArray(),
                Points = initial.Points,
                TargetPoints = initial.Points,
                JointOrientations = initial.JointOrientations,
                RollingAxis = RollingAxisEstimator.Estimate(initial.Points),
                Source = "档案"
            };
            displayedFrame.RollingAxis.Valid = false;
            displayedFrame.RollingAxis.Status = "尚未运行，仅显示后备投影";
            FrameGeometry.Prepare(displayedFrame, profile, null,
                applicationState.Display.CurveGammaDegrees * Math.PI / 180);
            RefreshCanvases();
            SelectedJointChanged(null, EventArgs.Empty);
        }

        private void ProfileChanged(object sender, EventArgs e)
        {
            if (profileBox.SelectedItem == null || runtime.Mode != RuntimeMode.Idle || playback != null)
            {
                return;
            }
            RobotProfile selected = profiles.FirstOrDefault(
                x => x.Name == Convert.ToString(profileBox.SelectedItem));
            if (selected != null && selected != activeProfile)
            {
                BindProfile(selected);
            }
        }

        private void StartRuntime(object sender, EventArgs e)
        {
            try
            {
                StopPlaybackInternal();
                RuntimeMode requested = inputModeBox.SelectedIndex == 0
                    ? RuntimeMode.Live
                    : inputModeBox.SelectedIndex == 2
                        ? RuntimeMode.SimulationAuto
                        : inputModeBox.SelectedIndex == 3
                            ? RuntimeMode.Mujoco : RuntimeMode.SimulationManual;
                runtime.Start(activeProfile, requested);
                mujocoStoppedByUser = false;
                mujocoReady = false;
                runtime.SetAutomaticSimulationJoint(
                    simulateSelectedOnlyBox.Checked ? SelectedJointIndex : -1);
                displayedSequence = -1;
                stateLabel.Text = requested == RuntimeMode.Live
                    ? "正在后台连接串口…"
                    : requested == RuntimeMode.Mujoco ? "等待 MuJoCo 实测关节角（不访问串口）"
                    : "模拟已启动（不会访问串口）";
                UpdateControlStates();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "无法开始",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void StopRuntimeClicked(object sender, EventArgs e)
        {
            if (runtime.Mode == RuntimeMode.Mujoco)
                mujocoStoppedByUser = true;
            mujocoReady = false;
            lock (mujocoSync) { pendingMujocoFrame = null; }
            StopRecordingInternal();
            runtime.Stop();
            stateLabel.Text = "已停止并释放串口；若曾使能，已尝试释放扭矩";
            UpdateControlStates();
        }

        private void RefreshTick(object sender, EventArgs e)
        {
            if (closing)
            {
                return;
            }
            if (playback != null)
            {
                AdvancePlayback();
                return;
            }
            MujocoFramePayload packet;
            lock (mujocoSync)
            {
                packet = pendingMujocoFrame;
                pendingMujocoFrame = null;
            }
            if (packet != null && !mujocoStoppedByUser)
            {
                try
                {
                    if (runtime.Mode == RuntimeMode.Idle ||
                        (runtime.Mode == RuntimeMode.Mujoco &&
                         (activeProfile.Joints.Count != packet.Angles.Length ||
                          activeProfile.Port != "MUJOCO-NO-SERIAL" || !MujocoLengthsMatch(activeProfile, packet))))
                    {
                        RobotProfile external = CreateMujocoProfile(packet);
                        profiles.RemoveAll(x => x.Port == "MUJOCO-NO-SERIAL");
                        profiles.Add(external);
                        LoadProfilesIntoCombo(external.Name);
                        BindProfile(external);
                        inputModeBox.SelectedIndex = 3;
                        runtime.Start(external, RuntimeMode.Mujoco);
                        mujocoReady = false;
                        displayedSequence = -1;
                        UpdateControlStates();
                    }
                    if (runtime.Mode == RuntimeMode.Mujoco)
                    {
                        runtime.UpdateMujocoPose(packet.Angles, packet.Targets,
                            packet.HeadOriginWorld, packet.HeadBasisWorld, packet.CapturedUnix, packet.SimTime);
                        lastMujocoSimTime = packet.SimTime;
                        lastMujocoCapturedUnix = packet.CapturedUnix;
                        mujocoReady = true;
                    }
                }
                catch (Exception ex)
                {
                    stateLabel.Text = "MuJoCo 数据错误：" + ex.Message;
                }
            }
            RobotFrame latest;
            if (runtime.TryGetLatest(out latest) && latest.Sequence != displayedSequence)
            {
                displayedSequence = latest.Sequence;
                Interlocked.Increment(ref displayedFrameCount);
                displayedFrame = latest;
                displayProfile = activeProfile;
                UpdateJointRows(latest, activeProfile);
                RefreshCanvases();
                if (recorder.IsRecording)
                {
                    recorder.Append(latest, activeProfile);
                }
            }
            DateTime now = DateTime.UtcNow;
            if ((now - lastStatisticsUiUtc).TotalMilliseconds >= 200)
            {
                lastStatisticsUiUtc = now;
                RuntimeStatistics statistics = runtime.GetStatistics();
                stateLabel.Text = statistics.Status;
                statisticsLabel.Text = String.Format(
                    "{0:0.0} 帧/s · 总帧 {1} · 通信错误 {2} · {3}",
                    statistics.SamplesPerSecond,
                    statistics.TotalSamples,
                    statistics.CommunicationErrors,
                    runtime.TorqueActive ? "扭矩已使能" : "扭矩未使能");
                if (recorder.IsRecording)
                {
                    recordStateLabel.Text = String.Format(
                        "记录 {0:mm\\:ss} · {1} 帧",
                        recorder.Duration,
                        recorder.FrameCount);
                }
            }
            if (runtime.Mode != RuntimeMode.Mujoco)
                UpdateControlStates();
        }

        private void UpdateJointRows(RobotFrame frame, RobotProfile profile)
        {
            if (jointGrid.Rows.Count != profile.Joints.Count)
            {
                BindProfile(profile);
            }
            int selected = SelectedJointIndex;
            for (int i = 0; i < frame.Joints.Length && i < jointGrid.Rows.Count; i++)
            {
                JointRuntimeState state = frame.Joints[i];
                DataGridViewRow row = jointGrid.Rows[i];
                row.Cells["Actual"].Value = state.AngleDegrees.ToString("0.0") + "°";
                row.Cells["Target"].Value = state.TargetDegrees.ToString("0.0") + "°";
                row.Cells["Status"].Value = state.Status;
                row.Cells["Raw"].Value = state.Raw;
                row.DefaultCellStyle.ForeColor = state.Online
                    ? jointGrid.DefaultCellStyle.ForeColor
                    : Color.FromArgb(205, 82, 82);
            }
            if (selected >= 0 && selected < jointGrid.Rows.Count)
            {
                jointGrid.Rows[selected].Selected = true;
            }
            SyncTargetFromFrame();
        }

        private void RefreshCanvases()
        {
            if (displayedFrame == null || displayProfile == null)
            {
                return;
            }
            threeDCanvas.SetFrame(displayedFrame, displayProfile, applicationState.Display);
            projectionCanvas.SetFrame(displayedFrame, applicationState.Display);
        }

        private void ApplyDisplaySettingsToControls()
        {
            DisplaySettings display = applicationState.Display;
            displayModeBox.SelectedIndex = display.DisplayMode == DisplayMode.Projection
                ? 1 : display.DisplayMode == DisplayMode.Split ? 2 : 0;
            themeBox.SelectedIndex = display.Theme == ThemeMode.Light ? 1 : 0;
            curveModeBox.SelectedIndex = (int)display.CurveMode;
            double gammaDegrees = display.CurveGammaDegrees;
            if (Double.IsNaN(gammaDegrees) || Double.IsInfinity(gammaDegrees)) gammaDegrees = 1.8;
            curveGammaBox.Value = (decimal)Math.Max(0.01, Math.Min(30, gammaDegrees));
            runtime.SetCurveGamma((double)curveGammaBox.Value * Math.PI / 180);
            axesBox.Checked = display.ShowWorldAxes;
            rollingAxisBox.Checked = display.ShowRollingAxis;
            localAxesBox.Checked = display.ShowLocalAxes;
            gridBox.Checked = display.ShowGrid;
            nodesBox.Checked = display.ShowNodes;
            labelsBox.Checked = display.ShowLabels;
            targetOverlayBox.Checked = display.ShowTarget;
            followBox.Checked = display.FollowRobot;
        }

        private void ApplyDisplayMode()
        {
            if (!IsHandleCreated)
            {
                return;
            }
            DisplayMode mode = applicationState.Display.DisplayMode;
            viewSplit.Panel1Collapsed = mode == DisplayMode.Projection;
            viewSplit.Panel2Collapsed = mode == DisplayMode.ThreeD;
            RefreshCanvases();
        }

        private void ConfigureToggle(CheckBox box, string text, Action changed)
        {
            box.Text = text;
            box.AutoSize = true;
            box.Margin = new Padding(5, 7, 3, 0);
            box.CheckedChanged += delegate
            {
                changed();
                RefreshCanvases();
            };
        }

        private void ToggleDebugPanel(object sender, EventArgs e)
        {
            debugExpanded = !debugExpanded;
            debugPanel.Visible = debugExpanded;
            leftLayout.RowStyles[3].Height = debugExpanded ? 370 : 0;
            debugToggleButton.Text = debugExpanded
                ? "▼ 舵机调试"
                : "▶ 舵机调试（默认折叠）";
        }

        private int SelectedJointIndex
        {
            get { return jointGrid.CurrentRow == null ? -1 : jointGrid.CurrentRow.Index; }
        }

        private void SelectedJointChanged(object sender, EventArgs e)
        {
            int index = SelectedJointIndex;
            if (index < 0 || activeProfile == null || index >= activeProfile.Joints.Count)
            {
                selectedJointLabel.Text = "未选择关节";
                return;
            }
            JointConfig joint = activeProfile.Joints[index];
            selectedJointLabel.Text = String.Format(
                "J{0} · ID {1} · {2}",
                index + 1,
                joint.ServoId,
                joint.AxisType == JointAxisType.Yaw ? "yaw" : "pitch");
            updatingTarget = true;
            targetSlider.Minimum = (int)Math.Ceiling(joint.MinimumAngle);
            targetSlider.Maximum = (int)Math.Floor(joint.MaximumAngle);
            if (targetSlider.Minimum >= targetSlider.Maximum)
            {
                targetSlider.Minimum = -90;
                targetSlider.Maximum = 90;
            }
            targetNumber.Minimum = (decimal)joint.MinimumAngle;
            targetNumber.Maximum = (decimal)joint.MaximumAngle;
            double value = displayedFrame != null && index < displayedFrame.Joints.Length
                ? displayedFrame.Joints[index].TargetDegrees
                : 0;
            value = Math.Max(joint.MinimumAngle, Math.Min(joint.MaximumAngle, value));
            targetNumber.Value = (decimal)value;
            targetSlider.Value = Math.Max(targetSlider.Minimum,
                Math.Min(targetSlider.Maximum, (int)Math.Round(value)));
            updatingTarget = false;
            if (simulateSelectedOnlyBox.Checked)
            {
                runtime.SetAutomaticSimulationJoint(index);
            }
        }

        private void SyncTargetFromFrame()
        {
            int index = SelectedJointIndex;
            if (displayedFrame == null || index < 0 || index >= displayedFrame.Joints.Length ||
                targetNumber.Focused || targetSlider.Focused)
            {
                return;
            }
            JointConfig joint = activeProfile.Joints[index];
            double value = Math.Max(joint.MinimumAngle,
                Math.Min(joint.MaximumAngle, displayedFrame.Joints[index].TargetDegrees));
            updatingTarget = true;
            targetNumber.Value = (decimal)value;
            targetSlider.Value = Math.Max(targetSlider.Minimum,
                Math.Min(targetSlider.Maximum, (int)Math.Round(value)));
            updatingTarget = false;
        }

        private void TargetSliderChanged(object sender, EventArgs e)
        {
            if (updatingTarget)
            {
                return;
            }
            updatingTarget = true;
            targetNumber.Value = Math.Max(targetNumber.Minimum,
                Math.Min(targetNumber.Maximum, targetSlider.Value));
            updatingTarget = false;
            TargetEdited((double)targetNumber.Value);
        }

        private void TargetNumberChanged(object sender, EventArgs e)
        {
            if (updatingTarget)
            {
                return;
            }
            updatingTarget = true;
            targetSlider.Value = Math.Max(targetSlider.Minimum,
                Math.Min(targetSlider.Maximum, (int)Math.Round(targetNumber.Value)));
            updatingTarget = false;
            TargetEdited((double)targetNumber.Value);
        }

        private void TargetEdited(double value)
        {
            int index = SelectedJointIndex;
            if (index < 0 || playback != null)
            {
                return;
            }
            if (runtime.Mode == RuntimeMode.SimulationManual)
            {
                runtime.SetManualSimulationAngle(index, value);
            }
            else
            {
                runtime.SetTargetPreview(index, value);
                if (runtime.Mode == RuntimeMode.Live && realtimeSendBox.Checked &&
                    safetyUnlockBox.Checked && runtime.TorqueActive &&
                    (DateTime.UtcNow - lastRealtimeSend).TotalMilliseconds >= 100)
                {
                    runtime.QueueTarget(index, value, (int)speedBox.Value,
                        (int)torqueLimitBox.Value, (int)accelerationBox.Value);
                    lastRealtimeSend = DateTime.UtcNow;
                }
            }
        }

        private void SendSelectedTarget(object sender, EventArgs e)
        {
            int index = SelectedJointIndex;
            if (index < 0 || activeProfile == null || index >= activeProfile.Joints.Count)
            {
                MessageBox.Show(this, "请先在主列表中选中一个关节。", "未选择关节",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (runtime.Mode != RuntimeMode.Live)
            {
                MessageBox.Show(this, "舵机输出只允许在实机模式下执行。模拟和回放模式不会访问串口。",
                    "输出已阻止", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (!safetyUnlockBox.Checked || !runtime.TorqueActive)
            {
                MessageBox.Show(this, "请先确认安全并安全使能扭矩。", "尚未解锁",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            runtime.QueueTarget(index, (double)targetNumber.Value, (int)speedBox.Value,
                (int)torqueLimitBox.Value, (int)accelerationBox.Value);
        }

        private void EnableTorque(object sender, EventArgs e)
        {
            if (runtime.Mode != RuntimeMode.Live)
            {
                MessageBox.Show(this, "扭矩只能在实机串口模式中使能。", "输出已阻止",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (!safetyUnlockBox.Checked)
            {
                MessageBox.Show(this, "请先勾选安全确认。", "尚未解锁",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            DialogResult result = MessageBox.Show(
                this,
                "将以当前实测位置作为首个目标，并使能所有在线舵机扭矩。\r\n请确认机器人周围安全。",
                "确认使能扭矩",
                MessageBoxButtons.OKCancel,
                MessageBoxIcon.Warning);
            if (result == DialogResult.OK)
            {
                runtime.QueueEnableTorque((int)torqueLimitBox.Value,
                    (int)speedBox.Value, (int)accelerationBox.Value);
            }
        }

        private void EmergencyRelease(object sender, EventArgs e)
        {
            if (runtime.Mode == RuntimeMode.Live)
            {
                runtime.QueueDisableTorque();
            }
            safetyUnlockBox.Checked = false;
            realtimeSendBox.Checked = false;
            stateLabel.Text = "已请求紧急释放全部扭矩";
        }

        private void AxisModeChanged(object sender, EventArgs e)
        {
            if (axisModeBox.SelectedIndex == 1)
            {
                runtime.UseCurrentAxisAsManual();
            }
            else
            {
                RollingAxisState current = runtime.GetAxisState();
                runtime.SetManualAxis(false, current.Point, current.Direction);
            }
        }

        private void ToggleAxisLock(object sender, EventArgs e)
        {
            RollingAxisState axis = runtime.GetAxisState();
            bool locked = !axis.Locked;
            runtime.SetAxisLocked(locked);
            lockAxisButton.Text = locked ? "解除轴锁定" : "锁定当前轴";
        }

        private void CurrentAxisToManual(object sender, EventArgs e)
        {
            runtime.UseCurrentAxisAsManual();
            axisModeBox.SelectedIndex = 1;
        }

        private void EditAxis(object sender, EventArgs e)
        {
            RollingAxisState current = runtime.GetAxisState();
            if (!current.Valid && displayedFrame != null)
            {
                current = displayedFrame.RollingAxis;
            }
            using (AxisEditorForm dialog = new AxisEditorForm(current))
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    runtime.SetManualAxis(true, dialog.AxisPoint, dialog.AxisDirection);
                    axisModeBox.SelectedIndex = 1;
                }
            }
        }

        private void EditPipe(object sender, EventArgs e)
        {
            using (PipeOutlineForm dialog = new PipeOutlineForm(applicationState.Display))
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    RefreshCanvases();
                }
            }
        }

        private void EditProfile(object sender, EventArgs e)
        {
            if (!EnsureIdleForConfiguration())
            {
                return;
            }
            using (ProfileEditorForm dialog = new ProfileEditorForm(activeProfile))
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    RobotProfile edited = dialog.ResultProfile;
                    store.SaveProfile(edited);
                    int existing = profiles.FindIndex(x => x.Name == activeProfile.Name);
                    if (existing >= 0)
                    {
                        profiles[existing] = edited;
                    }
                    else
                    {
                        profiles.Add(edited);
                    }
                    LoadProfilesIntoCombo(edited.Name);
                    BindProfile(edited);
                }
            }
        }

        private void DuplicateProfile(object sender, EventArgs e)
        {
            if (!EnsureIdleForConfiguration())
            {
                return;
            }
            string name = PromptText("复制机器人档案", "新档案名称", activeProfile.Name + " 副本");
            if (String.IsNullOrWhiteSpace(name))
            {
                return;
            }
            if (profiles.Any(x => x.Name == name))
            {
                MessageBox.Show(this, "同名档案已经存在。", "无法复制",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            RobotProfile copy = store.Duplicate(activeProfile, name.Trim());
            profiles.Add(copy);
            LoadProfilesIntoCombo(copy.Name);
            BindProfile(copy);
        }

        private bool EnsureIdleForConfiguration()
        {
            if (runtime.Mode != RuntimeMode.Idle || playback != null)
            {
                MessageBox.Show(this, "请先停止实时读取、模拟或回放，再修改机器人档案。",
                    "当前正在运行", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return false;
            }
            return true;
        }

        private void RefreshSerialPorts(object sender, EventArgs e)
        {
            if (inputModeBox.SelectedIndex != 0)
            {
                stateLabel.Text = "模拟模式不会访问串口；如需刷新，请先选择“实机串口”";
                return;
            }
            refreshPortsButton.Enabled = false;
            serialOperation = true;
            stateLabel.Text = "正在后台枚举串口…";
            ThreadPool.QueueUserWorkItem(delegate
            {
                string[] ports;
                string error = null;
                try
                {
                    ports = SerialPort.GetPortNames().OrderBy(x => x).ToArray();
                }
                catch (Exception ex)
                {
                    ports = new string[0];
                    error = ex.Message;
                }
                SafeBeginInvoke(delegate
                {
                    refreshPortsButton.Enabled = true;
                    serialOperation = false;
                    stateLabel.Text = error == null
                        ? "可用串口：" + (ports.Length == 0 ? "无" : String.Join("、", ports))
                        : "枚举串口失败：" + error;
                });
            });
        }

        private void CheckProfileIds(object sender, EventArgs e)
        {
            if (!CanRunBusProbe())
            {
                return;
            }
            checkIdsButton.Enabled = false;
            serialOperation = true;
            stateLabel.Text = "正在后台检查档案 ID…";
            RobotProfile profile = activeProfile.Clone();
            ThreadPool.QueueUserWorkItem(delegate
            {
                List<int> online = new List<int>();
                string error = null;
                try
                {
                    using (FeetechBus bus = new FeetechBus(profile.Port, profile.BaudRate))
                    {
                        bus.Open();
                        foreach (JointConfig joint in profile.Joints)
                        {
                            if (bus.Ping(joint.ServoId))
                            {
                                online.Add(joint.ServoId);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    error = ex.Message;
                }
                SafeBeginInvoke(delegate
                {
                    checkIdsButton.Enabled = true;
                    serialOperation = false;
                    if (error != null)
                    {
                        stateLabel.Text = "ID 检查失败：" + FriendlyPortError(error);
                    }
                    else
                    {
                        stateLabel.Text = String.Format(
                            "档案 ID 检查完成：{0}/{1} 在线（{2}）",
                            online.Count,
                            profile.Joints.Count,
                            online.Count == 0 ? "无" : String.Join("、", online));
                    }
                });
            });
        }

        private void ScanAllIds(object sender, EventArgs e)
        {
            if (!CanRunBusProbe())
            {
                return;
            }
            scanBusButton.Enabled = false;
            serialOperation = true;
            stateLabel.Text = "正在后台扫描总线 ID 0–253，界面仍可操作…";
            RobotProfile profile = activeProfile.Clone();
            ThreadPool.QueueUserWorkItem(delegate
            {
                List<int> found = new List<int>();
                string error = null;
                try
                {
                    if (busScanOverride != null)
                    {
                        found.AddRange(busScanOverride(profile));
                    }
                    else
                    {
                        using (FeetechBus bus = new FeetechBus(
                            profile.Port,
                            profile.BaudRate,
                            25))
                        {
                            bus.Open();
                            for (int id = 0; id <= 253; id++)
                            {
                                if (bus.Ping(id))
                                {
                                    found.Add(id);
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    error = ex.Message;
                }
                SafeBeginInvoke(delegate
                {
                    scanBusButton.Enabled = true;
                    serialOperation = false;
                    if (error != null)
                    {
                        stateLabel.Text = "总线扫描失败：" + FriendlyPortError(error);
                        return;
                    }
                    stateLabel.Text = "扫描完成，发现 ID：" +
                        (found.Count == 0 ? "无" : String.Join("、", found));
                    if (found.Count > 0)
                    {
                        ImportScanResults(found);
                    }
                });
            });
        }

        private bool CanRunBusProbe()
        {
            if (inputModeBox.SelectedIndex != 0)
            {
                stateLabel.Text = "模拟模式不会访问串口；总线检查仅在“实机串口”来源下可用";
                return false;
            }
            if (runtime.Mode != RuntimeMode.Idle || playback != null)
            {
                MessageBox.Show(this, "请先停止实时读取或回放，避免争用同一串口。",
                    "串口正在使用", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return false;
            }
            return true;
        }

        private void ImportScanResults(List<int> found)
        {
            List<int> existing = activeProfile.Joints.Select(x => x.ServoId).ToList();
            List<int> missing = found.Where(x => !existing.Contains(x)).ToList();
            if (missing.Count == 0)
            {
                MessageBox.Show(this, "扫描结果中没有需要新增到档案的 ID。",
                    "扫描结果", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            DialogResult answer = MessageBox.Show(
                this,
                "发现档案外 ID：" + String.Join("、", missing) +
                "\r\n是否确认追加到档案末尾？\r\n注意：软件不会猜测机械安装顺序，请在档案编辑器中复核并调整顺序。",
                "确认导入扫描结果",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question);
            if (answer != DialogResult.Yes)
            {
                return;
            }
            RobotProfile edited = activeProfile.Clone();
            foreach (int id in missing)
            {
                int index = edited.Joints.Count;
                edited.Joints.Add(new JointConfig
                {
                    Index = index + 1,
                    ServoId = id,
                    AxisType = index % 2 == 0 ? JointAxisType.Yaw : JointAxisType.Pitch,
                    Zero = 2048,
                    Direction = 1,
                    LinkLength = 89.5
                });
            }
            edited.Reindex();
            store.SaveProfile(edited);
            ReplaceActiveProfile(edited);
        }

        private void CalibrateSelected(object sender, EventArgs e)
        {
            int index = SelectedJointIndex;
            CalibrateIndices(index < 0 ? new int[0] : new int[] { index });
        }

        private void CalibrateAll(object sender, EventArgs e)
        {
            RobotFrame frame;
            if (!runtime.TryGetLatest(out frame))
            {
                MessageBox.Show(this, "没有可用于标定的实时实测帧。", "无法标定",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            CalibrateIndices(Enumerable.Range(0, frame.Joints.Length)
                .Where(i => frame.Joints[i].Online).ToArray());
        }

        private void CalibrateIndices(int[] indices)
        {
            RobotFrame frame;
            if (runtime.Mode != RuntimeMode.Live || !runtime.TryGetLatest(out frame))
            {
                MessageBox.Show(this, "零位标定只使用实机实时读取值。请先启动实机读取。",
                    "无法标定", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            indices = indices.Where(i => i >= 0 && i < activeProfile.Joints.Count &&
                i < frame.Joints.Length && frame.Joints[i].Online).ToArray();
            if (indices.Length == 0)
            {
                MessageBox.Show(this, "选定范围内没有在线关节。", "无法标定",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            string differences = String.Join("\r\n", indices.Select(i => String.Format(
                "J{0} / ID {1}: {2} → {3}（差值 {4:+#;-#;0}）",
                i + 1,
                activeProfile.Joints[i].ServoId,
                activeProfile.Joints[i].Zero,
                frame.Joints[i].Raw,
                Kinematics.ShortestCountDelta(frame.Joints[i].Raw, activeProfile.Joints[i].Zero))));
            DialogResult answer = MessageBox.Show(
                this,
                "将把当前实测位置设为零角。请复核变化：\r\n\r\n" + differences +
                "\r\n\r\n应用后实时读取会停止，可使用“撤销标定”恢复。",
                "确认零位标定",
                MessageBoxButtons.OKCancel,
                MessageBoxIcon.Warning);
            if (answer != DialogResult.OK)
            {
                return;
            }
            calibrationBackup = activeProfile.Clone();
            RobotProfile edited = activeProfile.Clone();
            foreach (int index in indices)
            {
                edited.Joints[index].Zero = frame.Joints[index].Raw;
            }
            runtime.Stop();
            StopRecordingInternal();
            store.SaveProfile(edited);
            ReplaceActiveProfile(edited);
            undoCalibrationButton.Enabled = true;
            stateLabel.Text = "零位标定已保存；实时读取已停止";
        }

        private void UndoCalibration(object sender, EventArgs e)
        {
            if (calibrationBackup == null || runtime.Mode != RuntimeMode.Idle)
            {
                return;
            }
            RobotProfile restored = calibrationBackup.Clone();
            store.SaveProfile(restored);
            calibrationBackup = null;
            ReplaceActiveProfile(restored);
            undoCalibrationButton.Enabled = false;
            stateLabel.Text = "已撤销最近一次零位标定";
        }

        private void ReplaceActiveProfile(RobotProfile edited)
        {
            int index = profiles.FindIndex(x => x.Name == activeProfile.Name);
            if (index >= 0)
            {
                profiles[index] = edited;
            }
            else
            {
                profiles.Add(edited);
            }
            LoadProfilesIntoCombo(edited.Name);
            BindProfile(edited);
        }

        private void StartRecording(object sender, EventArgs e)
        {
            if (playback != null || runtime.Mode == RuntimeMode.Idle)
            {
                MessageBox.Show(this, "请先启动实时读取或模拟，再开始记录。",
                    "无法记录", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            try
            {
                string root = Path.Combine(Application.StartupPath, "experiments");
                string folder = recorder.Start(
                    root,
                    experimentNameBox.Text,
                    activeProfile,
                    applicationState.Display);
                recordButton.Enabled = false;
                stopRecordButton.Enabled = true;
                recordStateLabel.Text = "正在记录：" + Path.GetFileName(folder);
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "无法开始记录",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void StopRecording(object sender, EventArgs e)
        {
            StopRecordingInternal();
        }

        private void StopRecordingInternal()
        {
            if (recorder.IsRecording)
            {
                string folder = recorder.Folder;
                recorder.Stop();
                recordStateLabel.Text = "记录已安全写完：" + Path.GetFileName(folder);
            }
            recordButton.Enabled = true;
            stopRecordButton.Enabled = false;
        }

        private void LoadPlayback(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog
            {
                Filter = "实验元数据 (metadata.json)|metadata.json|CSV 记录 (frames.csv)|frames.csv",
                Title = "选择实验文件夹中的 metadata.json 或 frames.csv"
            })
            {
                if (dialog.ShowDialog(this) != DialogResult.OK)
                {
                    return;
                }
                try
                {
                    StopRecordingInternal();
                    runtime.Stop();
                    playback = ExperimentPlayback.Load(dialog.FileName);
                    playbackIndex = 0;
                    playbackPositionSeconds = playback.ElapsedSeconds[0];
                    playbackPlaying = false;
                    playbackTrack.Maximum = Math.Max(0, playback.Frames.Count - 1);
                    displayProfile = playback.Profile;
                    RebuildGridForPlaybackProfile();
                    ShowPlaybackFrame(0);
                    stateLabel.Text = String.Format(
                        "已加载回放：{0} 帧（回放模式完全隔离舵机输出）",
                        playback.Frames.Count);
                    UpdateControlStates();
                }
                catch (Exception ex)
                {
                    playback = null;
                    MessageBox.Show(this, ex.Message, "无法加载回放",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void RebuildGridForPlaybackProfile()
        {
            jointGrid.Rows.Clear();
            foreach (JointConfig joint in playback.Profile.Joints)
            {
                jointGrid.Rows.Add(
                    "J" + joint.Index,
                    joint.ServoId,
                    joint.AxisType == JointAxisType.Yaw ? "yaw" : "pitch",
                    "—", "—", "回放", "—");
            }
            if (jointGrid.Rows.Count > 0)
            {
                jointGrid.Rows[0].Selected = true;
            }
        }

        private void TogglePlayback(object sender, EventArgs e)
        {
            if (playback == null)
            {
                return;
            }
            playbackPlaying = !playbackPlaying;
            playbackPlayButton.Text = playbackPlaying ? "暂停" : "播放";
            playbackTickUtc = DateTime.UtcNow;
        }

        private void StopPlayback(object sender, EventArgs e)
        {
            StopPlaybackInternal();
            BindProfile(activeProfile);
            stateLabel.Text = "已退出回放";
            UpdateControlStates();
        }

        private void StopPlaybackInternal()
        {
            playbackPlaying = false;
            playbackPlayButton.Text = "播放";
            playback = null;
            playbackIndex = 0;
            curveGammaBox.Value = (decimal)Math.Max(0.01, Math.Min(30, applicationState.Display.CurveGammaDegrees));
            playbackTrack.Maximum = 0;
            playbackTrack.Value = 0;
            displayProfile = activeProfile;
        }

        private void StepPlayback(int delta)
        {
            if (playback == null)
            {
                return;
            }
            playbackPlaying = false;
            playbackPlayButton.Text = "播放";
            ShowPlaybackFrame(Math.Max(0,
                Math.Min(playback.Frames.Count - 1, playbackIndex + delta)));
        }

        private void PlaybackTrackScrolled(object sender, EventArgs e)
        {
            if (!updatingPlaybackTrack && playback != null)
            {
                playbackPlaying = false;
                playbackPlayButton.Text = "播放";
                ShowPlaybackFrame(playbackTrack.Value);
            }
        }

        private void AdvancePlayback()
        {
            if (playback == null)
            {
                return;
            }
            if (!playbackPlaying)
            {
                return;
            }
            DateTime now = DateTime.UtcNow;
            double delta = (now - playbackTickUtc).TotalSeconds * PlaybackSpeed;
            playbackTickUtc = now;
            playbackPositionSeconds += Math.Max(0, delta);
            int index = playbackIndex;
            while (index + 1 < playback.Frames.Count &&
                playback.ElapsedSeconds[index + 1] <= playbackPositionSeconds)
            {
                index++;
            }
            if (index != playbackIndex)
            {
                ShowPlaybackFrame(index, false);
            }
            if (playbackIndex >= playback.Frames.Count - 1)
            {
                playbackPlaying = false;
                playbackPlayButton.Text = "播放";
            }
        }

        private double PlaybackSpeed
        {
            get
            {
                return playbackSpeedBox.SelectedIndex == 0 ? 0.25 :
                    playbackSpeedBox.SelectedIndex == 1 ? 0.5 :
                    playbackSpeedBox.SelectedIndex == 3 ? 2.0 : 1.0;
            }
        }

        private void ShowPlaybackFrame(int index)
        {
            ShowPlaybackFrame(index, true);
        }

        private void ShowPlaybackFrame(int index, bool resetClock)
        {
            if (playback == null)
            {
                return;
            }
            playbackIndex = Math.Max(0, Math.Min(playback.Frames.Count - 1, index));
            RobotFrame frame = playback.Frames[playbackIndex].Clone();
            if (recomputeAxisBox.Checked)
            {
                if (playback.RecomputedFrames != null)
                    frame = playback.RecomputedFrames[playbackIndex].Clone();
                else if (!recomputingPlayback)
                {
                    recomputingPlayback = true;
                    PlaybackData pending = playback;
                    ThreadPool.QueueUserWorkItem(delegate
                    {
                        Exception failure = null;
                        try { ExperimentPlayback.EnsureRecomputed(pending); }
                        catch (Exception ex) { failure = ex; }
                        if (closing || IsDisposed) return;
                        try
                        {
                            BeginInvoke((Action)delegate
                            {
                                recomputingPlayback = false;
                                if (playback != pending)
                                {
                                    if (playback != null && recomputeAxisBox.Checked) ShowPlaybackFrame(playbackIndex);
                                    return;
                                }
                                if (failure != null)
                                {
                                    recomputeAxisBox.Checked = false;
                                    stateLabel.Text = "重算失败：" + failure.Message;
                                }
                                else ShowPlaybackFrame(playbackIndex);
                            });
                        }
                        catch (InvalidOperationException) { }
                    });
                }
            }
            curveGammaBox.Value = (decimal)Math.Max(0.01, Math.Min(30, frame.CurveGammaRadians * 180 / Math.PI));
            displayedFrame = frame;
            displayProfile = playback.Profile;
            UpdateJointRows(frame, playback.Profile);
            RefreshCanvases();
            updatingPlaybackTrack = true;
            playbackTrack.Value = playbackIndex;
            updatingPlaybackTrack = false;
            playbackPositionSeconds = playback.ElapsedSeconds[playbackIndex];
            if (resetClock)
            {
                playbackTickUtc = DateTime.UtcNow;
            }
            stateLabel.Text = String.Format(
                "回放 {0}/{1} · {2:0.000}s · {3}",
                playbackIndex + 1,
                playback.Frames.Count,
                playbackPositionSeconds,
                recomputeAxisBox.Checked ? (playback.RecomputedFrames == null
                    ? "正在后台重算，暂显示记录轴" : "重新计算轴") : "记录轴");
        }

        private void UpdateControlStates()
        {
            bool idle = runtime.Mode == RuntimeMode.Idle;
            bool inPlayback = playback != null;
            curveGammaBox.Enabled = !inPlayback;
            bool liveSelected = inputModeBox.SelectedIndex == 0;
            bool liveRunning = runtime.Mode == RuntimeMode.Live;
            profileBox.Enabled = idle && !inPlayback && !serialOperation;
            inputModeBox.Enabled = idle && !inPlayback && !serialOperation;
            startButton.Enabled = idle && !inPlayback && !serialOperation;
            stopButton.Enabled = !idle;
            editProfileButton.Enabled = idle && !inPlayback && !serialOperation;
            duplicateProfileButton.Enabled = idle && !inPlayback && !serialOperation;
            refreshPortsButton.Enabled = idle && !inPlayback && liveSelected && !serialOperation;
            checkIdsButton.Enabled = idle && !inPlayback && liveSelected && !serialOperation;
            scanBusButton.Enabled = idle && !inPlayback && liveSelected && !serialOperation;
            calibrateJointButton.Enabled = liveRunning;
            calibrateAllButton.Enabled = liveRunning;
            targetSlider.Enabled = !inPlayback && runtime.Mode != RuntimeMode.SimulationAuto;
            targetNumber.Enabled = targetSlider.Enabled;
            safetyUnlockBox.Enabled = liveRunning;
            sendTargetButton.Enabled = liveRunning;
            realtimeSendBox.Enabled = liveRunning && runtime.TorqueActive && safetyUnlockBox.Checked;
            enableTorqueButton.Enabled = liveRunning && safetyUnlockBox.Checked && !runtime.TorqueActive;
            emergencyReleaseButton.Enabled = liveRunning;
            if (!liveRunning)
            {
                debugHintLabel.Text = "舵机输出仅在“实机串口”开始读取后可用。";
                debugHintLabel.ForeColor = Color.FromArgb(110, 110, 110);
            }
            else if (!safetyUnlockBox.Checked)
            {
                debugHintLabel.Text = "第 1 步：勾选下方安全确认，随后“安全使能扭矩”按钮会变为可用。";
                debugHintLabel.ForeColor = Color.FromArgb(174, 112, 20);
            }
            else if (!runtime.TorqueActive)
            {
                debugHintLabel.Text = "第 2 步：点击“安全使能扭矩”，确认后才能发送目标角。";
                debugHintLabel.ForeColor = Color.FromArgb(174, 112, 20);
            }
            else
            {
                debugHintLabel.Text = "扭矩已使能：可点击“发送目标”；需要拖动即发送时再勾选“滑动实时发送”。";
                debugHintLabel.ForeColor = Color.FromArgb(38, 132, 82);
            }
            recordButton.Enabled = !inPlayback && !idle && !recorder.IsRecording;
            loadPlaybackButton.Enabled = idle && !recorder.IsRecording;
            playbackPlayButton.Enabled = inPlayback;
            playbackStopButton.Enabled = inPlayback;
            stepBackButton.Enabled = inPlayback;
            stepForwardButton.Enabled = inPlayback;
            playbackTrack.Enabled = inPlayback;
            recomputeAxisBox.Enabled = inPlayback;
            if (!liveRunning)
            {
                safetyUnlockBox.Checked = false;
                realtimeSendBox.Checked = false;
            }
        }

        private Dictionary<string, object> HandleLocalCommand(Dictionary<string, object> request)
        {
            string command = request.ContainsKey("cmd")
                ? Convert.ToString(request["cmd"]).ToLowerInvariant()
                : "state";
            if (command == "geometry") return runtime.GeometryForApi();
            if (command == "target_frame") return runtime.SubmitTargetFrame(request, controlSpeed);
            if (command == "cancel_targets")
            {
                runtime.CancelTargetFrames(Convert.ToString(request["_connection_id"]));
                return Ok("本连接的待处理目标已清除");
            }
            if (command == "state")
            {
                RobotFrame frame;
                bool hasFrame = runtime.TryGetLatest(out frame);
                return new Dictionary<string, object>
                {
                    { "ok", true },
                    { "mode", runtime.Mode.ToString() },
                    { "torque_active", runtime.TorqueActive },
                    { "profile", activeProfile.Name },
                    { "sequence", hasFrame ? frame.Sequence : 0 },
                    { "displayed_sequence", Interlocked.Read(ref displayedSequence) },
                    { "displayed_count", Interlocked.Read(ref displayedFrameCount) },
                    { "painted_3d_count", Interlocked.Read(ref painted3DCount) },
                    { "painted_2d_count", Interlocked.Read(ref painted2DCount) },
                    { "source", hasFrame ? frame.Source : "" },
                    { "mujoco_sim_time", runtime.Mode == RuntimeMode.Mujoco ? lastMujocoSimTime : 0.0 },
                    { "mujoco_captured_unix", runtime.Mode == RuntimeMode.Mujoco ? lastMujocoCapturedUnix : 0.0 },
                    { "samples_per_second", runtime.GetStatistics().SamplesPerSecond },
                    { "angles_deg", hasFrame ? frame.Joints.Select(x => x.AngleDegrees).ToArray() : new double[0] },
                    { "targets_deg", hasFrame ? frame.Joints.Select(x => x.TargetDegrees).ToArray() : new double[0] },
                    { "online", hasFrame ? frame.Joints.Select(x => x.Online).ToArray() : new bool[0] },
                    { "points_mm", hasFrame ? frame.Points.Select(p => new double[] { p.X, p.Y, p.Z }).ToArray() : new double[0][] },
                    { "raw_axis", hasFrame ? AxisForApi(frame.RawAxis) : null },
                    { "display_axis", hasFrame ? AxisForApi(frame.RollingAxis) : null },
                    { "curve_valid", hasFrame && frame.Curve != null && frame.Curve.Valid },
                    { "curve_status", hasFrame && frame.Curve != null ? frame.Curve.Status : "" },
                    { "curve_segments", hasFrame && frame.Curve != null ? frame.Curve.Segments.Count : 0 },
                    { "curve_max_midpoint_error_mm", hasFrame && frame.Curve != null ? frame.Curve.MaximumMidpointError : 0.0 },
                    { "curve_modified_count", hasFrame && frame.Curve != null ? frame.Curve.ModifiedCount : 0 }
                    ,{ "curve_join_max_mm", hasFrame ? CurveJoinMax(frame.Curve, false) : 0.0 }
                    ,{ "curve_tangent_max", hasFrame ? CurveJoinMax(frame.Curve, true) : 0.0 }
                };
            }
            if (command == "mujoco_frame")
            {
                string sourceOwner = Convert.ToString(request["_connection_id"]);
                bool relay = request.ContainsKey("relay_enabled") && Convert.ToBoolean(request["relay_enabled"]);
                runtime.ValidateMujocoSource(sourceOwner);
                if (runtime.Mode != RuntimeMode.Idle && runtime.Mode != RuntimeMode.Mujoco)
                    throw new InvalidOperationException("当前正在使用其他数据源，未切换到 MuJoCo");
                if (mujocoStoppedByUser)
                    throw new InvalidOperationException("MuJoCo 已手动停止，请选择 MuJoCo 并点击开始读取");
                MujocoFramePayload packet = new MujocoFramePayload
                {
                    Angles = ReadFiniteArray(request, "angles_deg"),
                    Targets = ReadFiniteArray(request, "targets_deg"),
                    Lengths = ReadFiniteArray(request, "link_lengths_mm"),
                    SimTime = request.ContainsKey("sim_time") ? Convert.ToDouble(request["sim_time"]) : 0.0,
                    CapturedUnix = request.ContainsKey("captured_unix") ? Convert.ToDouble(request["captured_unix"]) : 0.0
                };
                double[] coordinates = ReadFiniteArray(request, "head_origin_world_mm");
                double[] basis = ReadFiniteArray(request, "head_basis_world");
                if (packet.Angles.Length < 2 || packet.Angles.Length > 253 ||
                    packet.Targets.Length != packet.Angles.Length ||
                    packet.Lengths.Length != packet.Angles.Length + 1 ||
                    coordinates.Length != 3 || basis.Length != 9 ||
                    coordinates.Any(x => Math.Abs(x) > 1000000) ||
                    packet.Lengths.Any(x => x < 1 || x > 2000) ||
                    Double.IsNaN(packet.SimTime) || Double.IsInfinity(packet.SimTime) ||
                    Double.IsNaN(packet.CapturedUnix) || Double.IsInfinity(packet.CapturedUnix) ||
                    packet.CapturedUnix < 0 || packet.CapturedUnix > RobotRuntime.UnixMs(DateTime.UtcNow) / 1000 + 1)
                    throw new ArgumentException("MuJoCo 角度或头部世界位姿数量不符或超出范围");
                packet.HeadOriginWorld = new Point3D(coordinates[0], coordinates[1], coordinates[2]);
                packet.HeadBasisWorld = new Matrix3
                {
                    M11 = basis[0], M12 = basis[1], M13 = basis[2],
                    M21 = basis[3], M22 = basis[4], M23 = basis[5],
                    M31 = basis[6], M32 = basis[7], M33 = basis[8]
                };
                if (runtime.Mode == RuntimeMode.Mujoco && mujocoReady)
                {
                    RobotProfile current = runtime.Profile;
                    if (current != null && current.Port == "MUJOCO-NO-SERIAL" &&
                        current.Joints.Count == packet.Angles.Length && MujocoLengthsMatch(current, packet))
                    {
                        if (relay) runtime.RegisterMujocoExecutor(sourceOwner);
                        runtime.UpdateMujocoPose(packet.Angles, packet.Targets,
                            packet.HeadOriginWorld, packet.HeadBasisWorld, packet.CapturedUnix, packet.SimTime);
                        lastMujocoSimTime = packet.SimTime;
                        lastMujocoCapturedUnix = packet.CapturedUnix;
                        if (relay) return runtime.MujocoReply(sourceOwner,
                            request.ContainsKey("applied_run_id") ? Convert.ToString(request["applied_run_id"]) : "",
                            request.ContainsKey("applied_delivery_id") ? Convert.ToInt64(request["applied_delivery_id"]) : 0);
                        return Ok("MuJoCo 实测帧已重建");
                    }
                }
                lock (mujocoSync) { pendingMujocoFrame = packet; }
                Dictionary<string, object> accepted = Ok("已接收 MuJoCo 最新帧");
                if (relay) accepted["relay_ready"] = false;
                return accepted;
            }
            if (command == "release")
            {
                runtime.CancelTargetFrames(null);
                if (runtime.Mode == RuntimeMode.Live)
                {
                    runtime.QueueDisableTorque();
                }
                return Ok("已请求释放扭矩");
            }
            if (command == "target")
            {
                if (runtime.Mode != RuntimeMode.Live || !runtime.TorqueActive)
                {
                    throw new InvalidOperationException("桌面程序未处于已使能扭矩的实机模式");
                }
                if (request.ContainsKey("joint_index") && request.ContainsKey("degrees"))
                {
                    int index = Convert.ToInt32(request["joint_index"]) - 1;
                    double degrees = Convert.ToDouble(request["degrees"]);
                    runtime.QueueTarget(index, degrees, controlSpeed,
                        controlTorqueLimit, controlAcceleration);
                }
                else
                {
                    if (request.ContainsKey("j1_deg"))
                    {
                        runtime.QueueTarget(0, Convert.ToDouble(request["j1_deg"]),
                            controlSpeed, controlTorqueLimit, controlAcceleration);
                    }
                    if (request.ContainsKey("j2_deg"))
                    {
                        runtime.QueueTarget(1, Convert.ToDouble(request["j2_deg"]),
                            controlSpeed, controlTorqueLimit, controlAcceleration);
                    }
                }
                return Ok("目标已进入串口命令队列");
            }
            if (command == "limits")
            {
                int index = request.ContainsKey("joint_index")
                    ? Convert.ToInt32(request["joint_index"]) - 1
                    : 0;
                int torque = request.ContainsKey("torque_limit")
                    ? Convert.ToInt32(request["torque_limit"])
                    : controlTorqueLimit;
                int acceleration = request.ContainsKey("acceleration")
                    ? Convert.ToInt32(request["acceleration"])
                    : controlAcceleration;
                runtime.QueueLimits(index, torque, acceleration);
                return Ok("限制参数已进入命令队列");
            }
            throw new InvalidOperationException("未知命令：" + command);
        }

        private static Dictionary<string, object> Ok(string message)
        {
            return new Dictionary<string, object>
            {
                { "ok", true },
                { "message", message }
            };
        }

        private static Dictionary<string, object> AxisForApi(RollingAxisState axis)
        {
            if (axis == null) return null;
            return new Dictionary<string, object>
            {
                { "direction", new double[] { axis.Direction.X, axis.Direction.Y, axis.Direction.Z } },
                { "confidence", axis.Confidence },
                { "valid", axis.Valid },
                { "status", axis.Status }
            };
        }

        private static double CurveJoinMax(CrossSectionCurve curve, bool tangent)
        {
            if (curve == null || !curve.Valid) return 0;
            double maximum = 0;
            for (int i = 1; i < curve.Segments.Count; i++)
            {
                CurveSegment prior = curve.Segments[i - 1];
                CurveSegment current = curve.Segments[i];
                double error = tangent
                    ? (prior.TangentAt(1) - current.Tangent).Length
                    : (prior.At(1) - current.Start).Length;
                maximum = Math.Max(maximum, error);
            }
            return maximum;
        }

        private static double[] ReadFiniteArray(Dictionary<string, object> request, string key)
        {
            if (!request.ContainsKey(key) || !(request[key] is System.Collections.IEnumerable))
                throw new ArgumentException("缺少 " + key);
            List<double> values = new List<double>();
            foreach (object item in (System.Collections.IEnumerable)request[key])
            {
                double value = Convert.ToDouble(item);
                if (Double.IsNaN(value) || Double.IsInfinity(value))
                    throw new ArgumentException(key + " 包含无效数值");
                values.Add(value);
            }
            return values.ToArray();
        }

        private static bool MujocoLengthsMatch(RobotProfile profile, MujocoFramePayload packet)
        {
            if (packet.Lengths.Length != profile.Joints.Count + 1 || Math.Abs(profile.HeadLength - packet.Lengths[0]) > 1e-6)
                return false;
            for (int i = 0; i < profile.Joints.Count; i++)
                if (Math.Abs(profile.Joints[i].LinkLength - packet.Lengths[i + 1]) > 1e-6) return false;
            return true;
        }

        private static RobotProfile CreateMujocoProfile(MujocoFramePayload packet)
        {
            RobotProfile profile = new RobotProfile
            {
                Name = "MuJoCo " + packet.Angles.Length + "关节（临时）",
                Port = "MUJOCO-NO-SERIAL",
                HeadLength = packet.Lengths[0]
            };
            for (int i = 0; i < packet.Angles.Length; i++)
                profile.Joints.Add(new JointConfig
                {
                    Index = i + 1,
                    ServoId = i + 1,
                    AxisType = i % 2 == 0 ? JointAxisType.Yaw : JointAxisType.Pitch,
                    LinkLength = packet.Lengths[i + 1]
                });
            return profile;
        }

        private void SafeBeginInvoke(Action action)
        {
            if (closing || IsDisposed)
            {
                return;
            }
            try
            {
                BeginInvoke(action);
            }
            catch
            {
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            closing = true;
            refreshTimer.Stop();
            StopRecordingInternal();
            runtime.Dispose();
            if (localServer != null)
            {
                localServer.Dispose();
            }
            if (activeProfile.Port != "MUJOCO-NO-SERIAL")
                applicationState.LastProfile = activeProfile.Name;
            try { store.SaveApplicationState(applicationState); } catch { }
            base.OnFormClosing(e);
        }

        private static string FriendlyPortError(string message)
        {
            return message.IndexOf("access", StringComparison.OrdinalIgnoreCase) >= 0 ||
                message.IndexOf("denied", StringComparison.OrdinalIgnoreCase) >= 0 ||
                message.IndexOf("拒绝", StringComparison.OrdinalIgnoreCase) >= 0
                ? "串口被占用，请关闭 FD 调试助手或其他串口程序"
                : message;
        }

        private static Control LabeledInline(string label, Control control, int width)
        {
            FlowLayoutPanel panel = new FlowLayoutPanel
            {
                AutoSize = true,
                WrapContents = false,
                Margin = new Padding(3)
            };
            panel.Controls.Add(new Label
            {
                Text = label,
                AutoSize = true,
                Margin = new Padding(0, 7, 3, 0)
            });
            control.Width = width;
            panel.Controls.Add(control);
            return panel;
        }

        private static Control WithCaption(string text, Control content)
        {
            Panel panel = new Panel { Dock = DockStyle.Fill };
            Label caption = new Label
            {
                Text = text,
                Dock = DockStyle.Top,
                Height = 26,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(8, 0, 0, 0),
                BackColor = Color.FromArgb(33, 41, 54),
                ForeColor = Color.White
            };
            content.Dock = DockStyle.Fill;
            panel.Controls.Add(content);
            panel.Controls.Add(caption);
            return panel;
        }

        private static void ConfigureButton(Button button, string text, EventHandler handler)
        {
            button.Text = text;
            button.AutoSize = true;
            button.Height = 30;
            button.Margin = new Padding(3);
            button.Click += handler;
        }

        private static void AddDebugField(
            TableLayoutPanel table,
            int row,
            int column,
            string text,
            NumericUpDown box)
        {
            table.Controls.Add(new Label
            {
                Text = text,
                AutoSize = true,
                Anchor = AnchorStyles.Left
            }, column, row);
            box.Dock = DockStyle.Fill;
            table.Controls.Add(box, column + 1, row);
        }

        private static string PromptText(string title, string label, string initial)
        {
            using (Form form = new Form
            {
                Text = title,
                StartPosition = FormStartPosition.CenterParent,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                MaximizeBox = false,
                MinimizeBox = false,
                Size = new Size(390, 155),
                Font = new Font("Microsoft YaHei UI", 9F)
            })
            {
                Label prompt = new Label
                {
                    Text = label,
                    AutoSize = true,
                    Location = new Point(12, 14)
                };
                TextBox value = new TextBox
                {
                    Text = initial,
                    Location = new Point(15, 40),
                    Width = 345
                };
                Button ok = new Button
                {
                    Text = "确定",
                    DialogResult = DialogResult.OK,
                    Location = new Point(205, 75),
                    Width = 75
                };
                Button cancel = new Button
                {
                    Text = "取消",
                    DialogResult = DialogResult.Cancel,
                    Location = new Point(285, 75),
                    Width = 75
                };
                form.Controls.AddRange(new Control[] { prompt, value, ok, cancel });
                form.AcceptButton = ok;
                form.CancelButton = cancel;
                return form.ShowDialog() == DialogResult.OK ? value.Text.Trim() : null;
            }
        }
    }
}
