using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO.Ports;
using System.Linq;
using System.Windows.Forms;

namespace SnakeRobotVisualizer
{
    internal sealed class ProfileEditorForm : Form
    {
        private readonly TextBox nameBox = new TextBox();
        private readonly ComboBox portBox = new ComboBox();
        private readonly NumericUpDown baudBox = new NumericUpDown();
        private readonly NumericUpDown headLengthBox = new NumericUpDown();
        private readonly DataGridView grid = new DataGridView();
        private readonly RobotProfile working;

        public RobotProfile ResultProfile { get; private set; }

        public ProfileEditorForm(RobotProfile source)
        {
            working = source.Clone();
            Text = "机器人档案编辑";
            StartPosition = FormStartPosition.CenterParent;
            MinimumSize = new Size(900, 520);
            Size = new Size(1050, 650);
            Font = new Font("Microsoft YaHei UI", 9F);

            TableLayoutPanel root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 4,
                Padding = new Padding(10)
            };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            Controls.Add(root);

            FlowLayoutPanel fields = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                WrapContents = true
            };
            fields.Controls.Add(Labeled("档案名称", nameBox, 190));
            portBox.DropDownStyle = ComboBoxStyle.DropDown;
            foreach (string port in SerialPort.GetPortNames().OrderBy(x => x))
            {
                portBox.Items.Add(port);
            }
            fields.Controls.Add(Labeled("串口", portBox, 105));
            baudBox.Minimum = 1200;
            baudBox.Maximum = 3000000;
            baudBox.Increment = 9600;
            fields.Controls.Add(Labeled("波特率", baudBox, 115));
            headLengthBox.Minimum = 1;
            headLengthBox.Maximum = 2000;
            headLengthBox.DecimalPlaces = 1;
            fields.Controls.Add(Labeled("头段长度 mm", headLengthBox, 105));
            root.Controls.Add(fields, 0, 0);

            ConfigureGrid();
            root.Controls.Add(grid, 0, 1);

            FlowLayoutPanel rowButtons = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true
            };
            Button add = MakeButton("新增关节", AddJoint);
            Button remove = MakeButton("删除关节", RemoveJoint);
            Button up = MakeButton("上移", delegate { MoveSelected(-1); });
            Button down = MakeButton("下移", delegate { MoveSelected(1); });
            Button template = MakeButton("奇数 yaw、偶数 pitch", ApplyAlternatingTemplate);
            rowButtons.Controls.AddRange(new Control[] { add, remove, up, down, template });
            root.Controls.Add(rowButtons, 0, 2);

            FlowLayoutPanel footer = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                FlowDirection = FlowDirection.RightToLeft
            };
            Button save = MakeButton("保存", SaveAndClose);
            save.Width = 100;
            Button cancel = MakeButton("取消", delegate { DialogResult = DialogResult.Cancel; Close(); });
            cancel.Width = 100;
            footer.Controls.Add(save);
            footer.Controls.Add(cancel);
            Label hint = new Label
            {
                AutoSize = true,
                Text = "方向只能为 +1 或 -1；ID 必须唯一。机械顺序由表格行顺序决定。",
                ForeColor = Color.DimGray,
                Margin = new Padding(4, 9, 20, 0)
            };
            footer.Controls.Add(hint);
            root.Controls.Add(footer, 0, 3);

            LoadProfile();
        }

        private void ConfigureGrid()
        {
            grid.Dock = DockStyle.Fill;
            grid.AllowUserToAddRows = false;
            grid.AllowUserToDeleteRows = false;
            grid.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            grid.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            grid.MultiSelect = false;
            grid.RowHeadersVisible = false;
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Index",
                HeaderText = "序号",
                ReadOnly = true,
                FillWeight = 45
            });
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "ServoId",
                HeaderText = "舵机 ID",
                FillWeight = 65
            });
            grid.Columns.Add(new DataGridViewComboBoxColumn
            {
                Name = "Axis",
                HeaderText = "转轴",
                DataSource = new string[] { "yaw", "pitch" },
                FillWeight = 70
            });
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Zero",
                HeaderText = "零位原始值",
                FillWeight = 85
            });
            grid.Columns.Add(new DataGridViewComboBoxColumn
            {
                Name = "Direction",
                HeaderText = "方向",
                DataSource = new string[] { "+1", "-1" },
                FillWeight = 55
            });
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Length",
                HeaderText = "连杆长度 mm",
                FillWeight = 90
            });
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Minimum",
                HeaderText = "最小角 °",
                FillWeight = 75
            });
            grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Maximum",
                HeaderText = "最大角 °",
                FillWeight = 75
            });
            grid.DataError += delegate(object sender, DataGridViewDataErrorEventArgs e)
            {
                e.ThrowException = false;
            };
        }

        private void LoadProfile()
        {
            nameBox.Text = working.Name;
            portBox.Text = working.Port;
            baudBox.Value = Math.Max(baudBox.Minimum, Math.Min(baudBox.Maximum, working.BaudRate));
            headLengthBox.Value = (decimal)Math.Max(
                (double)headLengthBox.Minimum,
                Math.Min((double)headLengthBox.Maximum, working.HeadLength));
            grid.Rows.Clear();
            foreach (JointConfig joint in working.Joints)
            {
                grid.Rows.Add(
                    joint.Index,
                    joint.ServoId,
                    joint.AxisType == JointAxisType.Yaw ? "yaw" : "pitch",
                    joint.Zero,
                    joint.Direction < 0 ? "-1" : "+1",
                    joint.LinkLength,
                    joint.MinimumAngle,
                    joint.MaximumAngle);
            }
        }

        private void AddJoint(object sender, EventArgs e)
        {
            int count = grid.Rows.Count;
            int nextId = 1;
            HashSet<int> used = new HashSet<int>(
                grid.Rows.Cast<DataGridViewRow>()
                    .Select(x => SafeInt(x.Cells["ServoId"].Value, -1)));
            while (used.Contains(nextId) && nextId < 253)
            {
                nextId++;
            }
            grid.Rows.Add(
                count + 1,
                nextId,
                count % 2 == 0 ? "yaw" : "pitch",
                2048,
                "+1",
                89.5,
                -90,
                90);
            grid.ClearSelection();
            grid.Rows[grid.Rows.Count - 1].Selected = true;
        }

        private void RemoveJoint(object sender, EventArgs e)
        {
            if (grid.Rows.Count <= 1)
            {
                MessageBox.Show(this, "至少保留一个关节。", "不能删除",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (grid.CurrentRow != null)
            {
                grid.Rows.Remove(grid.CurrentRow);
                ReindexRows();
            }
        }

        private void MoveSelected(int delta)
        {
            if (grid.CurrentRow == null)
            {
                return;
            }
            int index = grid.CurrentRow.Index;
            int target = index + delta;
            if (target < 0 || target >= grid.Rows.Count)
            {
                return;
            }
            object[] values = grid.Rows[index].Cells.Cast<DataGridViewCell>()
                .Select(x => x.Value).ToArray();
            grid.Rows.RemoveAt(index);
            grid.Rows.Insert(target, values);
            ReindexRows();
            grid.CurrentCell = grid.Rows[target].Cells[1];
            grid.Rows[target].Selected = true;
        }

        private void ApplyAlternatingTemplate(object sender, EventArgs e)
        {
            for (int i = 0; i < grid.Rows.Count; i++)
            {
                grid.Rows[i].Cells["Axis"].Value = i % 2 == 0 ? "yaw" : "pitch";
            }
        }

        private void SaveAndClose(object sender, EventArgs e)
        {
            try
            {
                grid.EndEdit();
                RobotProfile profile = new RobotProfile
                {
                    Name = nameBox.Text.Trim(),
                    Port = portBox.Text.Trim(),
                    BaudRate = (int)baudBox.Value,
                    HeadLength = (double)headLengthBox.Value
                };
                for (int i = 0; i < grid.Rows.Count; i++)
                {
                    DataGridViewRow row = grid.Rows[i];
                    string direction = Convert.ToString(row.Cells["Direction"].Value);
                    string axis = Convert.ToString(row.Cells["Axis"].Value);
                    JointConfig joint = new JointConfig
                    {
                        Index = i + 1,
                        ServoId = ParseInt(row, "ServoId", "舵机 ID"),
                        AxisType = String.Equals(axis, "pitch", StringComparison.OrdinalIgnoreCase)
                            ? JointAxisType.Pitch
                            : JointAxisType.Yaw,
                        Zero = ParseInt(row, "Zero", "零位"),
                        Direction = direction == "-1" ? -1 : 1,
                        LinkLength = ParseDouble(row, "Length", "连杆长度"),
                        MinimumAngle = ParseDouble(row, "Minimum", "最小角"),
                        MaximumAngle = ParseDouble(row, "Maximum", "最大角")
                    };
                    profile.Joints.Add(joint);
                }
                List<string> errors = profile.Validate();
                if (errors.Count > 0)
                {
                    throw new InvalidOperationException(String.Join("\r\n", errors.ToArray()));
                }
                ResultProfile = profile;
                DialogResult = DialogResult.OK;
                Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "档案配置无效",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private static int ParseInt(DataGridViewRow row, string name, string label)
        {
            int value;
            if (!Int32.TryParse(Convert.ToString(row.Cells[name].Value), out value))
            {
                throw new InvalidOperationException(
                    String.Format("第 {0} 行的{1}不是有效整数", row.Index + 1, label));
            }
            return value;
        }

        private static double ParseDouble(DataGridViewRow row, string name, string label)
        {
            double value;
            if (!Double.TryParse(Convert.ToString(row.Cells[name].Value), out value))
            {
                throw new InvalidOperationException(
                    String.Format("第 {0} 行的{1}不是有效数字", row.Index + 1, label));
            }
            return value;
        }

        private static int SafeInt(object value, int fallback)
        {
            int parsed;
            return Int32.TryParse(Convert.ToString(value), out parsed) ? parsed : fallback;
        }

        private void ReindexRows()
        {
            for (int i = 0; i < grid.Rows.Count; i++)
            {
                grid.Rows[i].Cells["Index"].Value = i + 1;
            }
        }

        private static Control Labeled(string text, Control control, int width)
        {
            Panel panel = new Panel
            {
                Width = width + 12,
                Height = 55,
                Margin = new Padding(3)
            };
            Label label = new Label
            {
                Text = text,
                AutoSize = true,
                Location = new Point(0, 0)
            };
            control.Location = new Point(0, 22);
            control.Width = width;
            panel.Controls.Add(label);
            panel.Controls.Add(control);
            return panel;
        }

        private static Button MakeButton(string text, EventHandler handler)
        {
            Button button = new Button
            {
                Text = text,
                AutoSize = true,
                Height = 30,
                Margin = new Padding(3, 5, 3, 5)
            };
            button.Click += handler;
            return button;
        }
    }

    internal sealed class AxisEditorForm : Form
    {
        private readonly NumericUpDown[] values = new NumericUpDown[6];
        public Point3D AxisPoint { get; private set; }
        public Point3D AxisDirection { get; private set; }

        public AxisEditorForm(RollingAxisState current)
        {
            Text = "手动 rolling 轴";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            Size = new Size(470, 280);
            Font = new Font("Microsoft YaHei UI", 9F);
            TableLayoutPanel table = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(12),
                ColumnCount = 4,
                RowCount = 4
            };
            Controls.Add(table);
            string[] labels = { "轴心 X", "轴心 Y", "轴心 Z", "方向 X", "方向 Y", "方向 Z" };
            double[] initial =
            {
                current.Point.X, current.Point.Y, current.Point.Z,
                current.Direction.X, current.Direction.Y, current.Direction.Z
            };
            for (int i = 0; i < values.Length; i++)
            {
                Label label = new Label
                {
                    Text = labels[i],
                    AutoSize = true,
                    Anchor = AnchorStyles.Left
                };
                NumericUpDown box = new NumericUpDown
                {
                    DecimalPlaces = 4,
                    Minimum = -100000,
                    Maximum = 100000,
                    Increment = i < 3 ? 1 : 0.01M,
                    Value = (decimal)Math.Max(-100000, Math.Min(100000, initial[i])),
                    Dock = DockStyle.Fill
                };
                values[i] = box;
                int row = i / 2;
                int col = (i % 2) * 2;
                table.Controls.Add(label, col, row);
                table.Controls.Add(box, col + 1, row);
            }
            Label hint = new Label
            {
                Text = "点向式输入：轴线 = 轴心 + t × 方向。方向会自动归一化。",
                AutoSize = true,
                Dock = DockStyle.Fill
            };
            table.SetColumnSpan(hint, 4);
            table.Controls.Add(hint, 0, 3);
            FlowLayoutPanel buttons = new FlowLayoutPanel
            {
                FlowDirection = FlowDirection.RightToLeft,
                Dock = DockStyle.Bottom,
                Height = 42,
                Padding = new Padding(8)
            };
            Button ok = new Button { Text = "应用", DialogResult = DialogResult.OK };
            Button cancel = new Button { Text = "取消", DialogResult = DialogResult.Cancel };
            ok.Click += Save;
            buttons.Controls.Add(ok);
            buttons.Controls.Add(cancel);
            Controls.Add(buttons);
        }

        private void Save(object sender, EventArgs e)
        {
            Point3D direction = new Point3D(
                (double)values[3].Value,
                (double)values[4].Value,
                (double)values[5].Value);
            if (direction.Length < 0.000001)
            {
                MessageBox.Show(this, "轴方向不能为零向量。", "输入无效",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                DialogResult = DialogResult.None;
                return;
            }
            AxisPoint = new Point3D(
                (double)values[0].Value,
                (double)values[1].Value,
                (double)values[2].Value);
            AxisDirection = direction.Normalized();
        }
    }

    internal sealed class PipeOutlineForm : Form
    {
        private readonly CheckBox enabled = new CheckBox();
        private readonly ComboBox shape = new ComboBox();
        private readonly NumericUpDown radiusX = MakeNumber(1, 5000, 160);
        private readonly NumericUpDown radiusY = MakeNumber(1, 5000, 160);
        private readonly NumericUpDown centerX = MakeNumber(-5000, 5000, 0);
        private readonly NumericUpDown centerY = MakeNumber(-5000, 5000, 0);
        private readonly DisplaySettings settings;

        public PipeOutlineForm(DisplaySettings source)
        {
            settings = source;
            Text = "静态管道轮廓";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            Size = new Size(380, 330);
            Font = new Font("Microsoft YaHei UI", 9F);
            TableLayoutPanel table = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(15),
                ColumnCount = 2,
                RowCount = 7
            };
            Controls.Add(table);
            enabled.Text = "显示静态参考轮廓";
            enabled.Checked = source.ShowPipeOutline;
            table.SetColumnSpan(enabled, 2);
            table.Controls.Add(enabled, 0, 0);
            shape.DropDownStyle = ComboBoxStyle.DropDownList;
            shape.Items.AddRange(new object[] { "圆形", "椭圆形" });
            shape.SelectedIndex = source.PipeEllipse ? 1 : 0;
            radiusX.Value = Clamp(radiusX, source.PipeRadiusX);
            radiusY.Value = Clamp(radiusY, source.PipeRadiusY);
            centerX.Value = Clamp(centerX, source.PipeCenterX);
            centerY.Value = Clamp(centerY, source.PipeCenterY);
            AddRow(table, 1, "形状", shape);
            AddRow(table, 2, "半径 X / 圆半径 mm", radiusX);
            AddRow(table, 3, "半径 Y mm", radiusY);
            AddRow(table, 4, "中心偏移 X mm", centerX);
            AddRow(table, 5, "中心偏移 Y mm", centerY);
            FlowLayoutPanel buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.RightToLeft
            };
            Button ok = new Button { Text = "应用", DialogResult = DialogResult.OK };
            Button cancel = new Button { Text = "取消", DialogResult = DialogResult.Cancel };
            ok.Click += Apply;
            buttons.Controls.Add(ok);
            buttons.Controls.Add(cancel);
            table.SetColumnSpan(buttons, 2);
            table.Controls.Add(buttons, 0, 6);
        }

        private void Apply(object sender, EventArgs e)
        {
            settings.ShowPipeOutline = enabled.Checked;
            settings.PipeEllipse = shape.SelectedIndex == 1;
            settings.PipeRadiusX = (double)radiusX.Value;
            settings.PipeRadiusY = (double)radiusY.Value;
            settings.PipeCenterX = (double)centerX.Value;
            settings.PipeCenterY = (double)centerY.Value;
        }

        private static NumericUpDown MakeNumber(decimal min, decimal max, decimal initial)
        {
            return new NumericUpDown
            {
                Minimum = min,
                Maximum = max,
                Value = initial,
                DecimalPlaces = 1,
                Dock = DockStyle.Fill
            };
        }

        private static decimal Clamp(NumericUpDown control, double value)
        {
            return Math.Max(control.Minimum, Math.Min(control.Maximum, (decimal)value));
        }

        private static void AddRow(TableLayoutPanel table, int row, string label, Control control)
        {
            table.Controls.Add(new Label
            {
                Text = label,
                AutoSize = true,
                Anchor = AnchorStyles.Left
            }, 0, row);
            table.Controls.Add(control, 1, row);
        }
    }
}
