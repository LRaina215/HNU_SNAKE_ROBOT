# Windows 可视化与目标转发

本目录是完整闭环的可视化源码，已纳入 Snake_Robot_Gait 仓库。启动整个闭环见[仓库 README](../README.md)；这里负责实测几何重建、显示和目标转发，顺应性目标与 Yamada 计算在 `../codes/`。

## 构建与运行

Windows 下双击 `启动程序.bat`，或在本目录执行：

```bat
windows_app\build.bat
windows_app\bin\SnakeRobotVisualizer.exe
```

构建使用 `%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe`，引用 .NET Framework 的 WinForms、Drawing、Web.Extensions 等内置程序集。`bin/` 中的 EXE、档案及 `app-state.json` 是生成/本地运行数据，不纳入 Git。首次运行自动创建默认档案；启动时不自动连接串口。

## 源码职责

| 文件 | 职责 |
| --- | --- |
| `windows_app/Program.cs` | 启动、DPI 设置和异常日志 |
| `windows_app/MainFormV2.cs` | 档案与界面交互、最新帧显示、本机命令分派 |
| `windows_app/RuntimeEngine.cs` | 数据源生命周期、模拟/实测处理、单一串口所有者 |
| `windows_app/Communication.cs` | 导出一致几何快照、验证整组目标、执行端归属、失效与执行确认 |
| `windows_app/LocalControlServer.cs` | 本机 TCP 逐行 JSON、连接生命周期 |
| `windows_app/Kinematics.cs` | 关节角到骨架点的正运动学 |
| `windows_app/RollingAxis.cs` | 螺旋轴估计与投影坐标系 |
| `windows_app/CrossSectionCurve.cs` | 二维分段直线/圆弧连续近似 |
| `windows_app/SnakeCanvas.cs` | 三维骨架和二维截面的绘制 |
| `windows_app/Models.cs` / `ProfileStore.cs` | 数据模型、档案与显示配置 |
| `windows_app/ExperimentData.cs` | 记录和回放 |
| `windows_app/FeetechProtocol.cs` | 串口协议，模拟测试以 FakeBus 替代 |
| `python_client.py` | 同步 TCP 客户端与后台 GeometryExchange |

数据采集/重建在运行时完成，UI 只取最新帧绘制。TCP 控制端提交的是目标角；MuJoCo 执行端上传的是物理步进后的实测角与当前 ctrl。不要将候选目标角充当实测反馈，也不要在 C# 内再实现一份顺应性算法。

## 通信

默认 `127.0.0.1:8765`。两端均可用 `SNAKE_VISUALIZER_PORT` 改端口。控制器与 MuJoCo 使用两条持久连接：前者取 geometry/提交 target_frame，后者上传 mujoco_frame/领取 simulation_target。协议版本仍为 2，v1.4.0 新增 MuJoCo 转发字段。

- [完整执行协议](COMMUNICATION_V3.md)
- [几何帧与目标提交协议](COMMUNICATION_V2.md)
- [验收记录](STAGE4_RESULTS.md)

实机路径继续由此 EXE 独占串口，且保留手动扭矩使能条件；当前 Adaptive 的自动执行入口使用 MuJoCo 模式。

## 验证

运行 `windows_app\run-tests.bat`：包括模型、几何、通信、协议、相机、UI 与持续运行测试。使用隔离端口 18781，串口操作使用 FakeBus，不操作真实舵机。

UI 测试会短暂将测试窗口置于前面，以测量实际绘制次数；它使用真实 WinForms 消息循环，检查 28 关节分屏至少 11 秒达到 9.5 Hz，不以后台产生的帧数替代画面刷新。任何非零退出码都使整套测试失败。

生成日志在 `artifacts/windows-native-acceptance/`，完整 MuJoCo 转发验证在仓库 `validation/adaptive_pole_stage4.py`。固定输入保存在 `../validation/fixtures/`，无需复制其他电脑的运行目录。
