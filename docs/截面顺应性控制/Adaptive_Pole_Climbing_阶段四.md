# 阶段四：经可视化程序转发的 MuJoCo 执行

完整启动见[主 README](../../README.md)。可视化源码统一在本仓库 `visualizer/`；构建后启动 `visualizer/windows_app/bin/SnakeRobotVisualizer.exe`，再运行 `codes/Adaptive_Pole_Climbing.py`。最低支持 MuJoCo 执行协议的发布版为 v1.4.0；本次表格绘制修复随 v1.4.1 发布。

```python
CONTROL_VIA_VISUALIZER = True  # 默认：EXE 转发后才执行
COMPLIANCE_ENABLED = True     # 实测截面顺应性算法
DEBUG = True                 # 接收曲线调试窗口
DEBUG_DRAW_CURVE = True       # False 时仅显示参数表
```

算法仍计算整组候选目标，但转发模式中不直接写 `d.ctrl`。`mujoco_execution_bridge.py` 接收 EXE 返回目标，仿真主线程在物理步进前校验并整组应用；后台连接不操作 MuJoCo 数据。上传给 EXE 的实际角来自物理步进后的 qpos，目标显示来自当前实际执行的 ctrl。

控制器和执行端在同一个 Python 进程中，用两条持久连接交换各自的数据。单个执行目标只领取一次，以运行标识和投递号去重；下一次实测快照附带已写入 ctrl 的投递号。它是软件执行确认，不表示关节已经到位。

当前默认单步：在 MuJoCo 窗口 Ctrl+U 应用标准螺旋，Ctrl+J/H 正反向一步，空格暂停。其他操作见[变径实验指南](Adaptive_Pole_Climbing_变径实验.md)。空闲时物理时间和相位冻结，通信及窗口刷新继续。等待 EXE 返回本次候选期间不执行物理步；最多等待 2 秒，所用顺应性反馈超过 500 ms 则取消本步。已开始的一步完成 N×10 个物理步后暂停。迟到的旧目标不得覆盖空闲 ctrl 或替代下一步候选。

历史直接驱动/预览回归可设置 `CONTROL_VIA_VISUALIZER=False` 或环境变量 `SNAKE_CONTROL_VIA_VISUALIZER=0`。关闭可视化通信时必须同时关闭转发执行模式。

协议见[COMMUNICATION_V3.md](../../visualizer/COMMUNICATION_V3.md)。两端默认使用本机 8765 端口。此处文件名用于区分阶段文档，线协议版本仍为 2。

复验：`python validation/adaptive_pole_stage4.py`，在隔离端口测试无 EXE、20/28 关节、输入校验、断开重启与执行来源逐项核对。固定输入在 `validation/fixtures/`，输出在 `validation/results/adaptive-pole-stage4/`。提交的验收摘要在 `validation/reports/`，完整状态见[验收记录](../../visualizer/STAGE4_RESULTS.md)。

阶段三提交 `6cb07e2` 已推送。阶段四与框架整理共同提交；验证范围为完整模拟执行闭环，动态稳定攀爬、接触力效果及实机仍需专项验收。
