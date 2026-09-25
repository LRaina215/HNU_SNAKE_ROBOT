# v1.4.0：MuJoCo 完整执行闭环

本版在现有协议版本 2 上增加可选的 MuJoCo 执行能力；旧 `geometry`、`preview`、`live` 和普通 `mujoco_frame` 调用仍兼容。版本号未更改为 3，文件名仅用于区分阶段四说明。

## 使用

1. 关闭旧 EXE，启动 `visualizer/windows_app/bin/SnakeRobotVisualizer.exe`（或发布版 v1.4.0 及以上）。默认端口仍为 8765；应用启动为空闲，不自动连接真实串口。
2. 在原 `mujoco` 环境运行 `codes/Adaptive_Pole_Climbing.py`。顶部 `CONTROL_VIA_VISUALIZER=True` 为默认。
3. 等待终端出现“转发执行端已连接”。现在目标必须经过 EXE，随后才写入 MuJoCo；EXE 没启动时仿真仍推进，初始目标保持零。程序不会自动改走直接输出路径。
4. DEBUG 曲线/表格开关仍有效。停止 Python 或关闭 EXE 后，新的目标不再执行。MuJoCo 在连接中断后保持最后目标并继续物理步进；保持目标不代表物理姿态被锁定，重力和接触仍然作用。

回退旧直接驱动方式：`CONTROL_VIA_VISUALIZER=False`，或环境变量 `SNAKE_CONTROL_VIA_VISUALIZER=0`。该方式用于历史算法回归，EXE 仅预览，不作为本版完整通路的验收依据。`COMPLIANCE_ENABLED` 独立控制顺应性算法。

## 通路与线程

Python 控制器计算整组目标 → `GeometryExchange(execution="mujoco")` → EXE 校验/保存最新待执行组 → `MujocoExecutionBridge` 上传实测帧时收到该组 → 仿真线程在物理步进前整组写入 `d.ctrl` → 实际 qpos 经 FK/轴估计/连续曲线形成下一轮反馈。

控制器和 MuJoCo 执行端仍可处于同一 Python 进程；两条后台 TCP 连接分别承担控制反馈与仿真实测/执行。网络线程不访问 mjData，物理步进不等待网络。没有增加新端口或同步往返等待。

## 协议增加项

`target_frame.execution` 增加 `mujoco`，要求当前数据源为 MuJoCo、已有活动执行连接，并沿用整组角度/配置/反馈时效/控制连接所有权校验。`preview` 永远不进入仿真执行；`live` 仍要求实机模式和已有的手动扭矩使能。

执行端在**持久连接**上的 `mujoco_frame` 请求附带：

```json
{"relay_enabled": true, "applied_run_id": "上次实际写入的运行标识", "applied_delivery_id": 12}
```

上述为新增字段，角度、长度、头部世界位姿等原有字段仍需提供。`targets_deg` 是执行端当前真正持有的 d.ctrl，不是控制算法刚计算的候选目标。

准备完成后回复含 `relay_ready=true`、`run_id` 和 `simulation_target`。无新目标时后者为 null；有目标时包含 `run_id/config_id`、`delivery_id`、`command_id`、`feedback_sequence`、`targets_deg`、`expires_unix_ms`。首次创建配置期间 `relay_ready=false`。不支持转发的旧 EXE 缺少该字段，Python 会明确提示使用 v1.4.0。

`delivery_id` 在同一运行内递增，执行端以 `(run_id, delivery_id)` 去重。目标只领取一次；主线程执行前再次检查运行标识、时效、数组长度/数值及模型关节范围。下一次实测上传携带已经写入 d.ctrl 的投递号，表示软件执行确认，不表示关节已到位。

`geometry.target_state` 增加 `mujoco_executor_ready`、`mujoco_delivery_id`、`mujoco_applied_delivery_id`；发送给执行端的状态为 `dispatched_mujoco`。接受、转发、实际写入三个环节可以分别核对。

## 失效与所有权

同一时刻只有一个持久连接可作为 MuJoCo 执行端，其上传的模型产生反馈。其他连接不能覆盖这个运行的实测帧。执行连接断开或超过 1 秒未更新会清除未执行目标；控制连接断开、取消、目标过期、配置切换和停止也会清除相应待执行目标。

已经返回给 Python 的包无法由 EXE 撤回，接收端仍受原截止时间约束。失联后最多可能执行仍在有效期内的在途包；过期包不应用。没有新执行时不推进下一轮控制相位。恢复连接后使用新反馈，不重放断连缓存。停止 EXE 不是紧急释放扭矩命令，实机策略不由这里的模拟保持行为替代。

## 验证范围

- `windows_app/CommunicationTests.cs`：执行端要求、独占、最新整组、预览隔离、不篡改实际角、执行确认、断连、过期、旧配置、停止；实机路径仅用 FakeBus 核对舵机 ID 和校准映射。
- `../validation/adaptive_pole_stage4.py`：无 EXE 不旁路；20/28 关节每次 d.ctrl 与实际收到的 EXE 包逐项对应；物理实际角与目标有动态差异；EXE 重启后的保持/恢复；旧包/重复包/越界包接收拒绝。
- 默认隔离端口 18787，结果在步态仓库 `validation/results/adaptive-pole-stage4/`。设置 `SNAKE_MUJOCO_TRACE` 时，除实际 ctrl/qpos，还保存候选 `requested_ctrl`、实际应用标识与 `.received.json`，用于验证执行来源；日常运行不记录这些历史。

本版验收是完整模拟通信/执行闭环，不等同于稳定攀爬或实机适配完成。本轮没有连接真实串口、使能真实舵机或进行硬件试验。
