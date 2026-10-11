# 已确认全局路线的真实机构与连续动作入口

本轮最终状态：**G01、G02 均失败，未搭管或完成跨越；2 个实际候选已用完，C01 未运行。** 详见 [RESULT.md](RESULT.md)。当前实现的是内侧下部顺势上绕候选，反勾备选未启用。

## 代码入口与每步做什么

`snake_control/approved_route_transfer.py` 中的 `ApprovedRouteTransfer.command(data, t)` 是步态入口，返回 20 个关节目标和可记录的反馈信息。

1. `geometry.Route` 直接加载已确认的完整全局曲线，始终保存原管螺旋、298 mm 过渡与目标螺旋。
2. `ExecutionCurve` 保留固定原管承载半径修正，按相关材料段的实际偏差施加有界局部曲线形变。它的 `window(shift)` 逐杆寻找 64.5 mm 前序弦，输出 P0 头到 P21 尾的 22 个点。
3. `ApprovedRouteTransfer` 读取允许的 MuJoCo 实际身体、关节角和驱动力矩，判断前部包绕、局部受阻及跟随情况，更新局部系数和共同推进速度。滚动相位严格与共同推进增量联动。
4. G01 使用固定实际承载位置/方向的拟合；G02 使用 `CoordinatedBearingMapper`，把私有承载框架与共同曲线增量一起输运，再通过全部 20 个真实交替关节拟合整条执行曲线。
5. 关节目标限制在 ±88°，每 50 ms 目标变化不超过 1°，驱动力矩沿用 ±2.355 N·m 限制。私有拟合根位姿只存在于独立 FK 数据；实际仿真初始化后只有关节驱动。

局部系数是过渡弧的 X/Y/Z 形变，范围分别为 [−10,+10]、[−12,+12]、[−18,+12] mm，系数每步变化不超过 0.2 mm；两端位置和切线不被该反馈场改变。实际仍靠原管的计划出管段在 G02 中参与调整，后部承载区不跟着放松。目标半径反馈仅在实际几何包绕后启用；本轮未触发。

## 本轮执行方式

在仓库根目录，用 `D:/Anaconda/envs/mujoco/python.exe` 运行：

```text
validation/approved_route_experiment.py --prepare
validation/approved_route_experiment.py --name G01 --variant 1
validation/approved_route_experiment.py --name G02 --variant 2
```

这些是本轮已执行的命令记录。入口自带本轮 90 分钟截止时间与两候选额度，当前不允许再开启第三次动力学运行，也没有为了消耗剩余时间重复尝试。已保存结果可只读重新绘图：

```text
validation/approved_route_audit.py
```

`approved_route_experiment.py --review --name G01` 或 `G02` 从保存的实际状态重建连续视频；它只做结果渲染，不运行新的动力学轨迹。

## 数据对应与文件

- `initial_fit.json`、`mechanism_fit_checks.json`、`mechanism_fit_overview.png`：必要机构拟合与初始原管接触检查。拟合物理步数为 0。
- `G02_CHANGE.md`、`G02_dispatch_fit.json`：针对第一次卡点的改法及必要首步拟合。
- `ledger.json`：本轮实际 2 候选、0 对照。
- 各候选 `global_local_actual_3d.png`、`curve_comparison.png`：同一身体节点的全局、局部和实际身体。
- 各候选 `curve_snapshots.csv`：图示时刻的三种曲线，包含时间、共同 shift、P0→P21 节点编号及毫米坐标。
- 各候选 `feedback_and_progress.png`、`observed_audit.json`：调形何时生效、空间误差和实际进展。
- 各候选 `actual_source_direction.png`：追踪固定材料杆，区分计划角度与实际角度。
- 各候选 `rear_feed_breakpoint.png`、`feed_breakpoint_audit.json`：共同 shift、后部轴向/绕行、承载注册与后部关节目标之间的断点。保留现有起点，不做起点重画。
- 各候选 `trace.npz`：100 Hz 实际状态，包含 `qpos/qvel/q/ctrl/torque_nm`、三种节点曲线、诊断接触、方向与推进记录。
- 各候选 `commands.json`：20 Hz 控制目标、真实杆长执行参考、材料曲线坐标、局部系数、拟合预测与反馈信息。
- 各候选 `actual_continuous_1x.mp4`、`video.json`：完整实际 1 倍速视频和首/中/末帧可读检查。
- 各候选 `sources/`：开始运行前保存的该版控制、驱动、模型与已确认曲线代码。

P0 是头部节点，P21 是尾部节点；杆 Mi 连接 Pi→P(i+1)，M14 为第 15 根杆。本轮方向图跟踪其中心，与之前曲线确认图所用的节点 P14 名称不同，但各图都固定同一个材料对象。尾看头沿 +X，右为 −Y，上为 +Z，正角为逆时针。

原始失败材料和 ≥2 MB 文件均保留本地，结果图、摘要和代码纳入本次提交。公共文档由监督会话维护，本目录只记录执行与证据。
