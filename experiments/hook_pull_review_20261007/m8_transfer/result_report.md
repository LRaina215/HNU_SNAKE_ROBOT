# M8局部交接动作原型：离线受阻，动力学未启动

2026-10-07。**已实现一个连贯动作候选，在6.05 s处数值求解停止；M8尚未靠到目标管，交接段未生成，因此没有启动动力学。当前候选停止。** 本次新增动力学0，审阅累计仍为2/4，余量不启用。现场准备/采集安排已停止；没有重做C0、静止研究、调参或改模型。

**动作怎么组织。** 从完整CLOSE1B出发，始终以 `ctrl_hold + (q_shape − q_shape_initial)` 接入。14 s靠管段将M8中心向目标管径向内收至84.8 mm、沿Y跟进30 mm，并将其朝向转为管面切向，延续前钩；M9从接头外侧跟进，目标中心从z=1095.491降至1077.011 mm，不是旧方案的笼统外移/上抬。前部M0/M3与后部M17/M20的接触材质点作为私有运动学支撑假设，M7/M14只保留法向贴管、允许切向滑移；后部M13送身6 mm。计划在实际M8满足区域/接触力条件持续0.2 s后，才进入4 s交接：M7/M14局部径向让位1.5 mm、M8再跟进5 mm、M13再送4 mm，末目标保持10 s。门控最多等待2 s，总窗口不超过30 s。运行器的门控明确使用仿真真值，不宣称实机反馈或统一方法成立。

**实际做到哪里。** 以下全部是保存的私有FK参考，**不是机器人动力学表现**：

| M8指标 | 初始 | 求解停止处6.05 s |
| --- | ---: | ---: |
| 中心X / Y / Z（mm） | 1036.625 / 70.119 / 1089.566 | 1042.491 / 82.103 / 1081.768 |
| 距目标轴（mm） | 109.720 | 99.967 |
| 目标管有符号几何距离查询（mm） | +11.710 | +5.268 |

已生成参考中M8径向缩小9.753 mm，但距92 mm区域边界仍差7.967 mm，且没有目标管几何接触。未完成14 s靠管，未进入原支撑让位，也没有实际新支撑、实际迁移或停后保持记录；此前C0/P1证据范围不扩大。

**卡点及检查边界。** 同一个候选只求解一次。SLSQP在6.05 s报 `Positive directional derivative for linesearch`，当时M8位置任务偏差0.340 mm；前一段J1/J3持续达到10°/s，末步J8/J15/J20也达到该速率。已生成0–6.05 s参考经同一次检查（每50 ms节点及10 ms中间姿态）：命令最大88°、速率10°/s；按原接触生成器 `contact.dist` 判据，最大管侵入2.541 mm、最小包绕267.753°，没有记录到关节/自碰/地面/支撑几何筛查超限。**完整18 s路径没有生成，所以完整局部路径未准入。** 这些筛查不证明承载、摩擦或实际跟随。

求解器报错不是结构无解或物理交接失败的证明。合理推测是固定支撑点、法向约束与靠管/转向任务在当前时标下限制了可用关节运动；没有分离这些因素，不能宣称唯一原因。附加 `mj_geomDistance` 查询还给出M9对源管负值（初始−4.606、停止处−11.687 mm），与上述接触生成判据不同，原值保留；本轮未解释该差异，也未围绕接触/模型继续修改或重跑。

**交付与后续判断。** [动作及单次运行入口](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/m8_local_transfer.py>)已写好；`run`会拒绝当前NOT_READY路径。[登记](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/m8_transfer/registration.json>)、[离线结果](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/m8_transfer/geometry_result.json>)、[参考原始记录](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/m8_transfer/planned_path.npz>)及[几何记录](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/m8_transfer/geometry_trace.npz>)保留。[正常速度离线片段](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/m8_transfer/reference/reference_prefix_1x.mp4>)逐帧标注“OFFLINE PARTIAL REFERENCE / NO PHYSICS RUN”，包含私有求解根变量；不是实验视频，也不是完整动作。没有实际运行，故没有实际失败视频。

当前只停止这一候选，不推导机器人不能过接头。若继续，应审阅承接点的切向/转动自由度是否过度固定，保留两管支撑来组织靠管转向，而不是继续延长旧轴向前缀、重做静止对照或修饰环境。该方向是待验证假设，本次不自动实施。实机的摩擦/受力、形状与预载响应、可用反馈和身体迁移观测仍未验证；当前不要求用户采集，也不生成实机动作。
