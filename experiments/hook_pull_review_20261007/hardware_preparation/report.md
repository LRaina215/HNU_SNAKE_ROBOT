# 现场短程验证准备：接口核对与资料采集

2026-10-07。本次仅沿现有实机入口及直接通信/记录调用只读核对，导出已有冻结候选的离线增量。未启动 EXE、串口、硬件网络、键盘监听或控制器；未使能、下发、求解或步进物理。新增动力学 **0**，本轮审阅累计 **2/4**，余下两次不启用。旧 B 与历史前缀准入字段未改。

**结论：已有接口支持整组位置指令、实际角和部分状态读取；现有实机入口不能等价播放这条 41 行、50 ms 前缀。当前交付停在现场资料采集，尚无可直接执行的硬件包。** 缺的是现场编号/方向/零位、已执行夹紧目标来源、起点几何与预载、反馈有效性、实际时序和停止依据。不能用一个固定角偏置替代这些核对。

## 1. 已有证据和本次边界

已审阅 [C0/P1 短动力学报告](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/short_physics/report.md>)：两次各 12 s，P1 为 2 s 冻结动作加 10 s 末目标保持，都通过两管共同支撑的保持门槛。2 s 时 P1 相对 C0 前部平均轴向差 −0.112 mm、后部 +1.203 mm，前后自转差约 −3.36°/+2.50°；没有新模块迁入目标管。自转和后部局部响应有依据，前后主动移送、重复性、新增反馈作用仍未成立。

已确认 `hook_pull_step4_rolling.solve_active` 的进度目标取当前接触模块轴向速度平均值，而身体评估取 M0–M7/M13–M20。P1 的 M0–M3/M7 相对 C0 向前、M4–M6 向后；不能把接触模块进度等同整个前部进度，也不能认定这是唯一失败原因，或要求每节同时向前。本次不重跑分析、不改目标。源码出处：[solve_active](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/hook_pull_step4_rolling.py:159>)。

## 2. 接口核对：源码事实与现场未知

调用链为 `Adaptive_Pole_Climbing_Real.main` → `GeometryExchange.publish_targets` → `SnakeControlClient.submit_targets` → EXE `SubmitTargetFrame/DispatchTargetFrame` → `FeetechBus.SetTargetPositions`。EXE 是唯一串口所有者；Python 只经本机 TCP 交给 EXE。这里的事实是仓库实现，未核对现场 EXE 是否为同版，也未确认已加载档案。

| 核对项 | 仓库确认事实 | 现场仍需确认 / 使用限制 |
| --- | --- | --- |
| 编号与轴 | 模型 actuator 按 `joint1..joint20`；协议数组按档案列表顺序，`joint_index=i+1`。仓库20关节档案 ID 为1–20，奇数 Yaw、偶数 Pitch，头段/每段64.5 mm。 | 现场从头到尾的机械顺序、ID、安装轴向及当前加载档案必须一致；编号相同不证明校准相同。 |
| 符号与单位 | 仿真桥与实机入口采用同一约定：奇数 `a_protocol=degrees(q_sim)`，偶数为负。控制器角为rad，协议角为deg，长度为mm；原始位置为4096计数/圈。 | 桥的符号是坐标约定；物理方向由独立 `Direction` 校准。不要再把这两个符号重复翻转。现场正向/零位是否匹配未确认。 |
| 零位与范围 | 档案逐节有独立 `Zero`，全部 `Direction=+1`，协议范围±90°。原始转换见下式，整帧越界拒绝；单关节旧入口会裁剪。 | 这些零位只是已保存档案。现场是否改装/重标定未知；本次不标定、不改限位，也不把仿真88°筛查变成实机已确认边界。 |
| 时序 | 实机串口线程顺序读取20节，再处理控制/整帧发送，每轮目标100 ms；耗时超过100 ms时不补偿。Python后台默认10 Hz、只有最新待发槽，EXE也只有最新待发槽。 | 41行前缀是20 Hz，旧帧可能被替换；当前入口还是单步动作、下发确认最多等2 s再等1 s。尚未有保留50 ms时标的执行入口，不能删帧/放慢后称同一候选。 |
| 目标历史 | 成功整帧串口写入后更新 `targets_deg` 和 `dispatched_command_id`；该确认表示软件完成写入，不表示电机到位。启动时 `TargetDegrees=0`。 | `targets_deg`不是读取舵机目标寄存器。须结合本次写入记录/命令确认确认现场夹紧目标来源；已有外部目标或启动默认零不能当成预载。 |
| 使能与接管 | “安全使能”先把在线节实测原始位置写成首目标，再设限值/使能。此分支没有同步更新 `TargetDegrees`；`Ctrl+U/apply` 在Python没有已有目标时令内部 `targets=实测角`。 | 重新使能会改目标，不能用于保留现有夹紧预载；`apply`虽不立即发角，其后续内部参考也不等于已执行夹紧目标。资料采集不按这些键。 |
| 角度/有效性 | `geometry`含角度、在线标记、逐节采样时间、帧号、配置和反馈年龄；Python超过0.5 s没有新反馈返回None，出指令前检查在线/反馈来源和时效。 | 20节数据是串行采集，不是同一瞬间实测；反馈丢失拒绝新目标，不保证停止电机执行最后目标。 |
| 负载/电流 | 状态块带符号 `load_permille`、`current_raw`，以及温度、电压、状态、moving、逐节状态采样时间。每5轮尝试状态块，累计3个失败周期后本次连接停用状态块。 | 负载按仓库定义是相对输出指标，不可换算N·m；电流原始值未标定，不能换成A或仿真 `qfrc_actuator`。字段存在不证明现场有新样本。 |
| 限制与停止 | Python检查档案角范围和每步最大30°跳变；整帧检查有限数、配置、在线、目标有效期20–2000 ms（客户端默认500 ms）及单一写入所有者。Python暂停保留最后目标；EXE紧急释放清待发并排队失能。 | 30°/步不是10°/s。协议速度是原始计数/s，UI默认800、转矩上限300、加速度20只是程序默认，不是批准设置/安全限值。入口没有负载、跟踪差或温度阈值自动急停；现场阈值、响应时间未确认。 |

J1–J20各项见 [最小对应表](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/hardware_preparation/joint_interface.csv>)，每行 `field_mapping_confirmed=NOT_CONFIRMED`。仿真局部轴与EXE Yaw/Pitch轴处于不同基，表内并列不代表直接比较轴向量。`COM9/115200`只是仓库档案值，不作自动连接依据。

校准实现：`a_protocol = Direction × shortest_mod4096(raw − Zero) × 360/4096`；`raw_target = (Zero + round(a_protocol/Direction × 4096/360)) mod4096`。无需将零位加入增量；现场下发前必须先核对当前档案与身体坐标映射。原始位置量化约0.08789°/计数，并不代表机械精度。UI速度默认800计数/s按代码比例为70.3125°/s，不等于仿真前缀的8.19°/s目标速率，也不证明真实跟踪速度。

对应源码/说明：

- [关节映射与协议配置](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/Communication.cs:326>)、[20关节档案](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/profiles/20关节实机档案.json>)、[校准转换](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/Kinematics.cs:149>)、[仿真桥符号](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/mujoco_visualizer_bridge.py:23>)、[实机入口转换](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/codes/Adaptive_Pole_Climbing_Real.py:83>)。
- [实机轮询与状态退化](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/RuntimeEngine.cs:447>)、[10Hz最新槽](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/python_client.py:145>)、[单步检查/等待](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/codes/Adaptive_Pole_Climbing_Real.py:221>)、[整帧接收及发送](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/Communication.cs:210>)。
- [启动目标零值](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/RuntimeEngine.cs:136>)、[使能分支](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/RuntimeEngine.cs:686>)、[安全使能实际写入](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/FeetechProtocol.cs:342>)、[apply参考](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/codes/Adaptive_Pole_Climbing_Real.py:383>)。
- [反馈字段和遥测时间](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/Communication.cs:156>)、[状态块解析](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/FeetechProtocol.cs:182>)、[速度参数及批量写入](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/FeetechProtocol.cs:312>)、[UI默认限制](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/MainFormV2.cs:532>)、[状态块协议说明](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/COMMUNICATION_V3.md:38>)。
- [暂停和退出](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/codes/Adaptive_Pole_Climbing_Real.py:302>)、[紧急释放UI](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/MainFormV2.cs:1125>)、[释放队列](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/RuntimeEngine.cs:374>)、[运行停止/串口退出](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/RuntimeEngine.cs:182>)。

## 3. 记录入口能留下什么

**EXE“开始记录/停止记录”**：写入正在运行EXE目录下的 `experiments/日期_实验名/metadata.json` 和 `frames.csv`。metadata保存档案；每个UI收到的新帧保存各节原始位置、实际角、软件目标角、在线状态和时间。UI按30 ms刷新，记录是它实际取到的帧，不保证每个串口帧都落盘。**此CSV没有负载/电流及遥测时间**。

**现有Python入口** `Adaptive_Pole_Climbing_Real.py --no-hotkeys`：启动不发目标，每1 s自动写 `validation/results/real-robot/real_时间.csv`，包含墙钟时间/反馈序号、实际角/软件目标角、带符号负载、电流原始值、温度、电压。只在操作者已有EXE实机读取条件下使用；本次未启动。采集期间只用 `status` 和结束时 `quit`，不使用 `step/back/apply/hold/reset`、几何调整或顺应切换。

**记录缺口**：Python CSV不保存逐节状态采样时间、online、命令确认/配置；不能分辨旧负载缓存与新样本。缺失遥测可能为空，某节从未读到时协议以0填值且采样时间0，**不能当成零负载**。EXE角记录与Python每秒状态记录互补，仍不足以重建2 s动作每次50 ms下发和即时负载过程。既有 `python_client.py geometry/state`只读查询可由现场操作者在采集前/后保存完整快照，核对配置、时效、命令确认与扭矩状态；快照不补齐期间全部遥测。若可用状态更新无法确认，停在采集，不用缓存字段构造闭环停止。

出处：[EXE开始记录](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/MainFormV2.cs:1552>)、[CSV字段/写入](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/ExperimentData.cs:90>)、[UI取帧记录](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/windows_app/MainFormV2.cs:827>)、[Python遥测日志](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/codes/Adaptive_Pole_Climbing_Real.py:117>)、[只读查询入口](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/visualizer/python_client.py:251>)。

## 4. 冻结动作的离线资料

[simulation_delta_deg.csv](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/hardware_preparation/simulation_delta_deg.csv>)直接取已冻结 `dispatch_time_s/dispatch_ctrl_rad`，41行、0–2 s，保留每个原时刻。列 `delta_sim_J1_deg..delta_sim_J20_deg` 仅为仿真坐标下相对首帧的度数增量：

`Δq_sim(t) = degrees(ctrl_saved(t) − ctrl_saved(0))`。

首行全零，最大绝对增量8.735372°（J6）；保存后重新读取CSV与原时间/增量逐项完全相同。未导出仿真绝对ctrl、根位姿、实际qpos或电机原始目标。生成过程只读取保存数组/档案/XML，无求解和动力学；[导出说明](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/hardware_preparation/offline_material.json>)记录来源与未准入状态，[离线导出代码](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/hardware_preparation/export_offline.py>)仅负责这两份表。

只有现场形状和预载相容、映射核对完成、时序与限值已确定时，才有资格评估“**现场已执行夹紧目标 + 映射后的固定起点增量**”这一接入关系。零增量应逐节返回现场原目标，不能用现场实测角替换该目标，不能把跟踪误差循环叠加。本式不是任意初态的运动保证，仍需检查运动余量与承载；当前没有生成这种硬件目标，也没有播放入口。

仿真是水平两管L型、管直段半径56.5 mm、模块外壳半径29 mm、段长64.5 mm。现场管径/接头形状、安装方向/重力、初形、接触支撑、预载、速度/转矩/加速度、线缆受力等不匹配时，仿真几何/驱动准入不继承。仿真摩擦/2.355 N·m饱和不能换成实机档位。[原模型](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml:16>)仅供核对。

## 5. 会改变下一步判断的现场信息

1. **映射**：实际20节头尾顺序、Servo ID、安装轴、正负方向、有效零位；实际EXE版本与加载档案。表内仓库值必须标为现场已核对或未核对。
2. **起点与预载**：同一时段20节实测角、已下发并持续执行的夹紧目标及其来源/确认历史、扭矩状态/现有速度转矩加速度设置；是否刚使能/重启、是否由外部程序写过。目标来源未知就不能搭增量。
3. **真实受力字段与停止**：各节在线/状态更新时间，负载和电流是否实际更新、缺失/停用情况；现场认可的停止阈值和人工停止方式。这里不编造N·m/A换算或安全数值。
4. **几何与身体观察**：两管外径、L接头几何、水平姿态；能辨认身体单元、接头和管轴的固定视角录像，管上尺度/定位标记、时间同步事件、是否有人/线缆托持。若约1 mm响应低于录像可分辨范围，应记“无法验证小位移”，不能用编码器替代。
5. **执行时序**：现场实际采样率、有效目标发送时刻/替换情况，以及50 ms前缀是否有经审阅的保留时标执行入口。当前源码对此不满足，不能靠速度框或人工连按推定。

最短操作顺序见 [一页现场说明](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/experiments/hook_pull_review_20261007/hardware_preparation/现场操作一页.md>)。资料充分后才讨论原目标12 s保持、小幅动作2 s加停后10 s；两次实机起点不可恢复同一完整积分状态，分别记录差异，不称严格同初态对照。角度/负载用于执行和受力核对；身体相对管道的位移、滑落、自转和按序迁移由外部观察判断。

本次交付到此停止。上述记录缺口与起点未知是未就绪依据，不是要求重构EXE/驱动或创建新控制框架；后续由审阅会话依据具体现场资料决定。
