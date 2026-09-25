> 2026-09-25 目录整理：可视化源码和客户端已纳入本仓库 `visualizer/`；完整启动、开关和验证入口以[主 README](../README.md)为准。下文旧外部目录为历史路径。

# 阶段三：基于实测截面的顺应性目标生成

实现位于 `pole_compliance.py`，由 `Adaptive_Pole_Climbing.py` 调用。依据 Takemori 等的 *Adaptive Helical Rolling of a Snake Robot to a Straight Pipe With Irregular Cross-Sectional Shape*，IEEE T-RO 39(1), 2023，III-B5—B7、式 (23)—(33)。本地原文位于 `D:\Desktop\SNAKE\顺应性运动\2023【 Adaptive Helical Rolling of a Snake Robot to a Straight Pipe with Irregular Cross-Sectional Shape.pdf`，印刷页 444—445。

## 运行和开关

先启动阶段二的 v1.3.0 可视化 EXE，再运行原来的 `Adaptive_Pole_Climbing.py`。顶部参数为：

```python
COMPLIANCE_ENABLED = True
COMPLIANCE_GAIN = 0.04
HELIX_RADIUS_MM = 110.0
HELIX_PITCH_MM = 180.0
COMPLIANCE_INSIDE = False
```

- `COMPLIANCE_ENABLED=False` 恢复阶段一的标准 Pole/Yamada 输出，双向通信和 DEBUG 窗口仍可使用。
- `COMPLIANCE_GAIN` 是各关节的 K_i，可为标量或长度等于关节数的列表。默认统一 0.04 是初始调试值；论文 28 关节实验采用首尾四关节 0.04、中间 0.08，并不意味着这组数值已适合当前机器人。
- 设计螺距角由初始几何统一确定：`alpha=atan2(P, 2*pi*R)`，在 F7 应用参数时更新，不从局部反馈反推操作参数。
- `COMPLIANCE_INSIDE=False` 使用管外公式，True 使用管内公式；管内要求所有 K_i 小于 1。修改这些参数后重启脚本。
- F7 应用标准螺旋；F5/F6 用实测反馈、K_i 和 α 生成顺应性形状，每次增减 `ROLL_STEP_RAD`。首次有效截面出现前，逐步形成标准螺旋；接管顺应性后无有效反馈不推进这一步；等待时 MuJoCo 物理时间与滚动相位一起暂停。

## 数据和计算

1. 读取阶段二同一份几何反馈。核心形状数据为按身体顺序排列的圆弧半径、带符号圆心角、直线长度和所属关节。机械长度用于归一化，已有轴/投影基用于三维坐标表示。没有增加扭矩读取或触发阈值。
2. 对关节 i 的圆弧施加式 (23)(24)：管外 factor=1+K_i、管内 factor=1-K_i，r_h=factor*r_p，phi_h=phi_p/factor。直线保持原长；变形前后圆弧长度 r*abs(phi) 不变。每次从实测截面出发，不重复变形上一轮生成的目标。
3. 以固定 α 将直线提升为三维直线、圆弧提升为圆螺旋。按同一关节的线/弧组合归一化；首尾端段分别归一化为对应端部连杆的一半。保留 A/B 情况中线和弧的原始先后顺序。
4. 计算分段常值 κ/τ，直线二者为零。用分段解析积分计算 Yamada 目标角，每轮使用同一骨架与滚动相位，再整组写入 MuJoCo。解析积分与 `scipy.integrate.quad` 独立核验一致。
5. 回传给 EXE 的仍是目标角预览。`GeometryExchange.publish_targets(..., source_geometry=feedback)` 显式保留用于计算的反馈帧号，避免把目标误关联到刚收到的新帧。未新增通信消息类型。

可视化端的 `python_client.py` 增加了可选 `source_geometry` 参数；请使用 `Paper Code\two_joint_visualizer` 中当前源码客户端。v1.3.0 的 EXE 协议和二进制无需更新。

## 与论文等长模型、符号的对应

当前机器人首尾段及内部连杆不完全等长。实现把关节 i 对应模块的三维长度设为相邻两连杆长度之和的一半，首尾各半段，保证总长度精确等于实际连杆长度之和；Yamada 积分覆盖关节前后两根连杆。等长时还原论文的 l 和 [(i-1)l,(i+1)l]。这是对非等长模型的工程推广，并非论文直接验证过的非等长结论。

协议的轴正向指向头，曲线按头到尾遍历，因此正 α 对应 z 随弧长减小。圆弧转角保留协议符号，得到 κ=cos²(α)/r、τ=-sign(phi)*sin(α)*cos(α)/r。用三维曲线数值求导核对该符号。

标准螺旋和重建曲线现在统一使用 `ψ(s)=∫τds+ψ_roll`，并按实际连杆长度积分。首次进入顺应性或运行配置更换时，以当前目标角及上一已执行相位拟合偏置，再计算本次相位的目标；这避免首次滚动增量被拟合抵消，但不保证形状切换严格无跳变。

## 失效行为与结果访问

首次收到可用截面前使用标准运动；进入顺应性后，如反馈超过 500 ms、轴无效、配置不匹配、关节离线、倒序或目标越界，则保持上一组关节目标并暂停相位推进，停止提交新的顺应性预览。反馈恢复后继续。不采用逐关节静默限幅。这里的保持是 MuJoCo/目标生成行为，不是已验证的实机失联策略。

`compliance.backbone` 保存三维目标的分段参数，`.sample(step_mm=5, world=True)` 可取得世界坐标曲线点；`.curvature_torsion(s)` 可查询弧长处的 κ/τ。主控制循环无需先生成大量点再数值差分。现有 DEBUG 窗口继续显示实测反馈，避免混淆实测曲线和生成目标。

设置已有 `SNAKE_MUJOCO_TRACE` 时，记录新增 `compliance_active`、`target_kappa`、`target_tau`（各关节弧长处）；旧 `kappa/tau` 列保留为初始标准步态参数，不能当成当前顺应性曲率。

## 验证

运行 `python validation/adaptive_pole_stage3.py`，证据位于 `validation/results/adaptive-pole-stage3/`。

- 录制的 28 关节、56 段截面：逐模块及总长度归一化，管内/管外变形与弧长保持。
- Yamada 解析积分与独立 quad 对照、整周周期性、半周反相，以及正反螺距、正反弯曲方向、平面退化。
- 从重建三维点数值求导验证 κ/τ；检查世界变换保持距离。
- 同一反馈帧不重复变形；过期、配置不匹配、无效、倒序、越界拒绝及恢复；预览来源帧绑定。
- 既有阶段一 120 组角度对照回归；隔离端口 EXE + 20 关节 MuJoCo 实际闭环调用新算法。

验证范围是目标生成算法和短程模拟联调，不包含稳定攀爬、摩擦接触充分性、实际舵机柔顺性或实机下发。论文实验使用带电流限制且有位置柔性的舵机控制，不能仅凭几何算法验证就认定当前实机会产生同等接触行为；硬件下发属于第四阶段。

2026-09-24 Windows 原生验证结果：56 段目标总长 1918 mm；Yamada 与 quad 最大差 `2.22e-16 rad`；重建曲线求导的 κ/τ 最大差分别为 `5.64e-11 / 4.57e-7 mm^-1`。最后一轮 8 秒墙钟联调包含 523 个仿真记录，其中 443 个使用顺应性目标，最大目标角约 1.326 rad，数值均有限；启动时一次 no_frame 后恢复。阶段一 120 组原目标角回归通过，DEBUG 表格/绘图/关闭三种模式回归通过。详细值以结果目录 JSON 为准。
