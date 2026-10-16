# U03接回全局曲线：阶段A实验材料

**当前阶段A未通过。A01–A15均未完成实际前部持续包绕和整身转移，阶段B未开始。** U03直接双窗成功材料只作参照，未当作曲线版成功或姿态播放。

A15已完成235秒并独立失败：95.5毫米目标半径使M0首次接触72.1→50.7秒，但承载与半圈未重叠，最大有效跨度172.27°时力全0；末源11.72N。90.5毫米只做未运行准备。入口：[结果](A15_OUTCOME.md)、[设计与代价](A15_DESIGN.md)、[接触对照](A15/contact_comparison.json)、[获取时序](A15/acquisition_timing.png)、[完整实际视频](A15/actual_side_oblique_1x.mp4)、[独立判定](A15/independent_outcome.json)。后续固定A15半径和A14 M，诊断首次承载保持与后续材料交接。

A14机制对照已完成235秒：仅改M衰减S5–9，早段320个实际样本与A13完全相同。40–54秒后部倾斜、M10位置误差和固定材料回退有所减轻；前六稳定获取与全21杆转移仍无，末态源分类力16.101N。入口：[完整结果](A14_OUTCOME.md)、[冻结假设](A14_DESIGN.md)、[事件比较](A14/comparison.json)、[机制图](A14/mechanism_comparison.png)、[隔离核对](A14/isolation_audit.json)。不追加M参数网格，B未开始。

A13已完成235秒并经独立核查失败：目标直段最多7杆，前六同时满足位置范围时最大跨度166.472°；没有稳定半圈或21杆转移，末态源分类力6.789N。入口：[完整结果](A13_OUTCOME.md)、[相同判据比较](A13/acquisition_comparison.json)、[独立判定](A13/independent_outcome.json)、[完整实际视频](A13/actual_side_oblique_1x.mp4)。下一步材料在A14_DIAGNOSIS目录，先查50–54秒的初次失接，不把后期饱和当作唯一解释。

A12已结束：同一物理轨迹完成235秒，23.701秒监督指示暂停后从完整检查点恢复，未重跑早段。无触地/自碰；局部原生修正改善部分J11跟踪，却没有前六稳定包绕或21杆转移。64秒锚点切换附近名义命令25毫秒变化6.48°，驱动仍按20°/秒执行；80.9秒后配准锚段含被修改内部关节，优化残差不能独立证明改善。入口：[设计](A12_DESIGN.md)、[A11对照](A12/comparison.json)、[映射诊断](A12/mapping_audit.json)、[切换诊断](A12/anchor_switch_25ms.json)、[实际验收](A12/audit.json)、[完整实际视频](A12/actual_side_oblique_1x.mp4)、[恢复记录](A12/resume_trace_audit.json)。冻结后提交独立复核。

A11已结束：完成235秒，无触地，但滞留于两管之间，前六未稳定包绕、整身未转移。40秒后段世界X误差约105→34毫米；94.9秒实际跟踪差仅.424°，原生及实际前端对齐形状误差却约218毫米。入口：[供料关系与设计](A11_DESIGN.md)、[预检](A11_PREFLIGHT_SUMMARY.json)、[源材料比较](A11_source_comparison.json)、[实际分解](A11_actual_diagnosis.json)、[验收](A11/audit.json)、[完整实际视频](A11/actual_side_oblique_1x.mp4)。全部冻结，待独立核查。

A10已结束并经监督核查不通过：127.401秒触地，M0仅73.501–76.001秒短暂目标接触，前六未稳定包绕。入口：[设计与预检](A10_DESIGN.md)、[完整预检摘要](A10_PREFLIGHT_SUMMARY.json)、[实际分解](A10_actual_diagnosis.json)、[验收](A10/audit.json)、[实际视频](A10/actual_side_oblique_1x.mp4)。原模型/初态/伺服未改，无高层反馈；取点异常未复发，但整体供料/朝向与离散形状误差仍在。

阅读入口：[结果与诊断](RESULTS.md)、[实现链](IMPLEMENTATION.md)、[候选台账](CANDIDATES.md)、[A08两窗设计](A08_DESIGN.md)、[A09连接净空设计](A09_DESIGN.md)。原机器人20关节、原水平L管、原驱动与物理、U03实际初态保持；运行实际位姿/接触只用于记录和失败停止。原XML逐字比较存于各audit.json，没有SHA校验。

最新证据：[A07全局/模板/名义C/实际身体](A07/global_local_actual.png)、[A07实际材料与接触](A07/material_evidence.png)、[A06/A07逐阶段实际比较](A06_A07_actual_comparison.json)、[A06形状与位姿分解](A06_shape_pose_decomposition.json)。各候选目录保留plan.json、冻结源码、model.xml与STL、initial.npz、trace.npz、积分末态及控制器检查点、连续实际视频。A04生成异常时缺失密集trace和末态，只有部分实际视频、图及日志，准确停止时刻未知；没有补跑冒充原记录。

## 当前入口

从仓库根目录运行，Python环境：`D:/Anaconda/envs/mujoco/python.exe`。必须选择新的候选目录名，入口不会覆盖已存在目录。

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' 'experiments/u03_global_curve_compliance_20261012/curve_runner.py' --case A07_replay --moving_family --anchor_frame --bezier_bridge --smooth_source --discrete_transport --axial_family --release_curve --radius 0.09680117714827459 --pitch 0.18306225015502264 --duration 235
```

这是A07参数；A06去掉`--release_curve`。A08使用`--case A08_replay --two_windows --duration 235`，A09再加`--clearance_connector`，A10只用`--case A10_replay --continuous_windows --duration 235`，不带A07其余几何开关。若中央源码以后修改，要复现历史版本，先将对应候选的全部冻结.py文件复制到仓库`experiments/<新实验目录>/`，再从该目录的runner启动；冻结候选目录中的路径计算依赖原中央层级，不直接在A07目录里启动。原历史物理模型与U03 initial.npz仍须可读。重跑为新的实验材料，不覆盖旧结果。

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' 'experiments/u03_global_curve_compliance_20261012/analyze_curve.py' --case A07
& 'D:/Anaconda/envs/mujoco/python.exe' 'experiments/u03_global_curve_compliance_20261012/render_curve.py' --case A07
```

analyze读取真实trace；render使用已保存0.1秒真实qpos生成1倍速侧视/斜视连续视频，不重跑动力学。原2倍速视频是运行时直接渲染。video_metadata.json记录帧率、时长和首/中/末帧读回。

## 下一步边界

两窗A08在38.9秒首次M0获取后，到45秒丢失；存在限速、受载跟踪及名义连接侵管问题。A09几何净空修正后M0在45秒仍有接触，却未形成前六杆持续包绕；47.876秒径向投影连接紧折导致参考生成异常。完整材料与[A09异常/获取诊断](A09_exception_diagnosis.json)保存。0.5秒名义取点/中点净空检查漏掉实际0.025秒命令时刻的折叠，不能把粗采样通过当全时域可执行。下一步须改变连接构造，不能只扩交点搜索范围或调整伺服；当前提交定位供监督复核，不继续盲试。阶段A通过独立核查前不接入局部顺应。

A10完成了新的极坐标构造与密集整杆检查，实际仍未接管。40秒源侧后段相对名义前移104.93毫米、极角偏差−36.86°，实际关节最大跟踪差3.20°；名义曲线净空和限速通过不足以说明实际姿态/供料对应。解卷期间材料站位与滚转带来的供料是否匹配，是待核查推测，不能先当唯一原因。提交完整材料，不继续同构造参数网格，B待A实际通过后派发。

A11已核清A10头站位锁在缩短连接末端、尾站位随之回退的名义关系，实际供料不由这项站位补偿约束。一个协调映射F/M/ψ修正了早期倒退，但后续曲线到原生关节的形状误差及受载滞留仍未解决。重现A11冻结构造使用`--case A11_replay --supply_windows --duration 235`；不会覆盖旧材料。当前提交完整实际结果供监督决定下一步，B仍需A真实通过后派发。


阶段A当前结论：A16已独立通过并冻结为B0，见A16_OUTCOME.md。阶段B设计/登记：B_DESIGN.md、B_preregistered.json；入口b_runner.py、local_curve_compliance.py，正式名义运行B1_NOM_V2（原MuJoCo3.2.7环境）。调试B1_NOM因观测forward改变物理缓存提前停止，排除贡献对比。所有后续结果汇总B_comparison.json/CSV及B_RESULTS.md。

推荐入口：`./experiments/u03_global_curve_compliance_20261012/run_B.ps1 -Case B1_fresh -Compliance`。无反馈版省去`-Compliance`；登记外扰加`-PulseX 3`或`-PulseX -3`。脚本固定原解释器和1线程，Case必须新命名。B1_NOM_V1跨版本探索结果不能计入反馈贡献主表。

当前阶段已按用户要求暂停。正式名义对照与结论见B_RESULTS.md；PUSH/PULL四次登记运行未启动。B1_NOM_V2是正式样本，B1_NOM与B1_NOM_V1排除主对照。

2026-10-14恢复执行已完成：源侧定位见B_SOURCE_COST_20261014/ANALYSIS.md，原登记四次扰动配对见B_HELDOUT_RESULTS_20261014.md及B_HELDOUT_comparison_20261014.json/CSV。两版扰动均2/2成功，有限局部收益伴随源侧侵入代价；当前按用户要求再次暂停。
