# 局部负载主动探测与急弯跨越：数据索引

实际执行日期为2026-10-04，目录名沿用10月3日方案。阶段0通过；阶段1训练0/12、冻结复核0/5，按门槛停止阶段2、3。尚未测试负载方向识别，不能据此否定其可行性。

报告：[阶段验证报告](../../docs/局部负载主动探测与急弯跨越/局部负载主动探测与急弯跨越_阶段验证报告.md)。实施代码和复现命令见报告；所有回放均只重算保存姿态，不推进物理时间。

| 文件或目录 | 用途 |
| --- | --- |
| `stage0_round2_clearance.json`、`stage0_round2_clearance/` | 最终有效三相位抱管、正反试探与真实回退 |
| `stage0_independent_audit.json` | 三相位100 Hz独立重算、反馈/力矩/几何核验 |
| `local_bend_geometry_checks.json`、`local_bend_fk_direction.json` | 统一Bishop弯曲场、积分与模型几何检查 |
| `stage1_training_plan.json`、`stage1_training_results.json` | 预声明12候选与全部训练结果 |
| `stage1_parameter_freeze.json` | 冻结参数、源码哈希、预声明5个复核状态 |
| `stage1_training_independent_audit.json`、`stage1_verification_independent_audit.json` | 决策时刻的独立接触归属、FK、力矩及命令审计 |
| `delivery_independent_audit.json` | 有效20次试验的原始命令匹配；另对17次转管的每个100 Hz姿态重算目标接触 |
| `stage1_release_gate.json` | 0/5，明确禁止放行阶段2、3 |
| `stage1_summary.csv`、`stage1_summary.png` | 17次转管的跟踪误差、力矩、接触汇总 |
| `completion_audit.json` | 已满足阶段失败的条件停止要求，不代表全部后续阶段已验证 |
| `delivery_manifest.json` | 逐文件哈希、依赖/资产来源说明与源码索引 |
| `source_snapshots/` | 最终有效源码及部分修复前源码的哈希快照 |
| `stage1/*/trace.npz`、`commands.json`、`control.json` | 全部训练、复核及探索的原始100 Hz轨迹、通信和控制状态 |

代表性回放：[三相位中的相位0抱管/试探](stage0_round2_clearance/phase0/replay.mp4)，[冻结参数对应的训练失败](stage1/f4_w64.5_turn45_p0_training_v1/replay.mp4)。每个视频目录同时包含诊断曲线和首/中/末帧PNG。

共有36次尝试：最终有效阶段0三次、训练12次、复核5次，另16次为排除的工程探索。旧批次包含初态穿透、反馈过期、试探保护遗漏，以及目标接触评分错误；均保留，不混入最终有效结论。`stage0_independent_audit_superseded.json`只记录旧审计，不代表最终通过。`f4_w64.5_turn60_p0_pilot01`的旧评分错误，视频仅供工程排查。

早期部分探索缺完整源码快照，仅有源码哈希及轨迹；最终有效批次的源码可核对。阶段1每次manifest未逐项记录所有依赖/资产版本，交付清单明确说明同解释器和阶段0共享环境来源，不能将交付时环境检查写成当时逐次记录。私有EXE工作副本由本目录`.gitignore`排除，保留仓库EXE及资产哈希。未操作实机。
