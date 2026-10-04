# 搭管后动态分段：2026-10-04

本轮八组有效试验均未通过“短程转移＋3秒支撑保持”的完整门槛。

说明与复现：[阶段报告](../../docs/搭管后动态分段顺应性_首轮验证.md)。

- `inputs/`：用户水平step92、竖直step95完整存档（MuJoCo 3.2.7）。
- `plan.json`：测试门槛、有限候选与探索数据排除原因。
- `freeze.json`：水平候选结果、源码哈希及竖直复核的冻结选择。
- `geometry_checks.json`：独立积分、重建和边界规则检查；`geometry_sample.json`为同一命令弯曲场的重建示例。
- `horizontal/final_*`：两基线与4/6/8节三个分段候选；`vertical/{original,source_axis,segmented}`为冻结复核。
- 有效组内：`trace.npz`（100 Hz真实物理状态）、`commands.json`（实际转发）、`controls.json`、`geometry.json`、`result.json`、`audit.json`、`sources/`（运行时源码）、`replay.mp4`和首尾图。
- `summary.json`：最终指标及额外诊断；场景级`comparison.png`显示头部位移、区域模块数和跟踪误差。
- `horizontal/segmented_w4`、`v1_*`、`v2_*`及相应日志：评分/恢复/提前停止检查阶段的探索数据，保留但不作有效对照。最早的`segmented_w4`无完整源码快照，不作可复现结论。

源码快照和原始输入按字节保存（本目录`.gitattributes`禁用换行转换）。原GUI内部跟踪历史仅恢复最后有效轴，控制步频的估计器复演并非原异步会话逐帧复现。两基线与分段版本均使用隔离EXE转发实际控制目标。

示例回放：

- [水平6节过渡](horizontal/final_segmented_w6/replay.mp4)
- [竖直冻结4节过渡](vertical/segmented/replay.mp4)
- [水平比较图](horizontal/comparison.png)
- [竖直比较图](vertical/comparison.png)
