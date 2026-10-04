# 窗口协调有界验证（2026-10-05）

**第1步未通过：六组中两组完整跨越，零组同时满足预声明过程质量。** 水平开发未过门槛，未运行竖直冻结复核。详见[结论报告](../../../docs/三段顺应性_窗口协调有界验证结论.md)与运行前固定的[PLAN](PLAN.md)。原规划及后续阶段未变。

## 结果与证据

- `runs.json`：假设、参数、实际执行核对、原完整跨越审核、过程质量和失败诊断。
- `archives/*.zip`：六组原始轨迹、控制量、实际执行目标、反馈、初态和源码快照。解压到本目录恢复对应试验文件夹；本机展开目录被gitignore，压缩包入库。
- `reviews/`：正常速度视频、过程与窗口协调曲线。V5/V6完整周期验收结束后才截断视频，原始停止时刻的轨迹仍在ZIP。
- `cmu_page-*.png`：仅本机阅读2025论文窗口图的缓存，未入库。

| 名称 | 改动 | 完整跨越 | 平顺过程 |
|---|---|---|---|
| v1_support_windows | 前后支撑反馈分速 | 未通过 | 未通过 |
| v2_bounded_phase | 限制前部领先相位，后部至少1/4供给 | 未通过 | 未通过 |
| v3_formation_lead4 | 提前成形范围改4关节 | 未通过 | 未通过 |
| v4_supported_shape | 已搭管前部恢复量改0.35 | 未通过 | 未通过 |
| v5_shared_phase | 前后共同滚动速率 | 通过 | 未通过；尾端1.237m/s |
| v6_continuous_middle | 中段参考跟随时间改0.2秒 | 通过 | 未通过；尾端1.161m/s、饱和更高 |

## 复现

在仓库根目录、安装MuJoCo3.2.7及既有依赖的环境中运行。自动恢复归档的水平搭管存档，启动隔离EXE执行端，不需要重新手动搭管。结果名使用新名称，避免覆盖已有证据。

```powershell
python validation/window_coordination_experiment.py v5_recheck --tail-mode same --maximum-lead 4 --supported-recovery .35 --seconds 180 --hypothesis "复核共同速率窗口候选"
python validation/window_coordination_experiment.py v6_recheck --tail-mode same --maximum-lead 4 --supported-recovery .35 --middle-follow .2 --seconds 180 --hypothesis "复核中段连续让位对照"
```

V1为默认参数、120秒；V2使用`--tail-mode coupled --seconds 180`；V3再加`--maximum-lead 4`；V4再加`--supported-recovery .35`。每组完整参数以`runs.json`和ZIP内`plan.json`为准，源码历史保存在各自`sources/`。

只重绘已保存数据，不重新运行物理实验：

```powershell
python validation/coordinated_winding_review.py experiments/segmented_pole_20261005/window_coordination/v5_shared_phase --video
python validation/window_coordination_archive.py
```

第二条检查插值执行、生成窗口曲线、汇总并压缩本目录所有已结束且有`stage1_gate.json`的试验，不做SHA校验。

## 已知限制

当前是已知两管轴、仿真接触标签辅助的独立控制实验。真值参与控制，也用于独立验收；尚非只依靠关节角和电流的自主实现。支撑代理量不是力闭合证明，控制分段颜色不是实际承载证据。20 ms指仿真时间，不能据此保证实机墙钟通信频率。不得将V5/V6功能通过解释为第1步完成。
