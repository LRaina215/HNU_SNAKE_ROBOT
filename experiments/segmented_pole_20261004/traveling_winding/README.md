# 同步成形与完整跨越数据

本目录对应2026年10月4日的[实现与验证报告](../../../docs/三段顺应性_同步成形与完整跨越验证.md)。最终候选为 `candidate.json`，水平开发运行 `v9_compliant_ends`，相同参数竖直复核 `v9_vertical_frozen`。

- `archives/`：每次运行的无损ZIP，包含计划、结果、实际命令、全部几何反馈、控制内部量、100 Hz轨迹、初始完整积分状态和运行源码快照。压缩包内保留运行目录名。
- `reviews/`：最终两场景2倍速回放、验收姿态及定量曲线。
- `runs.json`：全部试错索引，明确失败、阶段改善与完整验收结果；反转相位诊断不算第三个完整跨越成功样本。
- `cleanup_equivalence.json`：整理后控制器与两组成功运行共1024条目标的一致性检查，无物理步进或SHA。
- `online_stop_replay.json`：新入口完整周期验收停止点的存档回放检查。
- `direction_check.json`：竖直转移后相位反转诊断，排查单纯重力滑落。
- `middle_yield_diagnosis.json`：对齐前12.8秒的中段阻力对照，区分实际让位效果与参考跟随设置。

本机保留解压工作目录，但Git仅保存无损数据包和必要媒体，避免重复提交大体积JSON和失败视频。各轮失败视频可从归档的完整轨迹重新渲染。

## 解压与检查

在仓库根目录的PowerShell中执行：

```powershell
Expand-Archive -LiteralPath experiments/segmented_pole_20261004/traveling_winding/archives/v9_compliant_ends.zip -DestinationPath experiments/segmented_pole_20261004/traveling_winding
Expand-Archive -LiteralPath experiments/segmented_pole_20261004/traveling_winding/archives/v9_vertical_frozen.zip -DestinationPath experiments/segmented_pole_20261004/traveling_winding
python validation/traveling_winding_audit.py v9_compliant_ends
python validation/traveling_winding_audit.py v9_vertical_frozen
python validation/traveling_winding_checks.py
```

审计使用存档实际位姿与接触，不依赖分段颜色。旧 `assessment.json` 与新的 `whole_crossing_audit.json` 一并保留；完整判据见报告。归档源码用于溯源，复现当前成功候选使用仓库中的入口；不要在解压出的 `sources/` 内直接启动依赖仓库相对路径的历史脚本。
