# 2026-10-05 平顺转移首轮数据

结论与限制见 [验证报告](../../../docs/分段顺应性跨管/三段顺应性_平顺转移首轮验证.md)。保留候选为8°目标领先、20 ms插值、反馈减速关闭；水平和竖直存档均通过原完整跨越门槛，支撑释放瞬态仍未解决。使用已知管轴和接触信息，不代表自主或实机跨越。

- `runs.json`：全部六次试验的假设、参数、结果、完整验收及过程指标。
- `archives/<结果名>.zip`：含结果目录前缀的无损原始数据；解压到本目录，即恢复该次运行。包含plan/result/assessment、100 Hz轨迹、命令、控制量、几何反馈、初始状态及源码快照。
- `reviews/`：正常速度视频、峰值时刻图片、结束图片和过程曲线。颜色是控制归属，不是实际承载证据；成功视频到验收窗口结束，完整轨迹到脚本停止。
- `baseline_horizontal.json` / `baseline_vertical.json`：旧版v9原轨迹按相同统计规则重算的过程指标。
- `extension_check.json`：同参数延长组前160秒与先前运行逐数组一致的核对结果。

| 结果名 | 改动 | 结果 |
|---|---|---|
| lead8_horizontal | 8°领先限制，原0.2 s阶梯执行 | 完整通过，峰值动作仍大 |
| ramp20_horizontal | 再加0.02 s插值 | 完整通过，保留候选 |
| ramp20_vertical | 上述参数冻结，换竖直存档 | 完整通过，仍有尾部外张 |
| feedback_horizontal | 再按误差共同减慢相位/成形 | 160 s未完成，不判为永久失败 |
| feedback_extended_horizontal | 上述参数不变，延长时间 | 218.05 s完成验收，未设为默认 |
| lead2_ramp20_horizontal | 领先限制改2° | 38.2 s失去支撑 |

在仓库根目录解压单次数据并重绘（无需物理步进）：

```powershell
python -m zipfile -e experiments/segmented_pole_20261005/coordinated_winding/archives/ramp20_horizontal.zip experiments/segmented_pole_20261005/coordinated_winding
python validation/coordinated_winding_review.py experiments/segmented_pole_20261005/coordinated_winding/ramp20_horizontal --video
```

源码快照记录每轮实际运行版本。后续入口仅调整了默认插值周期、视频入口和删除未使用的配置字段；各轮实际参数以plan/configuration为准。原始场景和存档沿用 `experiments/segmented_pole_20261004/inputs/`。`coordinated_winding_archive.py` 核对命令插值及共同轨迹前缀后生成归档索引，不运行物理试验，不做SHA检查。
