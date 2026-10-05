# 公共控制与辅助模块

步态启动程序保留在 `../codes/`。本目录中的模块由步态或验证工具导入，不是独立的运动入口；从仓库根目录可用 `from snake_control.pole_compliance import ComplianceController` 等形式引用。

| 内容 | 模块 |
| --- | --- |
| 原截面顺应性与目标曲线 | `pole_compliance.py` |
| 通信、MuJoCo转发及存档 | `mujoco_visualizer_bridge.py`、`mujoco_execution_bridge.py`、`mujoco_checkpoint.py` |
| 调试显示与按键 | `adaptive_debug.py`、`geometry_debug_view.py` |
| 可选局部螺距功能及历史版本 | `local_pitch_*.py` |
| 弯曲轴与局部转弯 | `curved_axis_compliance.py`、`local_bend_backbone.py`、`contact_turn_controller.py` |
| 分段跨管、形状估计及协调原型 | 其余分段、包绕、窗口、编码角估计模块 |

2026-10-06：分段顺应性跨管方案已终止；局部螺距研究方向也已结束。相关代码保留为历史工作，不代表继续实验的待办。见[文档索引](../docs/README.md)与[跨管终止记录](../docs/分段顺应性跨管/顺应性跨管_当前实现终止结论.md)。

`experiments/` 中冻结的 `sources/codes/`、ZIP和原始结果仍使用当时的目录结构，不随当前源码迁移；历史复现需使用对应快照，不能将当前代码与冻结结果混用。
