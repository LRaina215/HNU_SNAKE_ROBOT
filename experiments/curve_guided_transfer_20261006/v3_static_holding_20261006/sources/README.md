# 源码及输入范围

本目录中`validation/v3_static_holding_experiment.py`及其运行依赖、原模型XML/两份外壳STL、EXE Python客户端和无实机档案的测试配置，均在本次首次物理步之前复制；它们对应唯一的V3_STATIC_C0实际运行。EXE复用仓库`visualizer/windows_app/bin/SnakeRobotVisualizer.exe`，私有隐藏副本只连接MuJoCo转发，实际返回配置保存在`../V3_STATIC_C0/config.json`。

`../inputs/`是首次步进前保存的V3参考、几何结果及上一轮说明，不改写原`structural_revision_v3/`资料。`../initial_pose.npz`来自其中唯一的d=0记录，未经再求解。

`../analysis_sources/validation/v3_static_holding_diagnostics.py`是运行结束后的分析代码；不属于运行前控制源码，不调用`mj_step`。失败调用在控制器返回前抛异常，`../rejected_control_reconstruction.json`明确标注离线重建，与原始成功控制日志分开。

维护入口为仓库`validation/v3_static_holding_experiment.py`，`--check-only`重建保存初态，`--run`执行唯一保持试验。当前记录和3次台账已存在，入口会拒绝再次运行；不能删除、改名或改台账来重置预算。完整路径入口及原NOT_READY状态保留。任何新的动力学工作需要新的用户执行范围。

无SHA校验。实际原始记录、视频与离线图的类型见上级结果报告。
