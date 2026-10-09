# 内侧反勾I01复现

I01是从人工预勾、零速度初态开始的一次整身跨管仿真成功。重复性尚未用另一轮动力学验证；本任务在成功后停止，复现入口仅做了准备检查，没有再运行物理步。

在仓库根目录使用原环境，创建尚不存在的输出目录：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_inner_hook_transfer.py --output-dir experiments/inner_hook_reproduction_new
```

入口复制I01冻结Python/XML/STL，并使用该例精确保存的 `initial_pose.npz`。无需重新优化起点，也不会加载CLOSE1B或U04作为动力学初态。U04初态JSON只用于内外侧对照图。实际运行输出位于新目录的 `experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009/I01/`；保持9前部、5内侧桥、7源段，shift774毫米、速度5毫米/秒、指令速率20°/秒、实际时长上限420秒、墙钟默认600秒。

只准备冻结工作目录、不启动物理步：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_inner_hook_transfer.py --output-dir experiments/inner_hook_prepare_new --prepare-only
```

本任务已做此检查，结果为 `REPRODUCE_READY/reproduction_entry.json`，没有动力学复验。完整复现依赖本地保存的NPZ与STL，不能仅凭Git里的JSON替换另一初态。

只读复核已有实际记录：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/review_inner_hook_transfer.py I01
```

该步骤不调用 `mj_step`；检查冻结模型FK、最后1秒严格21杆形状、原管清空、后部承载、Kp、真实入口方向、原物理一致性及世界参考误差。轨迹100Hz、命令20Hz，原物理步1ms。

优先回看 `I01/replay_1x.mp4`。方向看 `I01/finite_pipe_top_view.png` 和 `I01/inner_entry_top_side_comparison.png`；完整名义路线看 `I01/complete_reference_curve.png`；名义初中末身体窗口看 `I01/reference_initial_middle_final.png`，实际窗口看 `I01/actual_and_executed_reference.png`。这些参考图必须连同474.435毫米的末态世界执行参考RMS阅读，不能把几何成功称为精确空间跟踪。

原命令用于审计；其中截止时间已过，正式复现应使用上方会自动设定新执行截止时间的冻结入口：

```text
python validation/inner_hook_transfer_experiment.py --name I01 --front 9 --bridge 5 --speed .005 --rate 20 --seconds 420 --max-wall 600 --deadline 2026-10-09T07:36:07+00:00
```

若需要重新构造候选几何，应从 `InnerHookRoute.window(0)` 经 `WholeCurveMapper.initialize` 创建完整真实关节姿态，再核对源侧+Y、内側实际过渡、目标−X向上入口。这属于另一个初始化过程，不能替代本次精确初态的复现。
