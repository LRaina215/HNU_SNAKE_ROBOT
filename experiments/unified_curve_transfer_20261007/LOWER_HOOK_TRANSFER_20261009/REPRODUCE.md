# 最短复现入口

本批U01–U04均未完成整身跨管。复现是从同一个人工下方反勾初态开始的新动力学试验，不是已验证的成功方法；视频回看不需要重新仿真。本任务只做了4次正式动力学，没有为了复现再运行一次。

在仓库根目录使用原环境：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_lower_hook_trial.py --case U04 --output-dir experiments/lower_hook_reproduction_new
```

`--output-dir`必须尚不存在。入口把该试验的冻结Python/XML/STL复制到新工作目录，放入精确CLOSE1B父积分状态及一份仅含父末态qpos/qvel/ctrl的比较样本，再运行冻结的构造和控制代码。先生成同一个人工下方初态，随后在原重力/摩擦/驱动下连续驱动20关节；实际时长仍420秒、墙钟默认600秒。输出在新工作目录的 `experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009/U04/`。

只准备和检查复现入口、完全不启动物理步：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_lower_hook_trial.py --case U04 --output-dir experiments/lower_hook_reproduction_prepare_new --prepare-only
```

本任务已执行这一步，结果位于 `REPRODUCE_U04_READY/reproduction_entry.json`，物理步为0；这不代表已经做过第二次动力学复验。需要完整本地材料，包括每例 `initial_saved_state.npz` 和冻结STL；它们未强制加入Git。缺少时必须从本地归档恢复，不能替换成另一姿态。

回看已有U04实际视频：`U04/replay_1x.mp4`。其1339帧来自0–66.90秒的实际qpos采样；失败时悬垂尾部也保留在画面内。查看初态上下方向：`U04/initial_cross_section_vs_H03.png`。查看全路线与计划窗口：`U04/complete_reference_curve.png`、`U04/reference_initial_middle_final.png`，中、末名义窗口标为尚未达到。查看实际初、中、末：`U04/actual_and_executed_reference.png`。

只读审阅现有四次记录，不调用 `mj_step`：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/review_lower_hook_handoff.py U01 U02 U03 U04
```

审阅脚本核对人工初态与首帧、最后1秒真实FK/整身形状、穿透采样、源侧清空、世界执行参考误差、交接参考单步变化和原物理参数。原始轨迹100Hz、命令20Hz，原模型物理步1ms；审阅不能从100Hz记录推得每个1ms步的精确穿透持续时间。

U01–U03保留当时实际用过的计数停滞逻辑，U04冻结代码包含事件进展修正和连续前圈交接；入口不会用当前控制器偷偷覆盖旧试验。原始动作命令如下，截止时间已过，只用于审计，直接复现应使用上面的新工作目录入口：

```text
U01: --lower-entry --reform --posterior-handoff
U02: --lower-entry --reform --posterior-handoff --handoff-exit-count 2
U03: --lower-entry --lower-bridge --reform --posterior-handoff
U04: --lower-entry --lower-bridge --reform --posterior-handoff --continuous-reform-handoff
shared: --speed .005 --rate 20 --morph-length .20 --seconds 420 --max-wall 600 --deadline 2026-10-09T05:21:17+00:00
```
