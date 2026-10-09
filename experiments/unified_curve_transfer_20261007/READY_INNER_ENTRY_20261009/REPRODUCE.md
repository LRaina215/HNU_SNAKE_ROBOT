# 冻结版本准备与复现

在仓库根目录执行：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_ready_inner_entry.py --case R03 --out tmp/R03_frozen_ready
```

默认复制该次`sources/`及相同人工原管初态，前向运动学检查后保存`reproduction_ready.json`，物理步数为0。`tmp/R03R/`已成功准备，全21杆仍在原管附近。输出目录必须尚不存在，使用短路径可避免Windows长路径限制。

`--case`支持R01、R02、R03，每次使用自己的冻结控制版本；不会用当前源码替换历史实验。R01起始构造虽然调用通用WholeCurveMapper，其内部`fit.ready`使用旧人工目标钩的准入规则，不适用于全源管；本轮`initial_geometry.ready`依据全21杆源管半径、头方向、零目标接触和侵入限值独立判定。

添加`--run`会启动新的正式动力学，已耗尽的R01–R03额度不包含该运行。本轮没有执行此选项。准备时保留同一初始qpos、qvel、ctrl；运行参数见各次`plan.json`。模型网格资产、NPZ和完整视频保存在本地，跨机器复现需同时携带初态NPZ及模型网格资产。

实际视频为各次`replay_1x.mp4`，从人工全身原管起点连续到停止点；不包含被拒绝的候选，也不拼接I01。视频0／中间／末帧均已读回。
