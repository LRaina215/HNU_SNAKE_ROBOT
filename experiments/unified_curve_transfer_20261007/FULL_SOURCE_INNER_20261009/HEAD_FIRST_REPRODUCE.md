# F06/F07冻结版本复现

在仓库根目录使用MuJoCo环境运行：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_head_first.py --case F07 --out tmp/F07_frozen_ready
```

默认仅复制该次冻结源码、模型和相同人工原管初态，执行运动学前向计算，物理步数为0。`tmp/F07R/reproduction_ready.json`已记录一次成功准备，全21杆仍在原管附近。

F06和F07分别使用自己的`sources/`，不会拿当前后续草稿替换原实验代码。输出目录须尚不存在，宜采用较短路径以避免Windows长路径限制。网格资产、原始轨迹和视频留在本机；在其他机器复现需同时携带本地模型资产。

`--run`会启动一条新的正式动力学，已经用完的F06–F07额度不包含它。本轮没有执行该选项。原运行入口和参数保存在各次`plan.json`及冻结`validation/head_first_experiment.py`。当前工作源码包含未运行的返回值修正与可选运动学连续初始化，不应据此声称F07已执行送身。

完整实际1×视频：`F06/replay_1x.mp4`（220.05秒）、`F07/replay_1x.mp4`（111.70秒）。重绘视频只读取同一保存qpos，不重跑动力学；更正了旧渲染器遗留的人工钩头标签，其代码另存`review_sources/`，运行冻结源码保持原样。
