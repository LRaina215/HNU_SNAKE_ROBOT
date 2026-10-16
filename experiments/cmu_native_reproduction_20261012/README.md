# 原模型CMU跨L管：已完成的实际仿真

采用用户原始20关节蛇与原始L管XML，未改质量、摩擦、接触、力矩或目标管长度。N09、N10均实际完成整身转移并继续滚动。N10动作较清楚、无自碰；N09保留论文原负号窗相位差。二者均有按现有模型的选位与名义窗口速度适配，不能称作者硬件原样复刻。

主要材料：

- `N10/actual_continuous_1x.mp4`：全235秒实际连续运行的1倍回放；`actual_continuous.mp4`为原运行记录的2倍播放。
- `N10/close_contact_sheet.png`：近景阶段图；`N10/evidence_timeline.png`：真实接触、目标包绕与材料运动。
- `N09/actual_continuous.mp4`：原负号相位候选，全程2倍播放。
- `N09/verified_success.json`、`N10/verified_success.json`：包绕、退出、完整转移、持续滚动、限位与接触证据。
- `trace.npz`：每.1秒的真实qpos/qvel/ctrl、点轨迹、接触力、实际驱动力矩、穿透、名义nt和相位；每候选保存原XML、STL与执行时源码。
- `IMPLEMENTATION.md`：公式对应、原物理条件、失败台账、各项适配及成功边界。

用本机现有MuJoCo环境运行（新case名避免覆盖证据），在仓库根执行：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' 'experiments/cmu_native_reproduction_20261012/native_run.py' --case N10_REPRODUCE --A 1.2150592994538565 --ws .183889173151851 --phase 2.7925268031909267 --polar .08234860970373083 --nb 6 --seam .95 --rate_scale 2.23 --delta_map -1 --duration 235
```

N09用同一命令，将case改为`N09_REPRODUCE`、`--delta_map`改为`1`。没有重新放根、笛卡尔轨迹拟合、接住才推进的反馈，只有论文累计关节波、松开恢复和开环窗口。人工放置全身源管螺旋发生在第一步积分前，未预置目标包绕。

|阶段|N09（原式Δ）|N10（L分支Δ适配）|
|---|---:|---:|
|前6杆持续目标包绕|152.4 s|63.2 s|
|最后源管承载接触|171.7 s|114.5 s|
|全21杆持续目标包绕开始|205.9 s|153.5 s|
|之后持续滚动|29.0 s|81.4 s|
|末态螺旋轴与目标轴偏差|.699°|1.624°|

`success`由独立物理轨迹核对后写入，不由nt≥20或仅有目标接触决定。全部试次仍保留，未删除失败材料。16关节/T管偏离范围探索保留在相邻paper目录，不计本任务成果。
