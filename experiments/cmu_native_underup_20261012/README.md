# 下侧向上复现：运行与交付

从仓库根目录执行。Python环境为`D:/Anaconda/envs/mujoco/python.exe`，MuJoCo3.2.7。新case名必须不存在。已有U01–U03作为证据保留，不覆盖、不续跑。

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' experiments/cmu_native_underup_20261012/native_run.py --case U03_REPLAY --A 1.2150592994538565 --ws -0.183889173151851 --phase -2.7925268031909267 --polar -0.08234860970373083 --f -0.02 --nb 6 --seam 0.95 --rate_scale 2.8 --delta_map -1 --duration 235
& 'D:/Anaconda/envs/mujoco/python.exe' experiments/cmu_native_underup_20261012/analyze_underup.py --case U03_REPLAY
& 'D:/Anaconda/envs/mujoco/python.exe' experiments/cmu_native_underup_20261012/render_side.py --case U03_REPLAY
```

U01对应`--rate_scale 2.23 --low 0.4`；U02对应`--rate_scale 2.23 --low 0.6`。其余与U03相同。每个case有冻结的native_run.py/native_gait.py、原模型与STL、plan、初始状态、trace、完整积分末态和连续录像。

`analyze_underup.py`只检验物理整身状态并导出材料身份轨迹；方向验收仍需审阅实际路径与侧视。`direction_review.py`复查已保存U03及N10，只向本目录写方向材料，不改N10原证据。N10同侧视复核命令：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' experiments/cmu_native_underup_20261012/render_side.py --case N10_REFERENCE --source experiments/cmu_native_reproduction_20261012/N10
```

优先阅读`RESULTS.md`、`IMPLEMENTATION.md`、`METHOD_COMPARISON.md`；看`U03/actual_side_oblique_1x.mp4`与`direction_material_comparison.png`。三次正式候选已结束，本轮不再增加动力学。
