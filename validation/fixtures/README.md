# 固定验证输入

- `geometry_example.json`：阶段二隔离可视化程序导出的 28 关节/56 段连续截面。只作固定几何案例；时间戳和运行标识属于旧样本，测试按需要设置当前校验时刻，不用于实际控制。
- `pole28.xml`：原 28 关节 MuJoCo 验证模型，仅将网格引用改为相对 `../../mujoco_model/asset/`，没有修改关节、质量或动力学参数。
- `app-state.json`：测试 EXE 的初始显示配置，默认两关节档案/分屏显示。测试在各自结果目录复制此配置，不读取或改写用户正在使用的档案。

模型资源由仓库 `mujoco_model/asset/` 提供。`adaptive_pole_stage1/3/4.py` 与 `geometry_debug_smoke.py` 的固定输入均来自本目录；输出在 `../results/`，输出不作为运行的前置条件。
