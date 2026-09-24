# Adaptive_Pole_Climbing：阶段 1

本文件对应 `Adaptive_Pole_Climbing.py`。以现有 `Pole_Climbing.py` 为运行模板，将目标角生成替换为 Yamada 积分；没有加入顺应性控制或新的双向通信。原 Pole、其他步态、模型、可视化程序均未修改。

## 实现来源

- `H-Crawler.py`：复用其 `K/Psi/Kp/Ky` 曲率分解与 `scipy.integrate.quad` 计算关节角的组织方式。
- `Parallel_Crossing.py`：沿用 `HelixSegment.get_kappa/get_tau` 的标准螺旋公式。
- `Pole_Climbing.py`：保留模型加载、a/v 参数、键盘/控制台操作、可视化桥及无界面运行入口。

这些旧步态中部分 `T` 是占位函数，部分 `Psi` 专门针对分段履带形状，不能直接照搬。新脚本使用标准螺旋的常挠率积分 `Psi(s)=-tau*s+roll_phase+phase_offset`，按论文头到尾弧长方向积分。奇数关节使用 `-kappa*sin(Psi)`，偶数使用 `kappa*cos(Psi)`；再按当前 MuJoCo/可视化桥的 `[+1,-1,...]` 方向映射到执行器。

## 运行

在 Windows 的 Anaconda Prompt 中：

```bat
conda activate mujoco
cd /d "D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_Robot_Gait\codes"
python Adaptive_Pole_Climbing.py
```

默认仍为 `a=30, v=0`，即原 Pole 的平面圆弧横滚起点，不会默认直接抱住竖杆。上/下键改变 a，左/右键改变 v；9/0 设置反向/正向相位推进。沿用原程序行为，9/0 并不执行严格的单帧后暂停。

控制台支持原有命令，例如 `a=-76 v=0.05`、`time_step=0`、`time_sleep=0.01`、`step=0`。`time_step=0` 固定后续滚动相位，用于姿态观察；`step=0` 结束循环。`a=-76 v=0.05` 只是原脚本的参考参数，不代表新脚本已经通过稳定爬杆验收。

默认沿用现有单向可视化桥。如果 EXE 未启动，发送失败不会阻塞主控制循环。可用 `set SNAKE_VISUALIZER_ENABLED=0` 完全禁用桥接；不会访问实机串口。

## a/v 如何转换为 Yamada 输入

为避免更换算法时同时更换调参语义，保留原 Pole 的 a/v：

- `A=a/57.2958`：保持原脚本的角度换算常数。
- `tau=-v/l`。
- `kappa=abs(A)/(2*l*sinc(v))`，这里 `sinc(v)=sin(v)/v`，零点取 1。
- 相位偏置是 `-v + (pi if a>=0 else 0)`。

常曲率、常挠率的 Yamada 区间积分幅值为 `2*l*kappa*sinc(v)`，因此转换后可以和原 Pole 的整组目标角直接比较。实际运行仍通过曲率分量积分生成角度，没有直接调用旧 Pole 的逐关节目标公式。负 a 通过相位偏置表示，不使用负曲率。

当 kappa>0 时，参数对应 `r=kappa/(kappa²+tau²)`、`b=tau/(kappa²+tau²)`，每圈螺距为 `2*pi*b`。也可在代码中直接用 `HelixBackbone.from_radius_pitch(radius_mm, pitch_mm)` 构造目标；该入口的滚动相位需按其默认偏置约定使用。a=0 输出直线目标；v=0 为平面圆弧。a/v 转换要求 `|v|<pi`，避免 sinc 零点。

每个关节积分区间为 `[(j-1)l,(j+1)l]`，j 从 1 开始。l 取模型内部关节间距的平均值，是标准螺旋的名义等间距近似。当前 20 关节模型实际间距为 67.5–69.5 mm，名义值约 68.4474 mm；不声称生成的离散骨架精确通过连续螺旋，也不把短头尾段当成完整 l。

## 执行节奏

一轮先算齐所有目标并一次性赋给 `d.ctrl`，随后持有该目标完成 N×10 个物理步。每 10 步仍处理显示/桥接并等待 `time_sleep`。这保留原 Pole 每轮的总物理步数和等待量，但消除了逐关节更新造成的混合帧；瞬态轨迹可能与原脚本不同。

保留原程序 `t += time_step` 的相位时钟语义，滚动相位为 `omega*t`。omega 不是直接按 MuJoCo 秒计的滚动速度；每轮物理相位速度为 `omega*time_step/(N*10*模型 timestep)`。控制器换成按物理时间计速属于后续明确的行为调整。

## 验证结果（2026-09-24）

使用现有 Windows 原生 `D:\Anaconda\envs\mujoco\python.exe`，从 WSL 兼容入口运行。没有启动实机串口，没有对运行中的可视化程序发送测试帧。

- 120 组正/负幅值、正/负空间相位、20/28 关节及不同滚动相位：积分目标与原 Pole 公式最大误差 `3.50e-15 rad`。
- 直接指定半径/螺距，使用独立密集梯形积分核验曲率分解与奇偶方向：最大误差 `2.26e-11 rad`。
- 已检查整周周期性、半周符号反转、零曲率及非法参数。
- Windows 无界面 MuJoCo：20 关节平面、20 关节螺旋、28 关节螺旋各运行约 3 秒墙钟，分别产生 196、196、185 个记录样本，数值有限且仿真时间递增。
- 另用旧目标公式、相同的整帧提交节奏重放物理仿真，最大 qpos 分量差小于 `4.59e-8`。这不是与原脚本逐关节调度的轨迹等价声明。
- 原 Pole SHA-256 保持 `7c991ad721855d406aec8e38282fa25fd5b5ec8b5020bab06d14aa9259ff0caa`。

复验：在仓库目录运行 `python validation/adaptive_pole_stage1.py`。证据保存于 `validation/results/adaptive-pole-stage1/summary.json` 及同目录日志/轨迹。

未验收：Windows 交互 viewer 操作、稳定动态 rolling 爬管、真实舵机、双向接口、顺应性闭环。本阶段已验证算法替换和短程无界面执行，不能据此宣称上述项目完成。
