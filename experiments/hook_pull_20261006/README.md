# 钩挂回拉跨管：实验数据（2026-10-06）

方法与结论见 [钩挂回拉_方法与阶段结果](../../docs/钩挂回拉跨管/钩挂回拉_方法与阶段结果.md)。场景 `mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml`，20 关节，Python 3.9.21 / MuJoCo 3.2.7（`D:/Anaconda/envs/mujoco/python.exe`）。全部为仿真，直接写 ctrl，不经 EXE。台账 `trial_ledger.csv`，`key_frames.png` 为两个起点的关键帧。

| 目录 | 内容 | 结果 |
| --- | --- | --- |
| `offline/` | 第 1 步：A2 第 17 秒起点的整身滚动表、理想回拉路径 | 管轴已在钩内；纯平移约 23 mm、理想螺旋约 10 mm 后 M6 先碰管 |
| `offline_t40/` | 同上，A2 第 40 秒起点 | 纯平移约 46 mm、理想螺旋约 22 mm 后 M6 先碰管 |
| `PULL1` | 第 2 步，17 秒起点回拉 | 滚 79°、后退约 32 mm，M6 贴管后停，保持 5 秒 |
| `PULL2` | 第 2 步，40 秒起点回拉 | 滚 106°、后退约 48 mm，M6 贴管后停，保持 5 秒 |
| `CLOSE1` | 第 3 步，收紧后钩子继续按实测更新（第一版） | 通过，包绕 266° |
| `CLOSE2` | 同上，40 秒起点 | 退掉拉力后包绕 290°→228°，未通过（棘轮式放松） |
| `CLOSE1B` | 第 3 步，收紧完成后保持形状（修正版） | 通过，包绕 268°，目标管 3 接触 |
| `CLOSE2B` | 同上，40 秒起点 | 通过，包绕 287°，目标管 5 接触 |
| `EXTHOLD_1B` / `EXTHOLD_2B` | 收紧修正版终态再保持 30 秒 | 包绕不丢；蠕动 4.8 mm / 17 mm |
| `FRONT1` | 第 4 步探测：后部保持、前部滚动（接 CLOSE1） | 关节表只允许滚 10°，前进与不滚对照相当 |
| `HOLD_CONTROL1` | FRONT1 的对照：CLOSE1 终态不滚动保持 35 秒 | 前部自己蠕动 9.3 mm |
| `step4_feasibility/` | 第 4 步离线验证（2026-10-07，不推进物理时间） | 几何形状序列到 u=132 mm（约 2 节）不违反门限；但规定的是接触段沿自身轨迹滑移，滚动可行性未建立，`step4_feasibility.json` 的 verdict=EXISTS 为更正前判据（见方法文档）。逐 u 行 `step4_rows.json`，失败状态 `failing_state.npz` |

每个运行目录含 `plan.json`、`result.json`、`trace.npz`（100 Hz 位置、速度、指令、力矩、按管分组的接触力）、`commands.json`、`final_state.npz`（完整积分状态，供下一步接续）、`replay_1x.mp4`（实际物理回放，右侧为目标管截面）和 `sources/` 源码快照。`HOLD_CONTROL1`、`EXTHOLD_*` 当时用内联代码运行，之后整理为 `validation/hook_pull_hold.py`，逻辑相同，没有用脚本重跑。`FRONT1/sources/` 中的 `hook_pull_motion.py` 有一处类定义顺序错误（`RollSolver` 缺少 `table` 方法），只影响整身滚动表，不影响 FRONT1 使用的 `SplitRollSolver`；主目录已修正。

复现（运行目录已存在时拒绝覆盖，复现请换新名字）：

```powershell
$py = "D:/Anaconda/envs/mujoco/python.exe"
& $py -X utf8 validation/hook_pull_offline.py
& $py -X utf8 validation/hook_pull_experiment.py --name PULL1_REPRO
& $py -X utf8 validation/hook_pull_close.py --previous PULL1_REPRO --name CLOSE1B_REPRO --hold-mode freeze
& $py -X utf8 validation/hook_pull_offline.py --t0 40 --out experiments/hook_pull_20261006/offline_t40
& $py -X utf8 validation/hook_pull_experiment.py --name PULL2_REPRO --table experiments/hook_pull_20261006/offline_t40/roll_table.npz
& $py -X utf8 validation/hook_pull_close.py --previous PULL2_REPRO --name CLOSE2B_REPRO --hold-mode freeze
& $py -X utf8 validation/hook_pull_hold.py --previous CLOSE1B_REPRO --name EXTHOLD_REPRO --seconds 30
```
