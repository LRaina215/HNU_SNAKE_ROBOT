# 实验室急弯、T 型与 U 型仿真场景

更新：2026-10-03。此次完成场景搭建与静态验证，**尚未验证任何新增场景的攀爬通过能力**。没有更改顺应性控制器，也没有实现分支识别、主动转管或反向缠绕策略。

## 几何与实验含义

- 原 `pipe_bend90`（弯曲半径300 mm）和 `pipe_bend90_tight`（165 mm）均为连续圆弧弯管，保留为对照。90°是总转角，不代表轴线突然折转。
- 新 `pipe_elbow90_*` 用两段直管在45°斜切平面拼接。中心线直接转90°，两块凸网格共用同一个椭圆接缝，没有圆弧过渡；不将整个 L 做成一个会填平内角的凸包。
- 新 `pipe_tee_*` 用三条等径管臂形成 T，接头由相交圆柱近似。支臂长1.2 m，可容纳后续转管试验。旧 `pipe_tee` 的短小支管仍作为侧向障碍对照。
- 这些是实验室连接方式的理想化近似。没有实测接头外壳、套筒、胶接凸缘或小倒角尺寸，不能称为精确复刻。
- 两套机器人均采用现有实物参数，提供 `--joints 20` 和 `--joints 26`。管道裸外径110/160 mm，单层橡胶按1.5 mm建模，包胶外径113/163 mm；摩擦沿用暂定值，不保证无滑移。

## 启动与场景名称

在 `codes` 目录运行（当前实机对应20关节）：

```powershell
python Adaptive_Pole_Climbing_CurvedAxis_Mujoco.py --scene pipe_elbow90_up_horizontal --joints 20
python Adaptive_Pole_Climbing_CurvedAxis_Mujoco.py --scene pipe_tee_horizontal --joints 20
python Adaptive_Pole_Climbing_CurvedAxis_Mujoco.py --scene pipe_tee_vertical_up --joints 20
```

从仓库根目录运行时，脚本名前加 `codes/`。原方法使用 `Adaptive_Pole_Climbing_Mujoco.py`，场景名相同。所有名称都必须带 `pipe_`。

| `--scene` 参数 | 预定路线/布置 | 验证目的 |
| --- | --- | --- |
| `pipe_elbow90_up_horizontal` | 竖直向上 → 水平 +X | 实机上行后接水平管的急弯 |
| `pipe_elbow90_horizontal` | 水平 +X → 水平 +Y | 减少爬升因素，先查转向与包绕 |
| `pipe_elbow90_horizontal_up` | 水平 +X → 竖直向上 | 转向后建立承重缠绕 |
| `pipe_elbow90_horizontal_down` | 水平 +X → 竖直向下 | 解旋、下垂与向下重新缠绕 |
| `pipe_tee_horizontal` | 全部管臂水平；从 +X 横杆转入 +Y 支管 | 水平 T 选支路，另一路沿 +X 直行 |
| `pipe_tee_vertical_up` | 从水平 +X 横杆转入向上支管 | 竖直 T 向上转移，原横杆继续延伸 |
| `pipe_tee_vertical_down` | 从水平 +X 横杆转入向下支管 | 竖直 T 向下转移 |
| `pipe_tee_vertical_stem` | 从下方立管进入水平横杆，目标为 +X | 从 T 的竖干进入横杆；−X支路保留 |
| `pipe_u_horizontal` | 水平面内180°圆弧，弯曲半径300 mm | 连续轴方向场与双轴近似对照 |
| `pipe_u_horizontal_tight` | 水平面内180°圆弧，弯曲半径165 mm | 缩小曲率半径的对照 |
| `pipe_u_vertical` | 竖直上行绕顶后下行，弯曲半径300 mm | 持续曲率与重力方向变化 |
| `pipe_straight_small` | 竖直包胶外径113 mm | 小管固定/自适应螺距基线 |
| `pipe_straight_large` | 竖直包胶外径163 mm | 大管包绕长度与支撑基线 |
| `pipe_taper_horizontal` | 水平113→163 mm，轴向过渡约16.6 mm | 对应10月2日横向变径观察 |

水平入管轴沿 +X，普通水平管离地1.0 m；向下转管的水平段离地1.6 m，留出下行空间。竖直进入急弯/T 接头的中心高度为1.1 m。U 型从0.95 m处开始弯曲。这些长度、高度用于隔离地面影响，属于场景设计值。

## 进度统计与限制

`pipe_centerline` 为**预定目标路线**，用于 Ctrl+K 沿管进度和仿真轴误差诊断，不向控制器提供自动选路能力。T 型的另一支单独存为 `pipe_unused_branch`，不会被错误串成同一条折线。

T 型 `feature_span` 终点设在接头后约256.5 mm的目标支管上。静态检查已确认沿非目标支路直行的点不会跨过这条进度线。但“最近路线投影超过终点”仍不足以独立判定真实通过：未来实验须同时确认整身到达目标管、形成有效包绕且没有坠落。当前旧批量基线脚本不自动纳入这些场景；其高度跌落判据也不应直接套用到主动向下转移实验。

水平及向下段应看沿管进度，不应把关节平均高度下降直接解释为失败。原帮助中的“Ctrl+H向上”只适用于对应竖直入管的方向约定；新增场景的实际推进方向需要在短步运行中核对，不能仅凭按键名称判断。

这些场景支持检验昨天提出的三条假设，但搭建场景本身不证明可行：

1. 大小直管、横向变径：比较固定螺距角与局部调整，检查包绕角、滑移、电流及限位。
2. L/T急弯：对比原顺应性、完整局部轴方法和后续曲线拼接方法；先验证目标方向已知时能否接管，再研究逐关节负载是否能识别方向。
3. U 型：对比连续局部轴与分段轴，不把 U 型默认当作一定比直角急弯更困难。

## 重建与静态检查

在仓库根目录，使用与已有基线一致的 MuJoCo 3.3.0：

```powershell
python tools/build_pipe_network_scenes.py --joints 20
python tools/build_pipe_network_scenes.py --joints 26
python validation/lab_pipe_scenes.py --render
```

只生成某一个时，在生成脚本后加短名，例如 `elbow90_horizontal`（生成脚本不带 `pipe_` 前缀）。检验结果和预览保存至 `validation/results/lab-pipe-scenes/`，不入 Git。检查包括关节数、初始管面/地面间隙、90°折角、U型反向出口、L接头公共接缝与T型路线；不启动串口或控制器。
