# 冻结复现与本地材料

Python：`D:/Anaconda/envs/mujoco/python.exe`。工作目录：`D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait`。

默认不做动力学，克隆指定实验的冻结源码和同次人工全源管初态到短路径：

```powershell
& 'D:/Anaconda/envs/mujoco/python.exe' validation/reproduce_full_source_inner.py --case F05 --out tmp/F05R_NEW
```

已用`tmp/F05R`验证该入口，全源管几何ready、物理步数0。Windows输出路径应较短；过长的嵌套实验路径会超过文件API长度限制。`FULL_SOURCE_INNER_20261009/REPRODUCE_READY`是一次零步失败的长路径克隆，不能算动力学。

`--run`会新增一次正式动力学，本轮没有使用它。F01–F05各目录的sources是对应实际版本；当前主源码还包含已标注未运行的目标柱坐标修正。复现必须使用冻结sources，不能以当前主源码冒充F04/F05执行版本。F02另有运行中人工保存停止；仅重放其命令/轨迹可精确保留这一停止时刻，默认无限期运行原控制器不会复现相同停止标签。

每次的`trace.npz`包含实际qpos/qvel/ctrl、22节点、接触力、源接触、Kp、参考和阶段；`commands.json`包含20Hz完整参考、材料坐标、曲线投影和指令；`initial_state.npz`和`final_state.npz`为完整积分状态。STL随每次sources本地保存。上述大文件不入Git，但本机可用。

实际视频：F01 110.00秒，F02 51.25秒，F03 162.80秒，F04 179.85秒，F05 46.45秒，全部1倍速。从初态到停止连续，未拼接I01。
