# J03复现

成功运行使用J03/sources中的冻结版本。仓库活动文件已保留相同控制器和执行器；PREPARE/initial_pose.npz保存本次人工预包绕的真实27维初态，initial_geometry.json记录零物理步进的几何准备结果。9根前部杆预包绕约5.18 rad，后部通过入口弯曲过渡到悬垂供给。

在现有工作区再次运行会产生新的正式动力学记录；本轮已停止，没有执行以下复现命令。

```powershell
Set-Location -LiteralPath 'D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_Robot_Gait'
$trialDeadlineUtc = (Get-Date).ToUniversalTime().AddMinutes(20).ToString('o')
$trialName = 'J03_REPRO_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& 'D:\Anaconda\envs\mujoco\python.exe' validation/horizontal_pipe_experiment.py --name $trialName --true-hinge --front-only --connected-intake --pipe-end 1.65 --seconds 450 --max-wall 750 --deadline $trialDeadlineUtc
```

物理模型只保留沿Y轴的水平圆管，范围Y=0.15–1.65 m、X=1.1 m、Z=1 m、半径56.5 mm。管道加长在初始建模时发生，运行中不改变管道。原重力、摩擦、质量、网格、阻尼和驱动力上限保持。

完成条件在首次步进前写入plan.json：共同行程参考达到774 mm，全部21杆接近管面，各相邻杆段包括末杆均沿同一周向与轴向顺序排列，拟合螺距126–234 mm、轴向均方根误差小于12 mm，12根后部杆已被持续接收，J20刚度超过120且后部有实际管承托；连续保持1秒。触地、实际关节超过90.5°、明显实际干涉或登记的无进展条件停止并保存。

只读审阅已保存的J03，不做动力学：

```powershell
& 'D:\Anaconda\envs\mujoco\python.exe' validation/review_horizontal_pipe_j03.py
```

连续视频、trace.npz和commands.json为本地大文件证据，源码与结果摘要可单独提交。原始机器人STL来自mujoco_model/asset；本地J03/sources/mujoco_model/asset保留相同副本，独立审阅直接使用冻结模型及副本。
