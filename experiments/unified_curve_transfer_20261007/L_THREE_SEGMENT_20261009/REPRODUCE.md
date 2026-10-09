# 最短复现说明

已保存L01–L04四轮，不再执行动力学；第四轮为首轮三次审阅后明确新增的一次。活动代码为L04版本，每轮sources保存该轮冻结源码与模型；初态来自PREPARE_1/initial_pose.npz。原STL在mujoco_model/asset中，各轮本地sources/mujoco_model/asset也保留副本。

只读生成初中末参考/实际图及最后1秒复核，零物理步进：

```powershell
Set-Location -LiteralPath 'D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_Robot_Gait'
& 'D:\Anaconda\envs\mujoco\python.exe' validation/review_l_three_segment.py L01 L02 L03 L04
```

下面命令会新开正式动力学，仅在另行授权后运行；本轮未执行此命令：

```powershell
$trialName = 'L04_REPRO_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$trialDeadlineUtc = (Get-Date).ToUniversalTime().AddMinutes(10).ToString('o')
& 'D:\Anaconda\envs\mujoco\python.exe' validation/l_three_segment_experiment.py --name $trialName --initial PREPARE_1 --adaptive-supply --strict-window --speed 0.005 --rate 20 --seconds 240 --max-wall 280 --deadline $trialDeadlineUtc
```

首次执行几何准备可用`--prepare-only --prepare-name 新目录名`，几何准备零动力学；现存PREPARE_1已通过必要检查，不需重建。本地Python3.9.21、MuJoCo3.2.7、NumPy1.23.5、SciPy1.13.1、OpenCV4.11。必须使用`--strict-window`复现L04的完整窗口、尾杆几何及严格实际保持时间判据；L03的宽判据已被复核否定。
