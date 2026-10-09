# 运行与只读复核入口

三次正式试验H01A、H02、H03已结束。H03完整动作从experiments/hook_pull_20261006/CLOSE1B/final_state.npz恢复，不生成新的人工初态。每轮sources保存冻结源码、模型及本地STL；活动控制器使用相同H03选项。源码在snake_control/hook_reform_route.py、hook_start_transfer.py及l_three_segment_transfer.py，入口validation/hook_start_transfer_experiment.py。

零物理步的只读复核：

```powershell
Set-Location -LiteralPath 'D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_Robot_Gait'
& 'D:\Anaconda\envs\mujoco\python.exe' validation/review_hook_start_transfer.py H01A H02 H03
```

以下是H03最短重现命令，会产生新的正式动力学；本轮不再执行：

```powershell
$trialName = 'H03_REPRO_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$trialDeadlineUtc = (Get-Date).ToUniversalTime().AddMinutes(10).ToString('o')
& 'D:\Anaconda\envs\mujoco\python.exe' validation/hook_start_transfer_experiment.py --name $trialName --reform --posterior-handoff --morph-length 0.20 --speed 0.005 --rate 20 --seconds 360 --max-wall 500 --deadline $trialDeadlineUtc
```

必须保留--reform和--posterior-handoff：前者把反勾整形计入连续运动，后者在原保留后5杆退出后启用完整滚转供料并保持中段实际承载控制。完整窗口838.5 mm，全部实际几何和刚度条件按实际时间保持至少1秒才结束。局部接收计数不作为成功判断。

H01A没有这两项选项；H02只有--reform。三个版本的冻结副本用于审阅历史，不能拿最新活动文件假称重放旧轮。初始状态含76.9秒绝对历史时间，新动作时间从0起；保存ctrl及warmstart不被清零。目标管在建模时延长到Y2.0 m。运行前只做加载及必要几何检查，不重复静止保持。

环境为Python3.9.21、MuJoCo3.2.7、NumPy1.23.5、SciPy1.13.1、OpenCV4.11。原始大文件仅本地。H03的短时4.817 mm侵管、301.116 mm世界参考误差及尾杆直接承托力0必须与成功形状一起审阅，见RESULTS.md和H03/actual_evidence_review.json、contact_and_wrap_review.json。
