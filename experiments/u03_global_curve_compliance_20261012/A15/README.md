# A15冻结实际记录

唯一实际目标半径95.5毫米，235秒完成且独立失败。90.5毫米只做未运行准备。M0首次承载提前到50.7秒，但前六位置+半圈+承载从未同时成立，21杆未转移，B未开始。

OUTCOME.md给出事实、部分假设支持与限制；运行前A15_DESIGN.md冻结依据/代价，A15_PREPARATION.json、A15_CONTACT_PREPARATION.json为准备。原始trace.npz、积分末态、控制器检查点、model.xml/STL/initial.npz和冻结代码均保留。comparison.json、contact_comparison.json、front_event_timing.json、posterior_fit_sensitivity.json、isolation_audit.json支持诊断；independent_outcome.json为监督独立判定。

两个双视角视频均保留。完整1倍速由保存实际状态生成，4699帧、20fps、234.95秒，首/中/末读回成功见video_metadata.json；没有重跑动力学。actual_final_frame.png、mechanism_comparison.png和acquisition_timing.png已查看。

复现需将冻结运行.py复制到仓库experiments下的新中央目录，保持BASE/ROOT层级，启动全新case：--overlap_windows --phase_overlap_windows --target_fit_windows --duration 235。分析脚本从中央目录运行；本副本供源码审查。D:/Anaconda/envs/mujoco/python.exe，MuJoCo3.2.7；设置OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS为1。

大阈值距离查询矛盾负值保留在A15_wide_distance_rejected.json并明确无效/被有效结果替代；A15_radius905_rejected.json仅原mj_forward接触记录支持其准备风险，不能把假设位姿侵入当实际失败。没有SHA或高层反馈。
