# A14冻结记录

唯一一次M衰减S5–9机制对照，原条件235秒完成。阶段A原验收失败，等待独立复核，B未开始，不继续M参数网格。

OUTCOME.md给出事实与推断；A14_DESIGN.md是运行前冻结的假设/判据，A14_PREPARATION.json为必需准备。audit.json原验收为前六/整身稳定转移均无；comparison.json和posterior_fit_sensitivity.json给出与A13/U03同口径的源支撑、倾角和材料回退。isolation_audit.json确认初態/模型逐字相同、t≤32秒的320个实际样本完全一致，没有SHA。

原始trace.npz、integration_state.npz、final_state.npz、controller_checkpoint.pkl、model.xml、STL、initial.npz与冻结运行源码均保留。actual_side_oblique_2x.mp4来自在线实际采样；actual_side_oblique_1x.mp4来自保存状态，4699帧/20fps/234.95秒，首/中/末读回成功见video_metadata.json，没有重跑动力学。actual_final_frame.png和mechanism_comparison.png已查看。

运行复现需将本目录冻结运行.py复制到仓库experiments下的新中央目录，保持BASE/ROOT路径层级，从新中央目录runner启动并使用新的case名：--overlap_windows --phase_overlap_windows --duration 235。直接从本子目录运行会改变层级。分析脚本在中央运行，参数/输入仍可由本冻结副本审查。原环境D:/Anaconda/envs/mujoco/python.exe、MuJoCo3.2.7；设置OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、OMP_NUM_THREADS为1。
