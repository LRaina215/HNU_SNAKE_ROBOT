# -*- coding: utf-8 -*-
"""Freeze one objective revision, retaining all V5 failure evidence."""
from pathlib import Path
import json,difflib
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION';old=P/'CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS';C=old/'T_UPDATE';summary=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'));last=summary['selected_rows'][-1]
report=f'''# V5 实际单例：夹持放松与轴向漂移

原物理t0唯一T_UPDATE在5.000秒因source_boundary_drift_from_given停止，释放2.100秒，无目标接触、拟合、G写回或接管。最终给定参考位移{summary['final_given_motion_mm']:.6f}mm、稳定参考位移{summary['final_stable_motion_mm']:.6f}mm、最大稳定X位移{summary['final_stable_Xmotion_mm']:.6f}mm。alpha0.116,beta0,S0。完整Step2失败，不能将平均周向漂移小一些说成保持改善。

后段实际关节相对稳定参考变化到{last['maximum_posterior_actual_joint_delta_deg']:.6f}度，控制目标相对给定值变化到{last['maximum_posterior_ctrl_change_deg']:.6f}度；源侧请求法向载荷由约80N降到{last['allocated_normal_N']:.6f}N。初始稳定实测后段法向载荷约98.7N。QP的低力矩正则与较软关节形状任务允许过多内部载荷释放，实际线圈形状、轴向位移同时改变；尚不能唯一因果归因。请求/实际力矩、滤波因子与法向力均见OUTCOME_SUMMARY.json；某些帧的力矩修正发送仍受原公共限制，不能把模型目标当已实现。

V5.1保留V5动力学、持续源接触任务及所有主链，仅将力矩目标从零中心改为原给定后段位置舵机产生的力矩：clip(127*(given_ctrl-actual_q),±2.355Nm)，投影结果仍受2.30Nm软件界限。归一化力矩偏差权重由0.001改为1，以原持位预载为主要邻近目标；不改初态、硬件或源保护。四个保存状态及非零误差完整命令已通过，但均非动力学证明。下一步独立当前版本核对后一个真实T_UPDATE；最终四例仍待真实完整动作依据。

当前累计32次实际启动，V5.1为0。V3.2/V3.3/V4.1/V5和全部旧例均保存，不重复或覆盖。
'''
(P/'V5_ACTUAL_DIAGNOSTIC.md').write_text(report,encoding='utf-8')
design='''# V5.1 保留原夹持预载的动力学投影

本版是V5后段控制的单项修订，依据V5实际5秒早停及后段控制目标漂移超过3度、线圈松开/轴向漂移。动力学和接触点保持任务、前部G/重力/试探和原曲率链不变。原力矩正则偏好低力矩，会丢掉原给定后段持位预载；现以原给定位置舵机的力矩clip(127*(given_ctrl-actual_q),±2.355Nm)为邻近目标，做动力学/源接触约束投影。原物理kp127和2.355Nm不改，投影请求仍<=2.30Nm；归一化偏差权重由0.001改为1，其余控制任务和边界保持。新增日志分别列名义持位力矩和投影修正。

四个已保存姿态的分配法向载荷约90–98N，较V5原80N及后来45.7N更接近既有夹持，但这仍不是实际载荷证明。非零6.08mm接触误差的命令在世界线/角动量、原前部链、原生积分及2mm步长上通过。V5.1尚未积分。独立核对后仅做一个同一给定位姿的物理t0 T_UPDATE，原235秒预算、12mm源保护及所有物理停止条件保留，不新增离线全路径IK门槛。

'''+(old/'CONTROLLER_DESIGN.md').read_text(encoding='utf-8')
(B/'CONTROLLER_DESIGN.md').write_text(design,encoding='utf-8');diff=''.join(difflib.unified_diff((old/'source_wrench_hold.py').read_text(encoding='utf-8').splitlines(True),(B/'source_wrench_hold.py').read_text(encoding='utf-8').splitlines(True),fromfile='V5/source_wrench_hold.py',tofile='V5.1/source_wrench_hold.py'));(B/'PRELOAD_PROJECTION_CHANGES.diff').write_text(diff,encoding='utf-8')
f=B/'prepare_review_snapshot.py';s=f.read_text(encoding='utf-8').replace("'../V4_1_ACTUAL_DIAGNOSTIC.md'","'../V5_ACTUAL_DIAGNOSTIC.md'").replace("'../CONTROLLER_V4_1_ROBOT_MASS_HOLD/T_UPDATE/OUTCOME_SUMMARY.json'","'../CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS/T_UPDATE/OUTCOME_SUMMARY.json'");f.write_text(s,encoding='utf-8');print(str(B))
