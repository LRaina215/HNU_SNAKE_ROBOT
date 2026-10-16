# A12运行材料

设计与实际控制器：A12_DESIGN.md、handover_window_reference.py、plan.json。完整曲线、F/M、公共材料滚转、原机器人/管/初态/伺服均继承A11；最多8个交接关节作有限幅名义几何修正。

本次只有一条从U03初态开始的物理轨迹。23.701秒因监督先后指示暂停；随后要求被撤回，按指示从完整积分/控制检查点继续原冻结版本。暂停不是动力学失败，早段没有重新运行。pause_result.json、pause_trace.npz、pause_integration_state.npz、pause_controller_checkpoint.pkl、pause_final_state.npz、resume.json、resume_A12.py记录恢复。两个2倍速视频分段保留；最终1倍速视频覆盖合并后的完整真实轨迹，不重跑动力学。

验收见audit.json，结果见result.json，A11同标准对照见comparison.json。mapping_audit.json单列较长锚段的适用范围、锚点切换时命令变化及原限幅/限速后的FK形状；优化器残差不是实际包绕证据。actual_diagnosis.json采用后段10–21统一诊断，trace.npz保存实际位置、力、关节和原始/派发命令。

本目录代码是运行快照。原physics.py按实验材料根与仓库的层级定位依赖；若要复现，将这些Python文件复制到仓库experiments下一个新的材料根目录，再从该材料根调用curve_runner.py，使用全新的--case名称和--handover_windows --duration235参数。保留model.xml/STL和initial.npz核对原模型/初态。不要把复现结果写入本已冻结案例目录。
