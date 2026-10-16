"""Summarize preserved actual evidence and extract existing video, no dynamics."""
from pathlib import Path
import json, datetime, cv2, numpy as np
P=Path(__file__).resolve().parent; B=P/'CONTROLLER_V3_POSITIVE_POLAR'; C=B/'T_UPDATE'
result=json.loads((C/'result.json').read_text()); metrics=json.loads((C/'ACTION_CHAIN_METRICS.json').read_text()); audit=json.loads((C/'SOURCE_DRIFT_AND_GRAVITY_AUDIT.json').read_text())
video=C/'actual_dual_1x.mp4'; cap=cv2.VideoCapture(str(video))
metadata=dict(readable=cap.isOpened(),fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),path=str(video),untrimmed=True)
frames=[]
for index in [0,metadata['frames']//2,metadata['frames']-1]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
    if not ok:raise RuntimeError(f'Unreadable video frame {index}')
    cv2.imwrite(str(C/f'video_frame_{index:04d}.png'),frame); frames.append(frame)
cap.release();metadata['selected_frames_readable']=True;metadata['last_video_sample_s']=(metadata['frames']-1)/metadata['fps'];metadata['exact_final_state_s']=result['sim_time']
cv2.imwrite(str(C/'actual_video_contact_sheet.png'),np.concatenate(frames,axis=0))
(C/'VIDEO_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
summary=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),version='DIRECT_EXTENSION_V3_2_POSITIVE_POLAR',actual_result=result,action_chain_complete=False,step2_complete=False,
    total_step2_starts=29,legacy_source_starts=20,legacy_postwrap_starts=8,direct_extension_starts=1,
    final_source_guard=audit['final_source_guard'],true_receiver_contact_observed=False,first_raw_front_event_is_known_source_seam=True,
    confirmed_implementation_errors=['Fixed-bearing front gravity preload has opposite sign to potential gradient','Known source straight-end seam excluded for mixed edge normal between wall and cap thresholds'],
    causal_limit='Both errors are confirmed; their individual contributions to source drift have not been isolated.',
    no_fit_G_write_capture_transfer_or_rolling=True,video=metadata,source_support_retained=True,source_guard_metrics_include_final_integration_state=True,
    required_next='Separate focused correction and independent current-version review, then one actual diagnostic; no final four-case batch yet.')
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=f'''# V3.2 单例实际诊断：失败证据

从物理 t=0 完整运行一次 T_UPDATE，2.100 秒通过源侧稳定保持并开始释放，5.075 秒因源段位置漂移停止。没有可靠拟合、G 写回、新包绕、接收承载、整身转移或滚动；Step2 未完成。原始代码、快照、录像和状态均保留，不重跑或覆盖本例。

- 最终实际源段最大位移 {audit['final_source_guard']['maximum_source_motion_mm']:.6f} mm，X 位移 {audit['final_source_guard']['maximum_source_Xmotion_mm']:.6f} mm，最大姿态变化 {audit['final_source_guard']['maximum_source_orientation_deg']:.6f}°。触发的是 12 mm 位置保护。
- 源侧竖向载荷在释放后约 28 N，仍承担整机重量；接收直段和出口载荷均为零。S 和预接管滚动命令均为零。
- 控制器在 4.175 秒记录的首次“未知接触”来自 M3，点约 (0.949908, 0.043713, 1.035799) m，位于已知 X=0.95 m 源管接缝。混合边缘法向 nX=0.5036 落在现有壁面阈值 0.45 与端面阈值 0.65 之间，被错误排除；随后的三帧无未知接触。该单帧误判已触发 front_wrap，不能当成真实目标接触。
- 固定实测源承载端，按实际释放状态对前段重力势能做中心差分。现有重力补偿与应有保持力矩方向相反：同号最大差 {audit['gravity_preload_sign_audit']['same_sign_error_Nm']:.9f} Nm，反号误差 {audit['gravity_preload_sign_audit']['opposite_sign_error_Nm']:.3g} Nm。符号错误已确认；尚不能断言它是漂移的唯一原因。

下一版仅修正这两项实现错误，保持原初态、真实物理、增益、限制、试探与验收条件。新版本需独立核对后再运行一个实际诊断。最终两先验×冻结/更新 G 四例仍待后续动作链证据。

录像 `{video.relative_to(P)}` 为连续 1×，10 fps，覆盖 t=0…5.0 秒；末帧后到停止时刻 5.075 秒的准确状态见 integration_state.npz，已纳入漂移评估。trace.npz、contact_evidence.jsonl、ACTION_CHAIN_METRICS.json、SOURCE_DRIFT_AND_GRAVITY_AUDIT.json 和 OUTCOME_SUMMARY.json 为对应证据。

累计 Step2 启动 29 次：旧源侧 20 次、旧接管后 8 次、本直接伸出分支 1 次。完整动作成功数为 0。
'''
(P/'V3_2_ACTUAL_DIAGNOSTIC.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
