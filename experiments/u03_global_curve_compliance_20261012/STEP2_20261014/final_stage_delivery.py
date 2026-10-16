# -*- coding: utf-8 -*-
"""Record final supervision acceptance and stage stop; no new evaluation."""
from pathlib import Path
import json
base=Path(__file__).resolve().parent
report_path=base/'RESULTS.md';text=report_path.read_text(encoding='utf-8')
text=text.replace('大于3毫米的保存点连续区间仅0.1秒，未达到原0.3秒停止条件。',
    '按10Hz保存点计法，大于3毫米的连续区间估计为0.1秒，不是1kHz物理步上的精确持续时长；原停止条件也按相同采样累计，未达到0.3秒。')
text=text.replace('报告本身等待最终审阅。','阶段报告最终审阅也已完成。')
report_path.write_text(text,encoding='utf-8')
review_path=base.parent/'SUPERVISION/STEP2_R2_independent_outcome_20261015.json'
review=json.loads(review_path.read_text(encoding='utf-8'))
for name,independent in review['cases'].items():
    audit=dict(case=name,action_passed=independent['independent_action_pass'],independent_action_review_complete=True,
        independent_review=review_path.resolve().as_posix(),executor_metrics='metrics.json',raw_runner_result='result.json',
        raw_runner_pending_flag_scope='Pending at the moment of save; final verdict is this audit and independent review',
        report_review_complete=True,world_curve_exact_tracking_validated=False)
    (base/name/'acceptance_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
state_path=base/'CURRENT_STATUS.json';state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(state='R2_stage_delivered_paused_awaiting_user_instruction',paused=True,step2_complete=True,
    completion_scope='Four corrected given-contact development comparisons, artifacts and report independently reviewed; uniform benefit/world tracking/physical axial port not established',
    independent_report_review_complete=True,running=[],pending=[],new_scenes_added_in_batch=False,next='Wait for user instructions; no more runs in this stage')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
root_path=base.parent/'CURRENT_STATUS.json';root=json.loads(root_path.read_text(encoding='utf-8'))
root.update(state=state['state'],paused=True,pause_reason='STEP2 four R2 comparisons and final report independently reviewed; stage delivered; await user instructions',
    step2_complete=True,step2_completion_scope=state['completion_scope'],step2_independent_report_review_complete=True,
    supervision_heartbeat_status='PAUSED',running_cases=[],pending_registered_tests=[],new_runs_queued=False,
    next='Wait for user instructions. New pipe scenes remain authorized for necessary future preregistered validation; no additions in this stage',
    full_validation_matrix_complete=False,stage_B_complete=False,step3_authorized=False)
root_path.write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Stage delivered and paused; final report review complete; no new runs.')
