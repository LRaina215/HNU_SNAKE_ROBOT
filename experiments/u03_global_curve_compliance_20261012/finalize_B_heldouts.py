# -*- coding: utf-8 -*-
"""Publish the already analyzed local stage record; no dynamics."""
from pathlib import Path
import json,shutil

p=Path(__file__).resolve().parent
names=['B0_PUSH','B1_PUSH','B0_PULL','B1_PULL']
for name in names:
    shutil.copy2(p/'B_HELDOUT_RESULTS_20261014.md',p/name/'B_HELDOUT_RESULTS_20261014.md')
current=p/'B_RESULTS.md';old=current.read_text(encoding='utf-8')
if not old.startswith('> 2026-10-14更新：'):
    current.write_text('> 2026-10-14更新：已恢复并完成源侧分析和四次冻结扰动配对，当前再次按要求暂停。最新完整结果见[B_HELDOUT_RESULTS_20261014.md](B_HELDOUT_RESULTS_20261014.md)。以下保留2026-10-12名义阶段报告。\n\n'+old,encoding='utf-8')
paragraphs={
    'RESULTS.md':'\n\n## 2026-10-14冻结扰动配对完成\n\n已先定位名义源侧侵入：113.7秒源直管端部与M20/tail接触，邻近受载饱和和名义/实际材料角色滞后共存；不能确定唯一耦合原因。随后四次原登记+3N/−3N、72.0–72.5秒外扰全部完成235秒并通过整身转移与持续滚动，两版均2/2成功。B1完整转移在PUSH/PULL提前0.9/1.0秒、累计力矩饱和减少21.1/20.7关节·秒，但源侵入分别增加0.407/0.688毫米，共同窗口输送速度基本相同。前六失接0.7秒改善全在外扰前，不能作为外扰恢复收益。详见[B_HELDOUT_RESULTS_20261014.md](B_HELDOUT_RESULTS_20261014.md)。按用户要求暂停，没有其他实验。\n',
    'README.md':'\n2026-10-14恢复执行已完成：源侧定位见B_SOURCE_COST_20261014/ANALYSIS.md，原登记四次扰动配对见B_HELDOUT_RESULTS_20261014.md及B_HELDOUT_comparison_20261014.json/CSV。两版扰动均2/2成功，有限局部收益伴随源侧侵入代价；当前按用户要求再次暂停。\n',
    'CANDIDATES.md':'\n2026-10-14：B0_PUSH/B1_PUSH/B0_PULL/B1_PULL四次冻结配对全部完成235秒、整身转移与持续滚动通过。参数未改。两组B1提前约0.9/1.0秒、饱和累计下降但源侧侵入升高，没有成功数提升，当前暂停。\n',
    'B_DESIGN.md':'\n\n2026-10-14用户恢复后的执行记录：已有名义源侧接触/负载定位完成，随后冻结原B0/B1并完成原登记PUSH/PULL四次，不改参数/规则。具体结果见B_HELDOUT_RESULTS_20261014.md；两版都2/2通过，有限收益与代价并存，完成后再次按用户要求暂停。无固定偏置或其他实验。\n'
}
for filename,text in paragraphs.items():
    with (p/filename).open('a',encoding='utf-8') as fp:fp.write(text)
reg=json.loads((p/'B_preregistered.json').read_text())
reg.update(held_out_execution_status='all four original trials completed; summarized; user-requested pause',
    held_out_completed_cases=names,parameters_changed_on_holdouts=False,
    latest_user_scope='2026-10-14 resumed authorized scope completed, pause after four trials and summary')
(p/'B_preregistered.json').write_text(json.dumps(reg,indent=2))
status=json.loads((p/'CURRENT_STATUS.json').read_text())
status.update(state='paused_after_20261014_source_analysis_and_four_frozen_heldouts',paused=True,
    pause_reason='User requested stage summary and pause after these four trials',
    stage_B_complete=True,stage_B_execution_complete=True,held_out_validation_complete=True,
    held_out_tests_started=True,held_out_completed_cases=names,pending_registered_tests=[],
    running_cases=[],new_runs_queued=False,latest_candidate='B1_NOM_V2_frozen_heldouts',
    held_out_successes={'B0':2,'B1':2},held_out_conditions_per_version=2,
    universal_robustness_claim_supported=False,fixed_bias_ablation_done=False,
    independent_heldout_supervisor_review='pending',
    latest_summary='both methods pass2/2; B1 modest timing/saturation/response improvement with larger source intrusion; common-window transport nearly equal',
    next='No further experiments; await explicit user instructions',
    evidence=['B_SOURCE_COST_20261014/ANALYSIS.md','B_PAIRED_FREEZE_20261014/FREEZE.json',
        'B_PAIRED_FREEZE_20261014/final_validation.json','B_HELDOUT_RESULTS_20261014.md',
        'B_HELDOUT_comparison_20261014.json','B_HELDOUT_comparison_20261014.csv'])
(p/'CURRENT_STATUS.json').write_text(json.dumps(status,indent=2))
print('Saved stage report and paused status; all four registered trials complete, no further runs queued.')
