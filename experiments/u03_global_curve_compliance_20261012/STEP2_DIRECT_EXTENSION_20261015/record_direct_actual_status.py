"""Update executor facts from actual start/result files, without editing registration."""
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';C=B/'T_UPDATE';started=(C/'DYNAMICS_START.json').exists();finished=(C/'result.json').exists();running=started and not finished
reg=json.loads((B/'REGISTERED.json').read_text());approval=json.loads((B/'INDEPENDENT_CONTROLLER_REVIEW.json').read_text());approved=approval.get('approved') is True and approval.get('version')==reg['version']
result=json.loads((C/'result.json').read_text()) if finished else None;stamp=datetime.datetime.now().astimezone().isoformat()
stage='One reviewed physical-t0 T_UPDATE diagnostic running' if running else 'One physical-t0 T_UPDATE diagnostic saved; offline/independent outcome review pending' if finished else 'Current frozen single diagnostic has not started'
for file in [P/'IMPLEMENTATION_PROGRESS.json',P/'PRECHECK_STATUS.json',P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
    value=json.loads(file.read_text(encoding='utf-8-sig'))
    if file.name=='IMPLEMENTATION_PROGRESS.json':value.update(updated_at=stamp,current_stage=stage,new_dynamics_starts=int(started),dynamics_running=running,dynamics_authorized=reg['dynamics_authorized'],controller_current_independent_review_passed=approved,latest_actual_case=str(C),latest_actual_result=result)
    elif file.parent==P:value.update(updated_at=stamp,new_initial_dynamics_starts=int(started),dynamics_running=running,dynamics_authorized=reg['dynamics_authorized'],controller_independent_review_passed=approved,latest_actual_case=str(C),latest_actual_result=result)
    else:value.update(executor_current_stage=stage,executor_new_initial_dynamics_starts=int(started),executor_dynamics_running=running,executor_direct_controller_review_passed=approved,executor_latest_actual_case=str(C),executor_latest_actual_result=result)
    file.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(stage=stage,current_actual_starts=int(started),running=running,result=result),ensure_ascii=False))
