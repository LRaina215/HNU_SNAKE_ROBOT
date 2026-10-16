"""Publish current facts while keeping current-version dynamics gates false."""
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';stamp=datetime.datetime.now().astimezone().isoformat();stage='V5.1 actual evaluated; V6 source-axis balance frozen, exact independent review requested for one T_UPDATE';C=P/'CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION/T_UPDATE';out=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'))
for f in [P/'IMPLEMENTATION_PROGRESS.json',P/'PRECHECK_STATUS.json',P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
    v=json.loads(f.read_text(encoding='utf-8-sig'));v.update(executor_updated_at=stamp,executor_current_stage=stage,executor_current_controller=str(B),executor_current_controller_version='DIRECT_EXTENSION_V6_SOURCE_AXIS_BALANCE',executor_dynamics_running=False,executor_direct_controller_review_passed=False,executor_current_version_dynamics_authorized=False,executor_new_initial_dynamics_starts=5,executor_current_version_dynamics_starts=0,executor_latest_actual_case=str(C),executor_latest_actual_result=out['actual_result'])
    if f.parent==P:v.update(updated_at=stamp,dynamics_running=False,dynamics_authorized=False,step2_complete=False,latest_actual_case=str(C),latest_actual_result=out['actual_result'],controller_independent_review_passed=False)
    if f.name=='IMPLEMENTATION_PROGRESS.json':v.update(current_stage=stage,current_version_dynamics_starts=0,new_dynamics_starts=5,total_step2_starts=33,controller_current_independent_review_passed=False,next='Exact V6 review requested; run one physical-t0 T_UPDATE after explicit independent approval, with all original source and penetration guards.')
    if f.name=='PRECHECK_STATUS.json':v.update(new_initial_dynamics_starts=5,current_approved_controller_version=None,controller_current_initial_review_scope='Same original given pose; current frozen V6 not yet independently reviewed')
    f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(P/'CURRENT_GATE_WAIT_STATUS.json').write_text(json.dumps(dict(phase='no_active_actual_or_gate_waiter',branch=B.name,version='DIRECT_EXTENSION_V6_SOURCE_AXIS_BALANCE',new_version_dynamics_started=False,total_actual_starts=33,updated_at=stamp),indent=2)+'\n',encoding='utf-8')
f=P/'WORKING_STATE.md';s=f.read_text(encoding='utf-8');end='<!-- runtime-header-end -->\n'
if s.startswith('# CURRENT SELECTED RUNTIME') and end in s:s=s.split(end,1)[1]
header=f'''# CURRENT SELECTED RUNTIME - {stamp}

NO ACTIVE ACTUAL OR GATE WAITER. Latest actual V5.1 finished13.9s with source_boundary_drift_from_given. Exact integration-state given drift12.042325mm, stable11.016368mm; no target contact/fit/Gwrite/capture. Evaluated case/video140frames/exact motion CSV and soft-contact audit preserved. All starts33=20oldsource+8oldpostwrap+5direct; current V6 starts0, flagsfalse, no final4 or completed action.

Current V6 FROZEN REVIEW_REQUEST ready: restores original V3.3 given source position grip and adds a fore C objective reducing net gravity plus measured non-source moment about known sourceX. No physical initial/physics/G input change. Three isolated kinematic witnesses reduce moment, but tangent errors37/66/60deg and near-contact residual0.076Nm/head3.31mm are limitations; not a full-path or physical feasibility proof. Gradient/input/entry audits pass with0integration, gradient maximum1.382e-8Nm/rad. Code/model/assets frozen exactly, flagsfalse pending independent review for one T_UPDATE. Design and diff disclose limitations; no full offline-path perfection prerequisite.

Original elliptic soft contact does not implement rigid source anchors. At saved V5.1 exact pose13.9s, zero-velocity/acceleration inverse requires0.372853Nm unactuated source-axis moment; source joint actuators have zero virtual work on global rigid source-roll mode. This rules out that exact static pose, not every reachable shape or finite12mm tolerance. Exact delivered QP torques still yield source-body acceleration prediction errors0.043..0.388m/s2. Mechanical direction is reduce source-axis load via free fore geometry, not another rigid-source inverse variant. Evidence: SOFT_CONTACT_COMPATIBILITY_AUDIT.json; official3.2.7 computation docs.

Keep active work going; after exact independent approval run one physical-t0 T_UPDATE diagnostic, never duplicate oldcase. Parent reads shared artifacts; no authorized app-message back. Earlier runtime headings below are historical.

'''+end
f.write_text(header+s,encoding='utf-8');print(stage)
