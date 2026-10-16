"""Publish exact transmitted-C correction for one independent-reviewed diagnostic."""
from pathlib import Path
import json,difflib,datetime
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_1_TRANSMITTED_C_STATE';old=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';stamp=datetime.datetime.now().astimezone().isoformat();r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));r['initial_definition']=r['initial_definition'].replace('V6 preserves','V6.1 preserves');(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(B/'CONTROLLER_DESIGN.md').write_text('''# V6.1 transmitted geometric C state

V6 reached a real receiving contact (first positive10Hz truth sample18.4s) but failed source drift at28.225s, with only head material0 contacting, no reliable fit or capture. Its probe contact clock stopped at1.6s. At25s the private geometric candidate was32.27degrees from the sent total native command, while requested local-mode joint equivalents were at most1.15degrees. The saved-state continuation diagnosis separates rate and contact-step filtering; neither private geometric state nor alpha/beta timing accounted for the fraction actually transmitted. See ../V6_ACTUAL_DIAGNOSTIC.md and ../CONTROLLER_V6_SOURCE_AXIS_BALANCE/T_UPDATE/COMMAND_STALL_DIAGNOSTIC.json. This diagnosis does not isolate the entire cause of source drift.

## Runtime changes relative to exact frozen V6

Only direct_controller.py runtime behavior changes. The original rough planner, V6 source-axis balance, known-source classifier, G builder, estimator, original geometry and all physical/contact/command limits are byte-identical to V6.

1. Maintain transmitted_geometric in the fixed20-dimensional native coordinates, initialized from the original given geometric q. After the unchanged common command filter sends fraction f, update qG_sent <- qG_sent+f*(qG_candidate-qG_sent). Set private planner.q to this transmitted geometric state for the next optimization. The existing measured posterior boundary overwrite and compensating preload remain, so the total retained-source command stays given. G receives no actual whole-body FK or transmitted C points. The independent G phase/contact history is unchanged except for its rate below.
2. Existing alpha/beta monotonic increments use the previous command's common transmitted fraction as a multiplier. Existing joint tracking gates and original maximum rates remain; no source feed/rolling starts before actual capture. No new gain or threshold is introduced. Post-capture S/rolling timing remains unchanged. Capture's geometric seed is the last transmitted geometric C, avoiding an unsent candidate at the phase boundary.
3. Probe attribution now computes the joint-bounded marginal command clip(total)-clip(total-without-probe), then applies the same common transmission fraction to the existing applied-probe history. A probe pushing an already saturated joint receives zero marginal credit there. Existing contact, actual tracking, transmitted-probe tolerance, probe clock and estimator diversity/residual requirements remain.

Logs distinguish geometric_candidate_native_command from the transmitted geometric_native_command/field. The latter plus transmitted_local_curvature_field integrates exactly to raw_native_command. Requested local fields remain logged; command_limit_and_rate_curvature_field closes their exact sum with the sent geometric field. The phase-progress fraction is the previous command fraction, while command_filter_factor is the current fraction. Both are explicit. This makes the command history reviewable without claiming perfect physical following.

## Validation and limitations

CONTROLLER_ENTRY_AUDIT.json passes original t0/hold/native/source-feed/no-actual-pose-write checks. TRANSMISSION_HISTORY_AUDIT.json runs three constructed nonzero-error control commands without any physical integration: candidate steps0.5degrees, sent steps0.186..0.195degrees, next solve starts exactly from the sent free geometric C, the phase fraction matches the scalar increment, and sent geometric/local fields sum to native command within6.7e-16rad. Posterior total controls remain given and actual qpos is unchanged. The four saturation examples credit0/1/0/-1degrees from requested2/2/-2/-2degrees. These are bookkeeping checks, not physical source retention, completed probes, feasible wrapping or reliable geometry.

No clearance acceptance, actual12mm/12degree source guard,3mm penetration/self/floor rules, force relief threshold, physical mass/friction/actuator/solver, initial state or maximum phase speed is relaxed. Existing head tangent errors, local contact geometry, force balance and possible feasibility-restoration problems remain unresolved. A single original physical-t0 T_UPDATE diagnostic is requested after exact-version independent review, with failure retained. Do not rerun V6 or start the separate final4 comparison before complete actual action evidence.

## Review guard

direct_runner.py retains exact independent version/initial/code/registration/model/asset checks using byte comparison (no SHA). No approval is created by preparation. All34 prior actual starts are retained; this version has0 starts and no dynamic outcome yet.
''',encoding='utf-8')
s=(B/'prepare_review_snapshot.py').read_text(encoding='utf-8');start=s.index('    evidence=[');end=s.index('    limitations=',start);s=s[:start]+"    evidence=['CONTROLLER_DESIGN.md','CONTROLLER_ENTRY_AUDIT.json','TRANSMISSION_HISTORY_AUDIT.json','V6_1_CHANGES.diff','../V6_ACTUAL_DIAGNOSTIC.md','../CONTROLLER_V6_SOURCE_AXIS_BALANCE/T_UPDATE/COMMAND_STALL_DIAGNOSTIC.json'],\n"+s[end:];s=s.replace("'Isolated balance witnesses retain37..66degree head tangent errors; not a valid full path or touchdown proof'","'Command-history and marginal-probe checks are not actual source hold, physical wrapping or successful fit'");(B/'prepare_review_snapshot.py').write_text(s,encoding='utf-8')
diff=[]
for name in ['direct_controller.py','register_direct.py','prepare_review_snapshot.py']:
    diff.extend(difflib.unified_diff((old/name).read_text(encoding='utf-8').splitlines(True),(B/name).read_text(encoding='utf-8').splitlines(True),fromfile='V6/'+name,tofile='V6.1/'+name))
(B/'V6_1_CHANGES.diff').write_text(''.join(diff),encoding='utf-8')
stage='V6 actual failed28.225s after real receiving contact; V6.1 transmitted-C frozen review request for one diagnostic'
for f in [P/'IMPLEMENTATION_PROGRESS.json',P/'PRECHECK_STATUS.json',P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
    v=json.loads(f.read_text(encoding='utf-8-sig'));v.update(executor_updated_at=stamp,executor_current_stage=stage,executor_current_controller=str(B),executor_current_controller_version=r['version'],executor_dynamics_running=False,executor_direct_controller_review_passed=False,executor_current_version_dynamics_authorized=False,executor_new_initial_dynamics_starts=6,executor_current_version_dynamics_starts=0,executor_latest_actual_case=str(old/'T_UPDATE'),executor_latest_actual_result=json.loads((old/'T_UPDATE/result.json').read_text(encoding='utf-8')))
    if f.parent==P:v.update(updated_at=stamp,dynamics_running=False,dynamics_authorized=False,step2_complete=False,controller_independent_review_passed=False)
    if f.name=='IMPLEMENTATION_PROGRESS.json':v.update(current_stage=stage,current_version_dynamics_starts=0,new_dynamics_starts=6,total_step2_starts=34,controller_current_independent_review_passed=False,next='Freeze exact V6.1 and run sole physical-t0 T_UPDATE only after independent approval. No final4 yet.')
    if f.name=='PRECHECK_STATUS.json':v.update(new_initial_dynamics_starts=6,current_approved_controller_version=None,controller_current_initial_review_scope='Same original given pose; current V6.1 requires independent code review')
    f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
f=P/'WORKING_STATE.md';s=f.read_text(encoding='utf-8');end='<!-- runtime-header-end -->\n'
if s.startswith('# CURRENT SELECTED RUNTIME') and end in s:s=s.split(end,1)[1]
f.write_text(f'''# CURRENT SELECTED RUNTIME - {stamp}

NO ACTIVE ACTUAL OR GATE WAITER. Latest V6 completed28.225s, given-source guard12.012897mm exact state(M14..20), stable11.097254mm, no capture. Source release2.1s, raw non-source contact17.75s, true positive receiver10Hz sample18.4s, only material0. Alpha0.626/beta0.075625/S0. Probeclock1.6s, one direction only; no reliable fit/Gwrite/wrap/transfer. All starts34=20oldsource+8oldpostwrap+6direct, completed actions0. All rawstate/video283frames/contact/native evidence retained. V6_ACTUAL_DIAGNOSTIC.md and command-stall diagnosis saved; no old case reruns.

Current V6.1 transmitted-C correction prepared for frozen independent review: private C state now tracks the geometric component actually sent by the common filter; alpha/beta use prior sent fraction, same gates/maxrates. Probe credit is bounded marginal clip(total)-clip(total-without-probe) before common filter. Capture seed is sent C. G stays independent, original source grip/physics/limits/force thresholds/12mmguard unchanged. Nonzero bookkeeping/phase/native identities and entry audits pass with0integration; no physical success claimed. REVIEW_REQUEST/REVIEW_SNAPSHOT and exactdiff become available immediately after freeze. Current flagsfalse and starts0.

V6 beforecontact actual source-axis gravity moment followed load reduction toward0 and extended to true receiving contact. Aftercontact netmoment grew negative, private geometric candidate queued32degrees ahead of transmitted total command, filter~0.002, beta/probeclock stopped, source creep resumed. Geometry solver also rejects infeasible source-clearance candidates; restoration/force relief limitations remain unmodified. Next action is ONE exact-independent-reviewed V6.1 diagnostic, no perfect full offline IK gate or new initial search. Parent reads shared files, no authorized app-message back. Earlier headings below are historical.

'''+end+s,encoding='utf-8')
(P/'CURRENT_GATE_WAIT_STATUS.json').write_text(json.dumps(dict(phase='V6_1_review_request_ready_no_active_process',branch=B.name,version=r['version'],new_version_dynamics_started=False,total_actual_starts=34,updated_at=stamp),indent=2)+'\n',encoding='utf-8');print(stage)
