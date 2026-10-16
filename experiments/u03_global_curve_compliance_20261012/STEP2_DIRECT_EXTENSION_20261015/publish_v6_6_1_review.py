"""Freeze the concrete empty-contact regression correction."""
from pathlib import Path
import json,difflib,datetime
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_6_1_EMPTY_CONTACT_DIAGNOSTIC';old=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';actual=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF/T_UPDATE';stamp=datetime.datetime.now().astimezone().isoformat();r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));assert json.loads((B/'EMPTY_CONTACT_REGRESSION.json').read_text(encoding='utf-8'))['passed']
prefix='''# V6.6.1 correct empty added-contact diagnostics

Independent review rejected V6.6 before any integration: at the saved actual15.2s pose, measured contacts are onlyM13/14/16/18/20, so none is in the released first8 window. The newly logged command_incremental_clearance_violation_m attempted np.max on the empty added-plane array and raised ValueError. The rejected frozen branch and review remain intact. Waiter81953 has exited; no V6.6 physical start occurred.

V6.6.1 changes only the final rough_planner.solve diagnostic: keep the original maximum for a nonempty added-plane array; return0 when empty, and log command_incremental_plane_count. The optimization, all residuals/Jacobians/constraints/bounds/acceptance/restoration, returned commands and every other runtime module are byte-identical to frozen V6.6. No physical or safety change. A0 violation with count0 means no added plane is eligible, not that contacts were acquired.

EMPTY_CONTACT_REGRESSION.json uses the same actual15.2s physical pose/contact set as the independent rejection, with a saved-command private chart and identical saved controller-generated targets for both versions. Old V6.6 reproduces the same empty-array ValueError. V6.6.1 returns normally, accepted tracking with plane_count0 and added-plane diagnostic0. A22.3s nonempty comparison has4 eligible contacts and exactly0rad difference between old/new returned native commands. Actual snapshot qpos remains unchanged; simulation_steps0. These are required regressions, not a physical rollout or proof of new contact/source retention. CONTROLLER_ENTRY_AUDIT retains the original t0/native/source10 checks. Unchanged new-plane gradient and three bounded clearance witnesses are in the prior branch and remain limited static evidence.

Request renewed independent review for one original physical-t0 T_UPDATE diagnostic; current flagsfalse/starts0, cumulative39 actual starts unchanged. Full review of the V6.6 constraints and all actual action outcomes remain necessary. No final4 or completion claim.

The retained V6.6 design follows; references to its0 starts remain historical but its runtime mathematics are unchanged:

'''
(B/'CONTROLLER_DESIGN.md').write_text(prefix+(old/'CONTROLLER_DESIGN.md').read_text(encoding='utf-8'),encoding='utf-8')
f=B/'prepare_review_snapshot.py';s=f.read_text(encoding='utf-8');start=s.index('    evidence=[');end=s.index('    limitations=',start);s=s[:start]+"    evidence=['CONTROLLER_DESIGN.md','CONTROLLER_ENTRY_AUDIT.json','EMPTY_CONTACT_REGRESSION.json','V6_6_1_CHANGES.diff','../CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE/V6_6_CHANGES.diff','../CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE/COMMAND_CLEARANCE_AUDIT.json','../CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE/COMMAND_CONSTRAINT_GRADIENT_AUDIT.json','../V6_5_ACTUAL_DIAGNOSTIC.md','../../SUPERVISION/STEP2_V6_5_actual_V6_6_entry_block_20261016.json'],\n"+s[end:];f.write_text(s,encoding='utf-8')
diff=[]
for name in ['rough_planner.py','register_direct.py','prepare_review_snapshot.py']:diff.extend(difflib.unified_diff((old/name).read_text(encoding='utf-8').splitlines(True),(B/name).read_text(encoding='utf-8').splitlines(True),fromfile='V6.6/'+name,tofile='V6.6.1/'+name))
(B/'V6_6_1_CHANGES.diff').write_text(''.join(diff),encoding='utf-8')
stage='V6.6 blocked before dynamics by empty diagnostic; V6.6.1 exact15.2s regression fixed, frozen pending independent review'
for f in [P/'IMPLEMENTATION_PROGRESS.json',P/'PRECHECK_STATUS.json',P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
    v=json.loads(f.read_text(encoding='utf-8-sig'));v.update(executor_updated_at=stamp,executor_current_stage=stage,executor_current_controller=str(B),executor_current_controller_version=r['version'],executor_dynamics_running=False,executor_direct_controller_review_passed=False,executor_current_version_dynamics_authorized=False,executor_new_initial_dynamics_starts=11,executor_current_version_dynamics_starts=0,executor_latest_actual_case=str(actual),executor_latest_actual_result=json.loads((actual/'result.json').read_text(encoding='utf-8')))
    if f.parent==P:v.update(updated_at=stamp,dynamics_running=False,dynamics_authorized=False,step2_complete=False,controller_independent_review_passed=False)
    if f.name=='IMPLEMENTATION_PROGRESS.json':v.update(current_stage=stage,current_version_dynamics_starts=0,new_dynamics_starts=11,total_step2_starts=39,controller_current_independent_review_passed=False,next='Review exact V6.6.1 and unchanged V6.6 new constraints, then sole original-t0 T_UPDATE. No final4 yet.')
    if f.name=='PRECHECK_STATUS.json':v.update(new_initial_dynamics_starts=11,current_approved_controller_version=None,controller_current_initial_review_scope='Same original given initial; exact V6.6.1 independent review pending')
    f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
f=P/'WORKING_STATE.md';s=f.read_text(encoding='utf-8');end='<!-- runtime-header-end -->\n'
if s.startswith('# CURRENT SELECTED RUNTIME') and end in s:s=s.split(end,1)[1]
f.write_text(f'''# CURRENT SELECTED RUNTIME - {stamp}

{stage}. Current branch CONTROLLER_V6_6_1_EMPTY_CONTACT_DIAGNOSTIC,0starts/all3flagsfalse. Prior V6.6 review rejected empty np.max at actual saved15.2s: onlyM13/14/16/18/20 contacts, noneinfirst8. Waiter81953 stopped without integration; old cancel_wait.request records supersession. New sole waiter will be registered separately; do not duplicate or rerun preparers.

Runtime correction only rough_planner final metrics: originalmax if nonempty,0 ifempty, explicit command_incremental_plane_count. ALL optimizer/constraints/Jacobians/commands/other runtime modules identical to frozenV6.6. Required EMPTY_CONTACT_REGRESSION repeats old15.2s ValueError, corrected new solveaccepted/count0/violation0; nonempty22.3s old/new commands identical0rad. No integration or actual-q write. Originalentry10checks retained. New REVIEW_REQUEST includes V6.6 full diff/three saved-bounded clearance witnesses/command-plane FD4.5e-8 and independentreject, since full new-constraint review is still needed.

ActuallatestV6.5 failed27.075s sourcegiven12.032314mm, onlyM0contact/probeclock2.4s, nofit/Gwrite/capture/transfer. All39actualstarts=20oldsource+8oldpostwrap+11direct preserved,0fullactions, final4notyet. Costs2064.183000s simulation,15809.514436s wall. Current original-t0 task remains active; next sole exact-reviewed development run. No physics/initial/source12mm/probe/phase/native-history changes; no Step3/quota/SHA/agents/newinitial search. Parent reads shared files; no app-message authorization inferred. Historical headings below are not current processes.

'''+end+s,encoding='utf-8')
(P/'CURRENT_GATE_WAIT_STATUS.json').write_text(json.dumps(dict(phase='V6_6_1_review_request_ready_no_active_process',branch=B.name,version=r['version'],new_version_dynamics_started=False,total_actual_starts=39,updated_at=stamp),indent=2)+'\n',encoding='utf-8');print(stage)
