"""Executor-owned progress fields only; preserve supervisor acceptance decisions."""
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent;stamp=datetime.datetime.now().astimezone().isoformat()
report=dict(updated_at=stamp,current_stage='V7 all-source candidate static support and direct first contact checked; contact-triggered wrap and current controller independent review pending',
    controller_path='CONTROLLER_V1',controller_version='DIRECT_EXTENSION_V2_V7_CANDIDATE',code_frozen=False,
    original_selected_given_initial='GIVEN_INITIAL.npz',original_selected_initial_unchanged=True,
    original_initial_limited_review_passed=True,original_initial_full_step2_suitability_not_proven=True,
    controller_current_independent_review_passed=False,current_initial_independent_review_passed=False,
    current_initial='CONTROLLER_V1/GIVEN_INITIAL.npz',current_initial_source='INITIAL_V7_FRONT8_RELAXED/STATIC_PRELOAD_YZ/CANDIDATE_INITIAL.npz',
    dynamics_authorized=False,new_dynamics_starts=0,dynamics_running=False,
    old_source_starts_preserved=20,full_action_successes=0,step2_complete=False,
    implemented=['Private known-source plus rough receiver construction; no true receiver mesh or saved truth path in controller',
        'Separate source-hold, alpha front extension, beta front wrap, S only after capture',
        'Fixed-material256-cell curvature coordinates and native20-hinge integral',
        'Separate probe/gravity/preload/front/receiver/source force fields and command-limit contribution',
        'Unlabeled contact-plane constraints, source boundary drift logs, preserved estimator gates',
        'Global geometry writes with bounded signatures, including after capture',
        'Original-algorithm known-source collision constraint plus per-command native joint-step bounds',
        'Transmitted/tracked probe field, candidate-contact evidence gate; all-phase fixed64.5mm material grid',
        'Continuous original-model physical runner with initial-byte/code-review binding, not started'],
    confirmed=['V7 t0 command/native integral/state-write/source-S-zero audit passed before latest solver amendments',
        'Original selected given initial reaches first contact with fixed M8, but subsequent G-to-C wrapping did not complete',
        'Frame[axis,e,b] is left-handed; source-to-receiver mirror flips local-X/even-index hinge integrals. Endpoint audit185.818deg,84.59deg joints,~1.49Nm two-side static support exists, but not a continuous action',
        'V7 static initial: all21 source geometry,16 straight+5 source inlet, receiver0, M0tip[1.048755,.086859,.911143], tangent[.451512,.838796,.304234]',
        'V7 static initial original collision source0.959mm/others0, sourceFz27.9583N/weight28.01736N, peak torque2.2871Nm, qacc max0.1125; physical stability unverified',
        'V7 fixed M8..20 source-only static wrench feasible, peak1.287Nm; actual fixed-rear Fz17.668N at initial, feasibility is not realized full-weight load',
        'V7 first front non-source contact at8.45s of ideal kinematic command replay with stationary source; no simulation integration',
        'First early-wrap prototype stopped on source3.36mm; further variants exposed per-step interpolation and contact-plane stall. All logs/traces retained'],
    current_candidates=['INITIAL_V7_FRONT8_RELAXED/STATIC_PRELOAD_YZ: current unreviewed candidate, first8 loose source rods, retained M8..20 winding437.6deg',
        'INITIAL_V6_POSITIVE_RELAXED: released13 initial infeasible, posterior grip still minimum2.932Nm; rejected for current use',
        'INITIAL_V2_UPWARD_EXIT: earlier opposite-source-chirality candidate; exact endpoint proof retained, continuous wrap not established',
        'INITIAL_V3/V4/V5: earlier geometry/clearance/reach obstructions retained, not selected'],
    open_points=['Current V7 initial needs its own independent review; old limited approval does not transfer',
        'Continuous wrapping/capture, actual source hold, executed probe/geometry fit and post-transfer rolling have not been demonstrated',
        'Post rolling sign-1 follows mirror covariance; physical forward/spin direction still unverified'],
    next='Finish current kinematic wrap diagnosis and review package; independent current initial/controller review precedes one physical-t0 T_UPDATE diagnostic. Final separate four-case comparison remains required.')
report.update(current_stage='Original limited-reviewed initial retained; V3 positive-polar/same-physical-handed wrap under kinematic diagnosis, current controller review pending',
    controller_path='CONTROLLER_V3_POSITIVE_POLAR',controller_version='DIRECT_EXTENSION_V3_2_POSITIVE_POLAR',
    current_initial='CONTROLLER_V3_POSITIVE_POLAR/GIVEN_INITIAL.npz',current_initial_source='GIVEN_INITIAL.npz',
    current_initial_independent_review_passed=True,current_initial_review_scope='Exact same original initial covered by limited geometry/static review only; current controller chain remains unapproved',
    current_kinematic_diagnostic='CONTROLLER_V3_POSITIVE_POLAR/C_CONTINUOUS_WRAP_CONTINUATION',
    current_candidates=['Original GIVEN_INITIAL.npz is the current V3 prescribed starting pose; byte-identical copy',
        'V7 static source candidate and all earlier candidate failures remain archived, not selected'],
    next='Current V3 code/physics review snapshot is frozen. Independent initial-binding/controller-chain review precedes one physical-t0 T_UPDATE diagnostic; continue evaluator/kinematic evidence work while waiting. Final four-case comparison remains separate.',
    code_frozen=True,review_materials_ready=True,review_request='CONTROLLER_V3_POSITIVE_POLAR/REVIEW_REQUEST.json',review_snapshot='CONTROLLER_V3_POSITIVE_POLAR/REVIEW_SNAPSHOT')
report['confirmed']+=['V3 first raw front contact at22.225s, alpha0.889, in no-integration ideal-C command replay; source M8..20 held to numerical precision',
    'V3 receiver native template is a proper rigid rotation of the known source coil with no hinge sign flip: front6 span185.818deg, max FK error1.03e-12mm. This is model covariance only',
    'V3 post-contact polar winding+1 is a controller choice, not a user requirement; the earlier assumed mandatory upward wrap and V7 positive-source reconfiguration are not needed',
    'Template radius correction uses only rough target pipe radius minus known source pipe radius, avoiding the earlier erroneous3.687mm inward node shift']
(P/'IMPLEMENTATION_PROGRESS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=json.loads((P/'PRECHECK_STATUS.json').read_text(encoding='utf-8'));status.update(controller_phase_separation_implementation=report['current_stage'],controller_current_progress='IMPLEMENTATION_PROGRESS.json',controller_current_initial=report['current_initial'],controller_current_initial_independent_review_passed=report['current_initial_independent_review_passed'],controller_current_initial_review_scope=report['current_initial_review_scope'],controller_review_request=report['review_request'],controller_code_frozen=True,controller_independent_review_passed=False,dynamics_authorized=False,new_initial_dynamics_starts=0,dynamics_running=False,step2_complete=False)
(P/'PRECHECK_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for file in [P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
    if file.exists():
        value=json.loads(file.read_text(encoding='utf-8-sig'));value.update(executor_current_stage=report['current_stage'],executor_current_initial=report['current_initial_source'],executor_current_initial_independent_review_passed=report['current_initial_independent_review_passed'],executor_current_initial_review_scope=report['current_initial_review_scope'],executor_review_materials_ready=True,executor_review_request=str(P/report['review_request']),executor_code_frozen_for_review=True,executor_direct_controller_progress=str(P/'IMPLEMENTATION_PROGRESS.json'),executor_new_initial_dynamics_starts=0,executor_direct_controller_review_passed=False,executor_dynamics_running=False)
        file.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['current_stage','new_dynamics_starts','dynamics_authorized','step2_complete']},ensure_ascii=False))
