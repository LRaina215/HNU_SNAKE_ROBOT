"""Save narrow executor delivery status, preserving historical experiment data."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;E=P.parent
a=json.loads((P/'PRE_DYNAMICS_AUDIT.json').read_text());r=json.loads((P/'REGISTERED.json').read_text());proof=json.loads((P/'SOURCE_INITIAL_PROOF.json').read_text())
assert all(a['checks'].values()) and not r['dynamics_authorized']
assert not any((P/c['case']).exists() for c in r['cases'])
status=dict(implementation_and_review_materials_delivered=True,ready_for_independent_review=True,
    unique_initial='SOURCE_INITIAL.npz; full-source t0, all21 rods',initial_source_geometry_passed=True,target_and_elbow_initial_contact_N=0.,
    source_vertical_N=proof['source_straight_vertical_total_N'],robot_weight_N=proof['robot_weight_N'],
    exact_static_equilibrium=False,initial_dynamic_stability_verified=False,logged_full_source_stabilization_gate_implemented=True,
    full_action_chain_implemented=True,full_action_chain_dynamics_validated=False,preparation_static_checks_passed=True,
    independent_initial_and_chain_review_passed=False,dynamics_authorized=False,new_dynamics_starts=0,integration_calls=0,
    step2_complete=False,step2_requested_action_chain_complete=False,active_precontact_search_in_scope=False,
    quota_cutoff_in_scope=False,current_stage='Step2 full-source start; Step3 belongs to another chat',
    old_t30_and_t90_initials_eligible=False,old_materials_retained_as_diagnostic=True,
    report='DESIGN.md',registration='REGISTERED.json',self_audit='PRE_DYNAMICS_AUDIT.json',image='SOURCE_INITIAL_DUAL.png',per_rod_proof='SOURCE_INITIAL_21_RODS.csv',
    primary_code=['from_source_curve.py','source_controller.py','global_update.py','source_runner.py','evaluate_chain.py'],
    remaining=['independent unique-initial and actual-code-chain review','logged full-source dynamic stabilization','actual release/contact/refinement/acquisition/transfer/rolling four-case comparison'])
(P/'CURRENT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
old=E/'STEP2_20261014/CORRECTION_PREP_20261015'
(old/'SCOPE_SUPERSEDED_20261015.json').write_text(json.dumps(dict(selected_t30_initial_eligibility_retracted=True,reason='User requires all21 rods wrapped around source at physical t0; t30 has already extended front',
    old_data_and_preparation_retained=True,current_authoritative_directory='../../STEP2_FROM_SOURCE_20261015',no_new_dynamics=True),indent=2)+'\n',encoding='utf-8')
root_path=E/'CURRENT_STATUS.json';root=json.loads(root_path.read_text(encoding='utf-8'))
root.update(latest_executor_reported_progress='Full-source t0 implementation and review package delivered; all21 source geometry/zero target contact; exact static balance and dynamic stability not claimed; independent review pending; no new dynamics',
    step2_correction_preparation_directory='STEP2_FROM_SOURCE_20261015',step2_from_source_materials_delivered=True,
    step2_from_source_independent_review_passed=False,step2_from_source_dynamics_authorized=False,step2_from_source_new_dynamics_starts=0,
    step2_from_source_initial_dynamic_stability_verified=False,step2_corrected_future_run_ready=False,
    step2_current_required_initial='all21 source-wrapped at physical t0; SOURCE_INITIAL.npz',step2_t30_initial_eligibility_retracted=True,
    step2_complete=False,step2_requested_action_chain_complete=False)
root_path.write_text(json.dumps(root,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(materials_delivered=True,static_checks_passed=len(a['checks']),winding_deg=a['initial_geometry_and_contact_proof']['source_winding_span_deg'],
    initial_depth_mm=a['initial_geometry_and_contact_proof']['source_target_self_floor_depth_mm'],new_dynamics_starts=0,independent_review_passed=False),indent=2))
