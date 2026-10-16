from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent;a=json.loads((P/'GIVEN_INITIAL_PROOF.json').read_text());b=json.loads((P/'DIRECT_CLEARANCE_CERTIFICATE.json').read_text());c=json.loads((P/'DIRECT_SOURCE_SUPPORT_ALL.json').read_text());fixed=[r for r in c['records'] if r['contact_scope']=='fixed M8-M20 only']
checks=dict(all21_source=a['all21_source_branch_envelope'],no_receiver_precontact=a['receiver_outlet_and_straight_normal_N']==0.,no_receiver_pre_wrap=a['all_robot_centers_before_receiver_straight'],
 original_collision_and_swept_geometry=bool(b['certificate_passed']),fixed_source_all61_static_wrench=len(fixed)==61 and all(r['feasible'] for r in fixed),source_stays_put=b['maximum_fixed_source_node_motion_mm']<1e-6)
s=dict(version='DIRECT_GIVEN_INITIAL_PRECHECK_V1',checks=checks,initial_and_stationary_source_reach_materials_ready=all(checks.values()),
 report='DIRECT_EXTENSION_PRECHECK.md',initial='GIVEN_INITIAL.npz',initial_proof='GIVEN_INITIAL_PROOF.json',images=['GIVEN_INITIAL_SIDE.png','GIVEN_INITIAL_OBLIQUE.png','DIRECT_EXTENSION_KINEMATIC_SEQUENCE.png'],
 original_model_and_physics=True,old_initial_suitability_retracted=True,old_from_source_starts_preserved=20,new_initial_dynamics_starts=0,dynamics_running=False,dynamics_authorized=False,
 initial_independent_review_passed=False,actual_source_dynamic_stability_verified=False,controller_phase_separation_implementation='in progress',
 next_run_role='one T_UPDATE development diagnostic only; final four-case comparison remains separate and required',step2_complete=False,updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(P/'PRECHECK_STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
for path in [P.parent/'CURRENT_STATUS.json',P.parent/'STEP2_FROM_SOURCE_20261015/CURRENT_STATUS.json']:
 old=json.loads(path.read_text(encoding='utf-8'));old.update(current_stage='New given exit initial and stationary-source direct-reach precheck ready; separated-phase controller in progress',
  step2_direct_extension_directory=P.name,step2_direct_initial_materials_ready=all(checks.values()),step2_direct_initial_review_passed=False,step2_direct_initial_report=P.name+'/DIRECT_EXTENSION_PRECHECK.md',
  step2_new_initial_dynamic_starts=0,step2_source_preacquisition_rolling_allowed=False,step2_from_source_running=False,running=False,step2_complete=False,step2_requested_action_chain_complete=False,
  step2_suitable_exit_initial_accepted=False,updated_at=s['updated_at']);path.write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
print(json.dumps(s,indent=2))
