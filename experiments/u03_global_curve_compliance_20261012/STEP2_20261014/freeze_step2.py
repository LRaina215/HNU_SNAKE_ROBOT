"""Freeze the complete first round and record controller input separation."""
from pathlib import Path
import ast,json,shutil
base=Path(__file__).resolve().parent;registration=json.loads((base/'REGISTERED.json').read_text());frozen=base/registration['freeze_directory'];frozen.mkdir(exist_ok=False)
controllers=['contact_sensor.py','geometry.py','cylinder_estimator.py','native_mapping.py','source_mode.py','estimated_curve.py','estimated_observer.py','global_update.py']
audit=[]
for name in controllers:
    text=(base/name).read_text(encoding='utf-8');tree=ast.parse(text)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    forbidden=[value for value in ['pipe_straight_04','pipe_sharp_out','target_Y','[1.1,1.','[1.1, 1.','m.geom(','mesh_vert','geom_size','geom_pos','evaluation'] if value in text]
    assert not forbidden,(name,forbidden)
    audit.append(dict(file=name,imports=imports,forbidden_scene_geometry_or_labels_found=forbidden))
for path in base.glob('*.py'):shutil.copy2(path,frozen/path.name)
for name in ['run_step2.ps1','REGISTERED.json','DESIGN.md','source_constants.json','PREPARATION.json','OBSERVATION_PREPARATION.json','R2_CORRECTION_CHECKS.json']:shutil.copy2(base/name,frozen/name)
shutil.copytree(base/'GIVEN_INITIAL',frozen/'GIVEN_INITIAL')
metadata=dict(version=registration['version'],frozen_before_corrected_formal_runs=True,superseded_starts=4,new_corrected_dynamics_started=False,formal_limit=4,controller_input_audit=audit,
    model_initial_verification='byte equality against common given initial/model at dispatch; no hashes',
    development_scope='given contact, unknown target geometry with truth pose/unlabeled contact sensing; freeze/updateG paired',
    nominal_equivalence='PREPARATION.json; exact sameG points and<2e-14rad raw difference on savedS checks',
    changes_after_freeze_allowed='analysis/report/status only; no controller/prior/initial/physics/acceptance change within four runs')
(frozen/'FREEZE.json').write_text(json.dumps(metadata,indent=2));(base/'INPUT_AUDIT.json').write_text(json.dumps(metadata,indent=2))
root_path=base.parent/'CURRENT_STATUS.json';root=json.loads(root_path.read_text(encoding='utf-8'))
root.update(stage='STEP2',state='step2_R2_frozen_ready_for_four_trials',paused=False,pause_reason=None,current_batch='STEP2_20261014',step2_started=True,step2_complete=False,step2_authorized=True,
    step2_formal_result_limit=4,step2_freeze='STEP2_20261014/'+registration['freeze_directory'],step2_plan='STEP2_20261014/REGISTERED.json',step2_design='STEP2_20261014/DESIGN.md',running_cases=[],
    current_batch_complete=False,new_runs_queued=True,pending_registered_tests=['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE'],
    current_authorized_scope='STEP2 four given-contact coarse-target-prior paired comparisons; pause for independent review',
    latest_candidate=registration['version'],latest_summary='R2 receiver/unknown contact separation and safe G rollback corrected; R1 four starts saved/excluded after late supervision; original priors and gains unchanged; corrected outcomes pending',
    next='Execute four preregistered trials unchanged; retain failures and pause after report',active_search_in_scope=False,initial_pose_formation_in_scope=False,hardware_porting_in_scope=False,truth_reduction_in_scope=False)
root_path.write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(base/'CURRENT_STATUS.json').write_text(json.dumps(dict(stage='STEP2',state='R2_frozen_ready',formal_limit=4,superseded_starts=4,completed=[],running=[],pending=root['pending_registered_tests'],freeze=registration['freeze_directory'],step2_complete=False,next_stage_authorized=False),indent=2))
print('Frozen; input audit passed; four formal results pending.')
