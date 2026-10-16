from pathlib import Path
import json
base=Path(__file__).resolve().parent;registration=json.loads((base/'REGISTERED.json').read_text());cases=[x['case'] for x in registration['cases']]
checks=[]
for name in cases:
    p=base/name;plan=json.loads((p/'plan.json').read_text());checks.append(dict(case=name,configuration=plan['configuration'],runtime=plan['runtime'],
        original_model_equal=(p/'model.xml').read_bytes()==(base/'GIVEN_INITIAL/model.xml').read_bytes(),given_initial_equal=(p/'given_initial.npz').read_bytes()==(base/'GIVEN_INITIAL/integration_state.npz').read_bytes(),
        frozen_controller_files_equal={x:(p/x).read_bytes()==(base/registration['freeze_directory']/x).read_bytes() for x in ['step2_runner.py','contact_sensor.py','cylinder_estimator.py','estimated_curve.py','estimated_observer.py','global_update.py','REGISTERED.json']}))
assert all(x['original_model_equal'] and x['given_initial_equal'] and all(x['frozen_controller_files_equal'].values()) for x in checks)
(base/'DISPATCH.json').write_text(json.dumps(checks,indent=2))
state=json.loads((base/'CURRENT_STATUS.json').read_text());state.update(state='R2_four_formal_trials_running',running=cases,pending=[])
(base/'CURRENT_STATUS.json').write_text(json.dumps(state,indent=2))
path=base.parent/'CURRENT_STATUS.json';root=json.loads(path.read_text(encoding='utf-8'));root.update(state='step2_R2_four_formal_trials_running',running_cases=cases,new_runs_queued=False,pending_registered_tests=[],step2_formal_started=cases,step2_superseded_starts=4,next='Complete all4 corrected preregistered trials and saved-data report unchanged; pause for independent supervision',latest_summary='All4 R2 launched in original runtime with frozen model/common complete initial; R1 four interrupted starts retained/excluded; no final conclusions yet')
path.write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('All4 dispatch checks pass.')
