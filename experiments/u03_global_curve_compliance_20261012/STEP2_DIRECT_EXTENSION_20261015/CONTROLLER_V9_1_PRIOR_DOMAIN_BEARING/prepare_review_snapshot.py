"""Create a byte-preserving review packet; never grants dynamics approval."""
import datetime,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from physics import BASE,MODEL

reg=json.loads((BASE/'REGISTERED.json').read_text());freeze=BASE/'REVIEW_SNAPSHOT';freeze.mkdir(exist_ok=False)
files=list(BASE.glob('*.py'))+[BASE/name for name in ['REGISTERED.json','GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json','CONTROLLER_DESIGN.md']]
for file in files:shutil.copy2(file,freeze/file.name)
shutil.copy2(MODEL,freeze/'model.xml')
meshes=[]
for mesh in ET.parse(MODEL).findall('./asset/mesh'):
    if mesh.get('file'):
        relative=Path(mesh.get('file'))
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Expected original relative workspace asset')
        destination=freeze/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/relative,destination);meshes.append(str(relative))
request=dict(prepared_at=datetime.datetime.now().astimezone().isoformat(),version=reg['version'],review_snapshot=str(freeze),
    material_ready_for_independent_code_review=True,dynamics_authorized=False,new_dynamics_starts=0,step2_complete=False,
    requested_scope='Independent review for one physical-t0 T_UPDATE development diagnostic only. Check original given-pose binding, controller input boundary, phase/history continuity, force/probe layers and original physical safety. This is not Step2 outcome acceptance.',
    initial_file=str(BASE/'GIVEN_INITIAL.npz'),initial_identical_to_limited_review_file=(BASE/'GIVEN_INITIAL.npz').read_bytes()==(BASE.parent/'GIVEN_INITIAL.npz').read_bytes(),
    prior_limited_initial_review='../../SUPERVISION/STEP2_direct_given_pose_independent_review_20261015.json',
    initial_review_binding_file=reg['initial_review'],controller_decision_file=str(BASE/'INDEPENDENT_CONTROLLER_REVIEW.json'),
    decision_required_fields=['approved','version'],initial_binding_required_fields=['limited_geometry_and_static_support_review_passed','initial_file'],
    expected_code_py_files=len(list(BASE.glob('*.py'))),original_model=str(MODEL),original_mesh_assets=meshes,
    evidence=['CONTROLLER_DESIGN.md','CONTROLLER_ENTRY_AUDIT.json','CONTACT_CHART_AUDIT.json','INITIAL_FULL_TARGET_CHORD_AUDIT.json','V6_7_CHANGES.diff','../V6_6_1_ACTUAL_DIAGNOSTIC.md','../CONTROLLER_V6_6_1_EMPTY_CONTACT_DIAGNOSTIC/T_UPDATE/WRAP_ENTRY_CHART_DIAGNOSTIC.json','../CONTROLLER_V6_6_1_EMPTY_CONTACT_DIAGNOSTIC/T_UPDATE/OUTCOME_SUMMARY.json','../../SUPERVISION/STEP2_V6_6_1_command_constraints_independent_review_20261016.json'],
    limitations=['No integrated dynamics for this version','Sampled guide continuity/radii and G independence/circular-phase identities do not establish physical contact, pressure reduction, probing or source retention','Full action and final four comparisons remain unvalidated'],
    no_approval_created_by_this_script=True)
(BASE/'REVIEW_REQUEST.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:request[k] for k in ['version','review_snapshot','material_ready_for_independent_code_review','initial_identical_to_limited_review_file','dynamics_authorized']},indent=2))
