"""Minimal native-candidate checks and a reproducible local freeze; no rollout."""
import sys,json,ast,shutil,datetime,xml.etree.ElementTree as ET
from pathlib import Path
P=Path(__file__).resolve().parent;B=P/sys.argv[1];sys.path.insert(0,str(B))
from physics import *
from native_controller import DirectController
from native_mapping import NativeMapping
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
before=d.qpos.copy();c=DirectController(m,d,r,r['cases'][0]['condition'],True);q0,i0=c.command(d,0.);q1,i1=c.command(d,.025)
active_mapping=__import__(c.ref.__class__.__module__,fromlist=['NativeMapping']).NativeMapping
q_uncached,_=active_mapping.command(c.ref,0.);q_cached,_=c.ref.nominal_command(0.)
checks=dict(initial_ctrl_exact=bool(np.array_equal(q0,d.ctrl)),actual_qpos_unchanged=bool(np.array_equal(d.qpos,before)),native_integral_error_rad=float(abs(c.planner.coordinates.hinges(i1['curvature_field'])-q1).max()),info_keys_consistent=set(i0)==set(i1),given_state_bytes_same=(B/'GIVEN_INITIAL.npz').read_bytes()==(P/'GIVEN_INITIAL.npz').read_bytes(),model_bytes_same=MODEL.read_bytes()==(P/'CONTROLLER_V7_FINAL_CONTRACT_R1/LOCAL_FREEZE/model.xml').read_bytes(),future_native_max_deg=float(np.degrees(abs(q_uncached).max())),cache_exact=bool(np.array_equal(q_uncached,q_cached)))
assert checks['initial_ctrl_exact'] and checks['actual_qpos_unchanged'] and checks['native_integral_error_rad']<1e-12 and checks['info_keys_consistent'] and checks['given_state_bytes_same'] and checks['model_bytes_same'] and checks['future_native_max_deg']<88 and checks['cache_exact']
for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
r['current_code_self_checked']=True;(B/'REGISTERED.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
report=dict(version=r['version'],requirements=r['requirements'],passed=True,independently_reviewed=False,checks=checks,simulation_steps=0,limits=['Necessary entry checks only; actual action unproven','Current local curve/phase adaptation is not identical original Step1 dynamics'])
if 'source_balance' in r:
    balance_check=json.loads((B/'SOURCE_BALANCE_GRADIENT_CHECK.json').read_text());assert balance_check['passed'];report['source_balance_gradient_check']=balance_check
(B/'SELF_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(B/'CASE_REGISTRATION.json').write_text(json.dumps(dict(version=r['version'],prepared_at=datetime.datetime.now().astimezone().isoformat(),development_only=True,case=r['cases'][0],prior_actual_starts=r['all_previous_actual_starts'],duration_s=r['duration_s'],acquisition_timeout_s=r['direct_extension']['capture_timeout_after_release_s'],revision_reason=r['revision_reason'],numeric_criteria_origin='Self-selected EVALUATION_PROTOCOL comparison windows; not user hard requirements',target_truth_boundary='Only evaluation snapshot/goal stop/offline evaluator, no target truth in native controller or its input'),indent=2)+'\n',encoding='utf-8')
previous=P/'CONTROLLER_V8_NATIVE_STEP1/IMPLEMENTATION.md';s=previous.read_text(encoding='utf-8');s=s.replace('# V8 native Step1 development candidate','# '+r['version']+' development candidate');s+='\n\nCurrent revision: '+r['revision_reason']+'\nNo single-joint tracking pause on normal C release; actual20deg/s servo and original motor/contact limits remain. Probe-evidence and bearing evidence still use real actual motion. An exact native-command cache avoids recomputing identical parameter tuples; it rebuilds the same spatial spline and matches uncached q exactly.\n'
(B/'IMPLEMENTATION.md').write_text(s,encoding='utf-8')
with (B/'IMPLEMENTATION.md').open('a',encoding='utf-8') as f:
    if 'receiving_domain' in r:f.write('\nCurrent receiver chart/bearing definition supersedes prior wording: '+json.dumps(dict(domain=r['receiving_domain'],capture=r['capture_motion_evidence'],head_probe_tracking=r['probe'].get('tracking_hinges')),indent=2)+'\nPrior cylinder extrapolates to known source cut; no true target endpoint or labels enter control.\n')
    if 'handoff' in r:f.write('\nCurrent staged handoff supersedes absolute15s source retirement: '+json.dumps(r['handoff'],indent=2)+'\nActual strong receiving load permits gradual front12to14 release and initial0.1 shared feed. Two positively loaded receiving materials maintained2s permit gradual posterior/native release; source C remains active until this feedback release completes. Lost bearing stops shared S/roll; local C keeps working, with20s recovery timeout.\n')
    if (B/'NATIVE_VECTOR_EQUIVALENCE.json').exists():f.write('\nNative Bishop integration uses algebraically equivalent NumPy rotation prefix products. NATIVE_VECTOR_EQUIVALENCE.json records zero-integration comparisons at S0/9/20, command differences below3e-15rad and unchanged source/target/material/physics choices. Existing actual frozen runs are not modified.\n')
if 'source_balance' in r:
    with (B/'IMPLEMENTATION.md').open('a',encoding='utf-8') as f:f.write('''

Posterior local C now adjusts hinges12..19 to counter the change of actual gravity plus measured non-source contact X-moment from the initially held equilibrium. Exact robot COM Jacobians are reduced with a measured known-source bearing pose; no receiving mesh, true axis/radius, labels or G overwrite enters it. The local curvature offset is bounded25deg per native joint, rate4deg/s, gain0.7/s, leak0.05/s, and fades15s after measured receiving capture. Saturated drive directions that would increase unavailable torque are excluded. These are registered local-controller choices, not user tolerances. Original physical actuator/control/joint/penetration/support guards remain. This explicitly relaxes the earlier posterior-command lock for necessary source posture adjustment. It never advances source F/M/rolling before capture. A single actual-saved pose/load directional derivative check confirms the analytic moment gradient; it does not prove source bearing or full transfer. SOURCE_BALANCE_GRADIENT_CHECK.json records that limited check.
''')
F=B/'LOCAL_FREEZE';F.mkdir(exist_ok=False)
for f in list(B.glob('*.py'))+[B/n for n in ['REGISTERED.json','GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json','SELF_REVIEW.json','CASE_REGISTRATION.json','IMPLEMENTATION.md']]:shutil.copy2(f,F/f.name)
if (B/'SOURCE_BALANCE_GRADIENT_CHECK.json').exists():shutil.copy2(B/'SOURCE_BALANCE_GRADIENT_CHECK.json',F/'SOURCE_BALANCE_GRADIENT_CHECK.json')
if (B/'NATIVE_VECTOR_EQUIVALENCE.json').exists():shutil.copy2(B/'NATIVE_VECTOR_EQUIVALENCE.json',F/'NATIVE_VECTOR_EQUIVALENCE.json')
shutil.copy2(MODEL,F/'model.xml')
for mesh in ET.parse(MODEL).findall('./asset/mesh'):
    if mesh.get('file'):
        dest=F/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),dest)
print(json.dumps(report))
