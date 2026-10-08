"""Preserve concise actual evidence; no added simulation or hash tests."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]/'experiments/unified_curve_transfer_20261007'
p = BASE/'U62'
r = json.loads((p/'result.json').read_text(encoding='utf8'))
(p/'BASIS.txt').write_text(
    'U62 resumes full U61 13.15s actual handover state. Protect currently holding rods during physical prediction; detached rods may approach gradually. Bilateral forward motion is a soft short-step objective; actual continuous net motion determines feed success. Preserve original established carrier goals and pause during regrip. A missing next release candidate still caused carrier restoration, then a fresh release; U63 removes that mandatory cycle after successful feed.\n',encoding='utf8')
(p/'RESULT.txt').write_text(
    f"U62: {r['physical_seconds']:.2f}s actual dynamics, {r['outcome']}; complete={r['complete']}.\n"
    f"Fixed M0-M6 mean target-Y advance {r['fixed_front_M0_M6_mean_y_advance_mm']:.6f}mm; fixed M14-M20 mean source-X advance {r['fixed_rear_M14_M20_mean_x_advance_mm']:.6f}mm. Negative result retained; U61 earlier positive progress is not erased.\n"
    'M20 actual restoration confirmed at 8.50s (full state 8.55s). A forced new release/feed began at 18.95s (full state 19.00s), stalled, and restored at 55.00s (full state 55.05s). Next unload failed its original12s observation rule.\n'
    f"New sustained target modules: {r['new_sustained_target_modules']}; admissions: {r['admissions']}. No M15+ admission; whole body transfer incomplete.\n"
    f"Final actual source holders {r['final_actual_source_grip_modules']}; target holders {r['final_actual_target_grip_modules']}. Max actuator torque {r['max_actuator_torque_Nm']:.6f}Nm; pipe penetration {r['max_pipe_penetration_mm']:.6f}mm; max reference chord error {r['max_reference_link_error_mm']:.9f}mm.\n"
    'Only20 physical joint ctrl dispatched; private original-model predictions are not actual trials/progress. Original limits, drive, pressure rules and12mm curve cap retained. Frozen source, full states, actual trace, commands and1x replay saved. Large files remain local.\n',encoding='utf8')
failed = BASE/'U63_initialization_failed'
if failed.exists():
    (failed/'RESULT.txt').write_text('No actual control command or dynamics step executed. Initialization mistakenly requested a new grip-clearance registration of the already registered U61 checkpoint; existing admission guard rejected it. Corrected invocation retains saved clearance registration. This is not an additional actual trial.\n',encoding='utf8')
entries = [json.loads(line) for line in (BASE/'attempts.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
u63 = [i for i,e in enumerate(entries) if e['name']=='U63']
if len(u63)>1:
    entries[u63[0]].update(name='U63_initialization_failed',actual_dynamics_started=False)
(BASE/'attempts.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in entries),encoding='utf8')
for name in ['STATUS.txt','RESULTS.txt']:
    path = BASE/name
    content = path.read_text(encoding='utf8')
    old = '2026-10-08: 61 actual attempts started; U01-U60 ended; U61 coordinated actual feed running. U60: 36.10s, no sustained bilateral transport or new material frontier. Whole-body transfer remains incomplete. See U60/RESULT.txt and U61/plan.json.'
    new = '2026-10-08: 63 actual attempts started; U01-U62 ended; U63 running from U61 confirmed13.15s handover. U61 actual13.20s: fixed front+6.728668mm/rear+2.484844mm, original pitch-normalized slower-side material8.251066mm and1s M18 handover; no new M15+ admission. U62 forced next cycle:68.10s, fixed front-27.470200mm/rear-1.309566mm, no admission. U63 retains actual holders and continues common feed; restoration/switch only after actual stagnation. Whole-body transfer incomplete. Initialization failure without dynamics is listed separately, not counted as trial.'
    if old in content:
        content = content.replace(old,new,1)
    elif new not in content:
        content = new+'\n\n'+content
    path.write_text(content,encoding='utf8')
print('U62 evidence, U63 initialization record and running status preserved.')
