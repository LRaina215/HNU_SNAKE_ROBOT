"""Preserve intermediate and new-material facts without additional trials."""
from pathlib import Path
import json
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'experiments/unified_curve_transfer_20261007'
p=BASE/'U64'
(p/'RESULT.txt').write_text('Actual branch from U63 15.95s positive full state. Interrupted by task steering prioritizing U63 incoming M15 at50.90s. Last console evidence reached15.00s; exact final physical duration unavailable because Windows process interrupt did not produce final trace/result. Full9.00s actual physics/controller checkpoint retained. Not a completed validation or new material admission. No inferred final metrics.\n',encoding='utf8')
(p/'BASIS.txt').write_text('Continuation uses shared current curve tangent increment rather than independent old absolute tangent recovery, detached-source axial pose recovery removed, soft physical axial objective retained when phase braked. Original pressure/limits/12mm curve cap remain.15.95s positive U63 checkpoint is kept as alternate branch; U65 prioritizes actual incoming M15.\n',encoding='utf8')
for name in ['U65','U66']:
    p=BASE/name
    r=json.loads((p/'result.json').read_text(encoding='utf8'))
    a=np.load(p/'trace.npz')
    f=a['cylindrical_contact_force_N'][:,1,15]
    summary={k:r[k] for k in ['outcome','error','physical_seconds','complete','fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','new_sustained_target_modules','final_actual_source_grip_modules','final_actual_target_grip_modules','first_ctrl_error_rad','max_actuator_torque_Nm','max_pipe_penetration_mm','max_reference_link_error_mm']}
    summary['M15_target_contact_min_force_N']=float(f.min())
    summary['M15_target_contact_final_force_N']=float(f[-1])
    summary['M15_target_contact_all_trace_positive']=bool((f>.1).all() and a['target_grip_region_mask'][:,15].all())
    start=r['controller_initial_elapsed_s']
    summary['current_events']=[dict(e,t_since_resume_s=e['t_s']-start) for e in r['controller_events'] if e['t_s']>=start-1e-8]
    (p/'evidence_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    lines=[f"{name}: {r['physical_seconds']:.2f}s actual dynamics; {r['outcome']}; complete={r['complete']}; error={r['error']}.",
        f"Fixed front mean target-Y advance {r['fixed_front_M0_M6_mean_y_advance_mm']:.6f}mm; fixed rear mean source-X advance {r['fixed_rear_M14_M20_mean_x_advance_mm']:.6f}mm.",
        f"M15 qualified actual target-cylinder contact remains positive throughout this trace: {summary['M15_target_contact_all_trace_positive']}; min force {f.min():.6f}N, final force {f[-1]:.6f}N.",
        f"Original20-joint ctrl only; first saved ctrl error {r['first_ctrl_error_rad']}rad; max torque {r['max_actuator_torque_Nm']:.6f}Nm; pipe penetration {r['max_pipe_penetration_mm']:.6f}mm; reference chord error {r['max_reference_link_error_mm']:.9f}mm.",
        'New material admission is useful progress but is not whole-body transfer. Frozen source, full physical/controller states, actual100Hz trace, commands and1x replay retained; large files local.']
    if name=='U65':
        lines.extend(['Direct U63 full264-dimensional integration/ctrl continuation preserves M15 contact since U63 50.55s: total actual continuous contact3.35s. Original1s confirmation met with minimum2 real source and5 target holders.',
            'Old initial_target_end=max(initial transient target contact) masked M15 in admissions; incoming registry code was misplaced and did not execute. Actual handover is established by full trace, not registry claim. Both corrected in U66.',
            'At3.00s M20 unload-to-feed reference persistence would cross original pipe solid by8.539mm. Failed proposal not dispatched; transactional rollback keeps valid protected curve export. U66 uses existing actual-support route registration instead of relaxing guard.'])
    else:
        lines.extend(['M15 is formally registered with the unchanged1s actual rule at continuation start, sustained new-material admission checkpoint at0.75s. Initial frontier comes from historical transferred registry, not transient initial M15 contact.',
            'Existing actual-support route registration admits common feed; historical physical meter/root/ctrl unchanged. M20 true lift recorded0.70s. Feed reaches original9.336376mm material at4.60s then regrips as source cylinder holder M17 also loses contact; remaining M16 only. Original source<2 for1s stops5.55s. No M16+ admission.',
            'U67 branches from U65 still-supported incoming-M15 full state, cancelling stale M20 release after new target material admission and protecting all current source carriers.'])
    (p/'RESULT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (p/'BASIS.txt').write_text(json.loads((p/'plan.json').read_text(encoding='utf8'))['change']+'\n',encoding='utf8')
p=BASE/'U63'
r=json.loads((p/'result.json').read_text(encoding='utf8'))
a=np.load(p/'trace.npz');f=a['cylindrical_contact_force_N'][:,1,15];t=a['sim_time']-a['sim_time'][0]
(p/'incoming_M15.txt').write_text(f"M15 actual qualified target-cylinder contact began{t[np.flatnonzero(f>.1)[0]]:.2f}s and continued until50.90s;0.35s only, below original1s confirmation. Max force{f.max():.6f}N; final centre{a['centers_mm'][-1,15].tolist()}mm. This new material contact must not be ignored because front mean later retreats. Full264-dim integration/ctrl kept. U65 directly continues this state and confirms original1s by physical trace; no51s physics replay needed.\n",encoding='utf8')
text=(p/'RESULT.txt').read_text(encoding='utf8')
text=text.replace('reconstruct pre-proposal controller history from original dispatched commands before resuming.','direct resume labels the empty-span support wait, retains actual ctrl/reference/history and actual incoming M15 contact timer; no full replay required.')
(p/'RESULT.txt').write_text(text,encoding='utf8')
recovery=p/'resume_t50p90'
if recovery.exists():
    (recovery/'RESULT.txt').write_text('Checkpoint reconstruction interrupted by task steering; no completed/usable recovered state exported. Not an additional actual trial. Direct U65 resume uses original U63 full integration/ctrl and repaired logical wait label, preserving incoming M15.\n',encoding='utf8')
status='2026-10-08:67 actual attempts started; U01-U63/U65-U66 ended; U64 interrupted by steering; U67 running. U63 continuous feed checkpoints13.95s front+8.666872mm/rear+3.093185mm and15.95s front+8.857479mm/rear+5.289735mm; then retreat, incoming M15 contact50.55-50.90s, empty-span exception. U65 direct incoming-M15 resume preserves3.35s actual target contact; original1s confirmed. U66 formally admits M15 and advances fixed front+1.611010mm/rear+3.470604mm, then original source support stop5.55s after M20 lift. U67 retains all current source carriers after M15 handover. Whole-body transfer incomplete. No new M16+ admission yet. Initialization/recovery-only failures are separate, not additional actual trials.'
for name in ['STATUS.txt','RESULTS.txt']:
    path=BASE/name
    lines=path.read_text(encoding='utf8').splitlines()
    indices=[i for i,line in enumerate(lines) if line.startswith('2026-10-08:') and 'actual attempts' in line]
    if indices: lines[indices[0]]=status
    else: lines.insert(0,status)
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')
print('U63 incoming contact, interrupted U64, actual U65/U66 new-material evidence preserved; U67 running.')
