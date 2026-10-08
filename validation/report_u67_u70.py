"""Keep actual transfer facts, support-phase semantics and interrupted branches."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'experiments/unified_curve_transfer_20261007'
for name in ['U67','U69','U70']:
    p=BASE/name
    if not (p/'result.json').exists():
        continue
    r=json.loads((p/'result.json').read_text(encoding='utf8'))
    a=np.load(p/'trace.npz');f=a['cylindrical_contact_force_N'];c=a['centers_mm']
    summary={k:r[k] for k in ['outcome','error','physical_seconds','complete','fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','new_sustained_target_modules','final_actual_source_grip_modules','final_actual_target_grip_modules','first_ctrl_error_rad','max_actual_joint_deg','max_actuator_torque_Nm','max_pipe_penetration_mm','max_reference_link_error_mm']}
    summary['final_target_normal_force_sum_N']=float(f[-1,1].sum())
    summary['final_M15_center_mm']=c[-1,15].tolist()
    summary['final_M15_is_geometrically_target_side']=bool(a['target_grip_region_mask'][-1,15])
    summary['M15_min_target_force_N']=float(f[:,1,15].min())
    if len(c)>100:
        summary['tail_last1s_z_change_mm']=float(c[-1,20,2]-c[-101,20,2])
    start=r['controller_initial_elapsed_s']
    summary['current_events']=[dict(e,t_since_resume_s=e['t_s']-start) for e in r['controller_events'] if e['t_s']>=start-1e-8]
    (p/'evidence_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    lines=[f"{name}: {r['physical_seconds']:.2f}s actual dynamics; {r['outcome']}; complete={r['complete']}; error={r['error']}.",
        f"Fixed front mean target-Y advance {r['fixed_front_M0_M6_mean_y_advance_mm']:.6f}mm; fixed rear mean source-X advance {r['fixed_rear_M14_M20_mean_x_advance_mm']:.6f}mm.",
        f"Sustained target admissions in this branch: {r['new_sustained_target_modules']}; no whole-body success credit.",
        f"Final qualified source holders {r['final_actual_source_grip_modules']}; target holders {r['final_actual_target_grip_modules']}; target normal-force total {summary['final_target_normal_force_sum_N']:.6f}N is not alone proof of stability.",
        f"M15 final centre {summary['final_M15_center_mm']}mm; geometrically target-side {summary['final_M15_is_geometrically_target_side']}; minimum target force {summary['M15_min_target_force_N']:.6f}N.",
        f"Actual first saved ctrl error {r['first_ctrl_error_rad']}rad; max joint {r['max_actual_joint_deg']:.6f}deg; max torque {r['max_actuator_torque_Nm']:.6f}Nm; pipe penetration {r['max_pipe_penetration_mm']:.6f}mm; chord error {r['max_reference_link_error_mm']:.9f}mm.",
        'Original model, joint-only actuation, drive/ctrl/12mm curve/collision/complete rules retained. Only actual trace measures progress; private model predictions are not additional trials. Full states/frozen source/actual commands/1x replay preserved; large files remain local.']
    if name=='U67':
        lines.append('Cancelling stale M20 release after actual M15 handover did not retain two source contacts in this late branch. Actual M15 still bears. The two-source stop itself is insufficient evidence of physical instability after target support handover; U69 performs finite continuation from U66 actual state.')
    if name=='U69':
        lines.extend(['Finite actual target-carried continuation from U66. Old source minimum bypassed with verified M15 and4+ current target carriers; target motion side drives common phase when source group<2, while historical cumulative observer remains unchanged.',
            f"Tail last1s Z change {summary['tail_last1s_z_change_mm']:.6f}mm. M15 rolling contact gap incorrectly revoked handover even though centre stayed on target and5 actual target carriers remained; old source count stop fired. This count-based stop is not demonstrated loss of physical stability. U70 retains geometric handover through such individual contact gaps.",
            'U70 also omits empty source means from physical soft axial rows, avoiding NaN. No per-frame simultaneous positive motion requirement.'])
    (p/'RESULT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (p/'BASIS.txt').write_text(json.loads((p/'plan.json').read_text(encoding='utf8'))['change']+'\n',encoding='utf8')
p=BASE/'U68'
(p/'RESULT.txt').write_text('Actual branch from U63 incoming-M15 full state, captures current source M16/M17 loads10.846605/4.556142N and retains M20 established1.601603N. Last console reached35.00s; interrupted by steering to transfer support to receiving pipe rather than enforce all old source contacts. Exact final duration/full trace unavailable after Windows process interrupt. Full33.60s physics/controller checkpoint retained; M15 admission checkpoint retained. No final success/metrics inferred; branch not a completed validation.\n',encoding='utf8')
(p/'BASIS.txt').write_text(json.loads((p/'plan.json').read_text(encoding='utf8'))['change']+'\n',encoding='utf8')
path=BASE/'U63/retreat_observations.json'
if path.exists():
    (BASE/'U63/retreat_basis.txt').write_text('Recorded U63 actual onset14-15s: common axial task positive, target actual axial motion negative; physical candidate accepted for task residual/carrier contacts. At16s phase and shape axial feed zero but actual target cohort still retreats. No new unload/regrip preceded onset. Absolute old tangent recovery remained a separate task; braked phase dropped physical axial feature rows entirely. Shared current-curve tangent increments and always-present soft axial goal address these possible conflicts; recorded evidence does not prove a unique cause. Later M15 true target contact must be reported separately from front mean retreat. No U47 dispatch overwrite or old whole-side recovery-roll sign investigation repeated.\n',encoding='utf8')
status='2026-10-08:70 actual attempts started; U64/U68 interrupted by steering, U70 running; other U01-U69 ended. M15 new actual target handover confirmed from U63 incoming contact and U65/U66 actual continuation. U69 true source-reduced transport7.25s: front+5.433214mm/rear+3.289450mm, tail last1s Z-0.553112mm; old handover revocation on isolated M15 contact gap caused source-count stop while M15 remained target-side and5 targets held. U70 retains support handover/active motion side, omits empty means, observes actual tail/target/joint stability for finite30s. Whole-body transfer incomplete; no M16+ sustained admission yet. No history jump or fake bilateral material credit.'
for name in ['STATUS.txt','RESULTS.txt']:
    path=BASE/name;lines=path.read_text(encoding='utf8').splitlines()
    indices=[i for i,line in enumerate(lines) if line.startswith('2026-10-08:') and 'actual attempts' in line]
    if indices: lines[indices[0]]=status
    else: lines.insert(0,status)
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')
print('U67-U69 evidence and interrupted U68 preserved; U70 running.')
