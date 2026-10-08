"""Report actual U63 motion and saved feed milestones, without new trials."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'experiments/unified_curve_transfer_20261007'
p=BASE/'U63'
r=json.loads((p/'result.json').read_text(encoding='utf8'))
a=np.load(p/'trace.npz')
t=a['sim_time']-a['sim_time'][0]
c=a['centers_mm']
front=np.mean(c[:,:7,1]-c[0,:7,1],axis=1)
rear=np.mean(c[:,14:,0]-c[0,14:,0],axis=1)
summary={k:r[k] for k in ['outcome','error','complete','physical_seconds','fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','max_actual_joint_deg','max_actuator_torque_Nm','max_pipe_penetration_mm','max_reference_link_error_mm','first_ctrl_error_rad']}
summary['new_material_frontier_modules']=[i for i in r['new_sustained_target_modules'] if i>=15]
summary['historical_target_readmission_modules']=[i for i in r['new_sustained_target_modules'] if i<15]
summary['saved_feed_milestones']=[]
for checkpoint in r['action_checkpoints']:
    d=json.loads((p/checkpoint['controller_state']).read_text(encoding='utf8'))['controller']
    s=d['yield_state']
    if not s.get('continuation_checkpoint',0) or s['stage']!='feed':
        continue
    index=int(np.argmin(abs(t-checkpoint['t_s'])))
    summary['saved_feed_milestones'].append(dict(t_s=checkpoint['t_s'],
        fixed_front_mean_y_advance_mm=float(front[index]),fixed_rear_mean_x_advance_mm=float(rear[index]),
        original_action_pitch_normalized_material_mm=((np.asarray(d['physical_pipe_progress'])-s['feed_pipe_origin'])/d['physical_axial_sines']*1000).tolist(),
        physical_state=checkpoint['physical_state'],controller_state=checkpoint['controller_state']))
start=r['controller_initial_elapsed_s']
summary['current_events']=[dict(e,t_since_resume_s=e['t_s']-start) for e in r['controller_events'] if e['t_s']>=start]
(p/'evidence_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
(p/'BASIS.txt').write_text('Resume full actual U61 confirmed handover, retain observed holders and original progress, continue common feed and measured pressure-based shape. A completed feed no longer forces a fresh release/regrip. Original12mm curve cap, ctrl/drive/contact/collision/completion rules retained. Only35s actual stagnation triggers new release or restoration. New8mm continuous actual material milestones save full states.\n',encoding='utf8')
lines=[f"U63: {r['physical_seconds']:.2f}s actual dynamics; {r['outcome']}; complete={r['complete']}.",
    f"Final fixed M0-M6 mean target-Y advance {front[-1]:.6f}mm; fixed M14-M20 mean source-X advance {rear[-1]:.6f}mm.",
    'Continuous feed milestones before any new unload/regrip:']
for q in summary['saved_feed_milestones']:
    lines.append(f"  {q['t_s']:.2f}s: fixed front {q['fixed_front_mean_y_advance_mm']:.6f}mm, rear {q['fixed_rear_mean_x_advance_mm']:.6f}mm; original pitch-normalized material {q['original_action_pitch_normalized_material_mm']}mm; {q['physical_state']}.")
lines.extend([f"New M15+ target admission: {summary['new_material_frontier_modules']}. Historical target readmission: {summary['historical_target_readmission_modules']}; M14 is already transferred and is not a new material frontier.",
    f"Stop error: {r['error']}. The35s stagnation transition attempted to protect an empty released-module list; its failed proposal was not dispatched. Final physical state is valid but controller export includes that partial proposal; reconstruct pre-proposal controller history from original dispatched commands before resuming.",
    f"Max torque {r['max_actuator_torque_Nm']:.6f}Nm; max pipe penetration {r['max_pipe_penetration_mm']:.6f}mm; chord error {r['max_reference_link_error_mm']:.9f}mm; first saved ctrl error {r['first_ctrl_error_rad']}rad.",
    'Positive intermediate motion does not erase subsequent retreat or imply full transfer. See evidence_summary.json for actual stage transitions. Only20 actual joint ctrl; private physics predictions are not actual trials/progress. Actual traces/commands/1x replay/full states remain local; frozen source and concise reports preserved. Final same-version CLOSE1B full transfer still required.'])
(p/'RESULT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
for name in ['STATUS.txt','RESULTS.txt']:
    path=BASE/name
    text=path.read_text(encoding='utf8')
    text=text.replace('U01-U62 ended; U63 running from U61 confirmed13.15s handover.','U01-U63 ended. U63 retained actual holders and continuous feed; see U63/RESULT.txt.',1)
    text=text.replace('U63 retains actual holders and continues common feed; restoration/switch only after actual stagnation.',f"U63 ended after {r['physical_seconds']:.2f}s: {r['outcome']}; final front {front[-1]:.6f}mm/rear {rear[-1]:.6f}mm; new M15+ admission {summary['new_material_frontier_modules']}.",1)
    path.write_text(text,encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='current_events'},ensure_ascii=True))
