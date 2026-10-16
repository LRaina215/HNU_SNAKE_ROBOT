from pathlib import Path
import json
import numpy as np

case=Path(__file__).resolve().parent/'U74'
commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
trace=np.load(case/'trace.npz')
samples=[]
for stamp in [5.8,11.9,12.5,13.5,13.75,13.8,13.85,35.15]:
    x=min(commands,key=lambda z:abs(z['t_s']-stamp))
    index=round(x['t_s']/.01)
    physical=trace['centers_mm'][index]
    state=x['localized_yield']
    samples.append(dict(t_s=x['t_s'],stage=state['stage'],role=state['role'],side=state['side'],
        incoming_target_contact_seen_t=state['incoming_target_contact_seen_t'],
        incoming_target_material_modules=state['incoming_target_material_modules'],
        intended_material_modules=state['intended_material_modules'],
        target_M20_anchor_goal_N=state['anchor_goals_N'][1][20],
        source_M20_force_N=float(trace['cylindrical_contact_force_N'][index,0,20]),
        target_M20_force_N=float(trace['cylindrical_contact_force_N'][index,1,20]),
        M20_actual_center_mm=physical[20].tolist(),M20_reference_center_mm=x['reference_centers_mm'][20],
        actual_q19_q20_deg=np.degrees(trace['q'][index,-2:]).tolist(),
        dispatched_ctrl19_ctrl20_deg=np.degrees(x['target_rad'][-2:]).tolist(),
        actual_front_net_y_mm=float((physical[:7,1]-trace['centers_mm'][0,:7,1]).mean()),
        actual_rear_net_x_mm=float((physical[14:,0]-trace['centers_mm'][0,14:,0]).mean()),
        curve_unload_feedback=state.get('curve_unload_feedback'),
        curve_curvature_release_strength=x['curve_curvature_release_strength'],
        M20_center_task_velocity_mm_s=x['center_task_velocity_mm_s'][20],
        local_curve_bend_constraint=x['local_curve_bend_constraint'],
        tail_normal_execution=x['tail_normal_execution'],
        physics_execution_prediction=x['physics_execution_prediction'],
        physical_progress_cohort_modules=x['physical_progress_cohort_modules'],
        physical_pipe_axial_progress_mm=x['physical_pipe_axial_progress_mm'],
        local_pipe_material_progress_mm=x['local_pipe_material_progress_mm'],
        actual_material_estimate_mm=x['actual_material_estimate_mm'],
        occupied_motion_meter=x['occupied_motion_meter']))
report=dict(samples=samples,
    verified_fact='M20 remains unregistered while source cylinder force exceeds0.1N; rear_release side0 feed continues. Actual target pressure/rolling tasks are present.',
    inference='Source-unload curve and missing material-role handover conflict with preserving newly acquired target shape. First target loss is also an execution failure despite target inward task; do not attribute it solely to opening direction.',
    feed_timeout_fact='At35.15s source cohort is empty, source physical progress equals feed origin, so min((progress-origin)/axial_sines)=0; target progress is positive. No valid late-stage actual stall evidence follows from that0.',
    retained_checkpoint='admitted_M20.npz and admitted_M20_controller.json at5.80s',
    original_recovery_status='11.90s recovery was started and interrupted on supervisory instruction; no recovered checkpoint was written; original5.80s checkpoint remains authoritative')
(case/'handover_diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Saved handover_diagnosis.json; original checkpoint is5.80s; no new dynamics')
