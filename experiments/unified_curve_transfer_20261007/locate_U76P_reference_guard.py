from pathlib import Path
import sys,json,inspect
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from snake_control.unified_curve_transfer import UnifiedCurveTransfer
from snake_control.hook_pull_review_support import restore_saved_state
base=Path(__file__).resolve().parent;case=base/'U76P'
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m)
restore_saved_state(m,d,case/'action_t3p00_feed.npz')
c=UnifiedCurveTransfer(m,d);c.load_state(case/'action_t3p00_feed_controller.json')
a=np.load(case/'trace.npz');i=300
p=a['points_mm'][i]/1000;q=a['q'][i]
nodes,reference,locations=c.curve.reference(c.shift)
saved_parameters=c.curve.reference_node_parameters.copy()
saved_guide=c.curve.target_tail_guide.copy()
items=c.curve.clearance.items
records=[];original=c.curve.clearance.link_minimum
def record(nodes,samples=25):
    result=original(nodes,samples)
    frame=inspect.currentframe().f_back
    values=frame.f_locals
    fractions=np.linspace(0,1,samples)
    lines=nodes[:-1,None]*(1-fractions[None,:,None])+nodes[1:,None]*fractions[None,:,None]
    distances=np.stack([c.curve.clearance.distance(item,lines)[0].reshape(21,samples) for item in items])
    pipe,module,point=np.unravel_index(np.argmin(distances),distances.shape)
    parameters=np.interp(values['node_arc'],values['arc'],c.curve.arc_grid_u)
    u=parameters[module]*(1-fractions[point])+parameters[module+1]*fractions[point]
    raw=values['xyz'];raw_distances=np.stack([c.curve.clearance.distance(item,raw)[0] for item in items])
    body_mask=(c.curve.arc_grid_u>=parameters[-1])&(c.curve.arc_grid_u<=parameters[0])
    outside=np.flatnonzero(~body_mask)
    op,oi=np.unravel_index(np.argmin(raw_distances[:,outside]),raw_distances[:,outside].shape)
    oi=int(outside[oi])
    records.append(dict(clearance_attempt=values.get('_clearance_attempt'),minimum_reference_body_margin_mm=float(distances[pipe,module,point]*1000),
        reference_module=int(module),pipe=items[pipe]['name'],fraction_from_proximal_node=float(fractions[point]),
        point_mm=(lines[module,point]*1000).tolist(),parameter_u_mm=float(u*1000),
        current_reference_body_parameter_range_mm=[float(parameters[-1]*1000),float(parameters[0]*1000)],
        saved_current_body_parameter_range_mm=[float(saved_parameters[-1]*1000),float(saved_parameters[0]*1000)],
        within_current_body_sampling_window=bool(parameters[-1]<=u<=parameters[0]),
        within_frozen_actual_tail_parameter_range=bool(saved_guide['captured_parameters'][0]<=u<=saved_guide['captured_parameters'][-1]),
        invalid_reference_body_modules=[int(j) for j in np.flatnonzero(result<=0)],
        node_points_mm=(nodes[max(0,module-1):min(22,module+3)]*1000).tolist(),
        outside_body_extension_minimum_margin_mm=float(raw_distances[op,oi]*1000),outside_body_extension_pipe=items[op]['name'],
        outside_body_extension_point_mm=(raw[oi]*1000).tolist(),outside_body_extension_parameter_u_mm=float(c.curve.arc_grid_u[oi]*1000)))
    return result
c.curve.clearance.link_minimum=record
error=None
try:
    # One feedback/reference candidate from the saved state nearest the first
    # logged rejection. No physics step, parameter scan, or new path fitting.
    c.feedback_shape(p,q,reference,locations,abs(c.last-q),a['contact_force_N'][i],.5)
except ValueError as e:
    error=str(e)
report=dict(saved_checkpoint_s=3.,first_original_logged_rejection_s=3.15,
    original_logged_reason='Reference link crosses original pipe solid: -0.239 mm',
    calculation='one saved-state feedback/reference candidate using checkpoint3.00s observations; localizes the first failing rod, not claimed byte-identical to uncheckpointed3.15s controller',
    physics_steps=0,guide_blend_before=saved_guide['blend'],guide_blend_after=c.curve.target_tail_guide['blend'],error=error,
    first_intersection=next((r for r in records if r['minimum_reference_body_margin_mm']<=0),None),
    final_intersection=records[-1] if records else None,reference_guard_records=records)
(case/'reference_guard_location.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='reference_guard_records'},ensure_ascii=False,indent=2))
