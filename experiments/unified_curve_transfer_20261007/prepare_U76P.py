from pathlib import Path
import sys,json
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from snake_control.unified_curve_transfer import UnifiedCurveTransfer,RegisteredThreeRegionCurve
from snake_control.hook_pull_review_support import restore_saved_state
from validation.recover_unified_curve_state import save_state
base=Path(__file__).resolve().parent
out=base/'U76P_inputs';out.mkdir(exist_ok=True)
physical=base/'U76R/action_t6p80_feed.npz';saved=base/'U76R/action_t6p80_feed_controller.json'
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m)
restore_saved_state(m,d,physical)
controller=UnifiedCurveTransfer(m,d);elapsed=controller.load_state(saved)
qpos=d.qpos.copy();qvel=d.qvel.copy();ctrl=d.ctrl.copy();shift=controller.shift
old,_,_=controller.curve.reference(shift)
p,c=controller.observe(d)
previous_curve=controller.curve
# Re-register observed body correspondence, as in the existing route repair.
# This is a reference repair, not physical replay or a full-path optimizer.
curve=RegisteredThreeRegionCurve(p,source_node=20,target_node=18)
curve.radii=list(previous_curve.radii);curve.axial_sines=list(previous_curve.axial_sines)
curve.theta_rates=list(previous_curve.theta_rates)
curve.clearance=previous_curve.clearance
curve.clearance_enabled=previous_curve.clearance_enabled
curve.soft_positive_clearance=previous_curve.soft_positive_clearance
curve.capture_grip_clearance(p)
curve.deformation_anchor_parameter=float(curve.node_s0[0])
curve.deformation_anchor_shift=shift
curve.initial_head_arc-=shift
controller.curve=curve
curve.receive_tail_on_target_helix(shift,p,replace_existing=True)
new,_,_=controller.curve.reference(shift)
controller.events.append(dict(t_s=elapsed,mode='reregister_supported_tail_in_target_polar_coordinates',
    action='observed finite-link body re-registers reference correspondence; actual paired-checkpoint nodes17..21 become target-axis polar initial tail; same phase/feed drives curve and whole-body joint solve',
    maximum_reference_node_correction_mm=float(np.linalg.norm(new-old,axis=1).max()*1000),
    meaning='reference repair only; physical integration, ctrl, shared shift, measured progress and pressure history unchanged'))
controller.yield_state['target_tail_curve_step']=dict(blend=0.,maximum_node_step_mm=0.,action='new target-axis polar tail starts at observed supported body; reference registration separately reported')
save_state(out/'controller_state.json',controller,elapsed)
basis=dict(physical_checkpoint=str(physical),controller_checkpoint=str(saved),elapsed_s=elapsed,
    physical_sim_time_s=float(d.time),reference_repair_maximum_node_mm=float(np.linalg.norm(new-old,axis=1).max()*1000),
    tail_reference_centers_Y_before_after_mm=[float(old[20:22,1].mean()*1000),float(new[20:22,1].mean()*1000)],
    actual_tail_Y_mm=float(c[20,1]*1000),actual_tail_target_radius_mm=float(np.linalg.norm(c[20,[0,2]]-[1.1,1.])*1000),
    maximum_reference_link_error_mm=float(abs(np.linalg.norm(np.diff(new,axis=0),axis=1)-.0645).max()*1000),
    preservation_max_difference=dict(qpos=float(abs(d.qpos-qpos).max()),qvel=float(abs(d.qvel-qvel).max()),ctrl=float(abs(d.ctrl-ctrl).max()),shift=float(abs(controller.shift-shift))),
    previous_window_at_least_seconds=18.25,short_horizon_s=10.,
    note='No dynamics or physical replay in preparation; reference shape repaired from complete supported physical checkpoint, not from old source-driven tail reference')
(out/'basis.json').write_text(json.dumps(basis,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(basis,ensure_ascii=False,indent=2))
