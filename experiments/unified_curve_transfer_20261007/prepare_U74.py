from pathlib import Path
import json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.unified_curve_transfer_experiment import UnifiedSim, write
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.unified_curve_transfer import UnifiedCurveTransfer, RegisteredThreeRegionCurve, physical_wrap_masks

base = Path(__file__).resolve().parent
source = base/'U73'
inputs = base/'U74_inputs'
inputs.mkdir(exist_ok=True)
saved = json.loads((source/'controller_state.json').read_text(encoding='utf8'))
t = saved['elapsed_s']
old_failure = saved['controller']['yield_state'].pop('failure')
saved['controller']['yield_state'].update(stage='unload', started_t=t)
write(inputs/'repaired_action_controller.json', saved)
sim = UnifiedSim(inputs)
restore_saved_state(sim.m, sim.d, source/'final_state.npz')
sim.snapshot()
controller = UnifiedCurveTransfer(sim.m, sim.d)
controller.load_state(inputs/'repaired_action_controller.json')
p, centers = controller.observe(sim.d)
forces = sim.rows[-1]['cylindrical_contact_force_N']
reference_before, _, _ = controller.curve.reference(controller.shift)
registration_reason = None
try:
    error = controller.curve.persist_tail_recovery_reference(controller.shift)
except ValueError as rejected:
    registration_reason = str(rejected)
    controller.curve = RegisteredThreeRegionCurve(p, source_node=20, target_node=18)
    controller.curve.phase_origin_shift = controller.shift
    controller.curve.clearance_enabled = True
    controller.curve.clearance = controller.pipe_geometry
    controller.curve.soft_positive_clearance = controller.soft_positive_clearance_enabled
    controller.curve.capture_grip_clearance(p)
    arc, _ = controller.curve.arc_map()
    controller.curve.initial_node_arc = np.interp(controller.curve.node_s0, controller.curve.arc_grid_u, arc)
    controller.curve.initial_head_arc = float(controller.curve.initial_node_arc[0])-controller.shift
    _, registered_centers, locations = controller.curve.reference(controller.shift)
    controller.projected_locations = locations.copy()
    error = float(np.linalg.norm(registered_centers-centers,axis=1).max()*1000)
    controller.last_route_registration_time = t
    controller.last_shape_time = t
held = np.asarray(physical_wrap_masks(centers)) & (forces > .1)
held[0, controller.transferred_material_modules] = False
controller.protect_local_carrying_rods(p, held, forces, t, 20, [20])
reference_after, _, _ = controller.curve.reference(controller.shift)
state = controller.yield_state
normal = p[20]-np.array([0., 0., 1.])
normal[0] = 0.
normal /= max(np.linalg.norm(normal), 1e-9)
state.update(initial_force_N=float(forces[0,20]),
             initial_load_Nm=float(controller.filtered_load[19]),
             initial_angle_deg=float(abs(np.degrees(sim.d.qpos[controller.qa][19]))),
             outward_normal=normal.tolist(), anchor_recovery_active=False)
controller.events.append(dict(t_s=t, mode='new_supported_tail_coordination_after_failed_U73',
    previous_failure=old_failure, initial_source_force_N=float(forces[0,20]),
    reference_persistence_or_actual_registration_error_mm=error, registration_reason=registration_reason,
    widened_boundary_reference_change_mm=float(np.linalg.norm(reference_after-reference_before,axis=1).max()*1000),
    meaning='U73 failed action remains in history; new current-load unload attempt with neighbouring receiving curve motion; complete physical integration, ctrl, shared shift and measured history preserved'))
controller.save_state(inputs/'controller_state.json', t)
write(inputs/'basis.json',dict(previous_failure=old_failure, physics=str(source/'final_state.npz'),
    source_force_N=float(forces[0,20]), target_holders=np.flatnonzero(held[1]),
    maximum_reference_change_mm=float(np.linalg.norm(reference_after-reference_before,axis=1).max()*1000),
    registration_reason=registration_reason, registration_center_error_mm=error))
print((inputs/'basis.json').read_text(encoding='utf8'))
