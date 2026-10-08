"""Compare a recorded release solve with its dispatch and actual contact motion."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snake_control.hook_pull_review_support import restore_saved_state

parser = argparse.ArgumentParser()
parser.add_argument('case', type=Path)
args = parser.parse_args()
case = args.case.resolve()
plan = json.loads((case/'plan.json').read_text(encoding='utf8'))
commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
trace = np.load(case/'trace.npz')
spec = importlib.util.spec_from_file_location('release_frozen', case/'sources/snake_control/unified_curve_transfer.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
data = mujoco.MjData(model)
restore_saved_state(model, data, ROOT/plan['initial'])
controller = frozen.UnifiedCurveTransfer(model, data)
controller.load_state(ROOT/plan['controller_state'])
initial = json.loads((case/'initial_controller_state.json').read_text(encoding='utf8'))
controller.soft_positive_clearance_enabled = initial['controller']['soft_positive_clearance_enabled']
controller.curve.soft_positive_clearance = initial['curve']['soft_positive_clearance']
controller.curve.arc_cache = None
controller.curve.shape_revision += 1
controller.localized_yield_enabled = initial['controller']['localized_yield_enabled']
controller.alternating_yield_enabled = initial['controller']['alternating_yield_enabled']
controller.capture_execution_tasks = True
rows = []
maximum_error = 0.
for tick, recorded in enumerate(commands):
    index = tick*5
    data.qpos[:] = trace['qpos'][index]
    data.qvel[:] = trace['qvel'][index]
    data.ctrl[:] = trace['ctrl'][index]
    previous = controller.last.copy()
    target, info = controller.command(data, recorded['controller_t_s'], trace['contact_force_N'][index],
        trace['torque'][index], trace['cylindrical_contact_force_N'][index])
    maximum_error = max(maximum_error, float(np.max(abs(target-recorded['target_rad']))))
    state = info['localized_yield']
    if state['stage'] not in ['unload', 'feed']:
        continue
    solved = controller.execution_tasks['solution']
    dispatch = (target-previous)/frozen.DT
    selected = state['selected_joint']-1
    deepest = None
    for contact in controller.d.contact:
        for robot, pipe in [(contact.geom1, contact.geom2), (contact.geom2, contact.geom1)]:
            if int(model.geom_bodyid[robot]) == controller.head+20 and (model.geom(pipe).name or '') in ['pipe_straight_01', 'pipe_sharp_in_02']:
                if deepest is None or contact.dist < deepest.dist:
                    deepest = contact
    row = dict(t_s=recorded['t_s'], stage=state['stage'], anchor_recovery=state.get('anchor_recovery_active',False),
        selected_joint=selected+1, solved_rate_deg_s=float(np.degrees(solved[controller.va[selected]])),
        dispatched_rate_deg_s=float(np.degrees(dispatch[selected])),
        actual_selected_rate_deg_s=float(np.degrees(data.qvel[controller.va[selected]])),
        M18_actual_source_N=float(trace['cylindrical_contact_force_N'][index,0,18]),
        M20_actual_source_N=float(trace['cylindrical_contact_force_N'][index,0,20]),
        M20_small_step_approach=next((v for v in info['small_step_grasp_approaches'] if v['module']==20),None),
        M20_center_execution=info['tail_normal_execution'],
        shift_speed_mm_s=info['shift_speed_mm_s'])
    jp_front, jr_front = np.zeros((3,model.nv)), np.zeros((3,model.nv))
    # Actual centre for the contact-carrying M1, rather than the reference.
    front_center = .5*(controller.d.joint('joint1').xanchor+controller.d.joint('joint2').xanchor)
    mujoco.mj_jac(model,controller.d,jp_front,jr_front,front_center,controller.head+1)
    row.update(predicted_root_velocity=solved[:6].tolist(), measured_root_velocity=data.qvel[:6].tolist(),
        M1_requested_y_mm_s=info['center_task_velocity_mm_s'][1][1],
        M1_predicted_y_mm_s=float(jp_front[1]@solved*1000),
        M1_dispatch_with_measured_root_y_mm_s=float((jp_front[1,:6]@data.qvel[:6]+jp_front[1,controller.va]@dispatch)*1000),
        M1_actual_y_mm_s=float(jp_front[1]@data.qvel*1000))
    if deepest is not None:
        normal = deepest.pos-np.array([0.,0.,1.]);normal[0]=0.;normal/=np.linalg.norm(normal)
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jac(model,controller.d,jp,jr,deepest.pos,controller.head+20)
        n_jac=normal@jp
        root=float(n_jac[:6]@data.qvel[:6])
        row.update(M20_contact_distance_mm=float(deepest.dist*1000),
            M20_contact_actual_normal_mm_s=float(n_jac@data.qvel*1000),
            M20_contact_predicted_normal_mm_s=float((root+n_jac[controller.va]@solved[controller.va])*1000),
            M20_contact_dispatch_normal_mm_s=float((root+n_jac[controller.va]@dispatch)*1000))
    rows.append(row)
    if recorded['t_s'] >= 6.:
        break
if maximum_error > 1e-10:
    raise RuntimeError(f'Frozen command reconstruction differs: {maximum_error}')
report=dict(method='Private frozen-controller reconstruction of actual recorded observations; no physics steps or physical control dispatch',
    maximum_dispatched_ctrl_reconstruction_error_rad=maximum_error, rows=rows)
(case/'release_execution_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(maximum_error_rad=maximum_error, sample_rows=rows[::max(1,len(rows)//10)])))
