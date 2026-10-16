from pathlib import Path
import sys, json, importlib.util
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from snake_control.hook_pull_review_support import restore_saved_state
from validation.recover_unified_curve_state import save_state

case = Path(__file__).resolve().parent/'U74'
destination = case/'resume_t11p90'
destination.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location('frozen_U74',case/'sources/snake_control/unified_curve_transfer.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
physical = mujoco.MjData(model)
restore_saved_state(model,physical,case/'admitted_M20.npz')
controller = frozen.UnifiedCurveTransfer(model,physical)
elapsed = controller.load_state(case/'admitted_M20_controller.json')
controller.first_dispatch = False
commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
trace = np.load(case/'trace.npz')
start, end = round(5.80/frozen.DT), round(11.90/frozen.DT)
reference_error = ctrl_error = 0.
for tick in range(start,end):
    command = commands[tick]
    index = tick*5
    target,info = controller.command(physical,command['controller_t_s'],
        trace['contact_force_N'][index],trace['torque'][index],trace['cylindrical_contact_force_N'][index])
    ctrl_error = max(ctrl_error,float(abs(target-command['target_rad']).max()))
    reference_error = max(reference_error,float(abs(np.asarray(info['reference_points_mm'])-command['reference_points_mm']).max()))
    physical.ctrl[:] = command['target_rad']
    for _ in range(50):
        mujoco.mj_step(model,physical)
    if tick%50==0:
        print(f'Recovering original U74 through {(tick+1)*frozen.DT:.2f}s; ctrl difference {ctrl_error:.3g}, reference {reference_error:.3g}mm',flush=True)
index = end*5
errors = {key:float(abs(getattr(physical,key)-trace[key][index]).max()) for key in ['qpos','qvel','ctrl']}
if reference_error>1e-5 or ctrl_error>1e-10 or max(errors.values())>1e-10:
    raise RuntimeError(f'Recovered checkpoint differs: {errors}, ctrl {ctrl_error}, reference {reference_error}')
state = np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
mujoco.mj_getState(model,physical,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(destination/'final_state.npz',integration_state=state,ctrl=physical.ctrl.copy(),sim_time=physical.time)
controller.save_state(destination/'controller_state.json',elapsed+6.10)
c = trace['centers_mm'][index]
radius = np.linalg.norm(c[:,[0,2]]-[1100,1000],axis=1)
report = dict(checkpoint_s=11.90, recovered_from_s=5.80, original_command_replay_s=6.10,
    meaning='recover original recorded action only; no new trial or modified controls',
    physical_boundary_max_differences=errors, maximum_ctrl_difference_rad=ctrl_error,maximum_reference_difference_mm=reference_error,
    absolute_physical_time=float(physical.time),elapsed_controller_time=elapsed+6.10,
    min_body_Y_mm=float(c[:,1].min()),max_target_axis_radius_mm=float(radius.max()),
    source_modules=np.flatnonzero(trace['contact_force_N'][index,0]>.1).tolist(),
    target_modules=np.flatnonzero(trace['contact_force_N'][index,1]>.1).tolist(),
    M20_source_force_N=float(trace['cylindrical_contact_force_N'][index,0,20]),
    M20_target_force_N=float(trace['cylindrical_contact_force_N'][index,1,20]),
    transferred_material_modules=controller.transferred_material_modules)
(destination/'recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
