"""Archive an interrupted actual run from its recorded trace, without stepping physics."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from validation.unified_curve_transfer_experiment import video, write

case = Path(__file__).resolve().parent
trace = np.load(case/'trace.npz')
state = json.loads((case/'resume_t3p95/controller_state.json').read_text(encoding='utf8'))
commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
shutil.copyfile(case/'resume_t3p95/controller_state.json', case/'last_valid_controller_state.json')
seconds = float(trace['sim_time'][-1]-trace['sim_time'][0])
initial_elapsed = float(commands[0]['controller_t_s'])
result = dict(name='U46', outcome='controller_exception', complete=False,
    error='Degenerate current curve cannot define arclength: near-duplicate knots during reference persistence; failing command never dispatched.',
    physical_seconds=seconds, controller_initial_elapsed_s=initial_elapsed,
    controller_final_elapsed_s=initial_elapsed+seconds,
    final_shift_mm=float(trace['shift_mm'][-1]),
    fixed_front_M0_M6_mean_y_advance_mm=float((trace['centers_mm'][-1,:7,1]-trace['centers_mm'][0,:7,1]).mean()),
    fixed_rear_M14_M20_mean_x_advance_mm=float((trace['centers_mm'][-1,14:,0]-trace['centers_mm'][0,14:,0]).mean()),
    final_source_modules=np.flatnonzero(trace['contact_force_N'][-1,0]>.1).tolist(),
    final_target_modules=np.flatnonzero(trace['contact_force_N'][-1,1]>.1).tolist(),
    final_actual_source_grip_modules=np.flatnonzero((trace['cylindrical_contact_force_N'][-1,0]>.1)&trace['source_grip_region_mask'][-1]).tolist(),
    final_actual_target_grip_modules=np.flatnonzero((trace['cylindrical_contact_force_N'][-1,1]>.1)&trace['target_grip_region_mask'][-1]).tolist(),
    new_sustained_target_modules=[],
    max_actual_joint_deg=float(np.degrees(abs(trace['q'])).max()),
    max_actuator_torque_Nm=float(abs(trace['torque']).max()),
    max_pipe_penetration_mm=float(trace['pipe_penetration_mm'].max()),
    max_self_penetration_mm=float(trace['nonadjacent_self_penetration_mm'].max()),
    max_dispatch_step_deg=float(np.degrees(abs(np.diff(trace['ctrl'],axis=0))).max()),
    first_ctrl_error_rad=float(abs(trace['ctrl'][0]-commands[0]['target_rad']).max()),
    max_reference_link_error_mm=float(trace['maximum_reference_link_error_mm'].max()),
    controller_events=state['controller']['events'],
    report_reconstruction='Original actual trace; controller recovered using frozen observations and original dispatched controls. No new trial or physics steps in this preservation script.')
write(case/'result.json', result)
print(json.dumps({k:v for k,v in result.items() if k!='controller_events'}), flush=True)
video(case, [{key:trace[key][i] for key in trace.files} for i in range(len(trace['sim_time']))])
