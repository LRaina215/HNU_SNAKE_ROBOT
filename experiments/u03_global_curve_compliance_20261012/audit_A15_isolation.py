"""Frozen source/clock/init audit from A14/A15 saved states, no SHA/dynamics."""
from physics import *
import json
a=np.load(BASE/'A14/trace.npz');b=np.load(BASE/'A15/trace.npz');rows={}
for key in ['t','S','chi','roll_phase','supply_distance_m','source_chart_shift_m','source_interface_x_m']:
    rows[key]=dict(exact_equal=bool(np.array_equal(a[key],b[key])),max_abs_delta=float(np.max(abs(a[key]-b[key]))))
ix=a['t']<.5;jx=b['t']<.5;settle={key:bool(np.array_equal(a[key][ix],b[key][jx])) for key in ['qpos','qvel','ctrl','points','forces']}
out=dict(actual_trial_count=1,executed_target_radius_m=.0955,radius90p5_physical_trial_performed=False,initial_bytes_equal=(BASE/'A14/initial.npz').read_bytes()==(BASE/'A15/initial.npz').read_bytes(),model_bytes_equal=(BASE/'A14/model.xml').read_bytes()==(BASE/'A15/model.xml').read_bytes(),source_and_common_clock=rows,settle_actual_equal=settle,source_preparation_contract=json.loads((BASE/'A15_PREPARATION.json').read_text())['source_contract'])
(BASE/'A15/isolation_audit.json').write_text(json.dumps(out,indent=2));print(out)
