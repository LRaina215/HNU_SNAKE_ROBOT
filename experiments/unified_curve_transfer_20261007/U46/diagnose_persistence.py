from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from snake_control.hook_pull_review_support import restore_saved_state

case = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('u46_frozen', case/'sources/snake_control/unified_curve_transfer.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
data = mujoco.MjData(model)
restore_saved_state(model, data, case/'resume_t3p95/final_state.npz')
controller = frozen.UnifiedCurveTransfer(model, data)
elapsed = controller.load_state(case/'resume_t3p95/controller_state.json')
trace = np.load(case/'trace.npz')
grid_before = controller.curve.arc_grid_u.copy()
try:
    controller.command(data, elapsed, trace['contact_force_N'][-1], trace['torque'][-1], trace['cylindrical_contact_force_N'][-1])
    raise RuntimeError('Expected original persistence exception was not reproduced')
except ValueError as error:
    curve = controller.curve
    spacing = np.diff(curve.arc_grid_u)
    lengths = np.linalg.norm(np.diff(curve.sample(curve.arc_grid_u), axis=0), axis=1)
    bad = np.flatnonzero(lengths <= 1e-10)
    report = dict(error=str(error), method='Private command from recovered full checkpoint; no physics step or control dispatch',
        previous_grid_count=len(grid_before), failed_grid_count=len(curve.arc_grid_u),
        minimum_parameter_spacing_m=float(spacing.min()), minimum_sample_segment_m=float(lengths.min()),
        degenerate_segments=[dict(index=int(i), parameters=curve.arc_grid_u[i:i+2].tolist(), spacing_m=float(spacing[i]), segment_m=float(lengths[i])) for i in bad])
    (case/'persistence_diagnosis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))
