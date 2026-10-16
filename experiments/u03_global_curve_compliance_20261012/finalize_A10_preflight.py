"""Clarify the completed check's hold-end timestamps, no additional test."""
from pathlib import Path
import json
b=Path(__file__).resolve().parent;p=b/'A10_NATIVE_PREPARATION.json';z=json.loads(p.read_text())
for r in z['samples']:r.update(command_time_s=r['t'],ctrl_time_s=r['t']+.025)
z['timing']='q and C at command_time_s; limited ctrl after this 25ms hold at ctrl_time_s. Errors compare hold-end ctrl with held q/C, not instantaneous ctrl(t). Final command has an extra diagnostic hold ending235.025s; actual trial budget remains235s.'
z['worst_slew_row']=max(z['samples'],key=lambda r:r['raw_to_ctrl_deg'])
z['worst_native_front_row']=max(z['samples'],key=lambda r:r['raw_front_max_mm'])
clip=[r['t'] for r in z['samples'] if r['raw_max_deg']>88]
z['clip_time_range_s']=[min(clip),max(clip)] if clip else None
z['max_raw_front_error_mm']=max(r['raw_front_max_mm'] for r in z['samples'])
z['max_ctrl_front_error_mm']=max(r['ctrl_front_max_mm'] for r in z['samples'])
p.write_text(json.dumps(z,indent=2))
geo=json.loads((b/'A10_PREPARATION.json').read_text());out=dict(geometry=geo['geometric_bound_passed'],geometry_errors=geo['errors'],whole_rod_bound_mm=geo['worst_line_bound']['C_line_bound_mm'],native={k:v for k,v in z.items() if k not in ['samples','interface_event_times_s']},physical_steps=0,actual_success_not_verified=True)
(b/'A10_PREFLIGHT_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
