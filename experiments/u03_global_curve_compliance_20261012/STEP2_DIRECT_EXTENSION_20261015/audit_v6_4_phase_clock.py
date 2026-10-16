"""Original phase maxima remain task-gated despite a small common command fraction."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;s=(P/'audit_v6_3_task_gate.py').read_text(encoding='utf-8').replace('CONTROLLER_V6_3_HEAD_TASK_PROGRESS','CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE').replace('HEAD_TASK_PROGRESS_AUDIT.json','NOMINAL_PHASE_CLOCK_AUDIT.json')
s=s.replace('control.alpha=alpha','control.alpha=alpha;control.last_command_transmission_factor=.1')
s=s.replace("previous_ready=control.geometric_progress_ready;fraction=control.last_command_transmission_factor;", "previous_ready=control.geometric_progress_ready;control.last_command_transmission_factor=.1;fraction=1.;")
s=s.replace("constructed_alpha=alpha,solver_stub=", "constructed_alpha=alpha,previous_common_fraction=.1,logged_phase_fraction=next_info['phase_progress_transmission_fraction'],solver_stub=")
s=s.replace("row['actual_pose_unchanged'] and row['source_given_command_error_rad']", "row['logged_phase_fraction']==1. and row['actual_pose_unchanged'] and row['source_given_command_error_rad']")
exec(compile(s,str(P/'audit_v6_3_task_gate.py'),'exec'),{'__file__':str(P/'audit_v6_3_task_gate.py'),'__name__':'__main__'})
