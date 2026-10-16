"""Read the completed V6.3 actual head-task and full action evidence."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;s=(P/'analyze_v6_2_actual.py').read_text(encoding='utf-8').replace('CONTROLLER_V6_2_FEASIBILITY_RESTORE','CONTROLLER_V6_3_HEAD_TASK_PROGRESS').replace('total_actual_starts=36','total_actual_starts=37').replace('V6.2 one actual development run','V6.3 one actual development run')
exec(compile(s,str(P/'analyze_v6_2_actual.py'),'exec'),{'__file__':str(P/'analyze_v6_2_actual.py'),'__name__':'__main__'})
C=P/'CONTROLLER_V6_3_HEAD_TASK_PROGRESS/T_UPDATE';z=np.load(C/'trace.npz');out=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'));out.update(maximum_observed_head_task_error_mm=float(z['actual_head_task_error_m'].max()*1000),final_observed_head_task_error_mm=float(z['actual_head_task_error_m'][-1]*1000),final_phase_readiness=bool(z['geometric_progress_ready_next_command'][-1]),total_actual_starts=37)
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
