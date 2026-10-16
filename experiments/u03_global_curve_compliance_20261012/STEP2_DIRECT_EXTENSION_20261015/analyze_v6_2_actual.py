"""Read a completed V6.2 action and its actual restoration/task progress."""
from pathlib import Path
import json,ast,numpy as np
P=Path(__file__).resolve().parent;s=(P/'analyze_v6_1_actual.py').read_text(encoding='utf-8').replace('CONTROLLER_V6_1_TRANSMITTED_C_STATE','CONTROLLER_V6_2_FEASIBILITY_RESTORE').replace('total_actual_starts=35','total_actual_starts=36').replace('V6.1 one actual development run','V6.2 one actual development run')
exec(compile(s,str(P/'analyze_v6_1_actual.py'),'exec'),{'__file__':str(P/'analyze_v6_1_actual.py'),'__name__':'__main__'})
C=P/'CONTROLLER_V6_2_FEASIBILITY_RESTORE/T_UPDATE';z=np.load(C/'trace.npz');out=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'));metrics=[]
for value in z['planner_metrics']:
    try:metrics.append(ast.literal_eval(str(value)))
    except (SyntaxError,ValueError):metrics.append({})
out.update(sampled_restoration_attempts=sum(bool(m.get('restoration_attempted',False)) for m in metrics),sampled_accepted_restoration_steps=sum(m.get('solution_kind')=='feasibility_restoration' and m.get('accepted',False) for m in metrics),sampled_tracking_acceptance_count=sum(m.get('solution_kind')=='tracking' and m.get('accepted',False) for m in metrics),maximum_actual_head_departure_from_given_mm=float(np.linalg.norm(z['points'][:,0]-z['points'][0,0],axis=1).max()*1000),maximum_sent_geometric_head_task_error_mm=float(z['sent_geometric_head_tracking_error_m'].max()*1000),final_sent_geometric_head_task_error_mm=float(z['sent_geometric_head_tracking_error_m'][-1]*1000),restoration_count_scope='10Hz samples, not all25ms control commands',total_actual_starts=36)
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
