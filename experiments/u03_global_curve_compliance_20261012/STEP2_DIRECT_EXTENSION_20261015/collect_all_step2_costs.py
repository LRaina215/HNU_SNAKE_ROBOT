"""Combine historical ledger and real direct-extension starts, without reruns."""
from pathlib import Path
import json,csv,datetime
P=Path(__file__).resolve().parent;prior=json.loads((P/'PRIOR_STARTED_RUN_COSTS.json').read_text(encoding='utf-8'));rows=list(prior['rows']);direct=[]
for start in sorted(P.glob('CONTROLLER_*/*/DYNAMICS_START.json')):
    C=start.parent;file=C/'result.json';r=json.loads(file.read_text(encoding='utf-8')) if file.exists() else {};reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'))
    metric_file=C/'FINAL_CONTRACT_METRICS.json';metrics=json.loads(metric_file.read_text()) if metric_file.exists() else {};rolling=reg.get('requirements')=='STEP2_ROLLING_SCOPE_CLARIFIED_20261016'
    row=dict(group=reg['version'],case=C.name,result_file=str(file),scope=('Frozen final four: original Step1-t0 Step2 prior/G/C' if reg.get('formal_four_case_batch') else 'Original Step1-t0 coordinated Step2 prior/G/C development') if rolling else 'Historical all-source direct-extension diagnostic under superseded interpretations',stop=r.get('stop','running'),simulation_s=r.get('sim_time'),wall_s=r.get('wall_s'),reported_success=r.get('success'),counts_as_current_direct_step2_success=bool(rolling and metrics.get('action_chain_complete',False)),first_raw_front_contact_t=r.get('first_raw_front_contact_t'),first_reliable_t=r.get('first_reliable_t'),first_G_write_t=r.get('first_G_write_t'));direct.append(row);rows.append(row)
out=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),scope='All preserved actual Step2 starts, including historical post-wrap partial starts. Offline analysis/IK cost is not counted as integration.',actual_starts=len(rows),legacy_source_starts=prior['prior_source_starts'],legacy_postwrap_starts=prior['prior_postwrap_starts'],direct_extension_starts=len(direct),cumulative_simulation_s=sum(r['simulation_s'] or 0 for r in rows),summed_per_run_wall_s=sum(r['wall_s'] or 0 for r in rows),current_full_action_successes=sum(r.get('counts_as_current_direct_step2_success',False) for r in direct),rows=rows)
out['runs_with_pending_final_cost']=sum(r['simulation_s'] is None or r['wall_s'] is None for r in rows)
out['cost_totals_scope']='Reported completed-run durations only; active start is counted but its unfinished cost is pending.'
(P/'ALL_STEP2_STARTED_RUN_COSTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (P/'ALL_STEP2_STARTED_RUN_COSTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False,indent=2))
