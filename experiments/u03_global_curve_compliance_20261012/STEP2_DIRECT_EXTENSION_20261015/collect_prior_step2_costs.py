"""Read prior actual run results without modifying their original records."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent;E=P.parent;F=E/'STEP2_FROM_SOURCE_20261015';S=E/'STEP2_20261014';cases=['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE'];rows=[]
groups=[('postwrap_R1_interrupted',S/'R1_SUPERSEDED_STARTS','Already-receiving-wrap start, historical partial-action diagnostic'),
    ('postwrap_R2',S,'Already-receiving-wrap start, cannot establish complete Step2'),
    ('source_R1',F,'Old given source pose, direct-extension suitability retracted')]
groups += [(name,F/folder,'Old given source pose, direct-extension suitability retracted') for name,folder in [
    ('source_R2','R2_SUPPLY_CHART'),('source_R3','R3_STANCE_CURVE'),('source_R4','R4_GEOMETRIC_C_LAYERING'),('source_R5','R5_CONSTRAINED_CURVATURE')]]
for group,folder,scope in groups:
    for case in cases:
        result=folder/case/'result.json'
        if not result.exists():continue
        value=json.loads(result.read_text(encoding='utf-8-sig'));rows.append(dict(group=group,case=case,result_file=str(result),scope=scope,
            stop=value.get('stop'),simulation_s=value.get('sim_time'),wall_s=value.get('wall_s'),reported_success=value.get('success'),
            counts_as_current_direct_step2_success=False,first_raw_front_contact_t=value.get('first_raw_front_contact_t'),first_reliable_t=value.get('first_reliable_t'),first_G_write_t=value.get('first_G_write_t')))
with (P/'PRIOR_STARTED_RUN_COSTS.csv').open('w',newline='',encoding='utf-8-sig') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=dict(scope='Prior actual Step2 runs only; original files remain unchanged. Offline kinematic work has0 integration and is not mixed into rollout seconds.',
    prior_actual_starts=len(rows),prior_source_starts=sum(r['group'].startswith('source') for r in rows),prior_postwrap_starts=sum(r['group'].startswith('postwrap') for r in rows),
    cumulative_simulation_s=sum(r['simulation_s'] or 0 for r in rows),cumulative_wall_s=sum(r['wall_s'] or 0 for r in rows),
    current_direct_version_actual_starts=0,current_direct_step2_action_successes=0,rows=rows)
(P/'PRIOR_STARTED_RUN_COSTS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
