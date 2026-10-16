"""Read current logged contacts without touching the running process."""
from pathlib import Path
import sys,json,numpy as np
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';sys.path.insert(0,str(B))
from contact_sensor import known_source
reg=json.loads((B/'REGISTERED.json').read_text());path=B/'T_UPDATE/contact_evidence.jsonl'
with path.open() as handle:
    lines=handle.readlines()
row=None
for line in reversed(lines):
    try:row=json.loads(line);break
    except json.JSONDecodeError:pass
if row is not None:
    for c in row['contacts']:
        for key in ['point','normal','world_force_N']:c[key]=np.asarray(c[key],float)
    source=[c for c in row['contacts'] if known_source(c,reg['source_known'])];other=[c for c in row['contacts'] if not known_source(c,reg['source_known'])]
    print(json.dumps(dict(t=row['t'],known_source_contact_materials=[c['material'] for c in source],known_source_vertical_N=sum(c['world_force_N'][2] for c in source),non_source_contact_materials=[c['material'] for c in other],probe_requested=row.get('probe_requested'),probe_transmitted=row.get('probe_transmitted')),indent=2))
