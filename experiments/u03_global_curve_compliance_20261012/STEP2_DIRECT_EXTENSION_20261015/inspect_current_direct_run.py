"""Read the most recent complete evidence row without altering a live run."""
from pathlib import Path
import sys,json,argparse
ap=argparse.ArgumentParser();ap.add_argument('branch',nargs='?',default='CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS');args=ap.parse_args()
P=Path(__file__).resolve().parent;B=P/args.branch;C=B/'T_UPDATE'
if (C/'result.json').exists():print((C/'result.json').read_text(encoding='utf-8'));raise SystemExit
file=C/'contact_evidence.jsonl'
if not file.exists():print(json.dumps(dict(case_created=C.exists(),integration_started=(C/'DYNAMICS_START.json').exists(),complete_evidence_not_yet_available=True)));raise SystemExit
with file.open('rb') as f:
    f.seek(0,2);size=f.tell();f.seek(max(0,size-131072));lines=f.read().decode('utf-8',errors='ignore').splitlines()
row=None
for line in reversed(lines):
    try:row=json.loads(line);break
    except json.JSONDecodeError:pass
if row is None:print(json.dumps(dict(integration_started=(C/'DYNAMICS_START.json').exists(),complete_evidence_not_yet_available=True)));raise SystemExit
sys.path.insert(0,str(B))
import numpy as np
from contact_sensor import known_source
import contact_sensor
from geometry import Geometry
reg=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));g=Geometry.from_dict(reg['conditions']['DIRECTION']);source=reg['source_known'];fz=[0.,0.,0.];unknown=[]
lower=contact_sensor.receiver_axial_lower(g,source) if hasattr(contact_sensor,'receiver_axial_lower') else -.05
for ct in row['contacts']:
    for k in ['point','normal','world_force_N']:ct[k]=np.asarray(ct[k],float)
    src=known_source(ct,source);ax,rad,rho=g.coordinates(ct['point']);alignment=float(rad@ct['normal']/max(float(rho),1e-9));candidate=not src and lower<ax<1. and abs(rho-g.radius)<.012 and abs(g.axis@ct['normal'])<.25 and alignment>.90
    kind=0 if src else 1 if candidate else 2;fz[kind]+=float(ct['world_force_N'][2])
    if not src and ct['material']<8:unknown.append(ct['material'])
out=dict(integration_started=(C/'DYNAMICS_START.json').exists(),latest_saved_evidence_t_s=row['t'],known_source_Fz_N=fz[0],rough_receiver_candidate_Fz_N=fz[1],other_Fz_N=fz[2],current_front_non_source_materials=sorted(set(unknown)),probe_requested_per_m=row['probe_requested'],probe_transmitted_per_m=row['probe_transmitted'],evidence_input_per_m=row['probe_evidence_input'],note='Online rough/source classification only; not true receiver outcome evaluation.')
print(json.dumps(out,ensure_ascii=False))
