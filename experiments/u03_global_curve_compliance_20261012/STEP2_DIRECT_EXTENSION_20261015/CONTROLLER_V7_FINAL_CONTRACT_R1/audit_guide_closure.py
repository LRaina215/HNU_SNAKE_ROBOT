"""Geometric endpoint/material closure at phase landmarks; no integration."""
import json
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner

reg=json.loads((BASE/'REGISTERED.json').read_text());z=np.load(BASE/'GIVEN_INITIAL.npz');plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),z['qpos'],z['ctrl'],reg['kinematic_solver']);plan.start_wrap(.889)
rows=[]
for beta in [1e-12,.2,.225,.5,1.]:
    for native in [False,True]:
        G,goal,T,n=plan.wrap_guide(beta,0.,native);rows.append(dict(beta=beta,native_C_targets=native,head_error_mm=float(np.linalg.norm(G[0]-goal)*1000),material_length_error_mm=float(abs(np.linalg.norm(np.diff(G,axis=0),axis=1)-.0645).max()*1000),all_finite=bool(np.isfinite(G).all())))
report=dict(simulation_steps=0,scope='Virtual G/C target endpoint closure only; no bridge clearance, physical tracking or capture claim',rows=rows,passed=all(x['head_error_mm']<.001 and x['material_length_error_mm']<1e-7 and x['all_finite'] for x in rows))
(BASE/'GUIDE_CLOSURE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
