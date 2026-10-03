"""Complete the plan's adjacent-shell check without changing any model.

Reports convex-geom overlap separately from active parent-child contacts.
This cannot certify actual STL shell clearance or real hinge interference.
"""
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from pitch_adaptation_experiment_v5 import ROOT,SCENES,model_path
from pitch_adaptation_training_audit import BASE,sha,model_asset_hashes


def main():
    source=ROOT/'experiments/pitch_adaptation_20261003/stage0'
    with (source/'scan.csv').open(encoding='utf-8-sig',newline='') as f:cases=list(csv.DictReader(f))
    with np.load(source/'scan_targets.npz') as raw:targets=raw['targets_rad'];phases=raw['phases']
    model=mujoco.MjModel.from_xml_path(str(model_path('pipe_straight_small')));data=mujoco.MjData(model)
    robot=[i for i in range(model.ngeom) if model.geom_bodyid[i] and not (model.geom(i).name or '').startswith('pipe')]
    pairs=list(zip(robot[:-1],robot[1:]));pair_set={frozenset(p) for p in pairs}
    if len(pairs)!=20 or any(model.body_parentid[model.geom_bodyid[y]]!=model.geom_bodyid[x] for x,y in pairs):
        raise ValueError('Adjacent geoms are not the twenty direct parent-child links')
    data.qpos[:]=0.;data.qpos[3]=1.;mujoco.mj_forward(model,data)
    zero=[mujoco.mj_geomDistance(model,data,x,y,.1,None)*1000 for x,y in pairs]
    rows=[];distances=[];active_counts=[]
    for case,angles_by_phase in zip(cases,targets):
        measured=[];active=[]
        for angles in angles_by_phase:
            data.qpos[:]=0.;data.qpos[3]=1.;data.qpos[7:]=angles;mujoco.mj_forward(model,data)
            measured.append([mujoco.mj_geomDistance(model,data,x,y,.1,None)*1000 for x,y in pairs])
            active.append(sum(frozenset((int(c.geom1),int(c.geom2))) in pair_set for c in data.contact))
        measured=np.asarray(measured);distances.append(measured);active_counts.append(active)
        rows.append(dict(radius_mm=float(case['radius_mm']),candidate=case['candidate'],alpha_deg=float(case['alpha_deg']),
                         adjacent_minimum_convex_geom_distance_mm=float(measured.min()),
                         adjacent_overlap_observations=int((measured<0).sum()),maximum_active_adjacent_contacts=max(active)))
    parent_filter={scene:not bool(mujoco.MjModel.from_xml_path(str(model_path(scene))).opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)) for scene in SCENES}
    out=BASE/'adjacent_geometry';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'adjacent_distances.npz',distance_mm=np.asarray(distances),active_contacts=np.asarray(active_counts),
                        phases=phases,geom_pairs=np.asarray(pairs),zero_angle_distance_mm=np.asarray(zero))
    with (out/'adjacent_scan.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=dict(cases=len(rows),phase_samples=len(phases),adjacent_pairs=len(pairs),
                zero_angle_minimum_convex_geom_distance_mm=float(min(zero)),
                minimum_convex_geom_distance_mm=float(np.min(distances)),maximum_active_adjacent_contacts=int(np.max(active_counts)),
                parent_child_collision_filter_enabled=parent_filter,hardware_shell_clearance_proven=False,
                simulation_scope='original ideal hinge model excludes parent-child collision; nonadjacent/model-pipe behavior may be compared, actual hinge-shell clearance remains unverified',
                official_reference='https://mujoco.readthedocs.io/en/3.3.0/computation/index.html#collision-detection',
                inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'scan.csv',source/'scan_targets.npz',model_path('pipe_straight_small'),Path(__file__))},
                model_assets=model_asset_hashes(),rows=rows)
    (out/'adjacent_geometry.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','inputs_sha256','model_assets')},indent=2))


if __name__=='__main__':main()
