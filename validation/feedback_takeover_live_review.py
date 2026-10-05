"""Score already-recorded causal estimates against held-out simulator truth."""
import argparse
import json
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context


def run(folder):
    result,z,m,_,_=context(folder)
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    corner,source,target,radius=pipe_frame(m)
    t=z['time']-z['time'][0];rows=[]
    qpos=z['qpos'];forces=z['normal_force'];regions=z['region']
    for info in controls:
        obs=info['geometry_observer'];est=obs['estimate'];at=info['time']-result['step_seconds']
        i=int(np.argmin(abs(t-at)));q=qpos[i]
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q[3:7]);rot=rot.reshape(3,3)
        target_hat=rot@est['target'];source_hat=rot@est['source']
        point_hat=rot@est['corner_mm']+1000*q[:3]
        angle=lambda x,y:float(np.degrees(np.arccos(np.clip(x@y,-1,1))))
        rows.append(dict(time=float(t[i]),target_axis_error_deg=angle(target_hat,target),
            source_axis_error_deg=angle(source_hat,source),
            target_line_offset_mm=float(np.linalg.norm(np.cross(point_hat-corner,target))),
            junction_error_mm=float(np.linalg.norm(point_hat-corner)),radius_error_mm=float(est['pipe_radius_mm']-radius),
            rear_active=bool(np.any(np.asarray(info['weights'])[:,2]>1e-12)),
            target_only=obs.get('target_only',False),inference_seconds=obs.get('inference_seconds'),
            actual_target_modules=int(np.count_nonzero(forces[i,1]>.1)),
            actual_source_modules=int(np.count_nonzero(forces[i,0]>.1)),
            measured_target_region=int(regions[i].sum())))
    selected=[]
    for at in [0,20,40,60,80,90,100,110,150,rows[-1]['time']]:
        row=min(rows,key=lambda x:abs(x['time']-at));selected.append(row);print(row)
    summary=dict(run=folder.name,selected=selected,rows=rows,
        scoring_only='Actual floating pose, pipe frame and contact labels are read after the trial, never corrected into its controller.')
    write_json(folder.parent/(folder.name+'_estimate_review.json'),summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=__import__('pathlib').Path)
    run(p.parse_args().folder)
