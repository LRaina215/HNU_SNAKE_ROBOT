"""Score saved geometric trigger against physical support; never control input."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from segmented_pole_experiment import pipe_frame,write_json


def diagnose(folder):
    m=mujoco.MjModel.from_binary_path(str(folder/'initial/model.mjb'))
    corner,source,target,radius=pipe_frame(m)
    with np.load(folder/'approach/trace.npz') as data:
        z={key:data[key] for key in ('time','points_mm','normal_force','qpos')}
    hypotheses=json.loads((folder/'hypotheses.json').read_text(encoding='utf8'))
    h=np.eye(3)[np.argmin(abs(target))];h-=h@target*target;h/=np.linalg.norm(h);v=np.cross(target,h)
    rows=[]
    for row in hypotheses:
        i=int(np.argmin(abs(z['time']-row['time'])))
        points=z['points_mm'][i];centers=(points[:-1]+points[1:])/2
        ids=np.flatnonzero(np.mean(z['normal_force'][max(0,i-39):i+1,1]>.1,axis=0)>.25)
        theta=np.sort(np.mod(np.arctan2((centers[ids]-corner)@v,(centers[ids]-corner)@h),2*np.pi))
        span=2*np.pi-np.max(np.diff(np.r_[theta,theta[0]+2*np.pi])) if len(theta)>1 else 0.
        est=row['estimate'];rot=np.empty(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
        error=float(np.degrees(np.arccos(np.clip((rot@est['target'])@target,-1,1))))
        rows.append(dict(time=row['time'],phase=row['phase'],geometric_candidate=est['geometric_candidate'],
            quality=est['quality'],estimated_wrap=est['front']['wrap'],actual_target_modules=ids,
            actual_contact_span_deg=float(np.degrees(span)),target_axis_error_deg=error,
            actual_source_contact=bool(np.any(z['normal_force'][i,0]>.1))))
    distributed=[row for row in rows if row['actual_contact_span_deg']>90 and row['actual_source_contact']]
    report=dict(rows=rows,distributed_rows=distributed,truth_scoring_only=True,
        limitation='Distributed contacts are a diagnostic, not a proof of transfer feasibility.')
    write_json(folder/'candidate_diagnosis.json',report)
    print(folder.name,'samples',len(rows),'distributed',len(distributed),'candidates',sum(r['geometric_candidate'] for r in rows),flush=True)
    for row in distributed[:10]:print({k:v for k,v in row.items() if k!='actual_target_modules'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);diagnose(p.parse_args().folder)
