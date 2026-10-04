"""Read recorded physics; report transients separately from initial capture."""
import argparse
import json
from pathlib import Path
import numpy as np
from segmented_pole_experiment import write_json
from three_segment_review import context
from traveling_winding_audit import evaluate


def metrics(folder, output=None):
    result,z,m,d,meta=context(folder)
    t=z['time']-z['time'][0]
    error=np.abs(z['error_deg'])
    velocity=np.abs(np.degrees(z['qvel'][:,m.jnt_dofadr[m.actuator_trnid[:,0]]]))
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    center_speed=np.r_[np.zeros((1,centers.shape[1])),
        np.linalg.norm(np.diff(centers,axis=0),axis=2)/np.diff(t)[:,None]]
    saturated=abs(z['torque'])>=2.355*.99
    source=np.any(z['normal_force'][:,0]>.1,axis=1)
    target=np.any(z['normal_force'][:,1]>.1,axis=1)
    audit=evaluate(z,m)
    end=audit['accepted_window']['end_seconds'] if audit['accepted_window'] else t[-1]
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    finish=next((c.get('time',(i+1)*result.get('step_seconds',.2))
                 for i,c in enumerate(controls) if c.get('finish_local')),None)
    windows={}
    for name,a,b in [('whole',0,end+.001),('after_capture',1,end+.001),
                     ('transfer',1,finish if finish is not None else end+.001),
                     ('completion',finish or t[-1]+1,(finish or t[-1])+4)]:
        mask=(t>=a)&(t<b)&(t<=end+1e-6)
        if not np.any(mask):continue
        windows[name]=dict(seconds=float(mask.sum()*.01),maximum_error_deg=float(error[mask].max()),
            mean_error_deg=float(error[mask].mean()),maximum_joint_speed_deg_s=float(velocity[mask].max()),
            p99_joint_speed_deg_s=float(np.quantile(velocity[mask],.99)),
            maximum_module_speed_mm_s=float(center_speed[mask].max()),
            saturated_joint_sample_fraction=float(np.mean(saturated[mask])),
            saturated_joint_seconds=float(saturated[mask].sum()*.01),
            target_contact_fraction=float(target[mask].mean()))
    report=dict(name=folder.name,outcome=result['outcome'],physical_seconds=float(t[-1]),
                metrics_end_seconds=float(end),
                complete=audit['full_crossing_and_continued_rolling'],
                final_region=audit['final_target_region_modules'],max_region=audit['maximum_target_region_modules'],
                final_wrap=audit['final_wrap_turns'],finish_seconds=finish,
                last_source_contact_seconds=float(t[source][-1]) if np.any(source) else None,
                windows=windows,initial_error_deg=float(error[0].max()))
    write_json(output or folder/'process_metrics.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--output',type=Path);args=p.parse_args()
    print(json.dumps(metrics(args.folder,args.output),indent=2))
