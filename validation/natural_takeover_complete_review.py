"""Continuous mission audit: approach, probes, transfer and continued rolling.

Truth scores the saved motion only. The one-second initialization exclusion is
applied once to the mission, never again at automatic takeover.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from natural_takeover_review import review as review_approach
from window_coordination_archive import delivery
from segmented_pole_experiment import write_json
from three_segment_review import context


def review(root,transfer_name='transfer'):
    review_approach(root)
    folder=root/transfer_name
    delivery_check=delivery(folder);write_json(folder/'delivery_check.json',delivery_check)
    _,child,m,_,_=context(folder)
    keys=('time','qpos','qvel','ctrl','error_deg','points_mm','torque')
    with np.load(root/'approach/trace.npz') as data:
        parent={key:data[key] for key in keys}
    with child:
        for key in ('time','qpos','qvel','ctrl'):
            assert np.allclose(parent[key][-1],child[key][0],atol=1e-10,rtol=0),key
        trace={key:np.concatenate([parent[key],child[key][1:]]) for key in keys}
    audit=json.loads((folder/'whole_crossing_audit.json').read_text(encoding='utf8'))
    end=trace['time'][-1]
    if audit['accepted_window']:
        end=parent['time'][-1]+audit['accepted_window']['end_seconds']
    active=(trace['time']>=trace['time'][0]+1)&(trace['time']<=end+1e-8)
    centers=(trace['points_mm'][:,:-1]+trace['points_mm'][:,1:])/2000
    speed=np.linalg.norm(np.diff(centers,axis=0)/np.diff(trace['time'])[:,None,None],axis=2)
    metrics=dict(module_peak_m_s=float(speed[active[1:]].max()),
        joint_peak_deg_s=float(np.degrees(abs(trace['qvel'][active][:,m.jnt_dofadr[m.actuator_trnid[:,0]]])).max()),
        error_peak_deg=float(abs(trace['error_deg'][active]).max()),
        saturation=float(np.mean(abs(trace['torque'][active])>=.99*m.actuator_forcerange[:,1])))
    quality=dict(module_speed=metrics['module_peak_m_s']<=.5,joint_speed=metrics['joint_peak_deg_s']<=90,
        tracking=metrics['error_peak_deg']<=15,saturation=metrics['saturation']<=.297)
    functional=audit['full_crossing_and_continued_rolling']
    result=dict(complete=bool(functional and all(quality.values())),functional=bool(functional),quality=quality,metrics=metrics,
        continuous_physical_boundary_checked=True,initial_second_excluded_once=True,
        mission_seconds=float(end-trace['time'][0]),transfer_delivery=delivery_check,
        limitations='Nominal dynamics and right-angle comparable-radius/gravity prior; calibration pauses simulated time. Single mission is not varied-initial-condition qualification.')
    write_json(root/'complete_review.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--transfer',default='transfer');args=parser.parse_args();review(args.root,args.transfer)
