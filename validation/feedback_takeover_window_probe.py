"""Replay causal geometry tracking with fixed versus acquired-coil windows.

Recorded previous control state selects the window; simulator root/contact
columns are scoring-only. This counterfactual replay is not a physical trial.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import pipe_frame,write_json
from feedback_takeover_live_tracking import EncoderPipeTracker


def run(folder,stop,modes):
    _,z,m,_,meta=context(folder)
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    plan=json.loads((folder/'plan.json').read_text(encoding='utf8'))
    qpos=z['qpos'];commands=z['ctrl'];times=z['time']-z['time'][0]
    qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    corner,source,target,_=pipe_frame(m);runs=[]
    ids=np.flatnonzero(times<=stop+1e-6)[::2]
    for mode in modes:
        q=qpos[0,qadr]
        load=np.clip(m.actuator_gainprm[:,0]*(commands[0]-q),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        tracker=EncoderPipeTracker(m,q,load,meta['settings']['radius'],True,True,mode=='confidence')
        rows=[];ci=0
        for n,i in enumerate(ids):
            while ci+1<len(controls) and controls[ci+1]['time']<=times[i]+1e-6:ci+=1
            info=controls[ci];count=8
            if mode=='acquired':
                prefix=plan['configuration']['initial_front']
                while prefix<20 and info['acquired_weight'][prefix]>.9:prefix+=1
                count=max(8,min(prefix+2,len(tracker.previous)))
            q=qpos[i,qadr]
            load=np.clip(m.actuator_gainprm[:,0]*(commands[i]-q),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
            if n:tracker.update(q,load,n%10==0,bool(np.any(np.asarray(info['weights'])[:,2]>1e-12)),count,np.asarray(info['weights'])[:,2])
            if n%10:continue
            est=tracker.estimate;rot=np.zeros(9);mujoco.mju_quat2Mat(rot,qpos[i,3:7]);rot=rot.reshape(3,3)
            angle=lambda a,b:float(np.degrees(np.arccos(np.clip(a@b,-1,1))))
            rows.append(dict(time=float(times[i]),front_points=count,source_unobserved=tracker.source_unobserved,
                source_candidates=tracker.source_candidate_count,target_error_deg=angle(rot@est['target'],target),
                source_error_deg=angle(rot@est['source'],source),
                target_offset_mm=float(np.linalg.norm(np.cross(rot@est['corner_mm']+1000*qpos[i,:3]-corner,target)))))
        runs.append(dict(mode=mode,rows=rows))
        print(folder.name,mode,'target maximum',max(r['target_error_deg'] for r in rows),'offset maximum',max(r['target_offset_mm'] for r in rows),flush=True)
    suffix='_confidence_probe.json' if 'confidence' in modes else '_window_probe.json'
    write_json(folder.parent/(folder.name+suffix),dict(runs=runs,
        limitation='Same recorded movements, prior controller state only; no physical feedback rollout or truth correction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--stop',type=float,default=90.)
    p.add_argument('--modes',nargs='+',choices=['fixed','acquired','confidence'],default=['fixed','acquired'])
    a=p.parse_args();run(a.folder,a.stop,a.modes)
