"""Replay sticking-contact odometry without reading physical contacts/pose."""
import argparse
import json
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from feedback_takeover_live_tracking import EncoderPipeTracker
from feedback_takeover_contact_odometry import transport_contacts
from feedback_takeover_encoder_tracking import local_transport,transport
from feedback_takeover_cad_refinement import refine,refine_target


def run(stop,normal_only):
    root=ROOT/'experiments/segmented_pole_20261005/feedback_takeover';runs=[]
    for name in ('horizontal_encoder_live_whole_coil','vertical_encoder_live_whole_coil'):
        folder=root/name;_,z,m,_,meta=context(folder)
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];times=z['time']-z['time'][0]
        actual_corner,_,actual_target,_=pipe_frame(m)
        for use_refinement in (False,True):
            q=z['qpos'][0,qadr];load=np.clip(m.actuator_gainprm[:,0]*(z['ctrl'][0]-q),-2.355,2.355)
            tracker=EncoderPipeTracker(m,q,load,meta['settings']['radius'],True,True)
            estimate=tracker.estimate;before=q.copy();points=tracker.previous.copy();rows=[];ci=0;fallbacks=0
            for n,i in enumerate(np.flatnonzero(times<=stop+1e-9)[::2]):
                if n==0:continue
                while ci+1<len(controls) and controls[ci+1]['time']<=times[i]+1e-6:ci+=1
                source=bool(np.any(np.asarray(controls[ci]['weights'])[:,2]>1e-12))
                q=z['qpos'][i,qadr];current=tracker.points(q)
                predicted,info=transport_contacts(tracker.query,before,q,points,estimate,load,full_target=not source,normal_only=normal_only)
                if not info['observable']:
                    predicted=local_transport(points,current,estimate) if source else transport(points,current,estimate)
                    fallbacks+=1
                load=np.clip(m.actuator_gainprm[:,0]*(z['ctrl'][i]-q),-2.355,2.355)
                if use_refinement and n%10==0:
                    predicted=refine(tracker.query,q,current,predicted,load,fixed_radius=True) if source else refine_target(tracker.query,q,current,predicted,load)
                estimate=predicted;before=q.copy();points=current.copy()
                if n%10:continue
                # Physical pose is scoring-only, after the completed estimate.
                rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
                error=float(np.degrees(np.arccos(np.clip((rot@estimate['target'])@actual_target,-1,1))))
                offset=float(np.linalg.norm(np.cross(rot@estimate['corner_mm']+1000*z['qpos'][i,:3]-actual_corner,actual_target)))
                rows.append(dict(time=float(times[i]),target_error_deg=error,target_offset_mm=offset,odometry=info))
                if n%1000==0:print(name,use_refinement,float(times[i]),error,offset,flush=True)
            run=dict(name=name,refinement=use_refinement,fallbacks=fallbacks,rows=rows);runs.append(run)
            print(name,use_refinement,'max error/offset',max(r['target_error_deg'] for r in rows),max(r['target_offset_mm'] for r in rows),'fallbacks',fallbacks,flush=True)
            write_json(root/('normal_odometry_probe.json' if normal_only else 'contact_odometry_probe.json'),dict(runs=runs,normal_only=normal_only,limitation='Recorded q and commands only; contact constraint is a hypothesis. Offline estimate, not a crossing rollout.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stop',type=float,default=90.)
    p.add_argument('--normal-only',action='store_true')
    args=p.parse_args();run(args.stop,args.normal_only)
