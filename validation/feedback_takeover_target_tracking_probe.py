"""Replay the final-transfer segment to separate line fitting and frame drift."""
import json
import argparse
import copy
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from feedback_takeover_encoder_tracking import transport
from feedback_takeover_cad_refinement import refine_target
from encoder_pipe_proximity import EncoderPipeProximity
from encoder_pipe_geometry import cylinder


def run(modes,start_at=None,stop=115.,output='single_target_registration_probe.json',half_line=False):
    root=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'
    folder=root/'horizontal_encoder_live_conserved'
    result,z,m,_,_=context(folder)
    c=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    first=next(r for r in c if not np.any(np.asarray(r['weights'])[:,2]>1e-12)) if start_at is None else min(c,key=lambda r:abs(r['time']-.2-start_at))
    start=first['time']-.2
    qpos=z['qpos'];commands=z['ctrl'];times=z['time']-z['time'][0]
    ids=np.flatnonzero((times>=start-1e-6)&(times<=stop+1e-6))[::2]
    qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];own=mujoco.MjData(m)
    own.qpos[:7]=[0,0,0,1,0,0,0];query=EncoderPipeProximity(m)
    corner,_,target,_=pipe_frame(m)
    def points(q):
        own.qpos[qadr]=q;mujoco.mj_kinematics(m,own)
        return 1000*np.vstack([own.site('s_head').xpos,*[own.joint('joint'+str(i)).xanchor for i in range(1,21)],own.site('s_tail').xpos])
    path=root/output
    summaries=[r for r in json.loads(path.read_text(encoding='utf8'))['runs'] if r['registration'] not in modes] if path.exists() else []
    for mode in modes:
        estimate={k:np.asarray(v) if isinstance(v,list) else v for k,v in copy.deepcopy(first['geometry_observer']['estimate']).items()}
        previous=points(qpos[ids[0],qadr]);rows=[]
        for n,i in enumerate(ids):
            q=qpos[i,qadr];p=points(q)
            predicted=transport(previous[:8] if mode=='front' else previous,p[:8] if mode=='front' else p,estimate)
            load=np.clip(m.actuator_gainprm[:,0]*(commands[i]-q),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
            if n%10==0 and mode=='coil_fit':
                # Two free-end groups are already shaped separately by the
                # controller. The long interior coil supplies a fresh line,
                # rather than accumulating a frame prediction from the head.
                fit=cylinder(p[4:-4],predicted['pipe_radius_mm']+29.)
                axis=fit['axis']
                if axis@predicted['target']<0:axis=-axis
                predicted=dict(predicted,target=axis,corner_mm=fit['axis_point_mm'])
            estimate=refine_target(query,q,p,predicted,load,not half_line) if n%10==0 else predicted
            previous=p.copy()
            if n%10:continue
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,qpos[i,3:7]);rot=rot.reshape(3,3)
            angle=float(np.degrees(np.arccos(np.clip((rot@estimate['target'])@target,-1,1))))
            offset=float(np.linalg.norm(np.cross(rot@estimate['corner_mm']+1000*qpos[i,:3]-corner,target)))
            rows.append(dict(time=float(times[i]),target_error_deg=angle,target_offset_mm=offset))
        summaries.append(dict(registration=mode,rows=rows))
        print(mode,'start',start,'final',rows[-1],'max_angle',max(r['target_error_deg'] for r in rows),flush=True)
    write_json(path,dict(runs=summaries,start=start,stop=stop,half_line=half_line,
        interpretation='Recorded motions only; initial estimate from the causal trial, not true geometry. Truth scores but never corrects replay estimates.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes',nargs='+',choices=['front','whole','coil_fit'],default=['front','whole','coil_fit'])
    p.add_argument('--start',type=float);p.add_argument('--stop',type=float,default=115.)
    p.add_argument('--output',default='single_target_registration_probe.json');p.add_argument('--half-line',action='store_true')
    a=p.parse_args();run(a.modes,a.start,a.stop,a.output,a.half_line)
