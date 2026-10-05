"""Check the unified calibration API on the previously collected responses."""
import argparse
import json
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from feedback_takeover_response_calibration import calibrate


def run(scene):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine';at=0. if scene=='horizontal' else 2.
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,m,_,meta=context(folder);controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    initial=min(controls,key=lambda r:abs(r['time']-.2-at))['geometry_observer']['estimate']
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials']
    records=next(r['records'] for r in trials if r['scene']==scene and r['snapshot_seconds']==at and r['pattern']=='multisine' and not r['target_removed'])
    fit=calibrate(m,meta['source_model'],initial,[dict(time=r['time'],q=r['q'],command=r['command']) for r in records])
    # Truth only scores the returned, already selected result.
    i=int(np.argmin(abs(z['time']-z['time'][0]-at)));rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
    c,_,t,_=pipe_frame(m);est=fit['estimate'];g=fit['gravity']
    scores=dict(target_error_deg=float(np.degrees(np.arccos(np.clip((rot@est['target'])@t,-1,1)))),
        target_line_offset_mm=float(np.linalg.norm(np.cross(rot@est['corner_mm']+1000*z['qpos'][i,:3]-c,t))),
        gravity_error_deg=float(np.degrees(np.arccos(np.clip((rot@g)@m.opt.gravity/9.81**2,-1,1)))))
    write_json(root/('response_calibration_'+scene+'.json'),dict(fit,scene=scene,time=at,scoring=scores))
    print(scene,scores,'heldout',fit['held_out_rms_deg'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['horizontal','vertical'],required=True);run(p.parse_args().scene)
