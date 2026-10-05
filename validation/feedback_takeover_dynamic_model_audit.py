"""Oracle-only check of virtual dynamics; NEVER used to initialize observers.

Uses true geometry, gravity and velocity to distinguish optimizer error from
unmodelled mechanics. The oracle result is diagnostic and is not an accepted
joint-only estimate or crossing trial.
"""
import json
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from feedback_takeover_dynamic_observer_probe import virtual_model,predict


def run():
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials'];rows=[]
    for scene,at in [('horizontal',0.),('vertical',2.)]:
        folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
        _,z,m,_,meta=context(folder);i=int(np.argmin(abs(z['time']-z['time'][0]-at)))
        records=next(r['records'] for r in trials if r['scene']==scene and r['snapshot_seconds']==at and r['pattern']=='multisine' and not r['target_removed'])
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
        c,s,t,r=pipe_frame(m)
        estimate=dict(corner_mm=rot.T@(c-1000*z['qpos'][i,:3]),source=rot.T@s,target=rot.T@t,pipe_radius_mm=r)
        gravity=rot.T@m.opt.gravity
        # MuJoCo free-joint angular velocity is in the body frame; translation
        # is world-relative and must be rotated into the virtual world.
        velocity=z['qvel'][i,:6].copy();velocity[:3]=rot.T@velocity[:3]
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];vadr=m.jnt_dofadr[m.actuator_trnid[:,0]]
        prediction=predict(virtual_model(m,estimate,gravity,meta['source_model']),records,velocity,z['qvel'][i,vadr])
        row=dict(scene=scene,prediction=prediction,oracle_estimate=estimate,oracle_gravity=gravity)
        rows.append(row);print(scene,prediction['rms_deg'],prediction['maximum_error_deg'],flush=True)
    write_json(root/'dynamic_model_oracle_audit.json',dict(runs=rows,limitation='Truth-based mechanics check only; forbidden as observer initialization or controller feedback.'))


if __name__=='__main__':run()
