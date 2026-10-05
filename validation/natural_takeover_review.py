"""Read-only scoring of automatic approach; truth never returns to control."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco
from segmented_pole_experiment import write_json,pipe_frame
from snake_control.mujoco_checkpoint import load_checkpoint
from feedback_takeover_live_tracking import EncoderPipeTracker


def review(folder):
    with np.load(folder/'approach/trace.npz') as z:
        trace={k:z[k] for k in ('time','qpos','qvel','points_mm','normal_force','error_deg','torque','ctrl','applied_id')}
    commands=json.loads((folder/'approach/commands.json').read_text(encoding='utf8'))
    for command in commands:
        assert command['applied']
        frames=trace['applied_id']==command['id']
        # An initial q hold immediately followed by a ramp has no integration
        # frame. Its relay acknowledgement remains recorded; all others must.
        if np.any(frames):assert np.allclose(trace['ctrl'][frames],command['target'],atol=1e-12)
        else:assert command['time']==trace['time'][0]
    m,d,meta,_=load_checkpoint(folder/('candidate' if (folder/'candidate').exists() else 'initial'),verify_hashes=False)
    times=trace['time']-trace['time'][0];active=times>=1.
    centers=(trace['points_mm'][:,:-1]+trace['points_mm'][:,1:])/2000
    speeds=np.linalg.norm(np.diff(centers,axis=0)/np.diff(times)[:,None,None],axis=2)
    vadr=m.jnt_dofadr[m.actuator_trnid[:,0]];qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    report=dict(actual_commands=len(commands),duration=float(times[-1]),
        after_first_second=dict(module_peak_m_s=float(speeds[active[1:]].max()),
            joint_peak_deg_s=float(np.degrees(abs(trace['qvel'][active][:,vadr])).max()),
            maximum_error_deg=float(abs(trace['error_deg'][active]).max()),
            saturation=float(np.mean(abs(trace['torque'][active])>=.99*m.actuator_forcerange[:,1]))),
        candidate_contact_modules=[np.flatnonzero(row>.1) for row in trace['normal_force'][-1]],
        truth_is_scoring_only=True)
    if (folder/'candidate').exists():
        q=d.qpos[qa];load=np.clip(m.actuator_gainprm[:,0]*(d.ctrl-q),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        tracker=EncoderPipeTracker(m,q,load,meta['settings']['radius'])
        corner,source,target,radius=pipe_frame(m);rot=np.empty(9)
        mujoco.mju_quat2Mat(rot,d.qpos[3:7]);rot=rot.reshape(3,3)
        est=tracker.estimate
        report['candidate_estimate_scoring']=dict(
            source_error_deg=float(np.degrees(np.arccos(np.clip(est['source']@(rot.T@source),-1,1)))),
            target_error_deg=float(np.degrees(np.arccos(np.clip(est['target']@(rot.T@target),-1,1)))),
            corner_error_mm=float(np.linalg.norm(est['corner_mm']-rot.T@(corner-1000*d.qpos[:3]))),
            radius_error_mm=float(est['pipe_radius_mm']-radius))
    write_json(folder/'approach_review.json',report)
    print(json.dumps(report,default=lambda a:a.tolist(),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    review(p.parse_args().folder)
