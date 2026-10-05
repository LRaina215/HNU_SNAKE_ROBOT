"""Check that changing the expression frame does not change position targets.

This is a coordinate implementation check with known geometry, not evidence
that the geometry can be observed. No simulation advance or robot command.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,pipe_frame,write_json
from snake_control.mujoco_checkpoint import load_checkpoint
from feedback_takeover_live_tracking import EncoderPipeTracker
from snake_control.encoder_pipe_proximity import EncoderPipeProximity
from spatial_takeover_experiment import points_for_angles
from snake_control.target_frame_acquisition import TargetFrameAcquisition
from snake_control.window_coordinated_winding import WindowOptions
from snake_control.three_segment_compliance import ThreeSegmentOptions
from snake_control.traveling_compliant_winding import FormationOptions
from snake_control.pole_compliance import ComplianceParameters


def run():
    rows=[];out=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'
    for name,cp in [('horizontal_target_frame_conserved',HORIZONTAL),('vertical_target_frame_conserved',VERTICAL)]:
        m,d,meta,_=load_checkpoint(cp,verify_hashes=False)
        g=json.loads((out/name/'geometry.json').read_text(encoding='utf8'))[0]
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];q=d.qpos[qadr].copy()
        c,s,t,r=pipe_frame(m);phase=meta['settings']['phase']
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,d.qpos[3:7]);rotation=rotation.reshape(3,3)
        translation=1000*d.qpos[:3]
        head_g=dict(g,points_mm=(np.asarray(g['points_mm'])-translation)@rotation)
        own=mujoco.MjData(m);own.qpos[:7]=[0,0,0,1,0,0,0]
        targets=[];parts=[]
        for geom,corner,source,target,fk in [
            (g,c,s,t,lambda a:points_for_angles(m,d,a)),
            (head_g,rotation.T@(c-translation),rotation.T@s,rotation.T@t,lambda a:points_for_angles(m,own,a))]:
            controller=TargetFrameAcquisition(geom,ComplianceParameters((.01,.02),.4057779073,False),
                d.ctrl.copy(),phase,corner,source,target,r,measured=q,forward_points=fk,
                pitch_mm=meta['settings']['pitch'],options=ThreeSegmentOptions(follow_seconds=.06),
                formation=FormationOptions(),window=WindowOptions(tail_mode='same',supported_recovery=.35,support_centers=True),
                support_error_limit=m.actuator_forcerange[:,1]/m.actuator_gainprm[:,0],
                local_contact_projection=True,conserve_middle=True)
            load=np.clip(m.actuator_gainprm[:,0]*(d.ctrl-q),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
            proxy,_=EncoderPipeProximity(m).observe(q,np.asarray(geom['points_mm']),corner,source,target,r,load)
            command,_=controller.compute(geom,q,phase-.0375,.2,[proxy,proxy])
            targets.append(command);parts.append([controller.initial_front,controller.initial_rear])
        rows.append(dict(run=name,maximum_target_difference_deg=float(np.degrees(abs(targets[1]-targets[0])).max()),partitions=parts))
    write_json(out/'head_frame_control_audit.json',dict(rows=rows,
        purpose='Coordinate covariance only. Known axes are transformed for this diagnostic; no estimator claim.'))
    print(rows)


if __name__=='__main__':run()
