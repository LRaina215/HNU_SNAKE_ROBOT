"""Read-only CAD-gap/contact comparison; actual contacts are scoring labels."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json,pipe_frame
from snake_control.mujoco_checkpoint import load_checkpoint
from snake_control.encoder_pipe_proximity import EncoderPipeProximity

BASE=ROOT/'experiments/segmented_pole_20261005/support_transfer'
OUT=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'


def run():
    summaries=[]
    for name,cp in [('radial_front_relative_braking',HORIZONTAL),('vertical_initial_front_reference',VERTICAL)]:
        m,d,meta,_=load_checkpoint(cp,verify_hashes=False);corner,source,target,radius=pipe_frame(m)
        estimator=EncoderPipeProximity(m);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        with np.load(BASE/name/'trace.npz') as z:
            qpos=z['qpos'][::20];points=z['points_mm'][::20];torque=z['torque'][::20]
            truth=z['normal_force'][::20,:2]>.1;t=z['time'][::20]-z['time'][0]
        gaps=[];hypotheses=[];elapsed=[]
        for q,p,load in zip(qpos,points,torque):
            start=time.perf_counter()
            proxy,info=estimator.observe(q[qadr],p,corner,source,target,radius,load)
            gaps.append(info['cad_gap_mm']);hypotheses.append(proxy[:2]>.1);elapsed.append(time.perf_counter()-start)
        gaps=np.asarray(gaps);hypotheses=np.asarray(hypotheses)
        for label,pred in [('cad_gap',gaps<1.),('cad_gap_adjacent_load',hypotheses)]:
            valid=t>=1;tp=int(np.sum(pred[valid]&truth[valid]));fp=int(np.sum(pred[valid]&~truth[valid]));fn=int(np.sum(~pred[valid]&truth[valid]))
            row=dict(run=name,method=label,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                pipe_presence_agreement=float(np.mean(np.any(pred[valid],axis=2)==np.any(truth[valid],axis=2))),
                median_query_seconds=float(np.median(elapsed)),max_query_seconds=float(max(elapsed)))
            summaries.append(row);print(row,flush=True)
        np.savez_compressed(OUT/(name+'_cad_gap_shadow.npz'),time=t,gap_mm=gaps,
            hypothesis=hypotheses,scoring_contact=truth,joint_load=torque)
    write_json(OUT/'cad_gap_shadow.json',dict(rows=summaries,
        stage='Contact-label removal only; axes and pose expression remain known in this diagnostic.',
        inputs='Encoder angles, known robot CAD, provided pipe hypothesis, adjacent motor load.',
        physical_model_changed=False,
        query_model='Private non-stepping model copy. Native GJK distance query; synthetic 2m half-lines.',
        limitation='Geometric/load potential support is not normal force or force closure. Scoring is in-sample.'))


if __name__=='__main__':run()
