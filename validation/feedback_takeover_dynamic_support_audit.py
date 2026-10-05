"""Score support hypotheses from saved private filter states against truth.

The private model reconstructs contacts using inferred geometry, angles,
velocities, gravity and issued commands. Physical contacts are read only after
the inferred force labels have been produced, for precision/recall scoring.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,write_json
from feedback_takeover_dynamic_observer_probe import virtual_model
from snake_control.encoder_pipe_proximity import EncoderPipeProximity


def run(file):
    record=json.loads(file.read_text(encoding='utf8'));scene=record['scene'];root=file.parent
    fit=json.loads((root/record['fit_file']).read_text(encoding='utf8'))
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,robot,_,meta=context(folder);times=z['time']-z['time'][0];commands=z['ctrl'];truth=z['normal_force']
    models={};data={}
    for label,extension in [('none',None),('source','source'),('target','target')]:
        models[label]=virtual_model(robot,fit['estimate'],fit['gravity'],meta['source_model'],extension);data[label]=mujoco.MjData(models[label])
    slots=EncoderPipeProximity(robot).pipe_slots
    body_to_module={robot.body(name).id:i for i,name in enumerate(['head']+['link'+str(i) for i in range(1,20)]+['tail'])}
    rows=[];tp=fp=fn=0;zero_true_source_false=0
    for row in record['rows']:
        label=row.get('topology','none');m=models[label];d=data[label]
        d.qpos[:]=row['estimated_qpos'];d.qvel[:]=row['estimated_qvel']
        i=int(np.argmin(abs(times-row['time'])));d.ctrl[:]=commands[i];mujoco.mj_forward(m,d)
        predicted=np.zeros((2,21))
        for j in range(d.ncon):
            contact=d.contact[j]
            for axis,slot in enumerate(slots):
                if slot not in (contact.geom1,contact.geom2):continue
                other=contact.geom2 if contact.geom1==slot else contact.geom1
                module=body_to_module.get(int(m.geom_bodyid[other]))
                if module is None:continue
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,j,wrench);predicted[axis,module]+=max(0.,wrench[0])
        # Scoring begins here; real contacts never alter predicted labels.
        positive=predicted>.1;actual=truth[i,:2]>.1
        tp+=int(np.sum(positive&actual));fp+=int(np.sum(positive&~actual));fn+=int(np.sum(~positive&actual))
        zero_true_source_false+=int(not np.any(actual[0]) and np.any(positive[0]))
        rows.append(dict(time=row['time'],topology=label,predicted_normal_N=predicted,
            true_positive=int(np.sum(positive&actual)),false_positive=int(np.sum(positive&~actual)),false_negative=int(np.sum(~positive&actual)),
            scoring_actual_normal_N=truth[i,:2],innovation_rms_deg=row['innovation_rms_deg']))
    summary=dict(precision=tp/(tp+fp),recall=tp/(tp+fn),tp=tp,fp=fp,fn=fn,
        zero_true_source_but_predicted_source_frames=zero_true_source_false)
    write_json(file.with_name(file.stem+'_support_audit.json'),dict(input=file.name,summary=summary,rows=rows,
        limitation='Private-model force estimates are not measured contact force or validated force closure. Actual labels only score frozen estimates; frames are sampled at 0.2 s.'))
    print(scene,summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('file',type=__import__('pathlib').Path);run(p.parse_args().file)
