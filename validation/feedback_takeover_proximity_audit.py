"""Oracle-only diagnostic: is the CAD contact proxy itself geometrically sound?

This audit intentionally uses true axes and pose to isolate the distance query
from estimator error. No result or true geometry is returned to a controller.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from snake_control.encoder_pipe_proximity import EncoderPipeProximity


def run():
    root=ROOT/'experiments/segmented_pole_20261005/feedback_takeover';reports=[]
    for name in ('horizontal_encoder_live_whole_coil','vertical_encoder_live_whole_coil'):
        folder=root/name;_,z,m,_,_=context(folder);query=EncoderPipeProximity(m)
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];corner,source,target,radius=pipe_frame(m)
        time=z['time']-z['time'][0];rows=[]
        for i in range(0,len(time),20):
            if time[i]>100:break
            q=z['qpos'][i,qadr];points=z['points_mm'][i]
            gaps=query.gaps(q,points,corner,source,target,radius)
            load=np.clip(m.actuator_gainprm[:,0]*(z['ctrl'][i]-q),-2.355,2.355)
            adjacent=np.maximum(np.r_[0,abs(load)],np.r_[abs(load),0])
            predicted=(gaps<1.)&(adjacent[None,:]>.1)
            actual=z['normal_force'][i,:2]>.1
            rows.append(dict(time=float(time[i]),gaps_mm=gaps,predicted=predicted,actual=actual,
                false_positive=int(np.count_nonzero(predicted&~actual)),
                false_negative=int(np.count_nonzero(~predicted&actual)),
                actual_contact_maximum_gap_mm=float(gaps[actual].max()) if actual.any() else None))
        actual=np.asarray([r['actual'] for r in rows]);pred=np.asarray([r['predicted'] for r in rows])
        stats=dict(precision=float((actual&pred).sum()/pred.sum()),recall=float((actual&pred).sum()/actual.sum()),
            maximum_actual_gap_mm=max(r['actual_contact_maximum_gap_mm'] for r in rows if r['actual_contact_maximum_gap_mm'] is not None),
            zero_actual_source_false_positive_frames=sum(not r['actual'][0].any() and r['predicted'][0].any() for r in rows))
        print(name,stats,flush=True);reports.append(dict(name=name,stats=stats,rows=rows))
    write_json(root/'proximity_oracle_audit.json',dict(runs=reports,limitation='True geometry/pose used ONLY to audit virtual CAD gaps; not a deployable observer.'))


if __name__=='__main__':run()
