"""Inspect exact common-initialization matching; never relax the pairing gate."""
import json

import numpy as np

from pitch_adaptation_training_audit import BASE,ROOT,expected_jobs,label,load,sha


def main():
    plan=load(BASE/'training/training_plan.json');groups={}
    for job in expected_jobs(plan):
        case=BASE/'training'/label(job)
        if not (case/'result.json').exists() and job[1]=='D' and job[2]==1 and job[3]==0:case=BASE/'pilots'/label(job)
        result=load(case/'result.json')
        with np.load(case/'trace.npz') as raw:
            t=raw['sim_time'];indices=np.flatnonzero(np.isclose(t,2.2,atol=1e-9));i=int(indices[0])
            q=raw['qpos'][i];ctrl=raw['ctrl'][i]
            earlier=int(np.argmin(abs(t-.2)))
            item=dict(source_directory=str(case.relative_to(ROOT)),qpos=q.tolist(),ctrl=ctrl.tolist(),
                      native_apply_ctrl=raw['ctrl'][earlier].tolist(),initial_sim_time=float(t[i]),
                      initial_completed_steps=int(raw['completed_steps'][i]),initial_roll_phase=float(raw['roll_phase'][i]),
                      cancellation_retries=result['canceled_feedback_retries'],result_sha256=sha(case/'result.json'))
        groups.setdefault((job[0],job[2]),[]).append(item)
    summaries=[]
    for (scene,seed),items in groups.items():
        q=np.asarray([r['qpos'] for r in items]);d=np.max(abs(q[:,None]-q[None,:]),axis=2)
        reference=int(np.argmin(d.sum(axis=0)));ref=q[reference];different=[]
        for r,pose in zip(items,q):
            delta=pose-ref;r['maximum_qpos_difference']=float(max(abs(delta)))
            r['maximum_joint_difference_rad']=float(max(abs(delta[7:])))
            r['root_position_difference_m']=float(np.linalg.norm(delta[:3]))
            r['maximum_ctrl_difference_rad']=float(max(abs(np.asarray(r['ctrl'])-items[reference]['ctrl'])))
            if np.any(delta!=0):different.append(r)
        for item in different:
            with np.load(ROOT/item['source_directory']/'trace.npz') as raw,np.load(ROOT/items[reference]['source_directory']/'trace.npz') as baseline:
                rt,ri=np.unique(raw['sim_time'],return_index=True);bt,bi=np.unique(baseline['sim_time'],return_index=True)
                rt_mask=rt<=2.2+1e-9;bt_mask=bt<=2.2+1e-9
                if not np.allclose(rt[rt_mask],bt[bt_mask],atol=1e-10,rtol=0):raise ValueError('Initialization sampling grids differ')
                poses=raw['qpos'][ri[rt_mask]];reference_poses=baseline['qpos'][bi[bt_mask]]
                controls=raw['ctrl'][ri[rt_mask]];reference_controls=baseline['ctrl'][bi[bt_mask]]
                difference=np.max(abs(poses-reference_poses),axis=1);first=np.flatnonzero(difference>0)
                item['first_initial_pose_divergence_s']=float(rt[rt_mask][first[0]]) if len(first) else None
                item['maximum_initial_control_difference_rad']=float(np.max(abs(controls-reference_controls)))
                item['first_initial_pose_difference']=float(difference[first[0]]) if len(first) else None
        summary=dict(scene=scene,seed=seed,cases=len(items),reference=items[reference]['source_directory'],
                     maximum_qpos_difference=float(d.max()),nonmatching=different,all_cases=items)
        summaries.append(summary)
    out=BASE/'initial_diagnostic';out.mkdir(parents=True,exist_ok=True)
    (out/'common_initialization.json').write_text(json.dumps(dict(pairing_gate_relaxed=False,groups=summaries),indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps([dict(scene=r['scene'],seed=r['seed'],maximum_difference=r['maximum_qpos_difference'],
                           nonmatching=[dict(case=x['source_directory'],joint_difference=x['maximum_joint_difference_rad'],root_position_difference_m=x['root_position_difference_m'],ctrl_difference=x['maximum_ctrl_difference_rad'],first_divergence_s=x.get('first_initial_pose_divergence_s'),first_difference=x.get('first_initial_pose_difference'),maximum_initial_control_difference_rad=x.get('maximum_initial_control_difference_rad')) for x in r['nonmatching']])
                      for r in summaries if r['nonmatching']],indent=2))


if __name__=='__main__':main()
