"""Bounded stage1 training with original cross-section compliance retained.

Four fixed-alpha candidates plus three shared C/D k candidates: seven parameter
configurations, below the plan's limit of twelve. Only training seeds 1-3.
Formal tests remain locked. Completed same-source seed1 pilots may be reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT,SCENES,ALPHA,trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    out=ROOT/'experiments/pitch_adaptation_20261003/repair_v5/training'
    out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((out.parent/'audit.json').read_text(encoding='utf8'))
    if audit['completed_pilots']!=8 or len(audit['paired_initial_states'])!=4:
        raise RuntimeError('Complete all eight audited common-takeover pilots before training')
    source_paths=[ROOT/'snake_control'/name for name in ('local_pitch_controller_v5.py','local_pitch_feedback.py','local_pitch_geometry.py','pole_compliance.py')]
    source_paths.append(ROOT/'codes/Adaptive_Pole_Climbing_Mujoco.py')
    source_paths.extend([Path(__file__),ROOT/'validation/pitch_adaptation_experiment_v5.py'])
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    policy=dict(stage='training',version=5,seeds=[1,2,3],scenes=list(SCENES),
                B_alpha_deg=[12.,18.,math.degrees(ALPHA),28.],CD_shared_k=[0.,.5,1.],
                parameter_configurations=7,parameter_configuration_limit=12,
                fixed_filters=dict(window=7,sampler='measured module centres',spatial_sigma=1.,time_gain_at_0_2s=.2,
                                   alpha_rate_deg_per_0_2s=1.,physical_radius_timeout_s=.5,takeover_s=2.),
                B_selection='per-scene: success count descending, median maximum hold retreat ascending, median progress speed descending, smaller alpha as deterministic final tie',
                CD_selection='one k across all scenes/methods: pooled C/D success count descending, median maximum hold retreat ascending, median progress speed descending, smaller k as deterministic final tie',
                missing_hold_metric='infinite retreat; motion/support/target failures cannot win the retreat tie',
                formal_released=False,sources=source_hashes)
    plan_file=out/'training_plan.json'
    if plan_file.exists():
        previous=json.loads(plan_file.read_text(encoding='utf8'))
        if previous!=policy:raise RuntimeError('Existing training policy/sources differ; never mix versions')
    else:plan_file.write_text(json.dumps(policy,indent=2,ensure_ascii=False),encoding='utf8')
    jobs=[]
    for scene in SCENES:
        for seed in (1,2,3):
            jobs.extend((scene,'B',seed,0.,math.radians(a)) for a in policy['B_alpha_deg'])
            jobs.extend((scene,method,seed,k,ALPHA) for method in ('C','D') for k in policy['CD_shared_k'])
    np.random.default_rng(20261004).shuffle(jobs)
    result_rows=[]
    def run(job):
        for path,expected in source_hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Training source changed: '+path)
        scene,method,seed,k,alpha=job
        label=f'{scene}__{method}__s{seed}__k{k:g}__a{math.degrees(alpha):.4f}'
        saved=out.parent/'pilots'/label/'result.json'
        if method=='D' and seed==1 and k==0 and saved.exists():
            result=json.loads(saved.read_text(encoding='utf8'))
            for path,expected in result['sha256'].items():
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('Reused pilot source differs: '+path)
            result['training_reused_same_source_pilot']=True
            result['source_directory']=str(saved.parent.relative_to(ROOT))
            return result
        result=trial(scene,method,seed,k,alpha,out)
        result['source_directory']=str((out/label).relative_to(ROOT))
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,job):job for job in jobs}
        for future in as_completed(futures):
            result=future.result();result_rows.append(result)
            (out/'training_progress.json').write_text(json.dumps(dict(completed=len(result_rows),required=len(jobs),
                results=result_rows,formal_released=False),indent=2,ensure_ascii=False),encoding='utf8')
            print('TRAINING_PROGRESS '+json.dumps(dict(completed=len(result_rows),required=len(jobs),
                scene=result['scene'],method=result['method'],seed=result['seed'],k=result['k'],
                alpha_deg=result['alpha_deg'],outcome=result['outcome']),ensure_ascii=False),flush=True)
    print('Training complete; validate all raw outcomes and choose parameters before any formal freeze.',flush=True)


if __name__=='__main__':main()
