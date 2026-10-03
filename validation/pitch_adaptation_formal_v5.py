"""Execute the prescribed 160 paired trials only after an audited training freeze.

Default is a read-only preview. --run uses seeds 101-110 and frozen parameters;
there is no command-line tuning override. Reuses the unchanged v5 relay runner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT,SCENES,ALPHA,model_path,trial

BASE=ROOT/'experiments/pitch_adaptation_20261003/repair_v5'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf8'))


def verify_freeze(path):
    if not path.is_file():raise ValueError('Complete the stage1 raw-data audit and parameter freeze first: '+str(path))
    freeze=read(path)
    if freeze.get('schema')!=1 or freeze.get('version')!=5 or not freeze.get('formal_released'):
        raise ValueError('Missing audited v5 formal release')
    if freeze['training_seeds']!=[1,2,3] or freeze['test_seeds']!=list(range(101,111)) or freeze['scenes']!=list(SCENES) or freeze['methods']!=['A','B','C','D'] or freeze['formal_trials']!=160:
        raise ValueError('Frozen design differs from the experiment plan')
    training_plan=BASE/'training/training_plan.json'
    if digest(training_plan)!=freeze['training_plan_sha256']:raise ValueError('Training plan changed after freeze')
    plan=read(training_plan)
    if freeze['sources']!=plan['sources'] or freeze['filters']!=plan['fixed_filters']:
        raise ValueError('Frozen sources/filters differ from training')
    for name,expected in freeze['sources'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    for scene,expected in freeze['models'].items():
        if digest(model_path(scene))!=expected:raise ValueError('Frozen model changed: '+scene)
    if digest(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe')!=freeze['exe_sha256']:
        raise ValueError('Frozen EXE changed')
    if len(freeze['training_result_sha256'])!=120:raise ValueError('Training freeze lacks 120 distinct cases')
    if set(freeze.get('training_raw_sha256',{}))!=set(freeze['training_result_sha256']):
        raise ValueError('Training freeze lacks matching raw-artifact hashes')
    if digest(ROOT/'validation/pitch_adaptation_training_audit.py')!=freeze['audit_source_sha256']:
        raise ValueError('Training auditor changed after freeze')
    for folder,expected in freeze['training_result_sha256'].items():
        if digest(ROOT/folder/'result.json')!=expected:raise ValueError('Training result changed: '+folder)
        raw_hashes=freeze['training_raw_sha256'][folder]
        if set(raw_hashes)!={'trace.npz','control.json.gz','execution.log'}:
            raise ValueError('Training raw-artifact set differs: '+folder)
        for name,raw_expected in raw_hashes.items():
            if digest(ROOT/folder/name)!=raw_expected:raise ValueError('Training raw evidence changed: '+folder+'/'+name)
    if abs(freeze['A_alpha_deg']-math.degrees(ALPHA))>1e-9:
        raise ValueError('A baseline angle changed')
    if freeze['CD_k'] not in plan['CD_shared_k'] or set(freeze['B_alpha_deg'])!=set(SCENES):
        raise ValueError('Frozen candidate set differs')
    if any(angle not in plan['B_alpha_deg'] for angle in freeze['B_alpha_deg'].values()):
        raise ValueError('B angle was not a training candidate')
    conditions=dict(joints=20,mujoco='3.3.0',timestep_s=.001,kp=127.,torque_limit_nm=2.355,
                    gain=[.01,.02],movement_budget_s=120.,hold_s=10.,hold_retreat_limit_mm=20.,roll_step_rad=.15,step_s=.2,
                    straight_mean_progress_m=.5,taper_additional_progress_m=.2,no_progress_window_s=10.,no_progress_min_m=.01)
    if freeze['conditions']!=conditions:raise ValueError('Fixed trial conditions changed')
    return freeze


def jobs_for(freeze):
    jobs=[]
    for scene in SCENES:
        for seed in freeze['test_seeds']:
            for method in freeze['methods']:
                alpha=math.radians(freeze['B_alpha_deg'][scene]) if method=='B' else ALPHA
                k=freeze['CD_k'] if method in ('C','D') else 0.
                jobs.append((scene,method,seed,k,alpha))
    # Randomized, predeclared execution order; each scene/seed perturbation is
    # still identical across methods through the runner's seeded sampler.
    np.random.default_rng(20261005).shuffle(jobs)
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,default=BASE/'training_audit/parameter_freeze.json')
    parser.add_argument('--out',type=Path,default=BASE/'formal')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();freeze=verify_freeze(args.freeze);jobs=jobs_for(freeze)
    freeze_hash=digest(args.freeze)
    plan=dict(stage='formal',freeze_sha256=freeze_hash,runner_sha256=digest(Path(__file__)),required=160,
              seeds=freeze['test_seeds'],order=[dict(scene=s,method=m,seed=seed,k=k,alpha_deg=math.degrees(a)) for s,m,seed,k,a in jobs])
    if not args.run:
        print(json.dumps(dict(required=len(jobs),freeze_sha256=freeze_hash,B_alpha_deg=freeze['B_alpha_deg'],CD_k=freeze['CD_k'],
                             seeds=freeze['test_seeds'],preview_only=True),indent=2));return
    args.out.mkdir(parents=True,exist_ok=True);plan_path=args.out/'formal_plan.json'
    if plan_path.exists() and read(plan_path)!=plan:raise ValueError('Existing formal plan/source differs; no mixed batches')
    if not plan_path.exists():plan_path.write_text(json.dumps(plan,indent=2),encoding='utf8')
    results=[]
    def execute(job):
        verify_freeze(args.freeze)
        if digest(Path(__file__))!=plan['runner_sha256']:raise ValueError('Formal orchestrator changed during execution')
        if digest(args.freeze)!=freeze_hash:raise ValueError('Parameters changed during execution')
        result=trial(*job,args.out)
        if (result['scene'],result['method'],result['seed'])!=job[:3] or abs(result['k']-job[3])>1e-12 or abs(result['alpha_deg']-math.degrees(job[4]))>1e-9:
            raise ValueError('Saved result parameters differ')
        if result.get('formal_freeze_sha256',freeze_hash)!=freeze_hash:raise ValueError('Saved result belongs to another freeze')
        result.update(stage='formal',formal_freeze_sha256=freeze_hash,formal_orchestrator_sha256=plan['runner_sha256'])
        folder=args.out/f'{job[0]}__{job[1]}__s{job[2]}__k{job[3]:g}__a{math.degrees(job[4]):.4f}'
        (folder/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            result=future.result();results.append(result)
            progress=dict(completed=len(results),required=160,freeze_sha256=freeze_hash,results=results)
            (args.out/'formal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False),encoding='utf8')
            print('FORMAL_PROGRESS '+json.dumps(dict(completed=len(results),required=160,scene=result['scene'],method=result['method'],
                                                   seed=result['seed'],outcome=result['outcome']),ensure_ascii=False),flush=True)
    print('Formal execution complete; independent raw-data/statistical audit remains required.',flush=True)


if __name__=='__main__':main()
