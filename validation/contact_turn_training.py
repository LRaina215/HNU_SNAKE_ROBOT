"""Bounded, sequential oracle-motion training: at most twelve configurations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from contact_turn_experiment import BASE,write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args()
    plan_path=BASE/'stage1_training_plan.json'
    jobs=[dict(front=f,width=w,turn=a) for f in (4,6) for w in (64.5,129.) for a in (45.,60.,90.)]
    sources=[ROOT/'codes/local_bend_backbone.py',ROOT/'codes/contact_turn_controller.py',
             ROOT/'validation/contact_turn_experiment.py',ROOT/'validation/contact_turn_stage1.py']
    plan=dict(version=1,maximum_configurations=12,jobs=jobs,training_phase=0.,
              source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
              selection_rule='success first, then actual target contact modules, then actual target wrap, then head target progress; no test states used',
              test_states=[dict(phase=p,source_position_offset_m=o) for p,o in [(0.,-.02),(1.2566370614359172,0.),
                  (2.5132741228718345,.02),(3.7699111843077517,-.01),(5.026548245743669,.01)]],
              diagnostic_revisions_used=1,known_direction=True,geometry_truth_is_privileged=True,
              autonomous_recognition_released=False,physical_budget_s=60.)
    if plan_path.exists():
        old=json.loads(plan_path.read_text(encoding='utf8'))
        if old!=plan:raise RuntimeError('training plan changed; use a new version explicitly')
    else:write_json(plan_path,plan)
    if not args.run:
        print(json.dumps(plan,ensure_ascii=False));return
    results=[]
    for job in jobs:
        f,w,a=job['front'],job['width'],job['turn']
        # The repaired first pilot is the same frozen configuration, not a 13th.
        attempt='geometry_scoring_repair' if (f,w,a)==(4,64.5,60.) else 'training_v1'
        label=f'f{f}_w{w:g}_turn{a:g}_p0_{attempt}'
        result_path=BASE/'stage1'/label/'result.json'
        if not result_path.exists():
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'validation/contact_turn_experiment.py'),
                            '--stage1','--front',str(f),'--width',str(w),'--turn',str(a),'--attempt',attempt],check=True,cwd=ROOT)
        result=json.loads(result_path.read_text(encoding='utf8'));result['directory']=str(result_path.parent.relative_to(ROOT))
        results.append(result)
        write_json(BASE/'stage1_training_progress.json',dict(completed=len(results),required=12,results=results))
        if result['outcome']=='engineering_failure':
            raise RuntimeError('engineering failure; repair explicitly without discarding trace')
    write_json(BASE/'stage1_training_results.json',dict(completed=12,results=results,plan_sha256=sha(plan_path)))
    print(json.dumps(dict(completed=12,successes=sum(x['success'] for x in results)),ensure_ascii=False))


if __name__=='__main__':main()
