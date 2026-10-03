"""Separate native-parameter reference required by the experiment plan.

Seed1 only, descriptive; never included in B training selection or formal 160.
Small/taper native alpha equals A and reuses the unchanged audited pilots.
Large pipe native alpha differs from A and gets its own fixed-angle reference.
"""
import argparse
import csv
import json
import math

import mujoco

from pitch_adaptation_experiment_v5 import ROOT,SCENES,ALPHA,model_path,trial
from pitch_adaptation_training_audit import BASE,audit_case,label,sha,load,runtime_versions


def main(run=False):
    training=load(BASE/'training/training_plan.json')
    for name,expected in training['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Training/reference control source differs: '+name)
    design={scene:mujoco.MjModel.from_xml_path(str(model_path(scene))).numeric('experiment_helix_mm').data.tolist() for scene in SCENES}
    angles={scene:math.atan2(p,2*math.pi*r) for scene,(r,p) in design.items()}
    plan=dict(kind='native default alpha reference; descriptive seed1 only',seed=1,native_helix_mm=design,
              native_alpha_deg={s:math.degrees(a) for s,a in angles.items()},sources=training['sources'],
              included_in_training_selection=False,included_in_formal160=False,
              reference_runner_sha256=sha(__file_path()),dependencies=runtime_versions())
    if not run:print(json.dumps(plan,indent=2));return
    out=BASE/'native_reference';out.mkdir(parents=True,exist_ok=True);plan_file=out/'reference_plan.json'
    if plan_file.exists() and load(plan_file)!=plan:raise ValueError('Existing native reference plan differs')
    if not plan_file.exists():plan_file.write_text(json.dumps(plan,indent=2),encoding='utf8')
    rows=[]
    for scene in SCENES:
        alpha=angles[scene]
        if abs(alpha-ALPHA)<1e-12:
            job=(scene,'A',1,0.,ALPHA);case=BASE/'pilots'/label(job)
            if not (case/'result.json').exists():raise ValueError('Required identical native A pilot missing: '+str(case))
        else:
            job=(scene,'B',1,0.,alpha);result=trial(*job,out);case=out/label(job)
            result.update(stage='native_parameter_reference',included_in_training_selection=False,included_in_formal160=False,dependencies=runtime_versions())
            (case/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
        row,_=audit_case(case,job);row['reference_native_alpha_deg']=math.degrees(alpha);rows.append(row)
    with (out/'native_reference.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'native_reference.json').write_text(json.dumps(dict(plan=plan,rows=rows),indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps([dict(scene=r['scene'],native_alpha_deg=r['reference_native_alpha_deg'],outcome=r['outcome'],hold_max_retreat_mm=r['hold_max_retreat_mm']) for r in rows],indent=2))


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();main(args.run)
