"""Audit formal raw trials, then summarize the full 160-case paired design.

No training/tuning or simulation is performed. Partial batches get integrity
reports only. Complete batches also get seed-level statistics and scoring-only
mechanism replay, with geometry contacts explicitly separate from support.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT
from pitch_adaptation_formal_v5 import BASE,verify_freeze,jobs_for,digest,read
from pitch_adaptation_training_audit import audit_case,label
from pitch_adaptation_mechanism import replay
from pitch_adaptation_statistics import summarize


def plot_statistics(statistics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenes=list(dict.fromkeys(r['scene'] for r in statistics['methods']))
    index={(r['scene'],r['method']):r for r in statistics['methods']}
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    width=.18;x=np.arange(len(scenes));labels=['Small pipe','Large pipe','Horizontal taper','Vertical taper']
    panels=[('success','Success fraction (exact 95% interval)'),('hold_max_retreat_mm','Maximum hold retreat (mm)'),
            ('speed_m_s','Mean movement speed (m/s)'),('torque_rms_nm','Torque RMS (N m)'),
            ('torque_squared_per_m','Torque-squared / distance (N² m² s / m)'),('saturation_fraction','Torque saturation fraction')]
    for ax,(metric,title) in zip(axes.flat,panels):
        for j,method in enumerate('ABCD'):
            points=[];lo=[];hi=[];counts=[]
            for scene in scenes:
                r=index[(scene,method)]
                if metric=='success':
                    value=r['successes']/r['trials'];lower,upper=r['success_interval95'];count=r['trials']
                else:
                    summary=r['metrics'][metric];value=summary['median'];lower=summary['q25'];upper=summary['q75'];count=summary['n']
                points.append(np.nan if value is None else value);lo.append(np.nan if value is None else value-lower);hi.append(np.nan if value is None else upper-value);counts.append(count)
            locations=x+(j-1.5)*width
            ax.errorbar(locations,points,yerr=[lo,hi],fmt='o',capsize=3,label=method)
            if metric=='hold_max_retreat_mm':
                for location,value,count in zip(locations,points,counts):
                    if np.isfinite(value):ax.annotate(f'n={count}',(location,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
        ax.set_xticks(x,labels,rotation=15);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.25)
        if metric=='success':ax.set_ylim(-.05,1.05);ax.legend(ncol=4)
        if metric=='hold_max_retreat_mm':ax.axhline(20.,color='grey',ls='--',lw=1)
    fig.suptitle('Formal paired simulation: seeds 101-110, 160 trials\nContinuous panels show median / IQR; failed trials retained, missing hold denominators shown. Torque-squared is a burden proxy.',fontsize=12)
    fig.savefig(out/'formal_summary.png',dpi=150);plt.close(fig)


def audit(freeze_path,source,out):
    freeze=verify_freeze(freeze_path);plan_path=source/'formal_plan.json'
    if not plan_path.exists():raise ValueError('No frozen formal batch has been started: '+str(plan_path))
    plan=read(plan_path)
    if plan['required']!=160 or plan['freeze_sha256']!=digest(freeze_path):raise ValueError('Formal plan/freeze differs')
    if plan['runner_sha256']!=digest(ROOT/'validation/pitch_adaptation_formal_v5.py'):
        raise ValueError('Formal orchestrator differs from saved plan')
    if plan['statistics_source_sha256']!=digest(ROOT/'validation/pitch_adaptation_statistics.py'):
        raise ValueError('Formal statistics source differs from prespecified version')
    jobs=jobs_for(freeze)
    order=[dict(scene=s,method=m,seed=seed,k=k,alpha_deg=float(np.degrees(a))) for s,m,seed,k,a in jobs]
    if len(plan['order'])!=len(order) or any((p['scene'],p['method'],p['seed'],p['k'])!=(q['scene'],q['method'],q['seed'],q['k']) or abs(p['alpha_deg']-q['alpha_deg'])>1e-9 for p,q in zip(plan['order'],order)):
        raise ValueError('Formal randomized schedule differs from prescribed design')
    rows=[];missing=[];initials={};counts={};folders={}
    for job in jobs:
        case=source/label(job)
        if not (case/'result.json').exists():missing.append(label(job));continue
        result=read(case/'result.json')
        if result.get('stage') is None:
            missing.append(label(job)+' (runner provenance annotation pending)');continue
        if result.get('stage')!='formal' or result.get('formal_freeze_sha256')!=digest(freeze_path) or result.get('formal_orchestrator_sha256')!=plan['runner_sha256']:
            raise ValueError('Trial lacks the required formal provenance: '+str(case))
        if result.get('dependencies')!=freeze['dependencies']:raise ValueError('Formal dependency versions differ: '+str(case))
        row,initial=audit_case(case,job);rows.append(row);folders[(job[0],job[1],job[2])]=case
        key=(job[0],job[2])
        if key in initials and not np.array_equal(initials[key],initial):raise ValueError('Formal paired initialization differs: '+str(key))
        initials[key]=initial;counts[key]=counts.get(key,0)+1
    out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'audited_formal.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=dict(required=160,audited=len(rows),complete=not missing,missing=missing,
                complete_paired_initial_groups=sum(n==4 for n in counts.values()),source_and_raw_checks_passed=True,
                formal_plan_sha256=digest(plan_path),freeze_sha256=digest(freeze_path),
                analysis_source_sha256=digest(Path(__file__)),stage3_released=False,rows=rows)
    (out/'formal_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    if not missing:
        if report['complete_paired_initial_groups']!=40:raise ValueError('Not all forty paired groups are complete')
        statistics=summarize(rows)
        (out/'formal_statistics.json').write_text(json.dumps(statistics,indent=2,ensure_ascii=False),encoding='utf8')
        plot_statistics(statistics,out)
        mechanisms=[]
        for row in rows:
            key=(row['scene'],row['method'],row['seed']);case=folders[key]
            target=out/'mechanisms'/case.name;existing=target/'mechanism.json'
            cached=read(existing) if existing.exists() else None
            if cached is not None and all(digest(ROOT/name)==expected for name,expected in cached['sources_sha256'].items()) and (target/'mechanism.npz').exists() and (target/'mechanism.png').exists():
                mechanism=cached
            else:mechanism=replay(case,target)
            mechanisms.append(mechanism)
        (out/'mechanism_summary.json').write_text(json.dumps(mechanisms,indent=2,ensure_ascii=False),encoding='utf8')
        report.update(statistics_complete=True,mechanism_cases=len(mechanisms),
                      stage3_reason='Full evidence now available for H1-H4 review; this auditor alone does not declare benefit.')
        (out/'formal_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','missing')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,default=BASE/'training_audit/parameter_freeze.json')
    parser.add_argument('--source',type=Path,default=BASE/'formal')
    parser.add_argument('--out',type=Path,default=BASE/'formal_analysis')
    args=parser.parse_args();audit(args.freeze,args.source,args.out)
