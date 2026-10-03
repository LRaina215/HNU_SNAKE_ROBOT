"""Prespecified seed-level summaries; no frame-level pseudo-replication.

Called only on independently audited formal rows. Exact binomial intervals and
McNemar tests; fixed-seed percentile bootstrap of paired median differences.
Small-sample secondary intervals are descriptive, not universal effect claims.
"""
from collections import Counter

import numpy as np
from scipy.stats import binomtest

from pitch_adaptation_experiment_v5 import SCENES

POLICY=dict(unit='paired scene/seed trial; never a physical frame',confidence=.95,
            success_interval='Clopper-Pearson exact binomial',
            success_comparison='two-sided exact McNemar on discordant pairs',
            primary_family='D versus B in four scenes; D versus C in two taper scenes',
            primary_multiplicity='Holm over all six predeclared success comparisons',
            secondary_interval='20000 paired bootstrap resamples of median difference; seed 20261006',
            secondary_interpretation='descriptive, available-pair denominators explicit; no confirmatory significance claim',
            stage3='positive primary success evidence still requires reviewed matching mechanism evidence; ceiling/secondary benefits require explicit evidence review')

METRICS=('motion_s','rolling_cycles','speed_m_s','hold_max_retreat_mm','mean_progress_m','progress_per_cycle_m',
         'cumulative_retreat_m','torque_rms_nm','torque_squared_per_m','saturation_fraction',
         'maximum_tracking_deg','joint_clip_events','radius_valid_fraction','fallback_node_fraction')


def values(rows,metric):
    return np.asarray([r[metric] for r in rows if r.get(metric) is not None and np.isfinite(r[metric])],float)


def distribution(rows,metric):
    x=values(rows,metric)
    if not len(x):return dict(n=0,missing=len(rows),median=None,q25=None,q75=None)
    return dict(n=len(x),missing=len(rows)-len(x),median=float(np.median(x)),
                q25=float(np.percentile(x,25)),q75=float(np.percentile(x,75)),minimum=float(min(x)),maximum=float(max(x)))


def paired_difference(left,right,metric,both_success=False):
    missing=[];differences=[];seeds=[]
    for seed in sorted(left):
        a,b=left[seed],right[seed]
        if both_success and not (a['success'] and b['success']):continue
        x,y=a.get(metric),b.get(metric)
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):missing.append(seed);continue
        differences.append(float(x-y));seeds.append(seed)
    if not differences:
        return dict(n=0,seeds=[],missing_seeds=missing,median_difference=None,bootstrap95=None)
    d=np.asarray(differences);rng=np.random.default_rng(20261006)
    bootstrap=np.median(d[rng.integers(0,len(d),size=(20000,len(d)))],axis=1)
    return dict(n=len(d),seeds=seeds,missing_seeds=missing,differences=differences,
                median_difference=float(np.median(d)),bootstrap95=np.percentile(bootstrap,[2.5,97.5]).tolist())


def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);adjusted=[None]*len(order);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(order)-rank)*pvalues[index]);adjusted[index]=min(1.,previous)
    return adjusted


def summarize(rows):
    expected={(scene,method,seed) for scene in SCENES for method in 'ABCD' for seed in range(101,111)}
    actual=[(r['scene'],r['method'],r['seed']) for r in rows]
    if len(actual)!=160 or len(set(actual))!=160 or set(actual)!=expected:
        raise ValueError('Formal statistics require all 160 prescribed unique paired trials')
    methods=[];comparisons=[];primary=[]
    for scene in SCENES:
        groups={m:{r['seed']:r for r in rows if r['scene']==scene and r['method']==m} for m in 'ABCD'}
        for method,group in groups.items():
            trials=list(group.values());successes=sum(r['success'] for r in trials)
            ci=binomtest(successes,len(trials)).proportion_ci(confidence_level=.95,method='exact')
            methods.append(dict(scene=scene,method=method,successes=successes,trials=len(trials),
                                success_interval95=[float(ci.low),float(ci.high)],
                                outcomes=dict(Counter(r['outcome'] for r in trials)),
                                metrics={metric:distribution(trials,metric) for metric in METRICS}))
        for comparator in 'ABC':
            left,right=groups['D'],groups[comparator]
            wins=sum(left[s]['success'] and not right[s]['success'] for s in left)
            losses=sum(right[s]['success'] and not left[s]['success'] for s in left)
            discordant=wins+losses
            p=float(binomtest(wins,discordant,.5).pvalue) if discordant else 1.
            item=dict(scene=scene,left='D',right=comparator,wins=wins,losses=losses,ties=10-discordant,
                      exact_mcnemar_two_sided_p=p,primary_family_member=comparator=='B' or (comparator=='C' and not scene.startswith('pipe_straight')),
                      paired_metrics={metric:paired_difference(left,right,metric) for metric in METRICS},
                      paired_metrics_both_success={metric:paired_difference(left,right,metric,True) for metric in METRICS})
            comparisons.append(item)
            if item['primary_family_member']:primary.append(item)
    if len(primary)!=6:raise ValueError('Prespecified primary comparison family changed')
    for item,p in zip(primary,holm([x['exact_mcnemar_two_sided_p'] for x in primary])):
        item['primary_holm_p']=p;item['positive_primary_success_evidence']=p<=.05 and item['wins']>item['losses']
    eligible=[]
    for scene in ('pipe_taper_horizontal','real_s2l'):
        relevant=[x for x in primary if x['scene']==scene and x['right'] in ('B','C')]
        if all(x['positive_primary_success_evidence'] for x in relevant):eligible.append(scene)
    return dict(policy=POLICY,methods=methods,comparisons=comparisons,
                positive_primary_taper_scenes=eligible,stage3_released=False,
                stage3_reason='Mechanism evidence and any ceiling/secondary-benefit case remain to be reviewed; statistics alone do not release stage3.')
