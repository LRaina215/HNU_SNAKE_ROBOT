"""Independently reconstruct limiting stages from complete captured v5 calls."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT
from pitch_adaptation_training_audit import BASE,audit_case,sha,load
from pitch_adaptation_capture import reconstruct_limits


def audit_capture(case):
    with gzip.open(case/'capture.json.gz','rt',encoding='utf8') as f:capture=json.load(f)
    with gzip.open(case/'control.json.gz','rt',encoding='utf8') as f:control=json.load(f)['control']
    if capture['schema']!=1:raise ValueError('Unsupported controller capture schema')
    calls=capture['unchanged_controller_calls'];accepted=[c for c in calls if c['accepted']]
    if len(accepted)!=len(control) or any(c['original_control_record']!=r for c,r in zip(accepted,control)):
        raise ValueError('Captured attempts differ from original control records')
    limits=[reconstruct_limits(call) for call in calls]
    if any(x['maximum_joint_change_deg']>30.+1e-8 for x in limits if x['accepted']):raise ValueError('Captured joint jump violates the protocol')
    with gzip.open(case/'capture_limits.json.gz','wt',encoding='utf8') as f:
        json.dump(dict(schema=1,derived_from_capture_sha256=sha(case/'capture.json.gz'),
                       audit_source_sha256=sha(Path(__file__)),updates=limits),f,ensure_ascii=False,allow_nan=False)
    return dict(attempts=len(calls),accepted=len(accepted),rejected=len(calls)-len(accepted),
                maximum_capture_call_ms=max((c['capture_total_call_ms'] for c in calls),default=None),
                limit_reconstruction_passed=True,capture_sha256=sha(case/'capture.json.gz'),
                limits_sha256=sha(case/'capture_limits.json.gz'))


def pilot_gate():
    source=BASE/'capture_pilots';summaries=[];initials=[]
    for method in ('A','D'):
        case=source/f'pipe_taper_horizontal__{method}__s1__k0__a23.2494'
        result=load(case/'result.json');job=(result['scene'],method,1,result['k'],np.radians(result['alpha_deg']))
        row,initial=audit_case(case,job);initials.append(initial)
        record=audit_capture(case)
        if record['rejected'] or row['outcome']!='success':raise ValueError('Logged pilot was not a clean complete successful engineering trial')
        summaries.append(dict(method=method,result_sha256=sha(case/'result.json'),raw=row,capture=record))
    checks=load(source/'capture_checks.json')
    if checks['accepted_call_pairs']!=128 or checks['maximum_target_difference_rad']!=0 or not checks['limit_replay_passed'] or not checks['rejected_calls_captured_without_state_change']:
        raise ValueError('Logging output-equivalence checks failed')
    if not np.array_equal(initials[0],initials[1]):raise ValueError('Logged common initialization differs')
    gate=dict(passed=True,engineering_pilots=2,common_initial_max_difference_rad=float(abs(initials[0]-initials[1]).max()),
              logging_changes_control_policy=False,output_equivalence_checks=checks,pilots=summaries,
              sources={str((ROOT/'validation'/name).relative_to(ROOT)):sha(ROOT/'validation'/name)
                       for name in ('pitch_adaptation_capture.py','pitch_adaptation_experiment_v5_logged.py','pitch_adaptation_capture_checks.py','pitch_adaptation_capture_audit.py')})
    (source/'capture_gate.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(passed=gate['passed'],pilots=2,initial_max_difference=gate['common_initial_max_difference_rad'],
                          captured_accepted_calls=sum(p['capture']['accepted'] for p in summaries)),indent=2))
    return gate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path)
    args=parser.parse_args()
    if args.case:print(json.dumps(audit_capture(args.case),indent=2))
    else:pilot_gate()
