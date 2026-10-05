"""Freeze the scalar pulse rule on horizontal data, then score vertical data.

This detects additional mechanical constraint, not force closure or a safe
release certificate. Negative controls remove only target-pipe collision.
"""
import json
from pathlib import Path
import numpy as np
from segmented_pole_experiment import ROOT,write_json


def run():
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials']
    active=[r for r in trials if r['pattern']=='multisine']
    train=[r for r in active if r['scene']=='horizontal']
    positive=max(r['features']['minimum_magnitude'] for r in train if not r['target_removed'])
    negative=min(r['features']['minimum_magnitude'] for r in train if r['target_removed'])
    threshold=(positive+negative)/2
    rule=dict(feature='Minimum diagonal harmonic response magnitude over three simultaneous disjoint joint patterns',
        threshold=threshold,positive_training_max=positive,negative_training_min=negative,
        training_only='Four horizontal saved states; vertical values do not select threshold',
        interpretation='Response below threshold suggests mechanical constraint; does not identify force closure or prove support margin.')
    write_json(root/'horizontal_rule.json',rule)
    rows=[]
    for r in active:
        frames=r['records'];q=np.array([v['q'] for v in frames]);cmd=np.array([v['command'] for v in frames])
        predicted=r['features']['minimum_magnitude']<threshold
        rows.append(dict(scene=r['scene'],time=r['snapshot_seconds'],target_removed=r['target_removed'],
            response=r['features']['minimum_magnitude'],constraint_detected=bool(predicted),
            matched=bool(predicted== (not r['target_removed'])),
            target_contact_fraction=r['scoring']['target_contact_fraction'],
            head_displacement_mm=1000*r['scoring']['head_displacement_m'],
            maximum_joint_speed_deg_s=float(np.degrees(abs(np.diff(q,axis=0))/.02).max()),
            maximum_command_increment_deg=float(np.degrees(abs(np.diff(cmd,axis=0))).max())))
    result=dict(rule=rule,rows=rows,
        horizontal_correct=sum(r['matched'] for r in rows if r['scene']=='horizontal'),
        vertical_correct=sum(r['matched'] for r in rows if r['scene']=='vertical'),
        limitation='8 paired starting states from two existing trajectories; target-deletion counterfactuals, not natural obstacle traversal or force-bearing stability validation.')
    write_json(root/'classification.json',result)
    print(json.dumps(result,indent=2))
    # Repair the human-readable protocol label to the actual recorded times;
    # no command, response or physical result is modified.
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    protocol['validation']='vertical checkpoints at 2,10,22,34s; same pulses and feature formula'
    write_json(root/'protocol.json',protocol)


if __name__=='__main__':run()
