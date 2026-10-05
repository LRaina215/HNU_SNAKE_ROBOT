"""Offline parallel comparison of three topology hypotheses on one probe.

Workers read robot parameters and measured response only. No physical state,
contacts or axes are used. This does not run another physical experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import mujoco
from segmented_pole_experiment import ROOT,write_json
from feedback_takeover_response_calibration import calibrate
from feedback_takeover_response_gate import judge


def worker(checkpoint,probe,output,extension):
    robot=mujoco.MjModel.from_binary_path(str(checkpoint/'model.mjb'))
    meta=json.loads((checkpoint/'checkpoint.json').read_text(encoding='utf8'))
    records=json.loads((probe/'response_probe.json').read_text(encoding='utf8'))
    old=json.loads((probe/'response_calibration.json').read_text(encoding='utf8'))
    initial=old.get('unrefined',old['initial'])
    fit=calibrate(robot,meta['source_model'],initial,records,True,extension)
    name=extension or 'none';write_json(output/(name+'.json'),fit)
    return name,fit['training_rms_deg']


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    sources=args.output/'sources';sources.mkdir()
    for name in ('response_calibration_batch.py','feedback_takeover_response_calibration.py',
                 'feedback_takeover_response_gate.py','feedback_takeover_dynamic_observer_probe.py'):
        shutil.copy2(ROOT/'validation'/name,sources/name)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,args.checkpoint,args.probe,args.output,extension)
                 for extension in (None,'source','target')]
        scores=[f.result() for f in as_completed(futures)]
    selected=min(scores,key=lambda row:row[1])[0]
    fit=json.loads((args.output/(selected+'.json')).read_text(encoding='utf8'))
    write_json(args.output/'response_calibration.json',fit)
    shutil.copy2(args.probe/'response_probe.json',args.output/'response_probe.json')
    robot=mujoco.MjModel.from_binary_path(str(args.checkpoint/'model.mjb'))
    meta=json.loads((args.checkpoint/'checkpoint.json').read_text(encoding='utf8'))
    records=json.loads((args.probe/'response_probe.json').read_text(encoding='utf8'))
    gate=judge(robot,meta['source_model'],fit,records)
    write_json(args.output/'selection.json',dict(training_scores=scores,selected=selected,gate=gate,
        selection_uses_training_only=True,no_new_physical_trial=True))
    print('SELECTED',selected,'training',scores,'gate',gate['criteria'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('probe',type=Path);p.add_argument('output',type=Path)
    run(p.parse_args())
