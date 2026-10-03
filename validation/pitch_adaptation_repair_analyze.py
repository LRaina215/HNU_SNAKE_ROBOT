"""Audit repaired training pilots; does not authorize formal comparisons."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCENES=('pipe_straight_small','pipe_straight_large','pipe_taper_horizontal','real_s2l')


def analyze(version):
    out=ROOT/f'experiments/pitch_adaptation_20261003/repair_v{version}'
    cases=[]
    for path in sorted((out/'pilots').glob('*/result.json')):
        result=json.loads(path.read_text(encoding='utf8'));folder=path.parent
        hashes={key:hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==expected
                for key,expected in result['sha256'].items()}
        assert all(hashes.values()),(folder,hashes)
        with gzip.open(folder/'control.json.gz','rt',encoding='utf8') as f:controls=json.load(f)
        updates=controls['control'];physics=controls['physics']
        with np.load(folder/'trace.npz') as trace:
            t=trace['sim_time'];initial=np.flatnonzero(np.isclose(t,2.2,atol=1e-9))[0]
            result['_initial_qpos']=trace['qpos'][initial].copy()
            motion=np.flatnonzero((t>2.2+1e-8)&trace['moving']&(abs(trace['roll_phase'])>1e-9))
            first_active=bool(trace['compliance_active'][motion[0]]) if len(motion) else False
            if not first_active:raise AssertionError('First motion not compliant: '+str(folder))
        p_errors=[];max_alpha_step=0.;zero=0
        for i,row in enumerate(updates):
            p=np.array([np.nan if x is None else x for x in row['implied_pitch_mm']])
            active=np.array(row['pitch_constraint_active'])
            if np.any((p[active]<100.-1e-8)|(p[active]>300.+1e-8)):p_errors.append(i)
            if i:
                previous=updates[i-1]
                rate=abs(np.array(row['alpha_rad'])-previous['alpha_rad']).max()
                assert rate<=np.radians(1.)*row['dt_s']/.2+1e-9,(folder,i,rate,row['dt_s'])
                max_alpha_step=max(max_alpha_step,float(np.degrees(rate)))
            zero+=not any(row['radius_valid'])
        assert not p_errors,(folder,p_errors)
        hold=[]
        if 'hold_start_s' in result:
            hold=[r for r in physics if r['sim_time']>=result['hold_start_s']-1e-8]
        retreat=max((result['hold_start_mean_m']-r['progress'][0] for r in hold),default=None)
        hold_s=hold[-1]['sim_time']-result['hold_start_s'] if hold else None
        hold_verified=None if not hold else retreat<=.02+1e-12 and hold_s>=10.-1e-8
        # Raw 10ms samples are authoritative over console checkpoints.
        if result['success'] and not hold_verified:raise AssertionError('Hold success contradicted by 10ms data')
        result['_control']=updates
        result['audit']=dict(source_hashes_passed=True,first_motion_compliant=first_active,
                             first_control_sim_time=updates[0]['sim_time'] if updates else None,
                             zero_valid_update_fraction=zero/len(updates) if updates else None,
                             maximum_alpha_step_deg=max_alpha_step,post_filter_pitch_bounds_passed=True,
                             hold_max_retreat_mm=None if retreat is None else 1000*retreat,
                             hold_duration_s=hold_s,hold_passed=hold_verified)
        cases.append(result)
    pairs=[]
    for scene in SCENES:
        same=[r for r in cases if r['scene']==scene and r['seed']==1]
        if len(same)==2:
            difference=float(abs(same[0]['_initial_qpos']-same[1]['_initial_qpos']).max())
            assert difference==0.,(scene,difference)
            pairs.append(dict(scene=scene,initial_qpos_max_difference=difference))
    rows=[]
    for r in cases:
        rows.append(dict(scene=r['scene'],method=r['method'],seed=r['seed'],outcome=r['outcome'],
                         radius_valid_fraction=r.get('radius_valid_fraction'),alpha_min_deg=r.get('alpha_range_deg',[None,None])[0],
                         alpha_max_deg=r.get('alpha_range_deg',[None,None])[1],movement_steps=r['movement_steps'],
                         hold_max_retreat_mm=r['audit']['hold_max_retreat_mm'],ground=r['any_ground_contact'],
                         first_motion_compliant=r['audit']['first_motion_compliant'],error=r.get('error')))
    if rows:
        with (out/'training_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        fig,axes=plt.subplots(4,2,figsize=(12,12))
        for scene,(a,b) in zip(SCENES,axes):
            for r in cases:
                if r['scene']!=scene:continue
                status=r['status'];ts=[x['sim_time'] for x in status]
                a.plot(ts,[x['mean'] for x in status],label=r['method']+' '+r['outcome'])
                controls=r['_control'];ct=[x['sim_time'] for x in controls]
                means=np.degrees([np.mean(x['alpha_rad']) for x in controls])
                b.plot(ct,means,label=r['method'])
                b.fill_between(ct,np.degrees([min(x['alpha_rad']) for x in controls]),
                               np.degrees([max(x['alpha_rad']) for x in controls]),alpha=.12)
            a.set_ylabel(scene+'\nMean progress (m)');b.set_ylabel('Alpha mean/range (deg)')
            for ax in (a,b):ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Physical time (s)')
        fig.suptitle(f'Repaired v{version} training pilots | Original cross-section compliance retained')
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'pilot_progress.png',dpi=150);plt.close(fig)
    gate=dict(version=version,completed_pilots=len(cases),required_pilots=8,
              paired_initial_states=pairs,source_and_control_audits_passed=True,
              formal_released=False,reason='B and common C/D parameters have not been trained/frozen; stage1 remains active.',
              pilots=[{k:v for k,v in r.items() if not k.startswith('_') and k not in ('status','sha256')} for r in cases])
    (out/'audit.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in gate.items() if k!='pilots'},indent=2))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',type=int,choices=(4,5),default=5)
    analyze(parser.parse_args().version)
