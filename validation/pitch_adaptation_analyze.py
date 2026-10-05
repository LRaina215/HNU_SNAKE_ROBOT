"""Analyze saved pitch pilots and the estimator release gate. No simulator launched.

Use --out experiments/pitch_adaptation_20261003 after the trials finish.
Plots are static research figures; each run remains a training diagnostic.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'validation')]
from snake_control.local_pitch_geometry import estimate_local_radius
from snake_control.local_pitch_controller import LocalPitchController
from snake_control.pole_compliance import ComplianceParameters,build_backbone
from adaptive_pole_single_step import math_checks
from pitch_adaptation_experiment import model_path


def mechanism_scores(case,out):
    """Truth axis/region/contact replay is scoring only; the controller never sees it."""
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(model_path(case['scene'])))
    data=mujoco.MjData(model)
    try:
        line=model.numeric('pipe_centerline').data.reshape(-1,4)
        origin=line[0,:3];axis=line[1,:3]-origin;axis=axis/np.linalg.norm(axis)
    except KeyError:origin=np.zeros(3);axis=np.array([0.,0.,1.])
    h=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
    h=h-axis*np.dot(h,axis);h/=np.linalg.norm(h);v=np.cross(axis,h)
    if case['scene'].startswith('pipe_straight'):boundaries=[]
    elif case['scene']=='real_s2l':boundaries=model.numeric('pipe_profile').data[2:4].tolist()
    else:boundaries=model.numeric('feature_span').data[:2].tolist()
    rows=[]
    with np.load(case['_directory']/'trace.npz') as raw:
        ends=np.flatnonzero(np.diff(np.r_[0,raw['completed_steps']])==1)
        phases=raw['roll_phase'][ends]
        hold=ends[np.flatnonzero(np.isclose(phases,phases[-1]))[0]]
        hold_time=float(raw['sim_time'][hold])
        for index in ends:
            data.qpos[:]=raw['qpos'][index];mujoco.mj_forward(model,data)
            points=np.vstack([data.site('s_head').xpos,[data.joint(f'joint{i}').xanchor for i in range(1,21)],data.site('s_tail').xpos])*1000
            rel=points-origin*1000.;progress=rel@axis/1000
            azimuth=np.unwrap(np.arctan2(rel@v,rel@h))
            radius=np.sqrt((rel@h)**2+(rel@v)**2)
            regions=[(-np.inf,np.inf)] if not boundaries else [(-np.inf,boundaries[0]),(boundaries[0],boundaries[1]),(boundaries[1],np.inf)]
            lengths=[];turns=[]
            for lo,hi in regions:
                length=0.;angle=0.
                for j in range(21):
                    p,q=progress[j:j+2]
                    if abs(q-p)<1e-12:
                        start,end=(0.,1.) if lo<=p<hi else (0.,0.)
                    else:
                        aa,bb=sorted([(lo-p)/(q-p),(hi-p)/(q-p)])
                        start,end=max(0.,aa),min(1.,bb)
                    if end<=start:continue
                    length+=(end-start)*64.5
                    a=rel[j]*(1-start)+rel[j+1]*start;b=rel[j]*(1-end)+rel[j+1]*end
                    pa,pb=math.atan2(a@v,a@h),math.atan2(b@v,b@h)
                    angle+=math.atan2(math.sin(pb-pa),math.cos(pb-pa))
                lengths.append(length);turns.append(angle/(2*np.pi))
            touched=set()
            for contact in data.contact:
                aa,bb=int(contact.geom1),int(contact.geom2)
                if (model.geom(aa).name or '').startswith('pipe') and model.geom_bodyid[bb]:touched.add(int(model.geom_bodyid[bb]))
                if (model.geom(bb).name or '').startswith('pipe') and model.geom_bodyid[aa]:touched.add(int(model.geom_bodyid[aa]))
            rows.append(dict(sim_time=float(raw['sim_time'][index]),phase=float(raw['roll_phase'][index]),
                             mean_progress_m=float(progress[1:-1].mean()),minimum_progress_m=float(progress[1:-1].min()),
                             signed_geometric_turns=float((azimuth[-1]-azimuth[0])/(2*np.pi)),
                             axial_coverage_m=float(np.ptp(progress)),mean_axis_radius_mm=float(radius[1:-1].mean()),
                             region_length_mm=lengths,region_signed_turns=turns,pipe_contact_robot_bodies=sorted(touched),
                             contacted_module_length_proxy_mm=64.5*len(touched)))
    # Exact floating-point progress adjudicates the rounded console threshold.
    physics=read_gzip(case['_directory']/'control.json.gz')['physics']
    start=min(physics,key=lambda r:abs(r['sim_time']-hold_time))
    hold_rows=[r for r in physics if r['sim_time']>=hold_time-1e-8]
    case['hold_time_s']=hold_time
    case['exact_hold_max_retreat_mm']=1000*(start['progress'][0]-min(r['progress'][0] for r in hold_rows))
    case['exact_hold_duration_s']=physics[-1]['sim_time']-hold_time
    if case['outcome']=='hold_slip':assert case['exact_hold_max_retreat_mm']>.02*1000
    if case['outcome']=='success':assert case['exact_hold_max_retreat_mm']<=20. and case['exact_hold_duration_s']>=10.-1e-6
    case['minimum_abs_geometric_turns']=min(abs(r['signed_geometric_turns']) for r in rows)
    case['region_lengths_partition_error_mm']=max(abs(sum(r['region_length_mm'])-1354.5) for r in rows)
    (out/(case['_directory'].name+'_mechanism.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def check_controller(out,folders):
    feedback=next(p for folder in folders for p in sorted(folder.glob('pipe_straight_small__D*/feedback_*.json.gz'),reverse=True))
    geometry=read_gzip(feedback);now=time.time()*1000
    geometry.update(age_ms=0,sample_unix_ms=now,feedback_usable=True)
    params=ComplianceParameters((.01,.02),LocalPitchController.fixed_alpha,False)
    old=build_backbone(geometry,params)
    reference=np.clip(old.joint_angles(0),-np.pi/2+.01,np.pi/2-.01)
    targets=[];fallback=[]
    for k in (0.,.5,1.):
        LocalPitchController.exponent=k;LocalPitchController.method='D';LocalPitchController.records=[]
        c=LocalPitchController(params,np.full(21,64.5))
        target=c.compute(copy.deepcopy(geometry),.15,reference,now_ms=now,reference_phase=0.)
        assert target is not None,c.status
        targets.append(target)
        fallback.append(not any(c.records[-1]['radius_valid']) and c.records[-1]['fallback_fixed'])
        state=c.alpha_state.copy();old_key=c.key
        bad=copy.deepcopy(geometry);bad['sample_unix_ms']=now-1000
        assert c.compute(bad,.3,target,now_ms=now) is None
        assert c.key==old_key and np.array_equal(c.alpha_state,state)
        backward=copy.deepcopy(geometry);backward['sequence']-=1
        assert c.compute(backward,.3,target,now_ms=now) is None and c.key==old_key
    report=dict(all_k_fallback_targets_max_difference_rad=float(np.ptp(np.array(targets),axis=0).max()),
                all_k_fallback=all(fallback),stale_and_out_of_order_state_preserved=True,
                baseline_math=math_checks(),source_feedback=str(feedback.relative_to(ROOT)))
    (out/'controller_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def analyze(out):
    folders=[out/'matched_pilots_v2',out/'estimator_repeats_v2']
    runs=[]
    for folder in folders:
        for path in sorted(folder.glob('*/result.json')):
            result=json.loads(path.read_text(encoding='utf-8'));result['_directory']=path.parent
            extra=read_gzip(path.parent/'control.json.gz')
            controls=extra['control']
            result['fallback_update_fraction']=float(np.mean([r['fallback_fixed'] for r in controls])) if controls else None
            result['all_windows_invalid_update_fraction']=float(np.mean([not any(r['radius_valid']) for r in controls])) if controls else None
            result['max_mean_retreat_m']=max((a['mean']-b['mean'] for a,b in zip(result['status'][:-1],result['status'][1:])),default=0.)
            mechanism_scores(result,out)
            runs.append(result)
    fields=['scene','method','seed','outcome','success','movement_steps','physical_s','wall_s',
            'radius_valid_fraction','fallback_update_fraction','all_windows_invalid_update_fraction',
            'compute_ms_max','canceled_feedback_retries','torque_rms_nm','saturation_fraction','torque_squared_integral',
            'exact_hold_max_retreat_mm','exact_hold_duration_s','minimum_abs_geometric_turns','region_lengths_partition_error_mm']
    with (out/'training_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(runs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    scenes=['pipe_straight_small','pipe_straight_large','pipe_taper_horizontal','real_s2l']
    fig,axes=plt.subplots(4,3,figsize=(14,13))
    for row,scene in enumerate(scenes):
        for method,color in [('A','#576579'),('D','#db6c38')]:
            case=next((r for r in runs if r['scene']==scene and r['method']==method and r['seed']==1),None)
            if case is None:continue
            status=case['status'];initial=case.get('initial',{}).get('mean',status[0]['mean'])
            axes[row,0].plot([r['sim_time'] for r in status],[r['mean']-initial for r in status],color=color,
                             label=method+': '+case['outcome'])
            extra=read_gzip(case['_directory']/'control.json.gz');controls=extra['control']
            if controls:
                times=np.array([r['sim_time'] for r in controls]);angles=np.degrees([r['alpha_rad'] for r in controls])
                axes[row,1].plot(times,np.mean(angles,axis=1),color=color,label=method)
                axes[row,1].fill_between(times,np.min(angles,axis=1),np.max(angles,axis=1),color=color,alpha=.14)
                axes[row,2].plot(times,[np.mean(r['radius_valid']) for r in controls],color=color,label=method)
        axes[row,0].set_ylabel(scene+'\nMean progress (m)');axes[row,0].legend(fontsize=8)
        axes[row,1].set_ylabel('Alpha mean / range (deg)');axes[row,1].legend()
        axes[row,2].set_ylabel('Valid radius-window fraction');axes[row,2].set_ylim(-.02,1.02)
        for ax in axes[row]:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.18)
    fig.suptitle('Training pilots, seed 1 | Not the 160-trial formal comparison',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'pilot_progress.png',dpi=160);plt.close(fig)
    # Save every estimator variant, including unsuccessful ones.
    estimator=[]
    for case in runs:
        if case['method']!='D':continue
        feedback=sorted(case['_directory'].glob('feedback_*.json.gz'))
        for window in (7,9,13):
            for alternating in (False,True):
                fractions=[];residual=[];bias=[];axis_errors=[]
                for path in feedback:
                    g=read_gzip(path)
                    rows=estimate_local_radius(g['points_mm'],g['projection_axis'],window,alternating=alternating)
                    fractions.append(float(np.mean([r['valid'] for r in rows])))
                    residual += [r['residual_mm'] for r in rows if r['residual_mm'] is not None]
                    # In straight-only pilots, world pipe axis is z. This is
                    # offline scoring only, never an input to the controller.
                    if case['scene'] in ('pipe_straight_small','pipe_straight_large'):
                        direction=np.array(g['projection_axis']['direction'])
                        axis_errors.append(float(np.degrees(np.arccos(np.clip(abs(direction[2]),0,1)))))
                estimator.append(dict(scene=case['scene'],seed=case['seed'],window=window,alternating=alternating,
                                      mean_valid_fraction=float(np.mean(fractions)),
                                      zero_valid_frame_fraction=float(np.mean(np.array(fractions)==0)),
                                      residual_mm_median=float(np.median(residual)),
                                      axis_error_deg_median=float(np.median(axis_errors)) if axis_errors else None))
    (out/'estimator_feedback_diagnostics.json').write_text(json.dumps(estimator,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,4.8))
    labels=[];values=[]
    for window in (7,9,13):
        for alternating in (False,True):
            cases=[r for r in estimator if r['scene']=='pipe_straight_small' and r['window']==window and r['alternating']==alternating]
            labels.append(f'{window} points\n'+('two parity circles' if alternating else 'one circle'))
            values.append(float(np.mean([r['mean_valid_fraction'] for r in cases])) if cases else 0.)
    ax.bar(labels,values,color=['#617488','#db6c38']*3)
    for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.1%}',ha='center')
    ax.set_ylim(0,1.05);ax.set_ylabel('Mean valid-window fraction');ax.set_title('Small-pipe estimator diagnostics | Training feedback only')
    fig.tight_layout();fig.savefig(out/'estimator_gate.png',dpi=160);plt.close(fig)
    representative=next(r for r in runs if r['scene']=='pipe_taper_horizontal' and r['method']=='D')
    controls=read_gzip(representative['_directory']/'control.json.gz')['control']
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for stamp,color in zip((10.,22.,35.),('#577590','#d17a22','#42826a')):
        record=min(controls,key=lambda r:abs(r['sim_time']-stamp))
        distance=np.linspace(0,1354.5,len(record['alpha_rad']))
        label=f"t={record['sim_time']:.1f}s"
        for ax,key in zip(axes,('radius_mm','alpha_rad','implied_pitch_mm')):
            values=np.array(record[key]);values=np.degrees(values) if key=='alpha_rad' else values
            ax.plot(distance,values,color=color,label=label)
            invalid=~np.array(record['radius_valid']);ax.scatter(distance[invalid],values[invalid],s=17,marker='x',color=color)
    for ax,label in zip(axes,('Estimated / held radius (mm)','Alpha (deg)','Implied pitch (mm)')):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.18)
    axes[-1].set_xlabel('Body arc coordinate (mm)')
    fig.suptitle('Horizontal taper, D, seed 1 | x = invalid radius window / fallback node')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'body_pitch_distribution.png',dpi=160);plt.close(fig)
    checks=check_controller(out,folders)
    initial_errors=[]
    for scene in scenes:
        aa=next((r for r in runs if r['scene']==scene and r['method']=='A' and r['seed']==1),None)
        dd=next((r for r in runs if r['scene']==scene and r['method']=='D' and r['seed']==1),None)
        if aa and dd:
            pair=[]
            for case in (aa,dd):
                with np.load(case['_directory']/'trace.npz') as data:
                    index=int(np.argmin(abs(data['sim_time']-2.2)))
                    pair.append(data['qpos'][index])
            initial_errors.append(dict(scene=scene,max_initial_qpos_difference=float(abs(pair[0]-pair[1]).max())))
    small=[r for r in runs if r['scene']=='pipe_straight_small' and r['method']=='D']
    rejected=[dict(scene=r['scene'],seed=r['seed'],valid_fraction=r['radius_valid_fraction'],
                   zero_window_update_fraction=r['all_windows_invalid_update_fraction']) for r in small
              if r['radius_valid_fraction']==0.]
    stage0=json.loads((out/'stage0/summary.json').read_text(encoding='utf-8'))
    expected={(scene,method,1) for scene in scenes for method in ('A','D')} | {('pipe_straight_small',method,seed) for method in ('A','D') for seed in (2,3)}
    assert expected=={(r['scene'],r['method'],r['seed']) for r in runs},'Missing or unexpected verified pilots'
    for case in runs:
        for name,expected_hash in case['sha256'].items():
            if name.startswith('codes') or name.startswith('validation'):
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected_hash,('pilot source changed',name)
    manifest=dict(stage0_continuous_passed=stage0['geometry']['continuous_gate_passed'],stage1_estimator_passed=not bool(rejected),
                  stage2_released=False,formal_trials_executed=0,
                  reason='Necessary small-pipe feedback degenerates to fixed alpha; do not freeze B/C/D or run formal comparisons.',
                  rejected_cases=rejected,controller_checks=checks,initial_pairs=initial_errors,
                  pilots=len(runs),hypotheses=dict(H1='unresolved: hold slip is not proof of insufficient winding',
                  H2='unresolved: tuned B has not been selected',H3='unresolved: no frozen C versus D comparison',
                  H4='implementation gate failed for this estimator; not a refutation of all feedback methods'),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (ROOT/'snake_control/local_pitch_controller.py',ROOT/'snake_control/local_pitch_geometry.py',
                                  ROOT/'validation/pitch_adaptation_experiment.py')})
    (out/'release_gate.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/pitch_adaptation_20261003')
    analyze(parser.parse_args().out.resolve())
