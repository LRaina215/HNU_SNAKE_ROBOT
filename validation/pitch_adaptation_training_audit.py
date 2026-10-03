"""Independent raw-data audit and bounded stage1 parameter selection.

Can audit completed cases while the fixed-source training process is alive.
--freeze refuses incomplete/mismatched training or missing simulation evidence.
Never reads test seeds to tune parameters and never starts a simulation.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import scipy

from pitch_adaptation_experiment_v5 import ROOT,SCENES,ALPHA,model_path
sys.path.insert(0,str(ROOT/'codes'))
from adaptive_debug import pipe_progress

BASE=ROOT/'experiments/pitch_adaptation_20261003/repair_v5'


def runtime_versions():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__)


def model_asset_hashes():
    assets={}
    for scene in SCENES:
        model_file=model_path(scene);document=ET.parse(model_file).getroot()
        compiler=document.find('compiler')
        for element in document.findall('./asset/*'):
            filename=element.get('file')
            if filename is None:continue
            directory=compiler.get('meshdir' if element.tag=='mesh' else 'texturedir','') if compiler is not None else ''
            asset=(model_file.parent/directory/filename).resolve()
            if ROOT not in asset.parents:raise ValueError('Model asset is outside the repository: '+str(asset))
            assets[str(asset.relative_to(ROOT))]=sha(asset)
    return assets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def expected_jobs(plan):
    jobs=[]
    for scene in plan['scenes']:
        for seed in plan['seeds']:
            jobs.extend((scene,'B',seed,0.,math.radians(a)) for a in plan['B_alpha_deg'])
            jobs.extend((scene,method,seed,k,ALPHA) for method in ('C','D') for k in plan['CD_shared_k'])
    return jobs


def label(job):
    scene,method,seed,k,alpha=job
    return f'{scene}__{method}__s{seed}__k{k:g}__a{math.degrees(alpha):.4f}'


def audit_case(folder,job):
    result=load(folder/'result.json')
    scene,method,seed,k,alpha=job
    if (result['scene'],result['method'],result['seed'])!=(scene,method,seed) or abs(result['k']-k)>1e-12 or abs(result['alpha_deg']-math.degrees(alpha))>1e-9:
        raise ValueError('Trial identity differs: '+str(folder))
    for name,expected in result['sha256'].items():
        path=(ROOT/name).resolve()
        if ROOT.resolve() not in path.parents or sha(path)!=expected:
            raise ValueError('Source/model/EXE hash differs: '+name)
    missing=[name for name in ('trace.npz','control.json.gz','execution.log') if not (folder/name).exists()]
    if missing:raise ValueError('Missing raw evidence: '+str(folder)+' '+str(missing))
    if mujoco.__version__!='3.3.0':raise ValueError('Raw-data reconstruction requires MuJoCo 3.3.0')
    perturbation=np.random.default_rng(seed).uniform([.8,0.],[1.2,2*np.pi])
    log=(folder/'execution.log').read_text(encoding='utf8')
    observed=re.search(r'TRIAL seed=(\d+) 管道摩擦×([\d.]+) 初始绕管轴转动=([\d.]+)°',log)
    if observed is None or int(observed[1])!=seed or abs(float(observed[2])-perturbation[0])>.00050001 or abs(float(observed[3])-math.degrees(perturbation[1]))>.05000001:
        raise ValueError('Declared seed perturbation differs from execution log')
    with gzip.open(folder/'control.json.gz','rt',encoding='utf8') as f:raw=json.load(f)
    updates=raw['control'];physics=raw['physics']
    if not physics:raise ValueError('No physical record: '+str(folder))
    times=np.array([r['sim_time'] for r in physics]);progress=np.array([r['progress'] for r in physics])
    forces=np.array([r['actuator_force'] for r in physics])
    differences=np.diff(times)
    if not np.isfinite(progress).all() or not np.isfinite(forces).all() or np.any(differences<=0) or np.any(differences>.010000001):
        raise ValueError('Nonfinite data or physical recording gap: '+str(folder))
    if np.max(abs(forces))>2.355+1e-8:raise ValueError('Model torque limit exceeded: '+str(folder))
    model=mujoco.MjModel.from_xml_path(str(model_path(scene)))
    if model.nu!=20 or model.opt.timestep!=.001 or not np.allclose(model.actuator_gainprm[:,0],127.) or not np.allclose(model.actuator_forcerange,[-2.355,2.355]):
        raise ValueError('Fixed model conditions differ')
    native=model.numeric('experiment_helix_mm').data
    previous_alpha=np.full(22,math.atan2(native[1],2*np.pi*native[0]))
    max_step=0.;alpha_min=math.inf;alpha_max=-math.inf;valid_count=0;observations=0;zero_frames=0
    fallback_nodes=0;clipped=0;constraints=0;spreads=[]
    for row in updates:
        a=np.array(row['alpha_rad']);radius=np.array([np.nan if x is None else x for x in row['radius_used_mm']])
        active=np.array(row['pitch_constraint_active'],bool)
        if a.shape!=(22,) or not np.isfinite(a).all() or np.any(a<math.radians(8.)-1e-9) or np.any(a>math.radians(30.)+1e-9):
            raise ValueError('Alpha field invalid')
        change=float(np.degrees(abs(a-previous_alpha).max()))
        if change>row['dt_s']/.2+1e-7 or row['dt_s']<0:
            raise ValueError('Physical alpha rate exceeded: '+str(folder))
        expected_p=2*np.pi*radius*np.tan(a)
        stored_p=np.array([np.nan if x is None else x for x in row['implied_pitch_mm']])
        if np.any(active&~np.isfinite(expected_p)) or not np.allclose(expected_p[active],stored_p[active],atol=1e-8) or np.any(expected_p[active]<100.-1e-7) or np.any(expected_p[active]>300.+1e-7):
            raise ValueError('Active post-filter pitch bounds violated: '+str(folder))
        if method in ('A','B','C') and np.ptp(a)>1e-12:raise ValueError('Uniform control became local')
        if row['max_jump_deg']>30.+1e-8 or row['method']!=method or row['version']!=5:
            raise ValueError('Target policy/version differs')
        valid=np.array(row['radius_valid'],bool)
        valid_count+=int(valid.sum());observations+=len(valid);zero_frames+=not valid.any()
        fallback_nodes+=len(a)-sum(row['radius_usable'])
        clipped+=row['joint_clipped'];constraints+=int(active.sum())
        max_step=max(max_step,change);alpha_min=min(alpha_min,float(np.degrees(min(a))));alpha_max=max(alpha_max,float(np.degrees(max(a))))
        spreads.append(float(np.degrees(np.ptp(a))));previous_alpha=a
    with np.load(folder/'trace.npz') as trace:
        qpos=trace['qpos'];recorded_contacts=trace['pipe_contacts']
        t=trace['sim_time'];initial_indices=np.flatnonzero(np.isclose(t,2.2,atol=1e-9))
        if not len(initial_indices):raise ValueError('Common initialization absent')
        initial_q=qpos[initial_indices[0]].copy()
        motion=np.flatnonzero((t>2.2+1e-8)&trace['moving']&(abs(trace['roll_phase'])>1e-9))
        first_compliant=bool(trace['compliance_active'][motion[0]]) if len(motion) else None
        if len(motion) and not first_compliant:raise ValueError('Nominal motion after common initialization')
        ctrl=trace['ctrl'];limits=model.jnt_range[model.actuator_trnid[:,0]]
        if np.any(ctrl<limits[:,0]-1e-9) or np.any(ctrl>limits[:,1]+1e-9):raise ValueError('Delivered joint limit exceeded')
        last_phase=float(trace['roll_phase'][-1]);maximum_tracking=float(np.degrees(trace['tracking_error_rad']).max())
        if qpos.shape[1]!=27:raise ValueError('Wrong joint count in trace')
        # mj_step integrates qpos after its last kinematics pass. The bridge
        # refreshes poses on its wall-clock publication schedule, so a cached
        # 10ms xanchor can lag qpos by 1ms. Rebuild scoring positions from every
        # recorded qpos, independently of that schedule. No physical rerun.
        unique_t,unique_i=np.unique(t,return_index=True)
        positions=np.searchsorted(unique_t,times)
        if np.any(positions>=len(unique_t)) or not np.allclose(unique_t[positions],times,atol=1e-10,rtol=0):
            raise ValueError('Physical records lack matching qpos')
        data=mujoco.MjData(model);reconstructed=[]
        for index in unique_i[positions]:
            data.qpos[:]=qpos[index]
            mujoco.mj_kinematics(model,data)
            reconstructed.append(pipe_progress(model,data))
        rebuilt=np.asarray(reconstructed)
        maximum_cache_offset_mm=float(abs(rebuilt-progress).max()*1000)
        progress=rebuilt
    initial=int(np.argmin(abs(times-2.2)))
    hold_start=result.get('hold_start_s')
    if hold_start is not None:
        h=np.flatnonzero(times>=hold_start-1e-8)
        if not len(h):raise ValueError('Hold marker lacks physical data')
        if abs(times[h[0]]-hold_start)>1e-8:raise ValueError('Hold marker lacks exact-time qpos')
        start_mean=float(progress[h[0],0])
        retreat=max(0.,float(start_mean-min(progress[h,0])))*1000
        hold_duration=float(times[h[-1]]-hold_start)
        hold_ok=retreat<=20.+1e-9 and hold_duration>=10.-1e-8
        movement_end=h[0]
    else:
        retreat=None;hold_duration=None;hold_ok=False;movement_end=len(times)-1
    moved=float(progress[movement_end,0]-progress[initial,0]);motion_s=float(times[movement_end]-times[initial])
    speed=moved/motion_s if motion_s>0 else 0.
    if scene.startswith('pipe_straight'):
        goal_at_hold=hold_start is not None and moved>=.5-1e-8
    else:
        feature=result['goal_feature_m']
        rolling=[s for s in result['status'] if hold_start is None or s['sim_time']<=hold_start+1e-8]
        # Preserve the protocol's 1s checkpoint cadence, but score the actual
        # qpos at those instants instead of potentially cached anchor positions.
        checkpoint_indices=[int(np.argmin(abs(times-s['sim_time']))) for s in rolling]
        first=next((i for i in checkpoint_indices if progress[i,1]>feature),None)
        goal_at_hold=hold_start is not None and first is not None and progress[movement_end,1]>feature and progress[movement_end,0]-progress[first,0]>=.2-1e-8
        if hold_start is not None and result['success'] and np.any(progress[times>=hold_start-1e-8,1]<=feature):
            raise ValueError('Successful hold did not keep all joints beyond taper')
    ground=any(row['ground_contact'] for row in physics)
    if result['success'] and not (hold_ok and goal_at_hold and not ground):raise ValueError('Success contradicted by raw physical data')
    if result['outcome']=='hold_slip' and not (retreat is not None and retreat>20.):raise ValueError('Hold failure contradicted by raw data')
    if result['outcome']=='ground_collision' and not ground:raise ValueError('Ground failure lacks recorded ground contact')
    if result['outcome']=='support_loss' and recorded_contacts[-1]!=0:raise ValueError('Support loss lacks final zero-contact observation')
    if result['outcome']=='time_budget' and motion_s<result['movement_budget_s']-1e-8:raise ValueError('Time-budget failure ended before the physical budget')
    if result['outcome']=='hold_region_lost' and not (hold_start is not None and progress[-1,1]<=result['goal_feature_m']):
        raise ValueError('Hold region loss contradicted by reconstructed position')
    if result['outcome']=='no_progress_10s':
        checkpoints=result['status'];last=checkpoints[-1]
        earlier=next((s for s in reversed(checkpoints[:-1]) if last['sim_time']-s['sim_time']>=10.-1e-8),None)
        if earlier is None:raise ValueError('No-progress failure lacks a full checkpoint window')
        lo=int(np.argmin(abs(times-earlier['sim_time'])));hi=int(np.argmin(abs(times-last['sim_time'])))
        if progress[hi,0]-progress[lo,0]>=.01:raise ValueError('No-progress failure contradicted by reconstructed position')
    dt=np.diff(np.r_[0.,times]);torque_squared=float(np.sum(np.sum(forces**2,axis=1)*dt))
    row=dict(scene=scene,method=method,seed=seed,k=k,alpha_deg=math.degrees(alpha),outcome=result['outcome'],success=bool(result['success']),
             friction_multiplier=float(perturbation[0]),initial_axis_rotation_deg=float(np.degrees(perturbation[1])),
             perturbation_evidence='exact seeded draws from verified gait source; rounded execution log checked',
             motion_s=motion_s,mean_progress_m=moved,speed_m_s=speed,rolling_cycles=abs(last_phase)/(2*np.pi),
             cumulative_retreat_m=float(np.maximum(-np.diff(progress[initial:,0]),0.).sum()),
             progress_per_cycle_m=moved/(abs(last_phase)/(2*np.pi)) if last_phase else None,
             hold_max_retreat_mm=retreat,hold_duration_s=hold_duration,hold_passed=hold_ok,
             torque_rms_nm=float(np.sqrt(np.average(np.mean(forces**2,axis=1),weights=dt))),
             torque_squared_integral_nm2_s=torque_squared,torque_squared_per_m=torque_squared/moved if moved>0 else None,
             saturation_fraction=float(np.average(np.mean(abs(forces)>=2.355*.999,axis=1),weights=dt)),
             maximum_tracking_deg=maximum_tracking,joint_clip_events=clipped,maximum_alpha_step_deg=max_step,
             alpha_min_deg=alpha_min if updates else None,alpha_max_deg=alpha_max if updates else None,
             maximum_local_alpha_spread_deg=max(spreads) if spreads else None,
             radius_valid_fraction=valid_count/observations if observations else None,
             zero_valid_update_fraction=zero_frames/len(updates) if updates else None,
             fallback_node_fraction=fallback_nodes/observations if observations else None,
             active_pitch_constraint_observations=constraints,first_motion_compliant=first_compliant,
             source_directory=str(folder.relative_to(ROOT)),source_hashes_verified=True,
             result_sha256=sha(folder/'result.json'),physical_sample_max_gap_s=float(max(differences)) if len(differences) else None,
             raw_sha256={name:sha(folder/name) for name in ('trace.npz','control.json.gz','execution.log')},
             maximum_cached_pose_progress_offset_mm=maximum_cache_offset_mm,
             error=result.get('error'),ground_contact=ground,canceled_feedback_retries=result['canceled_feedback_retries'])
    return row,initial_q


def score(rows,tie):
    retreat=np.array([math.inf if r['hold_max_retreat_mm'] is None else r['hold_max_retreat_mm'] for r in rows])
    median=float(np.median(retreat));speed=float(np.median([r['speed_m_s'] for r in rows]))
    # Success count first. Missing hold cannot beat a measured retreat.
    rank=(-sum(r['success'] for r in rows),median,-speed,tie)
    serial=dict(successes=sum(r['success'] for r in rows),trials=len(rows),median_hold_retreat_mm=None if not math.isfinite(median) else median,
                median_hold_retreat_missing_or_infinite=not math.isfinite(median),median_progress_speed_m_s=speed)
    return rank,serial


def audit(out,freeze=False):
    training=BASE/'training';plan=load(training/'training_plan.json')
    if plan['seeds']!=[1,2,3] or plan['scenes']!=list(SCENES) or plan['parameter_configurations']>12:
        raise ValueError('Training plan changed')
    for name,expected in plan['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Immutable training source changed: '+name)
    stage0=load(ROOT/'experiments/pitch_adaptation_20261003/stage0/summary.json')
    if not stage0['geometry']['continuous_gate_passed'] or stage0['feasible_cases']!=35:raise ValueError('Offline geometry gate missing')
    for name,expected in stage0['provenance']['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Stage0 source changed')
    jobs=expected_jobs(plan);rows=[];missing=[];initials={};initial_counts={}
    for job in jobs:
        folder=training/label(job)
        if not (folder/'result.json').exists() and job[1]=='D' and job[2]==1 and job[3]==0:
            folder=BASE/'pilots'/label(job)
        if not (folder/'result.json').exists():missing.append(label(job));continue
        row,initial=audit_case(folder,job);rows.append(row)
        key=(job[0],job[2]);reference=initials.get(key)
        if reference is not None and not np.array_equal(initial,reference):raise ValueError('Paired common initial qpos differs: '+str(key))
        initials[key]=initial
        initial_counts[key]=initial_counts.get(key,0)+1
    out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'audited_training.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=dict(complete=not missing,audited=len(rows),required=len(jobs),missing=missing,
                audit_environment=runtime_versions(),
                audit_source_sha256=sha(Path(__file__)),
                common_initial_groups_observed=len(initials),
                common_initial_groups_verified=sum(count>=2 for count in initial_counts.values()),
                common_initial_group_case_counts={f'{scene}__s{seed}':count for (scene,seed),count in initial_counts.items()},
                source_and_raw_checks_passed=True,
                training_plan_sha256=sha(training/'training_plan.json'),formal_released=False,rows=rows)
    freeze_error=None
    if not missing:
        choices={};candidates=[]
        for scene in SCENES:
            groups=[]
            for angle in plan['B_alpha_deg']:
                group=[r for r in rows if r['scene']==scene and r['method']=='B' and abs(r['alpha_deg']-angle)<1e-9]
                if len(group)!=3:raise ValueError('B candidate count differs')
                rank,values=score(group,angle);groups.append((rank,angle));candidates.append(dict(scene=scene,method='B',alpha_deg=angle,**values))
            choices[scene]=min(groups)[1]
        common=[]
        for k in plan['CD_shared_k']:
            group=[r for r in rows if r['method'] in ('C','D') and r['k']==k]
            if len(group)!=24:raise ValueError('Shared C/D candidate count differs')
            rank,values=score(group,k);common.append((rank,k));candidates.append(dict(scene='all',method='C+D',k=k,**values))
        selected_k=min(common)[1]
        report.update(selected_B_alpha_deg=choices,selected_CD_k=selected_k,candidate_scores=candidates)
        # Communication/missing-init is an engineering failure requiring a
        # transparent audit, not a usable result for a parameter freeze.
        engineering=[r['source_directory'] for r in rows if r['outcome'] in ('communication_failure','initialization_failure')]
        selected_D=[r for r in rows if r['method']=='D' and r['k']==selected_k]
        usable_D=all(r['radius_valid_fraction'] is not None and r['radius_valid_fraction']>0 and
                     r['maximum_local_alpha_spread_deg'] is not None and r['maximum_local_alpha_spread_deg']>1e-6 for r in selected_D)
        report['freeze_conditions']=dict(no_engineering_failures=not engineering,selected_D_feedback_and_local_targets_present=usable_D,
                                         selected_D_complete=[r['source_directory'] for r in selected_D],engineering_failures=engineering)
        if freeze:
            if engineering or not usable_D:
                freeze_error='Training freeze gate failed; see audit report'
            frozen=dict(schema=1,version=5,stage='formal_parameter_freeze',training_seeds=[1,2,3],test_seeds=list(range(101,111)),
                        scenes=list(SCENES),methods=['A','B','C','D'],formal_trials=160,
                        A_alpha_deg=math.degrees(ALPHA),B_alpha_deg=choices,CD_k=selected_k,
                        filters=plan['fixed_filters'],conditions=dict(joints=20,mujoco='3.3.0',timestep_s=.001,kp=127.,torque_limit_nm=2.355,
                            gain=[.01,.02],movement_budget_s=120.,hold_s=10.,hold_retreat_limit_mm=20.,roll_step_rad=.15,step_s=.2,
                            straight_mean_progress_m=.5,taper_additional_progress_m=.2,no_progress_window_s=10.,no_progress_min_m=.01),
                        training_plan_sha256=report['training_plan_sha256'],training_result_sha256={r['source_directory']:r['result_sha256'] for r in rows},
                        training_raw_sha256={r['source_directory']:r['raw_sha256'] for r in rows},
                        audit_source_sha256=sha(Path(__file__)),
                        sources=plan['sources'],models={scene:sha(model_path(scene)) for scene in SCENES},
                        exe_sha256=sha(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe'),formal_released=True)
            frozen['dependencies']=runtime_versions()
            frozen['model_assets']=model_asset_hashes()
            file=out/'parameter_freeze.json'
            if file.exists() and load(file)!=frozen:
                freeze_error='Existing freeze differs; no silent refreeze'
            if freeze_error is None:
                file.write_text(json.dumps(frozen,indent=2,ensure_ascii=False),encoding='utf8')
                report['formal_released']=True
    if freeze_error is not None:report['freeze_error']=freeze_error
    (out/'training_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    if freeze_error is not None:raise ValueError(freeze_error)
    if freeze and missing:raise ValueError('Incomplete training: '+str(len(missing))+' required cases missing')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','missing','candidate_scores')},indent=2,ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE/'training_audit')
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();audit(args.out,args.freeze)
