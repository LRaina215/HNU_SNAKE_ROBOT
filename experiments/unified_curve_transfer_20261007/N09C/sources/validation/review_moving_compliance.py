"""Review saved measured compliance, ordered reception and curve transactions."""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_adaptive_front_winding import review as review_winding


def torque_metrics(torque, active):
    values=np.abs(torque[active])
    if not values.size:return dict(samples=0)
    return dict(samples=int(values.size),mean_abs_Nm=float(values.mean()),
        p95_abs_Nm=float(np.percentile(values,95)),maximum_abs_Nm=float(values.max()),
        physical_saturation_fraction=float(np.mean(values>=2.355*.999)),
        above_software_035_Nm_fraction=float(np.mean(values>.35)))


def review(case,include_winding=True):
    if include_winding:review_winding(case)
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    tc=np.array([r['t_s'] for r in commands]);meta=[r['moving_compliance'] for r in commands]
    action=[r['adaptive_front_winding'] for r in commands]
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];torque=z['torque'].copy()
        weight=z['moving_compliance_softness'].copy();q=z['q'].copy();ctrl=z['ctrl'].copy()
    projected=np.array([r['actual_projected_joint_curve_coordinates_mm'] for r in meta])
    filtered=np.array([r['filtered_actual_joint_curve_coordinates_mm'] for r in meta])
    nominal=np.array([r['nominal_joint_curve_coordinates_mm'] for r in meta])
    distance=np.array([r['actual_projection_distance_mm'] for r in meta])
    recovery=np.array([r['actual_bearing_recovery_allowed'] for r in meta])
    cmweight=np.array([r['softness'] for r in meta])
    soft_error=np.array([r['permitted_soft_posture_error_deg'] for r in meta])
    guard_error=np.array([r['rigid_speed_guard_error_deg'] for r in meta])
    changes=[];previous=None
    for at,row in zip(tc,meta):
        members=row['material_joint_members']
        if members!=previous:changes.append(dict(t_s=float(at),joints=members));previous=members
    reasons=Counter()
    for row in action:
        for reject in row['curve_shape_update_rejections']:
            reasons[reject['reason'].split(':')[0]]+=1
    report=dict(name=case.name,outcome=result['outcome'],complete=result['complete'],
        ordered_target_handover_modules=result.get('ordered_target_handover_modules'),
        membership_changes=changes,shape_rejection_counts=dict(reasons),
        actual_torque_all=torque_metrics(torque,np.ones_like(weight,dtype=bool)),
        actual_torque_compliant_weight_over_09=torque_metrics(torque,weight>.9),
        actual_torque_rigid_weight_under_005=torque_metrics(torque,weight<.05),
        software_band_caveat='0.35 Nm ctrl-q band at dispatch is not a measured torque cap during held commands; no changes to physical kp or force limits',
        maximum_actual_projection_distance_mm=float(distance.max()),
        maximum_actual_vs_nominal_station_difference_mm=float(np.abs(projected-nominal).max()),
        maximum_permitted_soft_geometric_error_deg=float(soft_error.max()),
        maximum_rigid_speed_guard_error_deg=float(guard_error.max()),
        final_rigid_guard_joint=int(guard_error[-1].argmax()+1),
        final_rigid_speed_guard_error_deg=float(guard_error[-1].max()),
        final_actual_projection_distance_mm=distance[-1].tolist(),
        final_filtered_actual_curve_station_mm=filtered[-1].tolist(),
        final_nominal_curve_station_mm=nominal[-1].tolist(),
        final_softness=cmweight[-1].tolist(),
        ever_compliant_joint_recovery=[])
    for j in range(20):
        entered=np.flatnonzero(cmweight[:,j]>.5)
        if not len(entered):continue
        recovered=np.flatnonzero(recovery[:,j] & (np.arange(len(tc))>=entered[0]))
        report['ever_compliant_joint_recovery'].append(dict(joint=j+1,
            first_compliant_t_s=float(tc[entered[0]]),
            first_actual_bearing_recovery_t_s=float(tc[recovered[0]]) if len(recovered) else None,
            measured_soft_torque=torque_metrics(torque[:,j],weight[:,j]>.9)))
    report['checkpoints']=[]
    for at in [5,15,30,45,60,90,120,float(t[-1])]:
        if at>t[-1]+1e-6:continue
        k=int(np.argmin(abs(tc-at)));i=int(np.argmin(abs(t-at)))
        report['checkpoints'].append(dict(t_s=float(t[i]),members=meta[k]['material_joint_members'],
            ordered_incoming=meta[k]['ordered_incoming_target_modules'],
            max_soft_error_deg=float(soft_error[k].max()),
            rigid_guard_max_error_deg=float(guard_error[k].max()),
            rigid_guard_joint=int(guard_error[k].argmax()+1),
            front_guard_error_deg=action[k]['front_speed_guard_error_deg'],
            rear_guard_error_deg=action[k]['rear_speed_guard_error_deg'],
            max_actual_motor_abs_torque_Nm=float(np.abs(torque[i]).max()),
            actual_motor_torque_Nm=torque[i].tolist(),
            ctrl_minus_actual_q_deg=np.degrees(ctrl[i]-q[i]).tolist()))
    baseline=case.parent/'N05L'/'trace.npz'
    if baseline.exists():
        with np.load(baseline) as z:
            bt=z['sim_time']-z['sim_time'][0];btorque=z['torque'].copy()
        report['N05_same_joint_time_observational_comparison']=dict(
            note='different trajectories after start; descriptive comparison, not isolation of compliance causality',
            N05_J9_J18_5_to_50s=torque_metrics(btorque[(bt>=5)&(bt<=50),8:18],np.ones_like(btorque[(bt>=5)&(bt<=50),8:18],dtype=bool)),
            current_J9_J18_5_to_50s=torque_metrics(torque[(t>=5)&(t<=50),8:18],np.ones_like(torque[(t>=5)&(t<=50),8:18],dtype=bool)))
    (case/'moving_compliance_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    axs[0].imshow(weight.T,origin='lower',aspect='auto',extent=[t[0],t[-1],.5,20.5],vmin=0,vmax=1)
    axs[0].set_ylabel('Material joint');axs[0].set_title('Actual-position compliance membership / softness')
    soft=weight>.9;den=np.maximum(soft.sum(1),1)
    axs[1].plot(t,(np.abs(torque)*soft).sum(1)/den,label='Mean measured |torque| of soft joints')
    axs[1].plot(t,(soft&(np.abs(torque)>=2.355*.999)).sum(1)/den,label='Soft physical saturation fraction')
    axs[1].axhline(.35,color='gray',linestyle='--',label='Software dispatch effort band');axs[1].legend();axs[1].set_ylabel('Nm / fraction')
    axs[2].plot(tc,soft_error.max(1),label='Allowed soft geometric error')
    axs[2].plot(tc,guard_error.max(1),label='Rigid speed guard error');axs[2].legend();axs[2].set_ylabel('deg')
    for ax in axs:ax.set_xlabel('Actual elapsed time / s')
    fig.savefig(case/'moving_compliance_review.png',dpi=140);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['name','membership_changes','actual_torque_compliant_weight_over_09',
        'ordered_target_handover_modules','final_rigid_guard_joint','final_rigid_speed_guard_error_deg','shape_rejection_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',type=Path,nargs='+')
    p.add_argument('--raw-only',action='store_true');args=p.parse_args()
    for case in args.cases:review(case,not args.raw_only)
