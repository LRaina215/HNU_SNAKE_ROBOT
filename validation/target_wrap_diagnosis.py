"""Round2 diagnosis of the existing R2 trial, no physics stepping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import csv
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from curve_local_path import LocalPathFit
from snake_control.local_material_reference import BoundaryMaterialCurve

PARENT=ROOT/'experiments/curve_guided_transfer_20261006'
OLD=PARENT/'stage_continuation_20261006'
BASE=PARENT/'target_wrap_round2'


def write(name,value):
    (BASE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def radial_geometry(points):
    relative=points-np.array([1100.,0.,1000.])
    radius=np.linalg.norm(relative[:,[0,2]],axis=1)
    polar=np.unwrap(np.arctan2(relative[:9,2],relative[:9,0]))
    return radius,float(np.degrees(abs(polar[-1]-polar[0]))),float(np.degrees(np.ptp(polar)))


def runs(time,mask):
    indices=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
    return [dict(begin_s=float(time[a]),end_s=float(time[min(b,len(time)-1)]),duration_s=float((b-a)*.01))
            for a,b in zip(indices[::2],indices[1::2]) if (b-a)*.01>=.5]


def main():
    BASE.mkdir(exist_ok=False)
    z=np.load(OLD/'R2_SPEED3_B/trace.npz')
    controls=json.loads((OLD/'R2_SPEED3_B/control.json').read_text(encoding='utf8'))
    t=z['sim_time'];ct=np.array([r['time_s'] for r in controls]);force=z['contact_force_N'][:,1]
    seedids=np.flatnonzero(force[(t>=4)&(t<=5)].mean(axis=0)>.1).tolist()
    fit=LocalPathFit();reference=BoundaryMaterialCurve(OLD/'local_path_boundary_speed3/reference_ready.npz')
    rows=[]
    times=np.unique(np.r_[np.arange(0,t[-1],.5),5,31.9,34.9,50,56.34,56.83,61.8,t[-1]])
    for moment in times:
        i=int(abs(t-moment).argmin());ci=int(abs(ct-t[i]).argmin());c=controls[ci]
        fit.d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(fit.m,fit.d);fit.last=None
        R=fit.d.body('head').xmat.reshape(3,3).copy()
        radius,turn,span=radial_geometry(z['points_mm'][i])
        actualgaps=[float(fit.radial_gap(m,1)) for m in range(10)]
        actualgeom=fit.d.qpos.copy()
        fit.set(reference.state(float(c['feed_mm'])))
        nominalgaps=[float(fit.radial_gap(m,1)) for m in range(10)]
        errors=[]
        for a,world in zip(c['material_axes'],[np.array([1.,0,0]),np.array([0.,1,0])]):
            errors.append(float(np.degrees(np.arccos(np.clip(a['direction_head']@(R.T@world),-1,1)))))
        row=dict(time_s=float(t[i]),stage=str(z['stage'][i]),feed_mm=float(z['feed_mm'][i]),
            source_axis_error_deg=errors[0],target_axis_error_deg=errors[1],
            old_front_contact_modules=[m for m in seedids if force[i,m]>.1],
            all_target_contact_modules=np.flatnonzero(force[i]>.1).tolist(),
            old_front_sum_target_force_N=float(force[i,seedids].sum()),
            M8_force_N=float(force[i,8]),M9_force_N=float(force[i,9]),
            front_M0_M7_polar_turn_deg=turn,front_M0_M7_polar_span_deg=span,
            front_node_radius_mm=radius[:11].tolist(),
            target_projected_shell_gaps_mm=actualgaps,nominal_target_shell_gaps_mm=nominalgaps,
            J1_J10_grasp_weights=c['grasp_weights'][:10],
            J1_J10_retained_deg=np.degrees(c['retained_shape_rad'][:10]).tolist(),
            J1_J10_AD_difference_deg=np.degrees(c['compliance_delta_rad'][:10]).tolist(),
            rate_projected_joints=c['rate_limited_joints'],
            torque_saturated_joints=np.flatnonzero(abs(z['torque'][i])>=2.355-1e-6).tolist(),
            front_M0_M7_Y_min_mm=float(z['points_mm'][i,:9,1].min()),
            old_front_max_radius_mm=float(radius[:9].max()))
        rows.append(row)
    intervals={str(m):runs(t,force[:,m]>.1) for m in sorted(set(seedids+[8,9]))}
    write('baseline_timeline.json',dict(physics_steps=0,rows=rows,initial_target_bearing_modules=seedids,
        sustained_target_contact_intervals=intervals,
        geometry_limitation='Projected shell gaps use infinite target cylinder; only meaningful with longitudinal overlap. Polar turn of an open body chain is descriptive, not a topological enclosure proof.',
        truth_use='OFFLINE ONLY; no controller input'))
    with (BASE/'baseline_timeline.csv').open('w',encoding='utf8',newline='') as f:
        fields=['time_s','stage','feed_mm','source_axis_error_deg','target_axis_error_deg','old_front_sum_target_force_N',
            'M8_force_N','M9_force_N','front_M0_M7_polar_turn_deg','old_front_max_radius_mm']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows([{k:r[k] for k in fields} for r in rows])
    moments=[0,5,30,50,56.34,61.8,float(t[-1])]
    selected=[min(rows,key=lambda r:abs(r['time_s']-m)) for m in moments]
    write('baseline_summary.json',dict(physics_steps=0,initial_target_bearing_modules=seedids,selected=selected,
        source_case='R2_SPEED3_B',prior_dynamic_trials=11,new_dynamic_trials=0))
    fig,ax=plt.subplots(3,2,figsize=(11,9),sharex=True)
    tt=np.array([r['time_s'] for r in rows])
    for m in seedids+[8,9]:ax[0,0].plot(t,force[:,m],label='M'+str(m))
    ax[0,0].set_ylabel('Actual target normal force (N)');ax[0,0].legend(ncol=3)
    ax[0,1].plot(tt,[r['source_axis_error_deg'] for r in rows],label='Source');ax[0,1].plot(tt,[r['target_axis_error_deg'] for r in rows],label='Target');ax[0,1].legend();ax[0,1].set_ylabel('Axis error (deg), offline truth')
    ax[1,0].plot(tt,[r['front_M0_M7_polar_turn_deg'] for r in rows]);ax[1,0].set_ylabel('Original front polar turn (deg)')
    for m in [0,3,6,8,9]:ax[1,1].plot(tt,[r['target_projected_shell_gaps_mm'][m] for r in rows],label='M'+str(m))
    ax[1,1].legend();ax[1,1].set_ylabel('Projected shell gap (mm)')
    for j in [5,6,7,8,9]:ax[2,0].plot(tt,[r['J1_J10_grasp_weights'][j] for r in rows],label='J'+str(j+1))
    ax[2,0].legend();ax[2,0].set_ylabel('Grasp participation')
    for j in [5,6,7,8,9]:ax[2,1].plot(tt,[r['J1_J10_retained_deg'][j] for r in rows],label='J'+str(j+1))
    ax[2,1].legend();ax[2,1].set_ylabel('Acknowledged deformation (deg)')
    for a in ax.flat:a.grid(alpha=.2);a.axvspan(31.9,34.9,color='.8',alpha=.5);a.axvspan(61.8,66.8,color='r',alpha=.1)
    ax[2,0].set_xlabel('Physics time (s)');ax[2,1].set_xlabel('Physics time (s)')
    fig.suptitle('Existing R2 baseline only; no new physical evolution');fig.tight_layout();fig.savefig(BASE/'baseline_diagnosis.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(initial_bearing_modules=seedids,selected=[{k:r[k] for k in ['time_s','old_front_contact_modules',
        'target_axis_error_deg','front_M0_M7_polar_turn_deg','old_front_max_radius_mm','M8_force_N','M9_force_N']} for r in selected]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
