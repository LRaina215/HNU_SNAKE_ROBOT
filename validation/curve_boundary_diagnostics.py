"""Saved boundary C0/B evidence and attempted-coordinate diagnosis, no mj_step."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import argparse
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from curve_local_path import BASE,LocalPathFit
from curve_guided_geometry import points
from snake_control.local_material_reference import BoundaryMaterialCurve


def main(name):
    case=BASE/name;z=np.load(case/'trace.npz')
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    controls=json.loads((case/'control.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    t=z['sim_time'];p=z['centers_mm'];ct=np.array([c['time_s'] for c in controls])
    h=np.degrees([c['retained_shape_rad'] for c in controls]);delta=np.degrees([c['compliance_delta_rad'] for c in controls])
    nominal=np.array([c['nominal_rad'] for c in controls]);target=np.array([c['target_rad'] for c in commands])
    counts=np.count_nonzero(z['contact_force_N']>.1,axis=2)
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
    rootvector=(z['qpos'][-1,:3]-z['qpos'][0,:3])*1000;centroid=p.mean(axis=1)
    fit=LocalPathFit();kin=mujoco.MjData(fit.m)
    def fk(q):
        kin.qpos[:]=0;kin.qpos[3]=1;kin.qpos[fit.qadr]=q;mujoco.mj_kinematics(fit.m,kin)
        return points(fit.m,kin).copy()
    axes=[]
    for moment in [0,5,10,20,float(ct[-1])]:
        ci=int(abs(ct-moment).argmin());i=int(abs(t-ct[ci]).argmin())
        fit.d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(fit.m,fit.d);R=fit.d.body('head').xmat.reshape(3,3)
        for a,world,side in zip(controls[ci]['material_axes'],[np.array([1.,0,0]),np.array([0.,1,0])],['source','target']):
            axes.append(dict(time_s=float(ct[ci]),side=side,cloud_rms_mm=a['cloud_rms_mm'],
                actual_axis_error_deg=float(np.degrees(np.arccos(np.clip(a['direction_head']@(R.T@world),-1,1))))))
    attempted=[]
    if result['outcome']=='controller_or_interface_stop':
        schedule_dir='local_path_boundary_speed3' if name.startswith('R2_SPEED3') else ('local_path_boundary_retimed' if name.startswith('R2_RETIMED') else 'local_path_boundary')
        schedule=np.load(BASE/schedule_dir/'command_schedule.npz')
        k=int(round(t[-1]/.05));feed=float(schedule['feed_mm'][k])
        curve=BoundaryMaterialCurve(case/'reference.npz')
        actual=fk(z['q'][-1]);nom=fk(curve.joint_angles(feed))
        center=curve.lambda0+feed-(np.arange(20)+1)*64.5
        smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
        for weight,side in zip([smooth((-center-64.5)/64.5),smooth((center-curve.bounds[-1]-64.5)/64.5)],['source','target']):
            ids=np.flatnonzero(weight>1e-8);lo,hi=int(ids[0]),int(ids[-1]+1)
            a,b=nom[lo:hi+2].copy(),actual[lo:hi+2].copy();a-=a.mean(0);b-=b.mean(0)
            u,s,v=np.linalg.svd(a.T@b);R=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
            attempted.append(dict(time_s=float(t[-1]),attempted_feed_mm=feed,side=side,
                cloud_rms_mm=float(np.sqrt(np.mean(np.sum((a@R.T-b)**2,axis=1)))),coverage_ratio=float(s[1]/s[0]),
                applied=False,diagnostic_uses_saved_joint_angles=True,encoder_quantization_limitation='Run received EXE-rounded angles; this uses saved simulation angles'))
    version='R2_SPEED3' if name.startswith('R2_SPEED3') else ('R2_RETIMED' if name.startswith('R2_RETIMED') else 'R2_BOUNDARY')
    paired=BASE/(version+('_PAIR_C0_PERTURB' if name.endswith('B_FREEZE') else '_PAIR_C0'))/'trace.npz'
    extra=None
    if name.endswith('_B') or name.endswith('B_FREEZE'):
        with np.load(paired) as c0:
            ci=int(abs(c0['sim_time']-t[-1]).argmin())
            extra=((p[-1]-p[0])[[8,9],1]-(c0['centers_mm'][ci]-c0['centers_mm'][0])[[8,9],1]).tolist()
    row=dict(physics_steps_added=0,physical_seconds=float(t[-1]),commands=len(commands),
        state_ack_identity_error_rad=float(abs(target-nominal-np.radians(h)).max()),
        after5_minimum_contacts_per_pipe=counts[t>=5-1e-8].min(axis=0).tolist(),
        maximum_retained_deg=float(abs(h).max()),maximum_compliance_difference_deg=float(abs(delta).max()),
        maximum_actual_torque_Nm=float(abs(z['torque']).max()),
        maximum_actual_command_speed_deg_s=float(np.degrees(abs(np.diff(z['ctrl'],axis=0))).max()/.05),
        angle_constrained_commands=sum(c['angle_constrained_joints']>0 for c in controls),
        rate_limited_commands=sum(c['rate_limited_joints']>0 for c in controls),
        root_displacement_vector_mm=rootvector.tolist(),geometric_centroid_displacement_mm=float(np.linalg.norm(centroid[-1]-centroid[0])),
        initial_to_final_head_rotation_deg=float(np.degrees((Rotation.from_quat(z['qpos'][-1,[4,5,6,3]])*
            Rotation.from_quat(z['qpos'][0,[4,5,6,3]]).inv()).magnitude())),
        scored_module_y_advance_mm=(p[-1]-p[0])[[8,9],1].tolist(),extra_scored_y_advance_over_matched_C0_mm=extra,
        added_external_force_maximum=float(z['external_applied_force_maximum'].max()),
        added_generalized_force_maximum=float(z['generalized_applied_force_maximum'].max()),
        ground_frames=int(np.count_nonzero(z['ground'])),nonadjacent_self_contact_frames=int(np.count_nonzero(z['self_contact'])),
        last_actual_feed_mm=float(z['feed_mm'][-1]),last_stage=str(z['stage'][-1]),
        conditional_axis_diagnostics=axes,coordinate_stop_attempt=attempted,
        world_truth_use='diagnostic/scoring only; not controller feedback')
    (case/'diagnostic_metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(3,2,figsize=(11,9),sharex=True)
    ax[0,0].plot(t,counts[:,0],label='Source');ax[0,0].plot(t,counts[:,1],label='Target');ax[0,0].axhline(2,color='k',ls=':');ax[0,0].legend();ax[0,0].set_ylabel('Contact modules >0.1 N')
    ax[0,1].plot(t[1:],speed.max(axis=1));ax[0,1].set_ylabel('Module speed (mm/s)')
    ax[1,0].plot(t,p[:,8,1]-p[0,8,1],label='Actual M8 Y');ax[1,0].plot(t,p[:,9,1]-p[0,9,1],label='Actual M9 Y');ax[1,0].plot(t,z['feed_mm'],ls='--',label='Commanded d');ax[1,0].legend();ax[1,0].set_ylabel('Material/command advance (mm)')
    ax[1,1].plot(t,np.degrees(abs(z['q']-z['ctrl'])).max(axis=1));ax[1,1].set_ylabel('Tracking error (deg)')
    ax[2,0].plot(ct,abs(h).max(axis=1),label='Executable residual');ax[2,0].plot(ct,abs(delta).max(axis=1),label='A-D difference');ax[2,0].legend();ax[2,0].set_ylabel('Adjustment (deg)')
    ax[2,1].plot(ct,[c['material_axes'][0]['cloud_rms_mm'] for c in controls],label='Source');ax[2,1].plot(ct,[c['material_axes'][1]['cloud_rms_mm'] for c in controls],label='Target');ax[2,1].axhline(16.125,color='r',ls=':');ax[2,1].legend();ax[2,1].set_ylabel('Cloud RMS (mm)')
    for a in ax.flat:a.grid(alpha=.2);a.axvline(5,color='.5',ls='--')
    ax[2,0].set_xlabel('Physical time (s)');ax[2,1].set_xlabel('Physical time (s)')
    fig.suptitle(name+' actual physics: '+result['outcome']);fig.tight_layout();fig.savefig(case/'actual_diagnostics.png',dpi=140);plt.close(fig)
    print(json.dumps({k:row[k] for k in ['physical_seconds','after5_minimum_contacts_per_pipe','last_actual_feed_mm','scored_module_y_advance_mm',
        'extra_scored_y_advance_over_matched_C0_mm','coordinate_stop_attempt','maximum_retained_deg']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);main(parser.parse_args().case)
