"""Diagnose saved interface holding; no physics steps or controller feedback."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from curve_local_path import BASE,OUT,LocalPathFit
from curve_guided_geometry import points
from snake_control.curve_grasp_revision import FittedMaterialCurve
from snake_control.local_material_reference import CoupledMaterialCurve
from snake_control.executable_grasp_state import EquilibriumGraspController


def main():
    case=BASE/'R2_PATH_PAIR_C0';z=np.load(case/'trace.npz')
    controls=json.loads((case/'control.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    t=z['sim_time'];p=z['centers_mm'];ct=np.array([r['time_s'] for r in controls])
    counts=np.count_nonzero(z['contact_force_N']>.1,axis=2)
    speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
    h=np.degrees([r['retained_shape_rad'] for r in controls])
    delta=np.degrees([r['compliance_delta_rad'] for r in controls])
    nominal=np.array([r['nominal_rad'] for r in controls]);targets=np.array([c['target_rad'] for c in commands])
    centroid=p.mean(axis=1);drift=np.linalg.norm(centroid-centroid[0],axis=1)
    fit=LocalPathFit();kin=mujoco.MjData(fit.m)
    def encoder_fk(q):
        kin.qpos[:]=0.;kin.qpos[3]=1.;kin.qpos[fit.qadr]=q
        mujoco.mj_kinematics(fit.m,kin)
        return points(fit.m,kin)
    axes=[]
    for moment in [0,5,15,30,float(ct[-1])]:
        ci=int(abs(ct-moment).argmin());i=int(abs(t-ct[ci]).argmin())
        fit.d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(fit.m,fit.d)
        R=fit.d.body('head').xmat.reshape(3,3)
        for a,world,side in zip(controls[ci]['material_axes'],[np.array([1.,0,0]),np.array([0.,1,0])],['source','target']):
            axes.append(dict(time_s=float(ct[ci]),side=side,cloud_rms_mm=a['cloud_rms_mm'],
                actual_axis_error_deg=float(np.degrees(np.arccos(np.clip(np.dot(a['direction_head'],R.T@world),-1,1))))))
    # Same saved held state, different initial reference/gauge, as offline
    # control arithmetic. This does not supply an alternative dynamic trial.
    old=FittedMaterialCurve(ROOT/'experiments/curve_guided_transfer_20261006/structural_revision_v3/fitted_reference.npz')
    new=CoupledMaterialCurve(OUT/'reference_ready.npz')
    prior=np.load(BASE/'R2_C0/trace.npz')
    config=json.loads((BASE/'R2_C0/config.json').read_text(encoding='utf8'))
    rotations=[]
    for qpos in [prior['qpos'][0],prior['qpos'][-1]]:
        fit.d.qpos[:]=qpos;mujoco.mj_kinematics(fit.m,fit.d)
        rotations.append(fit.d.body('head').xmat.reshape(3,3).copy())
    measured=prior['q'][-1];ack=prior['ctrl'][-1]
    arithmetic=[]
    for label,curve,registered in [('old_reference_old_prior',old,rotations[0]),
                                  ('old_reference_new_prior',old,rotations[1]),
                                  ('new_reference_new_prior',new,rotations[1])]:
        controller=EquilibriumGraspController(curve,config,ack.copy(),encoder_fk,registered)
        target,c=controller.compute(measured,0.,45.25,.05)
        arithmetic.append(dict(case=label,maximum_first_full_blend_step_deg=float(np.degrees(abs(target-ack)).max()),
            update_deg=np.degrees(target-ack).tolist(),initial_retained_deg=np.degrees(ack-curve.joint_angles(0.)).tolist(),
            compliance_difference_deg=np.degrees(c['compliance_delta_rad']).tolist(),material_axes=c['material_axes'],
            applied=False,physics_steps=0))
    tangent_old=old.feed_tangent(0.);tangent_new=new.feed_tangent(0.)
    tangent_angle=float(np.degrees(np.arccos(np.clip(np.dot(tangent_old,tangent_new)/np.linalg.norm(tangent_old)/np.linalg.norm(tangent_new),-1,1))))
    final10=t>=t[-1]-10-1e-8;pt=p[final10];dlast=np.linalg.norm(pt-pt[0],axis=2)
    im=np.unravel_index(dlast.argmax(),dlast.shape)
    crossings=np.flatnonzero(dlast.max(axis=1)>6.45)
    row=dict(physics_steps_added=0,actual_seconds=float(t[-1]),actual_commands=len(commands),
        state_ack_identity_error_rad=float(abs(targets-nominal-np.radians(h)).max()),
        after5_minimum_contacts_per_pipe=counts[t>=5-1e-8].min(axis=0).tolist(),
        geometric_centroid_displacement_mm=float(drift[-1]),root_displacement_vector_mm=((z['qpos'][-1,:3]-z['qpos'][0,:3])*1000).tolist(),
        initial_to_final_head_rotation_deg=float(np.degrees((Rotation.from_quat(z['qpos'][-1,[4,5,6,3]])*
            Rotation.from_quat(z['qpos'][0,[4,5,6,3]]).inv()).magnitude())),
        maximum_retained_deg=float(abs(h).max()),late_rate_limited_commands=sum(c['rate_limited_joints']>0 for c in controls if c['time_s']>=t[-1]-10),
        late_angle_constrained_commands=sum(c['angle_constrained_joints']>0 for c in controls if c['time_s']>=t[-1]-10),
        maximum_actual_torque_Nm=float(abs(z['torque']).max()),
        added_external_force_maximum=float(z['external_applied_force_maximum'].max()),
        added_generalized_force_maximum=float(z['generalized_applied_force_maximum'].max()),
        ground_frames=int(np.count_nonzero(z['ground'])),nonadjacent_self_contact_frames=int(np.count_nonzero(z['self_contact'])),
        final10s_drift_worst_module=int(im[1]),final10s_drift_worst_time_s=float(t[final10][im[0]]),
        first_6p45mm_crossing_from_final10s_origin_time_s=float(t[final10][crossings[0]]) if len(crossings) else None,
        conditional_axis_diagnostics=axes,axis_truth_use='OFFLINE ONLY',
        reference_change_maximum_joint_deg=float(np.degrees(abs(new.joint_angles(0.)-old.joint_angles(0.))).max()),
        reference_tangent_direction_change_deg=tangent_angle,control_arithmetic_only=arithmetic,
        inference='New nominal shape, initial axis registration and projection direction changed together. Small acknowledged tracking error and sustained contacts did not preserve the previous holding equilibrium. No isolation trial proves a unique cause.')
    (case/'diagnostic_metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(3,2,figsize=(11,9),sharex=True)
    ax[0,0].plot(t,counts[:,0],label='Source');ax[0,0].plot(t,counts[:,1],label='Target');ax[0,0].axhline(2,color='k',ls=':');ax[0,0].legend();ax[0,0].set_ylabel('Contact modules >0.1 N')
    ax[0,1].plot(t[1:],speed.max(axis=1));ax[0,1].set_ylabel('Module speed (mm/s)')
    ax[1,0].plot(t,drift,label='Body geometric centroid');ax[1,0].plot(t,np.linalg.norm((z['qpos'][:,:3]-z['qpos'][0,:3])*1000,axis=1),label='Head root');ax[1,0].legend();ax[1,0].set_ylabel('Displacement (mm)')
    ax[1,1].plot(t,np.degrees(abs(z['q']-z['ctrl'])).max(axis=1));ax[1,1].set_ylabel('Tracking error (deg)')
    ax[2,0].plot(ct,abs(h).max(axis=1),label='Executable residual');ax[2,0].plot(ct,abs(delta).max(axis=1),label='A-D difference');ax[2,0].legend();ax[2,0].set_ylabel('Adjustment (deg)')
    for side,index in [('Source',0),('Target',1)]:
        ax[2,1].plot(ct,[c['material_axes'][index]['cloud_rms_mm'] for c in controls],label=side)
    ax[2,1].legend();ax[2,1].set_ylabel('Cloud RMS (mm)')
    for a in ax.flat:a.grid(alpha=.2);a.axvline(5,color='.5',ls='--');a.axvspan(t[-1]-10,t[-1],alpha=.1,color='red')
    ax[2,0].set_xlabel('Physical time (s)');ax[2,1].set_xlabel('Physical time (s)')
    fig.suptitle('Actual interface C0: support persists, holding convergence fails');fig.tight_layout()
    fig.savefig(case/'actual_diagnostics.png',dpi=140);plt.close(fig)
    print(json.dumps({k:row[k] for k in ['after5_minimum_contacts_per_pipe','geometric_centroid_displacement_mm',
        'initial_to_final_head_rotation_deg','maximum_retained_deg','reference_change_maximum_joint_deg','reference_tangent_direction_change_deg',
        'late_rate_limited_commands','late_angle_constrained_commands','maximum_actual_torque_Nm','state_ack_identity_error_rad']}),flush=True)


if __name__=='__main__':main()
