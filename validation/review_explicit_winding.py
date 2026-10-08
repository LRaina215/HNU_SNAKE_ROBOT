"""Review saved direct-winding actions, rejection causes and actual geometry.

No physics is advanced. Intended window/curve changes remain separate from
measured body movement, contact and target-helix motion.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def review(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0]
        c=z['centers_mm'].copy()
        force=z['cylindrical_contact_force_N'].copy()
        shift=z['shift_mm'].copy()
        span=z['actual_front_target_winding_span_rad'].copy()
        source_mask=z['source_grip_region_mask'].copy()
    tc=np.array([row['t_s'] for row in commands])
    control=[row.get('explicit_winding',{}) for row in commands]
    requested=np.array([row.get('requested_speed_mm_s',np.nan) for row in control])
    accepted=np.array([row['shift_speed_mm_s'] for row in commands])
    following=np.array([row.get('maximum_integral_tracking_error_deg',np.nan) for row in control])
    shape_fraction=np.array([row.get('shape_update_fraction',0.) for row in control])
    field=np.array([row.get('maximum_bridge_clearance_field_mm',0.) for row in control])
    reasons=Counter()
    for row in control:
        for rejected in row.get('advance_rejections',[]):
            reason=rejected['reason'].split(':')[0]
            reasons[reason]+=1
    front=(c[:,:7,1]-c[0,:7,1]).mean(1)
    rear=(c[:,14:,0]-c[0,14:,0]).mean(1)
    radii=np.linalg.norm(c[:,:8,[0,2]]-[1100.,1000.],axis=2)
    phi=np.unwrap(np.arctan2(-(c[:,:8,2]-1000.),c[:,:8,0]-1100.),axis=0)
    phi_delta=phi-phi[0]
    initial_state=json.loads((case/'initial_controller_state.json').read_text(encoding='utf8'))
    spec=initial_state['curve']['planned_route_spec']
    slope=float(spec['target_right_hand_dphi_daxis'])/1000.
    guide=np.array([row['curve_guide_points_mm'] for row in commands])
    guide_centers=.5*(guide[:,:-1]+guide[:,1:])
    guide_phi=np.unwrap(np.arctan2(-(guide_centers[:,:8,2]-1000.),guide_centers[:,:8,0]-1100.),axis=0)
    guide_phi_delta=guide_phi-guide_phi[0]
    pressure=[row.get('curve_pressure_feedback',{}) for row in commands]
    vn=np.array([row.get('normal_curve_velocity_mm_s',np.zeros((2,21))) for row in pressure])
    vt=np.array([row.get('loaded_tracking_normal_velocity_mm_s',np.zeros((21,3))) for row in pressure])
    report={key:result.get(key) for key in ['name','outcome','complete','physical_seconds','final_shift_mm',
        'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','new_sustained_target_modules',
        'actual_front_winding_span_initial_rad','actual_front_winding_span_final_rad','max_actual_joint_deg',
        'max_actuator_torque_Nm','max_dispatch_step_deg','first_ctrl_error_rad','save_errors']}
    report.update(maximum_actual_front_mean_y_advance_mm=float(front.max()),
        maximum_actual_rear_mean_x_advance_mm=float(rear.max()),
        maximum_actual_front_winding_span_rad=float(span.max()),
        final_actual_front_temporal_azimuth_change_rad=phi_delta[-1].tolist(),
        final_actual_front_radius_mm=radii[-1].tolist(),
        target_helix_spatial_dphi_dy_rad_per_mm=slope,
        final_curve_guide_front_temporal_azimuth_change_rad=guide_phi_delta[-1].tolist(),
        temporal_azimuth_note='the spatial helix is also phase-rotated during rolling; its static dphi/dy is not a valid temporal motion law',
        rejection_counts=dict(reasons),
        maximum_bridge_clearance_field_mm=float(field.max()),
        accepted_pressure_shape_command_frames=int(np.count_nonzero(shape_fraction)),
        maximum_pressure_normal_curve_velocity_mm_s=float(abs(vn).max()),
        maximum_loaded_tracking_normal_curve_velocity_mm_s=float(np.linalg.norm(vt,axis=-1).max()),
        final_geometric_tracking_max_deg=float(following[-1]),
        evidence='saved actual trace and commands only; no simulation steps; world-axis displacement, nominal shift and field size are not by themselves physical handover')
    saturated=np.flatnonzero(field>=49.999)
    moving=np.flatnonzero(accepted>1e-6)
    report['bridge_field_first_50mm_cap_s']=float(tc[saturated[0]]) if len(saturated) else None
    report['last_accepted_shift_increment_s']=float(tc[moving[-1]]) if len(moving) else None
    report['maximum_integral_tracking_error_deg']=float(following.max())
    report['maximum_integral_tracking_error_time_s']=float(tc[following.argmax()])
    report['new_module_transient_contacts']=[]
    for module in range(8,21):
        active=force[:,1,module]>.1
        if not active.any():continue
        edges=np.diff(np.r_[False,active,False].astype(int))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        report['new_module_transient_contacts'].append(dict(module=module,maximum_force_N=float(force[:,1,module].max()),
            intervals_s=[[float(t[a]),float(t[b])] for a,b in zip(starts,ends)]))
    rows=[]
    for at in [0,5,15,30,45,60,90,120,float(t[-1])]:
        if at>t[-1]+1e-6:continue
        i=int(np.argmin(abs(t-at)));j=int(np.argmin(abs(tc-at)))
        row=dict(t_s=float(t[i]),actual_front_mean_y_advance_mm=float(front[i]),
            actual_rear_mean_x_advance_mm=float(rear[i]),actual_front_winding_span_rad=float(span[i]),
            actual_front_temporal_azimuth_change_rad=phi_delta[i].tolist(),
            curve_guide_front_temporal_azimuth_change_rad=guide_phi_delta[j].tolist(),
            actual_front_radius_mm=radii[i].tolist(),
            actual_front_mean_z_change_mm=float((c[i,:8,2]-c[0,:8,2]).mean()),
            posterior_geometrically_filtered_source_contacts=np.flatnonzero((force[i,0]>.1)&source_mask[i]).tolist(),
            target_cylinder_contacts=np.flatnonzero(force[i,1]>.1).tolist(),
            shared_shift_mm=float(shift[i]),requested_speed_mm_s=float(requested[j]),accepted_speed_mm_s=float(accepted[j]),
            geometric_tracking_max_deg=float(following[j]),shape_update_fraction=float(shape_fraction[j]),
            bridge_clearance_field_mm=float(field[j]),
            proposal_rejections=control[j].get('advance_rejections',[]))
        rows.append(row)
    report['checkpoints']=rows
    (case/'winding_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(4,2,figsize=(14,12),sharex=True,layout='constrained')
    axs[0,0].plot(t,front,label='Front M0-M6 mean +Y');axs[0,0].plot(t,rear,label='Rear M14-M20 mean +X')
    axs[0,0].set_ylabel('Actual displacement / mm');axs[0,0].legend()
    axs[0,1].plot(t,span);axs[0,1].axhline(span[0],color='grey',linestyle='--');axs[0,1].set_ylabel('Actual front wrap span / rad')
    axs[1,0].plot(tc,requested,label='Requested');axs[1,0].plot(tc,accepted,label='Accepted');axs[1,0].legend();axs[1,0].set_ylabel('Common shift speed / mm/s')
    axs[1,1].plot(tc,field,label='Bridge clearance field');axs[1,1].set_ylabel('Maximum curve offset / mm')
    for module in range(8):
        axs[2,0].plot(t,phi_delta[:,module],label=f'M{module}')
        axs[2,1].plot(t,radii[:,module],label=f'M{module}')
    axs[2,0].set_ylabel('Actual temporal azimuth change / rad');axs[2,0].legend(ncol=4,fontsize=7)
    axs[2,1].set_ylabel('Actual front radius / mm')
    axs[3,0].plot(t,np.count_nonzero(force[:,0]>.1,axis=1),label='Source cylinders')
    axs[3,0].plot(t,np.count_nonzero(force[:,1]>.1,axis=1),label='Target cylinders')
    axs[3,0].plot(t,np.count_nonzero((force[:,0]>.1)&source_mask,axis=1),label='Posterior source')
    axs[3,0].set_ylabel('Actual loaded contact modules');axs[3,0].legend(fontsize=8)
    axs[3,1].plot(tc,following,label='Integral q target error');axs[3,1].set_ylabel('Angle error / deg');axs[3,1].legend()
    for ax in axs.ravel():ax.grid(alpha=.2);ax.set_xlabel('Actual elapsed time / s')
    fig.suptitle(f'{case.name}: {result["outcome"]}; complete={result["complete"]}')
    fig.savefig(case/'winding_review.png',dpi=140);plt.close(fig)
    print(json.dumps({key:report[key] for key in ['name','outcome','physical_seconds','final_shift_mm','maximum_actual_front_mean_y_advance_mm',
        'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm','rejection_counts','maximum_bridge_clearance_field_mm']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',type=Path,nargs='+');args=p.parse_args()
    for case in args.cases:review(case)
