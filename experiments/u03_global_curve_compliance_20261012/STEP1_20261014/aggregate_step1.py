"""Reuse A16 baseline, combine first-round results; no dynamics."""
from physics import *
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=json.loads((BASE.parent/'A16/b_metrics.json').read_text())
base.update(case='N_OPEN_REUSE',condition='NOM',controller=dict(speed=False,shape=False),minimum_speed_ratio=1.,mean_speed_ratio=1.,source_gamma_max_per_m=0.,physical_data_reused='../A16',new_dynamics=False)
names=['N_SPEED','N_SHAPE','N_JOINT','L_OPEN','L_SPEED','L_SHAPE','L_JOINT','D_OPEN','D_SPEED','D_SHAPE','D_JOINT']
rows=[base]+[json.loads((BASE/name/'step1_metrics.json').read_text()) for name in names]
equal=[]
for label,a,b in [('NOM speed_vs_open','N_OPEN_REUSE','N_SPEED'),('NOM combined_vs_shape','N_SHAPE','N_JOINT'),('RECEIVE speed_vs_open','L_OPEN','L_SPEED'),('RECEIVE combined_vs_shape','L_SHAPE','L_JOINT'),('SOURCE speed_vs_open','D_OPEN','D_SPEED'),('SOURCE combined_vs_shape','D_SHAPE','D_JOINT')]:
    za=np.load(BASE.parent/'A16/trace.npz' if a=='N_OPEN_REUSE' else BASE/a/'trace.npz');zb=np.load(BASE/b/'trace.npz')
    diff={key:float(np.max(abs(za[key]-zb[key]))) for key in ['qpos','qvel','ctrl','raw_command','S','chi','centers']}
    equal.append(dict(comparison=label,max_differences=diff,all_equal=all(v==0 for v in diff.values())))
all_ratio_one=all(r['minimum_speed_ratio']==1. and r['mean_speed_ratio']==1. for r in rows)
decision=dict(speed_ratio_always_one=all_ratio_one,fixed_slow_required=False if all_ratio_one else 'review contribution',reason='No speed reduction occurred; fixed ratio1 would duplicate OPEN, no evidence that gains came from slowing',real_pause_and_shape_dynamics_observed=False,source_shape_benefit_isolated=False,step2_authorized=False)
(BASE/'COMPARISON.json').write_text(json.dumps(dict(results=rows,equality_checks=equal,conditional_controls=decision,scope='first development round;12 formal entries including1 reuse; given initial/truth assistance; no full research completion'),indent=2))
keys=['case','condition','full_transfer_and_rolling_success','front_stable_time_s','full_stable_time_s','final_full_target_s','minimum_each_module_post_Y_mm','post_minimum_abs_rod_spin_deg','minimum_speed_ratio','mean_speed_ratio','source_gamma_max_per_m','source_gamma_rate_max_per_m_s','source_coefficient_rate_sampled_max_per_m_s','local_curve_chord_error_max_mm','sampled_torque_saturated_joint_time_s','longest_any_joint_torque_saturation_s','max_depth_mm','max_matched_target_lead_m','max_post_anchor_unobserved_prediction_m','actual_load_steps','applied_load_impulse_N_s']
with (BASE/'COMPARISON.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=keys);w.writeheader()
    for row in rows:w.writerow({key:row.get(key) for key in keys})
for name in names:
    z=np.load(BASE/name/'trace.npz');t=z['t'];p=z['points'];g=z['global_reference_points'];c=z['nominal_curve_points'];roles=z['actual_material_role']
    fig=plt.figure(figsize=(15,9))
    for k,sec in enumerate([30.,70.,110.,113.7,150.,234.9]):
        i=int(np.argmin(abs(t-sec)));ax=fig.add_subplot(2,3,k+1,projection='3d')
        ax.plot(*g[i].T,color='cyan',label='current G material window');ax.plot(*c[i].T,color='magenta',label='current local C');ax.plot(*p[i].T,color='gray',lw=2,label='actual body')
        for material in range(21):
            color='orange' if roles[i,material]==1 else 'green' if roles[i,material] in [2,5] else 'purple' if roles[i,material] in [3,4] else 'gray'
            ax.plot(*p[i,material:material+2].T,color=color,lw=2)
        ax.set(xlim=(.5,1.4),ylim=(-.25,1.15),zlim=(.3,1.4),title=f'{name} t={t[i]:.1f} S={z["S"][i]:.2f}');ax.view_init(22,140)
        if k==0:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(BASE/name/'global_local_actual_current.png',dpi=140);plt.close(fig)
print(json.dumps(dict(all12_success=all(r['full_transfer_and_rolling_success'] for r in rows),equality=equal,conditional_controls=decision),indent=2))
