"""Summarize actual motion and measured gait feedback; no dynamics."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('case', type=Path)
args = parser.parse_args()
case = args.case
result = json.loads((case/'result.json').read_text(encoding='utf8'))
commands = json.loads((case/'commands.json').read_text(encoding='utf8'))
with np.load(case/'trace.npz') as trace:
    t = trace['sim_time']-trace['sim_time'][0]
    c = trace['centers_mm'].copy()
    f = trace['cylindrical_contact_force_N'].copy()
    shift = trace['shift_mm'].copy()
    ref = trace['reference_points_mm'].copy()
front = np.mean(c[:,:7,1]-c[0,:7,1],axis=1)
rear = np.mean(c[:,14:,0]-c[0,14:,0],axis=1)
tc = np.array([x['t_s'] for x in commands])
coord = [x.get('coordination',{}) for x in commands]
height = np.array([x.get('reservoir_height_mm',np.nan) for x in coord])
credit = np.array([x.get('measured_stored_material_mm',np.nan) for x in coord])
feed = [x.get('feed_policy',{}) for x in coord]
report = {key: result.get(key) for key in [
    'name','outcome','complete','physical_seconds','error','final_shift_mm',
    'fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm',
    'new_sustained_target_modules','final_actual_source_grip_modules','final_actual_target_grip_modules',
    'max_actual_joint_deg','max_actuator_torque_Nm','max_dispatch_step_deg','first_ctrl_error_rad']}
report['maximum_front_mean_advance_mm'] = float(front.max())
report['maximum_rear_mean_advance_mm'] = float(rear.max())
report['maximum_body_curve_rms_mm'] = float(np.sqrt(np.mean((c-(ref[:,:-1]+ref[:,1:])/2)**2,axis=(1,2))).max())
report['maximum_measured_stored_material_mm'] = float(np.nanmax(credit)) if np.isfinite(credit).any() else None
report['gait_events'] = [x for x in result['controller_events'] if x['mode'].startswith('actual_')]
report['evidence'] = 'trace actual qpos/centers/contact forces; commands feedback separate; nominal shift is not physical progress'
(case/'motion_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
axes[0].plot(t,front,label='Actual front M0-M6 mean +Y / mm')
axes[0].plot(t,rear,label='Actual rear M14-M20 mean +X / mm')
axes[0].legend(loc='upper left');axes[0].set_ylabel('Actual displacement / mm')
axes[1].plot(t,shift,label='Reference shared shift / mm',color='black')
axes[1].plot(tc,credit,label='Measured stored material estimate / mm')
axes[1].plot(tc,height,label='Commanded arch height / mm',linestyle='--')
axes[1].legend(loc='upper left');axes[1].set_ylabel('Reference / feedback / mm')
axes[2].plot(t,np.count_nonzero(f[:,0]>.1,axis=1),label='Actual source cylinder contacts')
axes[2].plot(t,np.count_nonzero(f[:,1]>.1,axis=1),label='Actual target cylinder contacts')
axes[2].legend(loc='upper left');axes[2].set_ylabel('Contact count');axes[2].set_xlabel('Actual elapsed time / s')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle(f"{case.name}: {result['outcome']} | complete={result['complete']}")
fig.savefig(case/'motion_review.png',dpi=150)
print(json.dumps(report,ensure_ascii=True,indent=2))
