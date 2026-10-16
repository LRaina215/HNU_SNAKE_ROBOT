from pathlib import Path
import json
import numpy as np

base = Path(__file__).resolve().parent
case = base/'U74'
r = json.loads((case/'result.json').read_text(encoding='utf8'))
a = np.load(case/'trace.npz')
previous = np.load(base/'U73/trace.npz')
t = a['sim_time']-a['sim_time'][0]
f = a['contact_force_N']
cf = a['cylindrical_contact_force_N']
c = a['centers_mm']
radius = np.linalg.norm(c[:,:,[0,2]]-np.array([1100.,1000.]),axis=2)

def windows(mask):
    edges = np.diff(np.r_[False,mask,False].astype(int))
    return [[float(t[start]),float(t[end-1]),float(t[end-1]-t[start]+.01)]
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]

new_events = [e for e in r['controller_events'] if e['t_s'] >= r['controller_initial_elapsed_s']]
complete = ((c[:,:,1]>80).all(1) & (radius<125).all(1) & ((f[:,0]>.1).sum(1)==0)
            & ((f[:,1]>.1).sum(1)>=4) & (f[:,1,17:]>.1).any(1))
summary = dict(
    outcome=r['outcome'],complete=r['complete'],physical_seconds=r['physical_seconds'],
    front_actual_mm=r['fixed_front_M0_M6_mean_y_advance_mm'],
    rear_actual_mm=r['fixed_rear_M14_M20_mean_x_advance_mm'],
    shared_shift_change_mm=float(a['shift_mm'][-1]-a['shift_mm'][0]),
    start_preservation_max_difference={key:float(abs(a[key][0]-previous[key][-1]).max()) for key in ['qpos','qvel','ctrl']},
    final_raw_source_modules=np.flatnonzero(f[-1,0]>.1).tolist(),
    final_cylindrical_source_modules=np.flatnonzero(cf[-1,0]>.1).tolist(),
    final_geometrically_filtered_source_modules=r['final_actual_source_grip_modules'],
    final_raw_target_modules=np.flatnonzero(f[-1,1]>.1).tolist(),
    final_cylindrical_target_modules=np.flatnonzero(cf[-1,1]>.1).tolist(),
    M20_source_force_range_N=[float(cf[:,0,20].min()),float(cf[:,0,20].max())],
    M20_final_source_force_N=float(cf[-1,0,20]),
    M20_target_force_range_N=[float(cf[:,1,20].min()),float(cf[:,1,20].max())],
    M20_target_contact_windows_s=windows(cf[:,1,20]>.1),
    M20_simultaneous_two_cylinder_contact_windows_s=windows((cf[:,0,20]>.1)&(cf[:,1,20]>.1)),
    whole_body_target_geometry_windows_s=windows((c[:,:,1]>80).all(1)&(radius<125).all(1)),
    M20_source_absent_windows_s=windows(cf[:,0,20]<=.1),
    all_raw_source_absent_windows_s=windows((f[:,0]>.1).sum(1)==0),
    full_completion_windows_s=windows(complete),
    final_min_body_Y_mm=float(c[-1,:,1].min()),final_max_target_axis_radius_mm=float(radius[-1].max()),
    final_M20_center_mm=c[-1,20].tolist(),final_M20_target_axis_radius_mm=float(radius[-1,20]),
    final_M19_target_force_N=float(cf[-1,1,19]),
    limits={key:r[key] for key in ['max_actual_joint_deg','max_actuator_torque_Nm','max_dispatch_step_deg',
                                  'max_pipe_penetration_mm','max_self_penetration_mm','max_reference_link_error_mm',
                                  'first_ctrl_error_rad']},
    new_events=new_events,
    basis=json.loads((base/'U74_inputs/basis.json').read_text(encoding='utf8')))
(case/'evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['new_events']},ensure_ascii=False,indent=2))
