"""Read existing CMU traces and initial geometry; no fitting or dynamics."""
from pathlib import Path
import json
import numpy as np
import sys
import mujoco

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT/'experiments/unified_curve_transfer_20261007/CMU_WINDOWED_20261009'
sys.path.insert(0, str(ROOT))
from snake_control.suspended_supply_transfer import nodes
from snake_control.approved_route_transfer import geometry as approved_geometry


def main():
    evidence = dict(new_dynamics_runs=0, new_joint_IK_fits=0,
                    source_theta_convention='theta=atan2(Z-1,-Y); viewed from tail toward +X: right=-Y, up=+Z',
                    angle_is_material_center_orbit_not_joint_roll_phase=True,
                    original_trials=[])
    for case in ['C01', 'C02']:
        geometry = json.loads((BASE/case/'initial_geometry.json').read_text(encoding='utf-8'))
        plan = json.loads((BASE/case/'plan.json').read_text(encoding='utf-8'))
        result = json.loads((BASE/case/'result.json').read_text(encoding='utf-8'))
        z = np.load(BASE/case/'trace.npz')
        t, c = z['sim_time'], z['centers_mm']/1000
        ids = [int(np.argmin(abs(t-s))) for s in [.5, 50.5, 100.5]]
        materials = []
        for rod in [14, 18]:
            theta = np.unwrap(np.arctan2(c[:, rod, 2]-1., -c[:, rod, 1]))
            samples = []
            for j in ids:
                radius = np.linalg.norm(c[j, rod, [1, 2]]-[0., 1.])
                samples.append(dict(t_s=float(t[j]), axial_displacement_from_0p5s_mm=float((c[j, rod, 0]-c[ids[0], rod, 0])*1000),
                                    orbit_from_0p5s_deg=float(np.degrees(theta[j]-theta[ids[0]])),
                                    radius_mm=float(radius*1000), source_contact=bool(z['source_contact_any'][j, rod]),
                                    original_source_band=bool(.074 < radius < .118 and c[j, rod, 0] < 1.035)))
            materials.append(dict(rod_index=rod, samples=samples))
        sl = slice(ids[0], ids[2]+1)
        item = dict(case=case, actual_seconds=result['actual_seconds'], outcome=result['outcome'],
                    front_acquisition=result['active_front_acquisition'], whole_transfer=result['whole_transfer_complete'],
                    initial_sine_amplitude_rad=geometry['amplitude_rad'], spatial_rad_per_hinge=geometry['spatial_rad_per_hinge'],
                    calibrated_signed_pitch_mm=geometry['calibrated_circle']['signed_pitch']*1000,
                    window_delta_rad=plan.get('window_phase_delta_rad', -np.pi/2),
                    phase_rate_rad_s=plan['phase_rate_rad_s'], window_rate_modules_s=plan['window_rate_modules_s'],
                    nominal_feed_per_cycle_mm=np.pi*.058*1000,
                    rear_joints_J9_to_J20_tracking_max_deg_first_two_cycles=float(np.degrees(abs(z['ctrl'][sl, 8:]-z['q'][sl, 8:])).max()),
                    rear_joints_J9_to_J20_actual_range_deg_first_two_cycles=np.degrees(np.ptp(z['q'][sl, 8:], axis=0)).tolist(),
                    same_material_source_measurements=materials)
        evidence['original_trials'].append(item)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'))
    data = mujoco.MjData(model)
    data.qpos[:] = np.load(ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_APPROVED_ROUTE_20261011/prepared_initial.npz')['qpos']
    mujoco.mj_fwdPosition(model, data)
    p = nodes(data)
    current = (p[:-1]+p[1:])/2
    phi = np.unwrap(np.arctan2(current[:, 2]-1., current[:, 1]))
    source_pitch = np.polyfit(phi, current[:, 0], 1)[0]*2*np.pi*1000
    route = approved_geometry.Route()
    p, _ = route.window(route.final_shift)
    target = (p[:-1]+p[1:])/2
    target_phi = np.unwrap(np.arctan2(-(target[:, 2]-1.), target[:, 0]-1.1))
    target_pitch = np.polyfit(target_phi, target[:, 1], 1)[0]*2*np.pi*1000
    evidence['chirality_comparison'] = dict(source_axis='+X, tail toward head', target_axis='+Y, tail toward head',
        source_right_hand_angle='atan2(Z-1,Y)', target_right_hand_angle='atan2(-(Z-1),X-1.1)',
        current_preserved_initial_source_signed_pitch_mm=float(source_pitch),
        approved_final_target_signed_pitch_mm=float(target_pitch),
        current_initial_centers_mm=(current*1000).tolist(), approved_target_centers_mm=(target*1000).tolist(),
        current_initial_static_FK_evaluations=1, initial_pose_replaced=False,
        note='Known-axis geometric pitch estimate from saved/fixed point coordinates; no new joint fitting')
    (OUT/'cmu_existing_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in evidence['original_trials']:
        print(item['case'], 'rear max tracking deg', round(item['rear_joints_J9_to_J20_tracking_max_deg_first_two_cycles'], 3))
        for entry in item['same_material_source_measurements']:
            print('M'+str(entry['rod_index']), [(round(s['t_s'], 2), round(s['axial_displacement_from_0p5s_mm'], 2),
                                                 round(s['orbit_from_0p5s_deg'], 2), s['source_contact']) for s in entry['samples']])


if __name__ == '__main__':
    main()
