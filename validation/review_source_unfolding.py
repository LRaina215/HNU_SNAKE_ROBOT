"""Saved source-release geometry, loading and ordered reception; no dynamics."""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np


def review(case):
    commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    release=[row['source_unfolding'] for row in commands]
    tc=np.array([row['t_s'] for row in commands]);amplitude=np.array([row['amplitude'] for row in release])
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];points=z['points_mm'].copy();force=z['cylindrical_contact_force_N'].copy()
        mask=z['target_grip_region_mask'].copy();q=np.degrees(z['q'].copy());torque=z['torque'].copy()
    shape_reasons=Counter()
    for row in commands:
        for item in row['adaptive_front_winding'].get('curve_shape_update_rejections',[]):shape_reasons[item['reason'].split(':')[0]]+=1
    actual_axial=(points[:,:-1,0]-points[:,1:,0])/64.5
    first_handover=[];age=np.zeros(21);next_module=8
    for index in range(len(t)):
        elapsed=0. if index==0 else t[index]-t[index-1]
        age=np.where((force[index,1]>.1)&mask[index],age+elapsed,0.)
        while next_module<21 and age[next_module]>=1.:
            first_handover.append(dict(t_s=float(t[index]),module=next_module));next_module+=1
    error=np.array(commands[-1]['joint_geometry_reference_deg'])-q[-1]
    state=json.loads((case/'controller_state.json').read_text(encoding='utf8'))
    us=state['curve']['planned_route_spec']['source_bound']*1000
    ut=state['curve']['planned_route_spec']['target_bound']*1000
    locations=np.array(release[-1]['nominal_material_station_coordinates_mm'])
    active=np.flatnonzero((locations>=release[-1]['posterior_boundary_curve_parameter_mm'])&(locations<=us))
    i=int(np.argmax(abs(error)))
    report=dict(name=case.name,outcome=result['outcome'],no_physics_steps=True,
        maximum_source_release_amplitude=float(amplitude.max()),final_source_release=release[-1],
        first_positive_release_s=float(tc[np.flatnonzero(amplitude>1e-8)[0]]) if np.any(amplitude>1e-8) else None,
        last_release_parameter_growth_s=float(tc[np.flatnonzero(np.r_[False,np.diff(amplitude)>1e-8])[-1]]) if np.any(np.diff(amplitude)>1e-8) else None,
        nominal_release_membership_changes=[],curve_shape_rejection_counts=dict(shape_reasons),
        new_sustained_raw_target_modules=result['new_sustained_target_modules'],
        ordered_reception_from_M8=first_handover,first_incoming_M8_maximum_target_force_N=float(force[:,1,8].max()),
        maximum_final_integral_error=dict(joint=i+1,error_deg=float(error[i]),torque_Nm=float(torque[-1,i]),
            curve_parameter_mm=float(locations[i]),curve_region='source' if locations[i]<us else 'bridge' if locations[i]<ut else 'target'),
        final_integral_error_deg=error.tolist(),final_actuator_torque_Nm=torque[-1].tolist(),
        final_nominal_release_joint_stations=(active+1).tolist(),
        final_actual_adjacent_rod_axial_fractions_in_nominal_release={str(index+1):actual_axial[-1,index:index+2].tolist() for index in active},
        maximum_bridge_clearance_field_mm=float(np.linalg.norm(np.asarray(state['curve']['bridge_clearance_offsets']),axis=1).max()*1000),
        evidence='nominal interval membership is recorded separately from actual rods; raw tail contact does not establish ordered body reception from M8; region uses continuous material curve coordinates')
    previous=None
    for at,row in zip(tc,release):
        membership=row['nominal_joint_stations_in_release_interval']
        if membership!=previous:
            report['nominal_release_membership_changes'].append(dict(t_s=float(at),joint_stations=membership,amplitude=row['amplitude']))
            previous=membership
    (case/'source_unfolding_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ['name','maximum_source_release_amplitude','first_positive_release_s','last_release_parameter_growth_s',
        'new_sustained_raw_target_modules','ordered_reception_from_M8','first_incoming_M8_maximum_target_force_N',
        'maximum_final_integral_error','curve_shape_rejection_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',type=Path,nargs='+');args=p.parse_args()
    for case in args.cases:review(case)
