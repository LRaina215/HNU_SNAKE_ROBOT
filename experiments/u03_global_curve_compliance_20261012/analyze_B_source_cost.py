"""Saved nominal states/contact geometry only. No dynamics or retuning."""
from physics import *
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=BASE/'B_SOURCE_COST_20261014'
OUT.mkdir(exist_ok=True)
records={};details={};curves={}
for case in ['A16','B1_NOM_V2']:
    z=np.load(BASE/case/'trace.npz')
    # Cache arrays; do not decompress whole trace per saved state.
    arrays={k:z[k] for k in ['t','qpos','qvel','ctrl','raw_command','torque',
        'centers','points','forces','depth','reference_points']}
    if case=='B1_NOM_V2':
        for k in ['compliance_delta_m','compliance_gap_m','compliance_mean_load_N',
            'compliance_active_material','global_reference_points',
            'global_reference_locations','global_target_arc_m']:
            arrays[k]=z[k]
    t=arrays['t'];selected=np.flatnonzero((t>=105)&(t<=125))
    peak=int(np.argmax(arrays['depth'][:,0]))
    m=mujoco.MjModel.from_xml_path(str(BASE/case/'model.xml'))
    d=mujoco.MjData(m);head=m.body('head').id
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    rows=[];contact_rows=[];geometric_error=0.
    for i in selected:
        d.qpos[:]=arrays['qpos'][i];d.qvel[:]=arrays['qvel'][i]
        d.ctrl[:]=arrays['ctrl'][i];mujoco.mj_forward(m,d)
        contacts=[]
        for k,ct in enumerate(d.contact):
            ids=[ct.geom1,ct.geom2];names=[m.geom(g).name or '<unnamed>' for g in ids]
            bodies=[int(m.geom_bodyid[g]) for g in ids]
            if not any(name in ['pipe_straight_01','pipe_sharp_in_02'] for name in names):continue
            if not any(head<=body<head+21 for body in bodies):continue
            material=next(body-head for body in bodies if head<=body<head+21)
            contacts.append(dict(geom_ids=[int(g) for g in ids],geom_names=names,
                geom_types=[int(m.geom_type[g]) for g in ids],
                body_names=[m.body(b).name for b in bodies],material=material,
                distance_mm=float(1000*ct.dist),position_m=ct.pos.tolist(),
                normal_world=ct.frame[:3].tolist()))
        recovered=max([max(0.,-c['distance_mm']) for c in contacts]+[0.])
        geometric_error=max(geometric_error,abs(recovered-1000*arrays['depth'][i,0]))
        center=arrays['centers'][i,20]
        row=dict(case=case,t_s=float(t[i]),source_depth_mm=float(1000*arrays['depth'][i,0]),
            target_depth_mm=float(1000*arrays['depth'][i,1]),
            M20_source_N=float(arrays['forces'][i,0,20]),
            source_total_N=float(arrays['forces'][i,0].sum()),
            target_total_N=float(arrays['forces'][i,1].sum()),
            M20_x_m=float(center[0]),M20_y_m=float(center[1]),M20_z_m=float(center[2]),
            M20_source_axis_radius_mm=float(1000*np.linalg.norm(center[1:]-[0.,1.])),
            delta_mm=float(1000*arrays['compliance_delta_m'][i]) if case=='B1_NOM_V2' else 0.,
            filtered_receiver_error_mm=float(1000*arrays['compliance_gap_m'][i]) if case=='B1_NOM_V2' else None,
            filtered_receiver_mean_N=float(arrays['compliance_mean_load_N'][i]) if case=='B1_NOM_V2' else None,
            feedback_observed_materials=np.flatnonzero(arrays['compliance_active_material'][i]).tolist() if case=='B1_NOM_V2' else [])
        for j in [17,18,19,20]:
            index=j-1
            row.update({f'J{j}_raw_deg':float(np.rad2deg(arrays['raw_command'][i,index])),
                f'J{j}_ctrl_deg':float(np.rad2deg(arrays['ctrl'][i,index])),
                f'J{j}_actual_deg':float(np.rad2deg(arrays['qpos'][i,qa[index]])),
                f'J{j}_ctrl_minus_actual_deg':float(np.rad2deg(arrays['ctrl'][i,index]-arrays['qpos'][i,qa[index]])),
                f'J{j}_raw_minus_ctrl_deg':float(np.rad2deg(arrays['raw_command'][i,index]-arrays['ctrl'][i,index])),
                f'J{j}_torque_Nm':float(arrays['torque'][i,index]),
                f'J{j}_torque_saturated':bool(abs(arrays['torque'][i,index])>=2.355-1e-5)})
        rows.append(row);contact_rows.append(dict(t_s=float(t[i]),source_contacts=contacts))
    contact=np.flatnonzero(arrays['forces'][:,0,20]>.1)
    near=contact[(t[contact]>=105)&(t[contact]<=125)]
    ownrow=min(rows,key=lambda r:abs(r['t_s']-t[peak]))
    same1137=min(rows,key=lambda r:abs(r['t_s']-113.7))
    owncontacts=min(contact_rows,key=lambda r:abs(r['t_s']-t[peak]))
    sametimecontacts=min(contact_rows,key=lambda r:abs(r['t_s']-113.7))
    summary=dict(global_source_peak_t_s=float(t[peak]),
        global_source_peak_mm=float(1000*arrays['depth'][peak,0]),
        near_M20_first_last_source_contact_s=[float(t[near[0]]),float(t[near[-1]])] if len(near) else None,
        recovered_contact_distance_max_difference_from_saved_depth_mm=geometric_error,
        own_source_peak=ownrow,own_peak_contacts=owncontacts,same1137=same1137,same1137_contacts=sametimecontacts)
    if case=='B1_NOM_V2':
        i=int(np.argmin(abs(t-113.7)))
        loc=arrays['global_reference_locations'][i]
        summary.update(M20_nominal_station_relative_target_m=float(.5*(loc[20]+loc[21])-arrays['global_target_arc_m'][i]),
            M20_nominal_center_m=(.5*(arrays['global_reference_points'][i,20]+arrays['global_reference_points'][i,21])).tolist(),
            M20_local_curve_center_m=(.5*(arrays['reference_points'][i,20]+arrays['reference_points'][i,21])).tolist(),
            M20_actual_rod_endpoints_m=arrays['points'][i,20:].tolist())
    records[case]=rows;details[case]=summary;curves[case]=contact_rows

keys=list(records['A16'][0])
with (OUT/'timeline_105_125.csv').open('w',newline='') as fp:
    writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader()
    for rows in records.values():writer.writerows(rows)
(OUT/'source_contact_geometry.json').write_text(json.dumps(curves,indent=2))
(OUT/'summary.json').write_text(json.dumps(dict(cases=details,
    force_source='original trace forces/torque, not recomputed per-contact forces',
    geometry_method='restore saved qpos/qvel/ctrl into frozen MuJoCo3.2.7 model; mj_forward only; no mj_step',
    limits='0.1s saved samples; per-contact live force split not stored; no isolated causal dynamics'),indent=2))
fig,axs=plt.subplots(5,1,figsize=(12,13),sharex=True)
for name,rows in records.items():
    tt=np.array([r['t_s'] for r in rows]);label='B0' if name=='A16' else 'B1'
    axs[0].plot(tt,[r['source_depth_mm'] for r in rows],label=label+' source')
    axs[1].plot(tt,[r['M20_source_N'] for r in rows],label=label+' M20 source')
    axs[1].plot(tt,[r['target_total_N'] for r in rows],ls=':',alpha=.7,label=label+' all target')
    axs[2].plot(tt,[r['M20_source_axis_radius_mm'] for r in rows],label=label+' M20 center radius')
    axs[3].plot(tt,[r['J19_ctrl_minus_actual_deg'] for r in rows],label=label+' J19 tracking')
    axs[3].plot(tt,[r['J20_ctrl_minus_actual_deg'] for r in rows],ls='--',label=label+' J20 tracking')
axs[4].plot([r['t_s'] for r in records['B1_NOM_V2']],[r['delta_mm'] for r in records['B1_NOM_V2']],label='B1 local delta')
for ax,ylabel in zip(axs,['source depth(mm)','normal load(N)','source axis radius(mm)','ctrl-actual(deg)','delta(mm)']):
    ax.set_ylabel(ylabel);ax.axvline(113.7,color='gray',lw=.8);ax.legend(ncol=2,fontsize=8)
axs[-1].set_xlabel('saved actual time(s)');fig.tight_layout();fig.savefig(OUT/'source_cost_timeline.png',dpi=140);plt.close(fig)
print(json.dumps(details,indent=2))
