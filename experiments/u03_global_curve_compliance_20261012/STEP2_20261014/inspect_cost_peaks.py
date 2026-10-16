"""Necessary saved-state cost/station diagnostics; no physics steps."""
from physics import *
import json
from analyze_step2 import scene_truth,longest
out=[]
for name in ['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE']:
    case=BASE/name;z=dict(np.load(case/'trace.npz'));m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);peaks=[]
    for kind in [0,1]:
        i=int(np.argmax(z['depth'][:,kind]));d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];mujoco.mj_forward(m,d);h=m.body('head').id;contacts=[]
        for ct in d.contact:
            geoms=[ct.geom1,ct.geom2]
            for j,g in enumerate(geoms):
                env=m.geom(g).name or '';body=int(m.geom_bodyid[geoms[1-j]])-h
                if not env.startswith('pipe_') or not 0<=body<21:continue
                role=0 if env in ['pipe_straight_01','pipe_sharp_in_02'] else 1
                if role==kind:contacts.append(dict(material=body,environment_evaluation_label=env,depth_mm=max(0.,-float(ct.dist))*1000))
        deepest=max(contacts,key=lambda x:x['depth_mm']) if contacts else None
        peaks.append(dict(component='source' if kind==0 else 'target',t=float(z['t'][i]),depth_mm=float(z['depth'][i,kind]*1000),deepest_evaluation_contact=deepest,
            applied_G_radius_m=float(z['applied_global_radius_m'][i]),probe_curvature_per_m=z['probe_curvature_per_m'][i].tolist(),source_gamma_per_m=float(z['source_gamma_per_m'][i]),receiving_delta_mm=float(z['compliance_delta_m'][i]*1000),
            saturated_joint_ids_1_based=(np.flatnonzero(abs(z['torque'][i])>=2.355-1e-5)+1).tolist(),same_index_world_G_rms_mm=float(np.sqrt(np.mean(np.linalg.norm(z['points'][i]-z['global_reference_points'][i],axis=1)**2))*1000)))
    truth=scene_truth(m,d);metric=json.loads((case/'metrics.json').read_text());extra=dict(
        maximum_sampled_contiguous_over3mm_s=longest(np.max(z['depth'][:,:2],axis=1)>.003,z['t']),
        estimated_planned_station_vs_true_cylinder_start_axial_mm=float((z['estimated_entry_m'][-1]-truth['start'])@truth['axis']*1000),
        G_planned_station_vs_true_cylinder_start_axial_mm=float((z['applied_global_entry_m'][-1]-truth['start'])@truth['axis']*1000),
        physical_port_along_axis_identified=False,planned_entry_is_helix_station_not_physical_cylinder_port=True,
        legacy_actual_target_support_count_semantics='Estimated receiver-candidate support count, NOT true target support; see TRACE_FIELDS.md')
    metric.update(extra);(case/'metrics.json').write_text(json.dumps(metric,indent=2));out.append(dict(case=name,peaks=peaks,**extra))
(BASE/'COST_AND_STATION_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
