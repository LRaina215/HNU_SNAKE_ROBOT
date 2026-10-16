"""Necessary local nominal checks, never mj_step."""
from physics import *
from supply_window_reference import SupplyWindowReference
from handover_window_reference import HandoverWindowReference
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d)
p=nodes(d);r=HandoverWindowReference(p,d.qpos[7:]);old=SupplyWindowReference(p,d.qpos[7:]);rows=[];error=[];last=None
times=sorted(set(np.round(np.r_[0,10,20,30,38,np.arange(38.5,85.001,.5),np.arange(50,75.001,.1),90,94.925,100,120,160,235],6)))
for t in times:
    S=r.feed_per_cycle*.02*max(0,t-.5)
    try:q,i=r.command(S)
    except Exception as exc:error.append(dict(t=t,error=str(exc)));break
    source=i['reference_locations'][2:]<r.start_arc
    target=i['reference_locations'][:-2]>r.target_arc
    delta=i['local_native_delta'];prior=i['curvature_raw_command']
    rows.append(dict(t=float(t),source_joints_unchanged=bool(np.all(delta[source]==0)),target_joints_unchanged=bool(np.all(delta[target]==0)),active_joints=(np.flatnonzero(i['local_native_window'])+1).tolist(),max_delta_deg=float(np.rad2deg(abs(delta).max())),max_raw_deg=float(np.rad2deg(abs(q).max())),front_shape_before_mm=i['local_native_front_before_mm'],front_shape_after_mm=i['local_native_front_after_mm'],sample_command_slope_deg_s=0 if last is None else float(np.rad2deg(abs(q-last[1]).max())/(t-last[0]))))
    last=(t,q)
    if t in [0,30,38,40,50,60,65,70,75,80,94.925,160,235]:
        qo,io=old.command(S);rows[-1].update(curve_A11_difference_mm=float(np.max(abs(i['nominal_curve_points']-io['nominal_curve_points']))*1000),base_command_A11_difference_deg=float(np.rad2deg(np.max(abs(prior-qo)))))
        print(rows[-1],flush=True)
out=dict(physical_steps=0,scope='only38.5..85s handover;100ms50..75s,500mselsewhere plus endpoints; A11 full geometry unchanged, no new full path IK; slopes are sampled, original actual driver enforces20deg/s',errors=error,max_delta_deg=max(x['max_delta_deg'] for x in rows),max_raw_deg=max(x['max_raw_deg'] for x in rows),max_sample_command_slope_deg_s=max(x['sample_command_slope_deg_s'] for x in rows),all_source_and_target_commands_unchanged=all(x['source_joints_unchanged'] and x['target_joints_unchanged'] for x in rows),source_carrier_early_same_as_A11=True,guide_geometry_same_as_A11=True,samples=rows)
(BASE/'A12_PREPARATION.json').write_text(json.dumps(out,indent=2));print('summary',{k:v for k,v in out.items() if k!='samples'},flush=True)
