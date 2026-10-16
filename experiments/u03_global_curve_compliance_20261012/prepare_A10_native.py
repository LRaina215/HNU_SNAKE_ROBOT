"""Native FK at command time, slew FK at HOLD END (t+25ms); no dynamics."""
from inspect_A10_native import fitted
from physics import *
from continuous_window_reference import ContinuousWindowReference
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=ContinuousWindowReference(nodes(d),d.qpos[qa]);last=z['ctrl'].copy();rows=[];events=[];previous=None
for k in range(9401):
    t=k*.025;q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));raw=q if t>=.5 else z['ctrl'].copy()
    # Twenty-five 1ms slew updates with unchanged raw command equal this bounded step.
    last=np.clip(np.clip(raw,last-np.radians(20)*.025,last+np.radians(20)*.025),-np.radians(88),np.radians(88))
    roles=np.column_stack([i['reference_locations']<ref.start_arc,i['reference_locations']>ref.target_arc])
    if previous is not None and np.any(previous!=roles):events.append(t)
    previous=roles
    d.qpos[qa]=q;mujoco.mj_forward(m,d);a=fitted(nodes(d),i['nominal_curve_points'])
    d.qpos[qa]=last;mujoco.mj_forward(m,d);b=fitted(nodes(d),i['nominal_curve_points'])
    row=dict(t=t,command_time_s=t,ctrl_time_s=t+.025,raw_max_deg=float(np.rad2deg(max(abs(q)))),raw_to_ctrl_deg=float(np.rad2deg(max(abs(q-last)))),raw_front_max_mm=a['front_max_mm'],ctrl_front_max_mm=b['front_max_mm'],raw_full_rms_mm=a['full_rms_mm'],ctrl_full_rms_mm=b['full_rms_mm'],guide_C_max_mm=i['guide_to_C_max_mm'])
    rows.append(row)
    if k%1000==0:print('native',k,'/9400',row,flush=True)
out=dict(physical_steps=0,dt_s=.025,command_limit_deg=88,slew_deg_s=20,registration='diagnostic rigid rear-node10..21 alignment only; no feedback',interface_event_times_s=events,max_raw_deg=max(r['raw_max_deg'] for r in rows),clip_count=sum(r['raw_max_deg']>88 for r in rows),max_raw_to_ctrl_deg=max(r['raw_to_ctrl_deg'] for r in rows if r['t']>=.5),critical_38_50=dict(max_raw_to_ctrl_deg=max(r['raw_to_ctrl_deg'] for r in rows if 38<=r['t']<=50),max_raw_front_error_mm=max(r['raw_front_max_mm'] for r in rows if 38<=r['t']<=50),max_ctrl_front_error_mm=max(r['ctrl_front_max_mm'] for r in rows if 38<=r['t']<=50)),max_guide_C_error_mm=max(r['guide_C_max_mm'] for r in rows),samples=rows)
(BASE/'A10_NATIVE_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))
