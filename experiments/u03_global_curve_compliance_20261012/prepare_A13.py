"""Local geometry, original slew and native shape checks for overlap handover."""
from physics import *
from overlap_window_reference import OverlapWindowReference
from snake_control.pipe_clearance import PipeClearance
from inspect_A10_native import fitted
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);r=OverlapWindowReference(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d)
times=sorted(set(np.round(np.r_[np.arange(0,80.001,.1),np.arange(25,42.001,.025),99.2,120,133.2,160,235],6)))
last=z['ctrl'].copy();prior=0.;rows=[];errors=[];qp=None
for t in times:
    try:q,i=r.command(r.feed_per_cycle*.02*max(0,t-.5))
    except Exception as exc:errors.append(dict(t=float(t),error=str(exc)));break
    dt=t-prior;prior=t;raw=q if t>=.5 else z['ctrl'];last=np.clip(np.clip(raw,last-np.radians(20)*dt,last+np.radians(20)*dt),-np.radians(88),np.radians(88));C=i['nominal_curve_points'];gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
    d.qpos[:]=z['qpos'];d.qpos[7:]=q;mujoco.mj_forward(m,d);a=fitted(nodes(d),C);d.qpos[7:]=last;mujoco.mj_forward(m,d);b=fitted(nodes(d),C)
    slope=0 if qp is None or dt==0 else float(np.rad2deg(abs(q-qp).max())/dt);qp=q
    rows.append(dict(t=float(t),connector_m=r.connector_arc,source_nodes=int(round(i['source_reference']*22)),target_nodes=int(round(i['target_reference']*22)),head_center=((C[0]+C[1])/2).tolist(),C_whole_rod_bound_mm=float(gap.min()*1000),worst_module=int(gap.argmin()),raw_max_deg=float(np.rad2deg(abs(q).max())),hold_end_raw_ctrl_deg=float(np.rad2deg(abs(q-last).max())),raw_native_front_mm=a['front_max_mm'],limited_native_front_mm=b['front_max_mm'],sample_slope_deg_s=slope,guide_C_mm=i['guide_to_C_max_mm']))
out=dict(spec=r.spec,physical_steps=0,scope='local0..80s100ms,25..42s25ms receiving entry/finite lead staging plus few late endpoints; no full path precision IK',slew_interpretation='hold-end estimate on mixed sample intervals; original actual driver enforces20deg/s and88deg. Late sparse endpoints do not estimate full slew history.',errors=errors,min_whole_rod_bound_mm=min(a['C_whole_rod_bound_mm'] for a in rows),max_raw_deg=max(a['raw_max_deg'] for a in rows),max_sample_slope_deg_s=max(a['sample_slope_deg_s'] for a in rows),max_local_hold_end_error_deg=max(a['hold_end_raw_ctrl_deg'] for a in rows if a['t']<=80),samples=rows)
(BASE/'A13_PREPARATION.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['spec','samples']},flush=True)
for t in [20,30,35,38,40,45,50,53.8,55,60,65,70,80,133.2]:
    print(min(rows,key=lambda a:abs(a['t']-t)),flush=True)
