"""Necessary delta checks for the one A14 contrast, no physical steps."""
from physics import *
from overlap_window_reference import OverlapWindowReference
from phase_overlap_reference import PhaseOverlapReference
from snake_control.pipe_clearance import PipeClearance
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);p=nodes(d);qinit=d.qpos[7:].copy()
r=PhaseOverlapReference(p,qinit);base=OverlapWindowReference(p,qinit);clear=PipeClearance(m,d)
initial_delta=float(np.max(abs(r.command(0)[0]-base.command(0)[0])))
times=sorted(set(np.round(np.r_[np.arange(0,32.001,.5),np.arange(32,100.001,.1),np.arange(40,60.001,.025),np.arange(100,235.001,.5),235],6)))
last=z['ctrl'].copy();prior=0.;qp=None;rows=[];errors=[]
for t in times:
    try:q,i=r.command(r.feed_per_cycle*.02*max(0,t-.5))
    except Exception as exc:errors.append(dict(t=float(t),error=str(exc)));break
    dt=t-prior;prior=t;raw=q if t>=.5 else z['ctrl'];last=np.clip(np.clip(raw,last-np.radians(20)*dt,last+np.radians(20)*dt),-np.radians(88),np.radians(88))
    C=i['nominal_curve_points'];gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
    slope=0 if qp is None or dt==0 else float(np.rad2deg(abs(q-qp).max())/dt);qp=q
    rows.append(dict(t=float(t),S=float(i['S']),F_m=float(i['supply_distance_m']),M_m=float(i['source_chart_shift_m']),source_interface_x_m=float(i['source_interface_x_m']),C_whole_rod_bound_mm=float(gap.min()*1000),raw_max_deg=float(np.rad2deg(abs(q).max())),hold_end_raw_ctrl_deg=float(np.rad2deg(abs(q-last).max())),sample_slope_deg_s=slope,nominal_source_nodes=int(round(i['source_reference']*22)),nominal_target_nodes=int(round(i['target_reference']*22))))
    if len(rows)%250==0:print('checked',len(rows),'time',round(t,3),flush=True)
out=dict(spec=r.spec,physical_steps=0,scope='A13 unchanged throughS5; delta sampling32..100s100ms,40..60s25ms,late100..235s500ms plus unchanged early checkpoints. Original actual driver retains1ms slew and25ms commands.',initial_command_delta_from_A13_deg=float(np.rad2deg(initial_delta)),errors=errors,min_whole_rod_bound_mm=min(x['C_whole_rod_bound_mm'] for x in rows),max_raw_deg=max(x['raw_max_deg'] for x in rows),max_sample_slope_deg_s=max(x['sample_slope_deg_s'] for x in rows),max_hold_end_error_deg=max(x['hold_end_raw_ctrl_deg'] for x in rows),slew_interpretation='mixed-interval hold-end estimate, not exact25ms full235s history. Raw angle/slope may exceed original driver; real driver remains88deg and20deg/s.',samples=rows)
(BASE/'A14_PREPARATION.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['spec','samples']},flush=True)
