"""Necessary delta checks for A15 receiving shape, no physical steps."""
from physics import *
from overlap_window_reference import OverlapWindowReference
from phase_overlap_reference import PhaseOverlapReference
from target_fit_reference import TargetFitReference
from snake_control.pipe_clearance import PipeClearance
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);p=nodes(d);qinit=d.qpos[7:].copy()
r=TargetFitReference(p,qinit);base=PhaseOverlapReference(p,qinit);clear=PipeClearance(m,d)
initial_delta=float(np.max(abs(r.command(0)[0]-base.command(0)[0])))
source_keys=['radius','pitch','beta','a','phi0','x0','feed_early','feed_late','feed_per_cycle','gauge_angle','tail_u0','anchor_u0']
source_contract={key:float(getattr(r,key)-getattr(base,key)) for key in source_keys}
source_contract['anchor_normal_max_delta']=float(np.max(abs(r.supply_anchor_normal-base.supply_anchor_normal)))
source_contract['F_M_source_point_checks']=[]
for S in [0,5,7,9,20,37]:
    r.set_nominal_family(S);base.set_nominal_family(S);u=np.array([-4.,-2.,-1.,r.exit_u])
    source_contract['F_M_source_point_checks'].append(dict(S=S,F_delta=float(r.feed_distance(S)-base.feed_distance(S)),M_delta=float(r.source_chart_shift-base.source_chart_shift),source_max_point_delta_m=float(np.max(abs(r.source(u,0)-base.source(u,0))))))
times=sorted(set(np.round(np.r_[np.arange(0,100.001,.1),np.arange(25,60.001,.025),np.arange(100,235.001,.5),235],6)))
last=z['ctrl'].copy();prior=0.;qp=None;rows=[];errors=[]
for t in times:
    try:q,i=r.command(r.feed_per_cycle*.02*max(0,t-.5))
    except Exception as exc:errors.append(dict(t=float(t),error=str(exc)));break
    dt=t-prior;prior=t;raw=q if t>=.5 else z['ctrl'];last=np.clip(np.clip(raw,last-np.radians(20)*dt,last+np.radians(20)*dt),-np.radians(88),np.radians(88))
    C=i['nominal_curve_points'];gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
    slope=0 if qp is None or dt==0 else float(np.rad2deg(abs(q-qp).max())/dt);qp=q
    rows.append(dict(t=float(t),S=float(i['S']),F_m=float(i['supply_distance_m']),M_m=float(i['source_chart_shift_m']),source_interface_x_m=float(i['source_interface_x_m']),C_whole_rod_bound_mm=float(gap.min()*1000),raw_max_deg=float(np.rad2deg(abs(q).max())),hold_end_raw_ctrl_deg=float(np.rad2deg(abs(q-last).max())),sample_slope_deg_s=slope,nominal_source_nodes=int(round(i['source_reference']*22)),nominal_target_nodes=int(round(i['target_reference']*22))))
    if len(rows)%250==0:print('checked',len(rows),'time',round(t,3),flush=True)
out=dict(spec=r.spec,physical_steps=0,source_contract=source_contract,scope='Receiving shape delta0..100s100ms,25..60s25ms,late100..235s500ms. Source/F/M/psi frozen A14; no dynamics. Actual driver retains1ms slew and25ms commands.',initial_command_delta_from_A14_deg=float(np.rad2deg(initial_delta)),errors=errors,min_whole_rod_bound_mm=min(x['C_whole_rod_bound_mm'] for x in rows),max_raw_deg=max(x['raw_max_deg'] for x in rows),max_sample_slope_deg_s=max(x['sample_slope_deg_s'] for x in rows),max_hold_end_error_deg=max(x['hold_end_raw_ctrl_deg'] for x in rows),slew_interpretation='mixed-interval hold-end estimate, not exact25ms full235s history. Raw angle/slope may exceed original driver; real driver remains88deg and20deg/s.',samples=rows)
(BASE/'A15_PREPARATION.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['spec','samples']},flush=True)
