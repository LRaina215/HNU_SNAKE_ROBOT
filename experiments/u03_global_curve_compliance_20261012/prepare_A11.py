"""Necessary geometry and command checks; same generator, no physical steps."""
from physics import *
from supply_window_reference import SupplyWindowReference
from snake_control.pipe_clearance import PipeClearance
from inspect_A10_native import fitted
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);ref=SupplyWindowReference(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];cache={};errors=[];events=[];previous=None
def command(t):
    key=round(float(t),6)
    if key not in cache:
        q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));C=i['nominal_curve_points'];gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
        row=dict(t=key,max_raw_deg=float(np.rad2deg(max(abs(q)))),C_line_bound_mm=float(gap.min()*1000),worst_module=int(np.argmin(gap)),head_center_m=((C[0]+C[1])/2).tolist(),guide_C_error_mm=i['guide_to_C_max_mm'],min_curvature_radius_mm=float(1/max(np.linalg.norm(i['curvature_field'],axis=1))),source_fraction=i['source_reference'],target_fraction=i['target_reference'],connector_arc_m=ref.connector_arc)
        cache[key]=(q.copy(),i,row)
    return cache[key]
for t in np.arange(0,235.001,.5):
    try:q,i,row=command(t)
    except Exception as e:errors.append(dict(t=float(t),error=str(e)));break
    roles=np.column_stack([i['reference_locations']<ref.start_arc,i['reference_locations']>i['prefix_end_arc_m'],i['reference_locations']>ref.target_arc])
    if previous is not None:
        for node,side in np.argwhere(roles!=previous):events.append(dict(t=float(t),node=int(node),interface=['source','prefix-to-receiver','target'][side]))
    previous=roles
print('coarse',len(cache),'errors',errors,'crossings',len(events),flush=True)
times=set(np.round(np.arange(0,50.0001,.025),6))
for e in events:
    times.update(np.round(np.arange(max(0,e['t']-1),min(235,e['t']+1)+.001,.025),6))
# 0.1s outside the dense windows only estimates slew history there.
times.update(np.round(np.arange(0,235.0001,.1),6));last=z['ctrl'].copy();prior=-.025;native=[];prevq=None;prevtime=None;C0=None
for k,t in enumerate(sorted(times)):
    try:q,i,row=command(t)
    except Exception as e:errors.append(dict(t=float(t),error=str(e)));break
    dt=t-prior;prior=t;raw=q if t>=.5 else z['ctrl'];last=np.clip(np.clip(raw,last-np.radians(20)*dt,last+np.radians(20)*dt),-np.radians(88),np.radians(88))
    C=i['nominal_curve_points'];cc=(C[:-1]+C[1:])/2
    if C0 is None:C0=cc.copy()
    d.qpos[:]=z['qpos'];d.qpos[qa]=q;mujoco.mj_forward(m,d);a=fitted(nodes(d),C)
    d.qpos[qa]=last;mujoco.mj_forward(m,d);b=fitted(nodes(d),C)
    slope=float(np.rad2deg(max(abs(q-prevq)))/(t-prevtime)) if prevq is not None else 0.;prevq=q.copy();prevtime=t
    native.append(dict(t=float(t),hold_end_time_s=float(t+dt),interval_s=float(dt),max_raw_to_limited_deg=float(np.rad2deg(max(abs(q-last)))),max_sample_slope_deg_s=slope,raw_front_error_mm=a['front_max_mm'],limited_front_error_mm=b['front_max_mm'],rear_dx_mm=float((cc[10:,0]-C0[10:,0]).mean()*1000)))
    if k%500==0:print('dense',k,'/',len(times),'t',t,'minimum',min(v[2]['C_line_bound_mm'] for v in cache.values()),flush=True)
rows=[v[2] for v in cache.values()];worst=min(rows,key=lambda r:r['C_line_bound_mm']);alltarget=[r['t'] for r in rows if r['target_fraction']==1]
rear=np.arange(20);ws=-.183889173151851;bb=np.column_stack([np.sin(ws*rear+(rear%2)*np.pi/2),np.cos(ws*rear+(rear%2)*np.pi/2)])[12:];ts=np.arange(.6,20.001,.2);qq=np.array([command(t)[0][12:] for t in ts]);co=np.linalg.lstsq(bb,qq.T,rcond=None)[0].T;ph=np.unwrap(np.arctan2(co[:,1],co[:,0]));carrier=dict(rate_deg_s=float(np.rad2deg(np.polyfit(ts,ph,1)[0])),fit_rms_deg=float(np.rad2deg(np.sqrt(np.mean((co@bb.T-qq)**2)))))
out=dict(spec=ref.spec,physical_steps=0,errors=errors,geometry_sampling='whole rod65points, subtract29mm envelope and half max sample interval; original cylinder plus conservative compiled mesh planes',slew_sampling='0..50s every25ms; all source/prefix/target material-crossing windows +/-1s every25ms;0.1s elsewhere approximates history there. limited error is HOLD END against held current q/C, not instantaneous ctrl(t).',dense_samples=len(native),geometry_samples=len(cache),worst_line_bound=worst,max_raw_deg=max(r['max_raw_deg'] for r in rows),min_curvature_radius_mm=min(r['min_curvature_radius_mm'] for r in rows),nominal_full_target_time=min(alltarget) if alltarget else None,nominal_remaining_s=235-min(alltarget) if alltarget else None,geometric_passed=not errors and worst['C_line_bound_mm']>=0,carrier=carrier,critical0_40=dict(max_raw_deg=max(v[2]['max_raw_deg'] for t,v in cache.items() if t<=40),max_hold_end_slew_error_deg=max(v['max_raw_to_limited_deg'] for v in native if .5<=v['t']<=40),max_raw_native_front_error_mm=max(v['raw_front_error_mm'] for v in native if v['t']<=40)),max_hold_end_slew_error_deg=max(v['max_raw_to_limited_deg'] for v in native),max_sample_slope_deg_s=max(v['max_sample_slope_deg_s'] for v in native),events=events,native_samples=native,geometry_rows=sorted(rows,key=lambda r:r['t']))
(BASE/'A11_PREPARATION.json').write_text(json.dumps(out,indent=2));tt=sorted(cache);np.savez_compressed(BASE/'A11_nominal_samples.npz',t=tt,q=np.array([cache[t][0] for t in tt]),guide=np.array([cache[t][1]['reference_points'] for t in tt]),C=np.array([cache[t][1]['nominal_curve_points'] for t in tt]))
fig=plt.figure(figsize=(15,9))
for k,t in enumerate([0,10,20,30,40,60]):
    q,i,r=command(t);ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*i['reference_points'].T,'g.-',label='guide');ax.plot(*i['nominal_curve_points'].T,'b.-',label='integrated C');ax.set(xlim=(.5,1.4),ylim=(-.3,.8),zlim=(.5,1.4),title=f't={t}s bound={r["C_line_bound_mm"]:.2f}mm');ax.view_init(22,140)
    if k==0:ax.legend()
fig.tight_layout();fig.savefig(BASE/'A11_nominal_geometry.png',dpi=130)
print(json.dumps({k:v for k,v in out.items() if k not in ['spec','events','native_samples','geometry_rows']},indent=2),flush=True)
