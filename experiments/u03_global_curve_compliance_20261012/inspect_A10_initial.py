from physics import *
from continuous_window_reference import ContinuousWindowReference
from snake_control.pipe_clearance import PipeClearance
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=ContinuousWindowReference(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d);rows=[]
for t in [0,10,20,30,38,40,45,50,60,150,200]:
    S=ref.feed_per_cycle*.02*max(0,t-.5)
    try:
        q,i=ref.command(S);p=i['reference_points'];c=i['nominal_curve_points'];gap=clear.link_minimum(c,samples=65)-.029-np.max(np.linalg.norm(np.diff(c,axis=0),axis=1))/128
        rows.append(dict(t=t,S=S,source_exit_m=ref.source_exit.tolist(),connector_arc_m=ref.connector_arc,template_head_m=p[0].tolist(),C_head_m=c[0].tolist(),max_raw_deg=float(np.rad2deg(np.max(abs(q)))),rod_line_clearance_bound_mm=float(gap.min()*1000),worst_module=int(np.argmin(gap)),source_station_transport_rad=i['source_station_transport_rad']))
    except Exception as e:rows.append(dict(t=t,error=str(e)))
(BASE/'A10_INITIAL_DIAGNOSIS.json').write_text(json.dumps(dict(physical_steps=0,spec=ref.spec,samples=rows),indent=2));print(json.dumps(rows,indent=2))
