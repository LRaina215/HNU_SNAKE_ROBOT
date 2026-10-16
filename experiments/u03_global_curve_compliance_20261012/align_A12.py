"""Read-only A11 handover alignment; no physical steps."""
from physics import *
from supply_window_reference import SupplyWindowReference
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d)
r=SupplyWindowReference(nodes(d),d.qpos[7:]);tr=np.load(BASE/'A11/trace.npz');rows=[]
for t in [40,45,50,55,60,65,70,75,80]:
    S=r.feed_per_cycle*.02*(t-.5);q,i=r.command(S);loc=i['reference_locations'];idx=np.argmin(abs(tr['t']-t));js=[]
    for j in [9,10,11,12]:
        js.append(dict(J=j+1,arc_relative_source_mm=((loc[j:j+3]-r.start_arc)*1000).tolist(),q_deg=float(np.rad2deg(q[j])),ctrl_deg=float(np.rad2deg(tr['ctrl'][idx,j])),q_actual_deg=float(np.rad2deg(tr['qpos'][idx,7+j])),torque=float(tr['torque'][idx,j]),source_force_adjacent_N=tr['forces'][idx,0,j:j+2].tolist(),target_force_adjacent_N=tr['forces'][idx,1,j:j+2].tolist()))
    rows.append(dict(t=t,S=S,F=r.feed_distance(S),M=r.source_chart_shift,source_x=r.source_exit[0],prefix_length=r.prefix_end_arc-r.start_arc,receiver_length=r.target_arc-r.prefix_end_arc,head_station_source=loc[0]-r.start_arc,source_nodes=int(sum(loc<r.start_arc)),head=i['nominal_curve_points'][0].tolist(),joints=js))
(BASE/'A12_A11_HANDOVER_ALIGNMENT.json').write_text(json.dumps(rows,indent=2))
for a in rows:
    print('t',a['t'],'F/M',round(a['F'],3),round(a['M'],3),'prefix/receiver',round(a['prefix_length'],3),round(a['receiver_length'],3),'source nodes',a['source_nodes'])
    for j in a['joints'][1:3]:print(j)
