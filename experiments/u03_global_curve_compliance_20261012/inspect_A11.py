from physics import *
from supply_window_reference import SupplyWindowReference
from snake_control.pipe_clearance import PipeClearance
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=SupplyWindowReference(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d);rows=[];c0=None
for t in [0,5,10,15,20,25,30,35,38,40,45,50,60,80,100,150,180,200,235]:
    try:q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5))
    except Exception as e:rows.append(dict(t=t,error=str(e)));break
    C=i['nominal_curve_points'];cc=(C[:-1]+C[1:])/2
    if c0 is None:c0=cc.copy()
    gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
    rows.append(dict(t=t,S=i['S'],head_center_m=cc[0].tolist(),max_raw_deg=float(np.rad2deg(max(abs(q)))),min_line_bound_mm=float(gap.min()*1000),worst_module=int(np.argmin(gap)),rear10_20_dx_mm=float((cc[10:,0]-c0[10:,0]).mean()*1000),source_fraction=i['source_reference'],target_fraction=i['target_reference'],connector_arc_m=ref.connector_arc,guide_C_max_mm=i['guide_to_C_max_mm']))
out=dict(spec=ref.spec,physical_steps=0,samples=rows);(BASE/'A11_INITIAL_INSPECTION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
