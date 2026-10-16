from physics import *
from supply_window_reference import SupplyWindowReference
from snake_control.pipe_clearance import PipeClearance
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);r=SupplyWindowReference(nodes(d),d.qpos[7:]);c=PipeClearance(m,d);out=[]
for t in [30,38,40,50]:
    q,i=r.command(r.feed_per_cycle*.02*(t-.5));C=i['nominal_curve_points'];p=i['reference_points'];a=np.linspace(0,1,257);xyz=C[:-1,None,:]*(1-a[None,:,None])+C[1:,None,:]*a[None,:,None];gap=c.signed(xyz.reshape(-1,3)).reshape(21,257)-.029;j,k=np.unravel_index(np.argmin(gap),gap.shape);x=xyz[j,k]
    out.append(dict(t=t,module=int(j),location=x.tolist(),C_endpoints=C[j:j+2].tolist(),G_endpoints=p[j:j+2].tolist(),line_bound_mm=float(gap[j,k]*1000),target_axis_distance_mm=float(np.linalg.norm(x[[0,2]]-[1.1,1])*1000),source_axis_distance_mm=float(np.linalg.norm(x[1:]-[0,1])*1000),interface=r.source_exit.tolist(),prefix_controls=r.connector_controls.tolist()))
(BASE/'A11_CLEARANCE_LOCATION.json').write_text(json.dumps(out,indent=2));print(json.dumps([{k:v for k,v in rr.items() if k!='prefix_controls'} for rr in out],indent=2))
