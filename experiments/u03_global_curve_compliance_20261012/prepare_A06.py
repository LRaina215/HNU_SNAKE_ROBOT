"""Necessary nominal mapping/continuity check before new actual candidate."""
from physics import *
from curve_reference import GlobalCurveReference
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d)
A=1.2150592994538565;w=.183889173151851;tau=w/.0645;k=A*tau/(2*np.sin(w));R=k/(k*k+tau*tau);pitch=2*np.pi*tau/(k*k+tau*tau)
ref=GlobalCurveReference(nodes(d),d.qpos[7:],radius=R,pitch=pitch,moving_family=True,anchor_frame=True,bezier_bridge=True,local_material=False,smooth_source=True,discrete_transport=True,axial_family=True)
n=np.arange(20);basis=np.column_stack([np.sin(-w*n+(n%2)*np.pi/2),np.cos(-w*n+(n%2)*np.pi/2)])
out=dict(physical_steps=0,projection=False,node_radius_m=R,pitch_m=pitch,calibration='smooth helical curvature/torsion equivalent of once-initialized source integral targets, not rod center radius');samples=[]
for label,tt,ids in [('source',np.arange(.6,20.001,.1),np.arange(8,20)),('target',np.arange(150,235.001,.5),np.arange(20))]:
    qs=[]
    for t in tt:q,i=ref.command(ref.feed_per_cycle*.02*(t-.5));qs.append(q)
    qs=np.array(qs);bb=basis[ids];co=np.linalg.lstsq(bb,qs[:,ids].T,rcond=None)[0].T;ph=np.unwrap(np.arctan2(co[:,1],co[:,0]))
    out[label]=dict(rate_deg_s=float(np.rad2deg(np.polyfit(tt,ph,1)[0])),amplitude_rad=float(np.mean(np.linalg.norm(co,axis=1))),fit_rms_deg=float(np.rad2deg(np.sqrt(np.mean((co@bb.T-qs[:,ids])**2)))))
    print(label,out[label],flush=True);(BASE/'A06_GEOMETRIC_PREPARATION.json').write_text(json.dumps(out,indent=2))
previous=None;node_step=0;target_step=0
for t in np.arange(0,235.01,.5):
    q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));current=dict(t=float(t),q=q,points=i['reference_points'],loc=i['reference_locations'])
    if previous is not None:node_step=max(node_step,float(np.max(np.linalg.norm(current['points']-previous['points'],axis=1))));target_step=max(target_step,float(np.max(abs(q-previous['q']))))
    samples.append(current);previous=current
mask=np.array([np.max(abs(s['q']))>np.radians(88) for s in samples]);qq=np.array([s['q'] for s in samples]);peak=np.unravel_index(np.argmax(abs(qq)),qq.shape)
out.update(full_nominal_material_search_passed=True,nominal_sample_s=.5,max_reference_node_step_mm=node_step*1000,max_raw_step_deg=float(np.rad2deg(target_step)),max_raw_deg=float(np.rad2deg(np.max(abs(qq)))),nominal_clip_fraction=float(mask.mean()),clip_peak_t=samples[peak[0]]['t'],clip_peak_joint_1_based=int(peak[1]+1),first_clip_t=None if not mask.any() else samples[np.flatnonzero(mask)[0]]['t'],last_clip_t=None if not mask.any() else samples[np.flatnonzero(mask)[-1]]['t'],reference_spec=ref.spec)
(BASE/'A06_GEOMETRIC_PREPARATION.json').write_text(json.dumps(out,indent=2));np.savez_compressed(BASE/'A06_geometric_nominal_samples.npz',t=np.array([s['t'] for s in samples]),q=qq,points=np.array([s['points'] for s in samples]),locations=np.array([s['loc'] for s in samples]));print(json.dumps({k:v for k,v in out.items() if k!='reference_spec'},indent=2))
