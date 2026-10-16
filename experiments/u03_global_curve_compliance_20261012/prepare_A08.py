"""Initial registration and one nominal two-window mechanism check."""
from physics import *
from two_window_reference import TwoWindowReference
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=TwoWindowReference(nodes(d),d.qpos[7:])
out=dict(spec=ref.spec,physical_steps=0,samples=[]);fig=plt.figure(figsize=(15,9))
for k,t in enumerate([0,20,30,40,60,150]):
    S=ref.feed_per_cycle*.02*max(0,t-.5);q,i=ref.command(S);p=i['reference_points'];c=i['nominal_curve_points']
    out['samples'].append(dict(t=t,S_cells=S,connector_arc_m=i['connector_arc_m'],L=i['release_L'],template_head_m=p[0].tolist(),C_head_m=c[0].tolist(),C_front_Z_m=[float(c[:7,2].min()),float(c[:7,2].max())],front_q_deg=np.rad2deg(q[:6]).tolist(),max_raw_deg=float(np.rad2deg(np.max(abs(q)))),target_fraction=float(i['target_reference'])))
    ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*ref.global_points().T,color='gray',alpha=.4,label='G');ax.plot(*p.T,'g.-',label='two windows+connector');ax.plot(*c.T,'b.-',label='C release/recovery');ax.set(xlim=(.45,1.4),ylim=(-.3,.9),zlim=(.55,1.3),title=f't={t}s S={S:.2f} cells');ax.view_init(22,140)
    if k==0:ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(BASE/'A08_nominal_geometry.png',dpi=130)
qs=[];ts=np.arange(.6,20.001,.2)
for t in ts:q,i=ref.command(ref.feed_per_cycle*.02*(t-.5));qs.append(q)
qs=np.array(qs);w=-.183889173151851;n=np.arange(20);basis=np.column_stack([np.sin(w*n+(n%2)*np.pi/2),np.cos(w*n+(n%2)*np.pi/2)]);bb=basis[8:];co=np.linalg.lstsq(bb,qs[:,8:].T,rcond=None)[0].T;ph=np.unwrap(np.arctan2(co[:,1],co[:,0]))
out['source_command_mapping']=dict(rate_deg_s=float(np.rad2deg(np.polyfit(ts,ph,1)[0])),amplitude_rad=float(np.mean(np.linalg.norm(co,axis=1))),fit_rms_deg=float(np.rad2deg(np.sqrt(np.mean((co@bb.T-qs[:,8:])**2)))))
errors=[];peak=0;maxconn=0
for t in np.arange(0,235.01,.5):
    try:q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));peak=max(peak,float(np.rad2deg(np.max(abs(q)))));maxconn=max(maxconn,i['connector_arc_m'])
    except Exception as e:errors.append(dict(t=float(t),error=str(e)));break
out.update(nominal_sampling_s=.5,errors=errors,max_raw_deg=peak,max_connector_arc_m=maxconn)
(BASE/'A08_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
