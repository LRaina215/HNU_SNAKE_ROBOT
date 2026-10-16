"""One mechanism check: progressive front curvature release/recovery."""
from physics import *
from curve_reference import GlobalCurveReference
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d)
kw=dict(radius=.09680117714827459,pitch=.18306225015502264,moving_family=True,anchor_frame=True,bezier_bridge=True,smooth_source=True,discrete_transport=True,axial_family=True)
base=GlobalCurveReference(nodes(d),d.qpos[7:],**kw);ref=GlobalCurveReference(nodes(d),d.qpos[7:],release_curve=True,**kw)
samples=[];fig=plt.figure(figsize=(13,8))
for k,t in enumerate([0,10,20,30,40,60]):
    S=ref.feed_per_cycle*.02*max(0,t-.5);q0,j=base.command(S);q,i=ref.command(S);C=i['nominal_curve_points'];g=i['reference_points']
    samples.append(dict(t=t,S=S,nt=i['release_nt'],L=i['release_L'],weights=i['release_weights'].tolist(),front_q_deg=np.rad2deg(q[:6]).tolist(),base_front_q_deg=np.rad2deg(q0[:6]).tolist(),rear_q_max_change_deg=float(np.rad2deg(np.max(abs(q[8:]-q0[8:])))),C_head_m=C[0].tolist(),C_front_Z_range_m=[float(C[:7,2].min()),float(C[:7,2].max())],C_tail_m=C[-1].tolist()))
    ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*ref.global_points().T,color='grey',alpha=.35,label='G');ax.plot(*g.T,'g.-',label='G window');ax.plot(*C.T,'b.-',label='C release/recovery');ax.set(xlim=(.3,1.4),ylim=(-.3,.8),zlim=(.5,1.35),title=f't={t}s L={i["release_L"]:.2f}');ax.view_init(22,140)
    if k==0:ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(BASE/'A07_release_curve_preparation.png',dpi=130)
(BASE/'A07_PREPARATION.json').write_text(json.dumps(dict(physical_steps=0,projection=False,high_level_feedback=False,change='geometry-derived curvature scaled in first six bend cells; true C integrated, posterior template frame/point retained; no joint-wave overwrite',samples=samples,reference_spec=ref.spec),indent=2));print(json.dumps(samples,indent=2))
