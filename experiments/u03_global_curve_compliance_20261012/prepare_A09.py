"""Check whether a world tube corridor survives curve mapping, before physics."""
from physics import *
from two_window_reference import TwoWindowReference
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=TwoWindowReference(nodes(d),d.qpos[7:],clearance_connector=True)
out=dict(physical_steps=0,spec=ref.spec,nominal_center_distance_required_m=.0855,samples=[],errors=[]);fig=plt.figure(figsize=(13,8))
peak=0;worst=1.;worst_t=None;worst_module=None
for t in np.arange(0,235.01,.5):
    try:q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5))
    except Exception as e:out['errors'].append(dict(t=float(t),error=str(e)));break
    c=i['nominal_curve_points'];cc=(c[:-1]+c[1:])/2;r=np.linalg.norm(cc[:,[0,2]]-[1.1,1],axis=1);mask=(cc[:,1]>.15)&(cc[:,1]<1.2)
    if mask.any() and r[mask].min()<worst:worst=float(r[mask].min());worst_t=float(t);worst_module=int(np.arange(21)[mask][np.argmin(r[mask])])
    peak=max(peak,float(np.rad2deg(np.max(abs(q)))))
    if t in [20,30,35,40,45,50]:
        out['samples'].append(dict(t=float(t),C_head_m=c[0].tolist(),front_Y_m=cc[:6,1].tolist(),front_radius_mm=(r[:6]*1000).tolist(),max_raw_deg=float(np.rad2deg(np.max(abs(q))))))
        k=len(out['samples']);ax=fig.add_subplot(2,3,k,projection='3d');ax.plot(*ref.global_points().T,color='gray',alpha=.25);ax.plot(*i['reference_points'].T,'g.-',label='corridor template');ax.plot(*c.T,'b.-',label='C');ax.set(xlim=(.45,1.4),ylim=(-.3,.6),zlim=(.6,1.3),title=f't={t}s');ax.view_init(22,140)
        if k==1:ax.legend(fontsize=7)
out.update(max_raw_deg=peak,worst_nominal_center_radius_m=worst,worst_t=worst_t,worst_module_0_based=worst_module,clearance_check_passed=bool(worst>=.0855 and not out['errors']),note='integrated C midpoint diagnostic only; no claim of exact native-robot mesh/capsule clearance')
fig.tight_layout();fig.savefig(BASE/'A09_nominal_clearance.png',dpi=130);(BASE/'A09_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['samples','spec']},indent=2))
