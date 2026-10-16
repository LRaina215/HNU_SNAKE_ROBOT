"""Reference geometry only; no MuJoCo dynamics and no success claim."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hook_pull_experiment import Sim
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.unified_curve_transfer import UnifiedCurveTransfer
from snake_control.global_transfer_route import make_manual_route, sample_route, route_body

case=ROOT/'experiments/unified_curve_transfer_20261007/GLOBAL_ROUTE_REVIEW'
case.mkdir(exist_ok=True)
s=Sim(case);restore_saved_state(s.m,s.d,ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz')
c=UnifiedCurveTransfer(s.m,s.d);actual,_=c.observe(s.d)
spec=make_manual_route(actual)
u=np.linspace(-.8,3.4,1500);route=sample_route(spec,u)
final_shift=0.
while route_body(spec,final_shift)[1][-1]<spec['target_bound']+.02:
    final_shift+=.02
fig=plt.figure(figsize=(17,6.2),layout='constrained')
colors=['#2366b2','#df8809','#26964e']
for index,shift in enumerate([0.,final_shift/2,final_shift]):
    ax=fig.add_subplot(1,3,index+1,projection='3d')
    codes=np.where(u<=spec['source_bound'],0,np.where(u>=spec['target_bound'],2,1))
    for code in range(3):
        q=route[codes==code]
        ax.plot(q[:,0],q[:,1],q[:,2],c=colors[code],alpha=.30,lw=1.3)
    body,body_u=route_body(spec,shift)
    ax.plot(body[:,0],body[:,1],body[:,2],c='#1d2734',lw=3,label='global body reference')
    for j in [0,5,10,15,20,21]:
        ax.scatter(*body[j],c='#c33',s=13)
        ax.text(*body[j],f'  N{j}',fontsize=8)
    if index==0:
        ax.plot(actual[:,0],actual[:,1],actual[:,2],color='#9567bb',lw=2,alpha=.85,label='actual CLOSE1B / local entry')
    ax.plot([.5,1.1],[0,0],[1,1],color='#999',lw=9,alpha=.30)
    ax.plot([1.1,1.1],[0,.98],[1,1],color='#999',lw=9,alpha=.30)
    ax.set(xlim=(.48,1.25),ylim=(-.14,.99),zlim=(.86,1.15),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.set_box_aspect((.77,1.13,.29));ax.view_init(elev=26,azim=-52)
    ax.set_title(['Initial reference + actual entry','Middle: same body / same shared shift','Final: whole body in target helix'][index]+f'\nshift = {shift*1000:.0f} mm')
    ax.legend(fontsize=8,loc='upper right')
fig.suptitle('Frozen global route: source helix -> tangent-continuous bridge -> extended target helix\n'
    f'Source +X and target +Y right-hand winding: both dphi/daxis = {spec["source_right_hand_dphi_daxis"]:.2f} rad/m; head N0 -> tail N21',fontsize=13)
fig.savefig(case/'global_route_body_reference.png',dpi=150)
(case/'manual_global_route.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
q=sample_route(spec,np.linspace(spec['source_bound'],spec['target_bound'],501))
report=dict(final_reference_shift_mm=final_shift*1000,source_right_hand_dphi_daxis=spec['source_right_hand_dphi_daxis'],
    target_right_hand_dphi_daxis=spec['target_right_hand_dphi_daxis'],bridge_minimum_centerline_pipe_distance_mm=float(c.pipe_geometry.signed(q).min()*1000),
    source='geometric reference only; actual CLOSE1B posture preserved for local entry, not rewritten into global route')
(case/'reference_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
