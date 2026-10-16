"""Local post-run25ms anchor-switch audit and saved actual comparison figure."""
from physics import *
from handover_window_reference import HandoverWindowReference
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
case=BASE/'A12';events=json.loads((case/'mapping_audit.json').read_text())['anchor_switches']
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL);d.qpos[:]=initial['qpos'];mujoco.mj_forward(m,d);r=HandoverWindowReference(nodes(d),d.qpos[7:]);rows=[]
for event in events:
    sec=event['t'];samples=[];prior=None
    for t in np.arange(sec-.15,sec+.151,.025):
        S=r.feed_per_cycle*.02*max(0,t-.5);q,i=r.command(S)
        samples.append(dict(t=float(t),anchor_first=int(r.anchor_ids[0]),max_command_change_deg=None if prior is None else float(np.rad2deg(abs(q-prior).max()))))
        prior=q
    rows.append(dict(saved_event_s=sec,nominal_interval_s=.025,max_25ms_change_deg=max(a['max_command_change_deg'] or 0 for a in samples),samples=samples))
(case/'anchor_switch_25ms.json').write_text(json.dumps(dict(physical_steps=0,source='post-run frozen nominal generator, local windows only; actual slew remains original driver',windows=rows),indent=2))
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
for name,color in [('A11','tab:red'),('A12','tab:blue')]:
    z=np.load(BASE/name/'trace.npz');t=z['t'];mask=(t>=45)&(t<=85)
    for ax,j in zip(axes[:2],[10,11]):ax.plot(t[mask],np.rad2deg(abs(z['ctrl'][mask,j]-z['qpos'][mask,7+j])),color=color,label=name);ax.set_ylabel(f'J{j+1} tracking error (deg)');ax.grid(alpha=.25)
    axes[2].plot(t[mask],np.rad2deg(abs(z['raw_command'][mask]-z['ctrl'][mask]).max(1)),color=color,label=name)
for ax in axes:
    ax.axvline(64.001,color='gray',ls='--',lw=1);ax.legend()
axes[2].set(ylabel='max raw-to-ctrl (deg)',xlabel='actual simulation time (s)');axes[2].grid(alpha=.25)
fig.suptitle('Original driver: saved actual handover comparison; dashed = anchor switch near64s');fig.tight_layout();fig.savefig(case/'handover_comparison.png',dpi=150)
print([(a['saved_event_s'],a['max_25ms_change_deg']) for a in rows],flush=True)
