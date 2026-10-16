"""Actual six-rod qualification and U03 comparison; no physical steps."""
from physics import *
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
case=BASE/'A13';z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];rr=np.linalg.norm(c[:,:6,[0,2]]-[1.1,1],axis=2);rad=(rr>.065)&(rr<.13);y=(c[:,:6,1]>.15)&(c[:,:6,1]<1.2);g=rad&y;span=np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1);f=z['forces'][:,1,:6].sum(1);qualified=g.all(1)&(span>np.pi)&(f>.2);ids=np.flatnonzero(g.all(1));best=int(np.argmax(np.where(g.all(1),span,-1)))
allr=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);target=(allr>.07)&(allr<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2);rows=[]
for sec in [40,50,53.8,60,70,80,100,160,200,234.9,float(t[best])]:
    k=int(np.argmin(abs(t-sec)));rows.append(dict(t=float(t[k]),front_r_mm=(rr[k]*1000).tolist(),front_Y_m=c[k,:6,1].tolist(),front_geo=g[k].tolist(),span_deg=float(np.rad2deg(span[k])),front_target_N=float(f[k]),source_contacts=np.flatnonzero(z['forces'][k,0]>.1).tolist(),target_straight_modules=np.flatnonzero(target[k]).tolist()))
out=dict(max_front_geo_modules=int(g.sum(1).max()),any_all6_geometry=bool(ids.size),any_all6_geometry_halfturn_contact=bool(qualified.any()),max_span_when_all6_geometry_deg=float(np.rad2deg(span[ids].max())) if ids.size else None,max_target_straight_count_time=float(t[np.argmax(target.sum(1))]),best_all6_geometry_halfturn_time=float(t[best]),samples=rows)
(case/'front_failure_diagnosis.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
for name,path,color in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03','tab:green'),('A13',case,'tab:orange')]:
    zz=np.load(path/'trace.npz');tt=zz['t'];cc=zz['centers'];s=np.ptp(np.unwrap(np.arctan2(cc[:,:6,2]-1,cc[:,:6,0]-1.1),axis=1),axis=1)
    axes[0].plot(tt,cc[:,:6,1].min(1),label=name,color=color);axes[1].plot(tt,np.rad2deg(s),label=name,color=color);axes[2].plot(tt,zz['forces'][:,0].sum(1),label=name,color=color)
for ax in axes:ax.legend();ax.grid(alpha=.25);ax.set_xlim(0,235)
axes[0].axhline(.15,color='gray',ls='--');axes[0].set_ylabel('min front6 Y (m)');axes[1].axhline(180,color='gray',ls='--');axes[1].set_ylabel('front6 span (deg)');axes[2].set(ylabel='source normal sum (N)',xlabel='saved actual time (s)')
fig.suptitle('Actual acquisition: position, halfturn and source unloading are separate criteria');fig.tight_layout();fig.savefig(case/'acquisition_evidence.png',dpi=150)
print({k:v for k,v in out.items() if k!='samples'},flush=True)
