"""Actual contact/geometry/halfturn timing at acquisition, no physical steps."""
from physics import *
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def axis_r(P):
    x=P[:,:-1,[0,2]]-[1.1,1];v=P[:,1:,[0,2]]-P[:,:-1,[0,2]];u=np.clip(-(x*v).sum(2)/np.maximum((v*v).sum(2),1e-20),0,1)
    return np.linalg.norm(x+u[:,:,None]*v,axis=2)
rows=[];fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
for name in ['A14','A15']:
    z=np.load(BASE/name/'trace.npz');t=z['t'];c=z['centers'];r=np.linalg.norm(c[:,:6,[0,2]]-[1.1,1],axis=2);geo=((r>.065)&(r<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2)).all(1);span=np.rad2deg(np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1));N=z['forces'][:,1,:6].sum(1);win=(t>=49.99)&(t<=60.01);force=win&(N>.2);both=force&geo;ids=np.flatnonzero(force);best=int(np.argmax(np.where(geo,span,-1)))
    intervals=[];start=None
    for k,v in enumerate(force):
        if v and start is None:start=k
        if start is not None and (not v or k==len(force)-1):
            end=k-1 if not v else k;intervals.append(dict(first_s=float(t[start]),last_s=float(t[end]),sample_duration_s=float((end-start+1)*.1),peak_N=float(N[start:end+1].max()),max_span_deg=float(span[start:end+1].max()),all_geometry_sample_s=float(geo[start:end+1].sum()*.1)));start=None
    rows.append(dict(case=name,front_force_intervals_50_to_60=intervals,max_frontN_50_to_60=float(N[win].max()),max_span_when_all_geo_and_force_50_to_60_deg=float(span[both].max()) if both.any() else None,best_geometry_span_t=float(t[best]),best_geometry_span_deg=float(span[best]),best_geometry_span_front_N=float(N[best]),any_halfturn_geometry_contact=bool((geo&(span>180)&(N>.2)).any())))
    actual=(axis_r(z['points'][:,:7])-.0855)*1000;nominal=(axis_r(z['nominal_curve_points'][:,:7])-.0855)*1000
    axes[0].plot(t,c[:,:6,1].min(1),label=name);axes[1].plot(t,span,label=name);axes[2].plot(t,N,label=name);axes[3].plot(t,actual[:,0],label=name+' actual M0 simplified');axes[3].plot(t,nominal[:,0],'--',label=name+' C M0 simplified')
for ax in axes:ax.set_xlim(45,65);ax.axvspan(50,54,color='gray',alpha=.1);ax.legend();ax.grid(alpha=.25)
axes[0].axhline(.15,color='gray',ls='--');axes[0].set_ylabel('min front6 Y (m)');axes[0].set_ylim(.08,.2)
axes[1].axhline(180,color='gray',ls='--');axes[1].set(ylabel='front6 span (deg)',ylim=(90,190));axes[2].axhline(.2,color='gray',ls='--');axes[2].set(ylabel='front6 target normal (N)',ylim=(-.2,15));axes[3].axhline(0,color='gray',ls='--');axes[3].set(ylabel='M0 simplified surface gap (mm)',xlabel='saved actual time (s)',ylim=(-4,24))
fig.suptitle('A15: earlier contact precedes maximum wrap; source/world pose remains relevant');fig.tight_layout();fig.savefig(BASE/'A15/acquisition_timing.png',dpi=150)
out=dict(physical_steps=0,method='Saved actual front6 force>.2N intervals within50..60s; sample duration=count*.1s. Simplified axis/cylinder line gap diagnostic only, not actual mesh penetration. Geometry/halfturn/contact success still original simultaneous2s.',cases=rows)
(BASE/'A15/front_event_timing.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
