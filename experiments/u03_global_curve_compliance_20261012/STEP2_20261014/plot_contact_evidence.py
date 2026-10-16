"""Saved unlabeled contact evidence and estimated hypotheses, not scene truth."""
from physics import *
from geometry import Geometry
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for name in ['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE']:
    case=BASE/name;z=dict(np.load(case/'trace.npz'));contacts=[json.loads(line) for line in (case/'contact_evidence.jsonl').read_text().splitlines()]
    ct=np.array([x['t'] for x in contacts]);fig=plt.figure(figsize=(14,10))
    for panel,sec in enumerate([0.,15.4,30.,234.9]):
        i=int(np.argmin(abs(z['t']-sec)));j=int(np.argmin(abs(ct-z['t'][i])));record=contacts[j];g=Geometry(z['estimated_entry_m'][i],z['estimated_axis'][i],z['estimated_radius_m'][i]);e,b=g.frame()
        ax=fig.add_subplot(2,2,panel+1,projection='3d');ax.plot(*z['points'][i].T,color='gray',lw=2,label='actual body');ax.plot(*z['global_reference_points'][i].T,color='cyan',label='current G')
        for station in [-.05,.10,.35,.60]:
            angle=np.linspace(0,2*np.pi,60);ring=g.entry+station*g.axis+g.radius*(np.cos(angle)[:,None]*e+np.sin(angle)[:,None]*b);ax.plot(*ring.T,color='gold',alpha=.45,lw=.7)
        labels=set()
        for c in record['contacts']:
            kind=c['estimated_class'];color=['orange','green','purple'][kind];label=['known-source consistent','receiver candidate consistent','bridge/unknown'][kind]
            p=np.array(c['point']);n=np.array(c['normal']);ax.scatter(*p,color=color,s=18,label=label if kind not in labels else None);labels.add(kind);ax.quiver(*p,*n,length=.018,color=color,arrow_length_ratio=.3)
        ax.set(xlim=(.6,1.35),ylim=(-.15,1.2),zlim=(.45,1.3),title=f'{name} actual={z["t"][i]:.3f}s sensor={record["t"]:.3f}s fit={z["geometry_confidence"][i]:.2f}');ax.view_init(22,140)
        if panel==0:ax.legend(fontsize=7)
    fig.suptitle('Unlabeled contact points/normals + estimated cylinder, current G and actual body');fig.tight_layout();fig.savefig(case/'contact_points_normals_hypothesis.png',dpi=140);plt.close(fig)
print('Four saved contact geometry figures written; no scene labels used in these figures.')
