"""Material-identity acquisition evidence, comparing existing U03 and N10."""
from native_run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out={};fig,axs=plt.subplots(2,3,figsize=(15,8))
for row,name in enumerate(['U03','N10']):
    case=BASE/name if name=='U03' else ROOT/'experiments/cmu_native_reproduction_20261012/N10'
    z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);theta=np.unwrap(np.arctan2(c[:,:,2]-1,c[:,:,0]-1.1),axis=0)
    mask=(t>=30)&(t<=70);events=[]
    for n in range(6):
        crossed=np.flatnonzero((t[1:]>=35)&(t[1:]<=65)&((c[:-1,n,2]-1)*(c[1:,n,2]-1)<=0))
        crossings=[dict(t=float(t[i+1]),upward=bool(c[i+1,n,2]>c[i,n,2]),radius_m=float(r[i+1,n]),y_m=float(c[i+1,n,1])) for i in crossed]
        events.append(dict(module=n,axis_height_crossings=crossings))
        axs[row,0].plot(c[mask,n,0],c[mask,n,2],label=f'M{n}');axs[row,1].plot(t[mask],c[mask,n,2]);axs[row,2].plot(t[mask],r[mask,n])
    for sec in [35,38,40,46,54,60]:
        i=np.argmin(abs(t-sec));axs[row,0].annotate(f'{sec}s M0',c[i,0,[0,2]],fontsize=8);axs[row,0].scatter(*c[i,0,[0,2]],s=18,color='black')
    circle=plt.Circle((1.1,1),.0565,color='gray',alpha=.3);axs[row,0].add_patch(circle);axs[row,0].set_aspect('equal');axs[row,0].set(xlabel='world X (m)',ylabel='world Z (m)',title=name+' same material paths, 30–70s');axs[row,0].legend(ncol=3)
    axs[row,1].axhline(1,color='gray');axs[row,1].set(xlabel='time (s)',ylabel='world Z (m)',title='Same material heights')
    axs[row,2].axhspan(.065,.13,color='gray',alpha=.2);axs[row,2].set(xlabel='time (s)',ylabel='radius to target axis (m)',title='Near-coil radius band')
    out[name]=dict(events=events,samples=[dict(t=float(t[i]),centers_m=c[i,:6].tolist(),radius_m=r[i,:6].tolist()) for i in [np.argmin(abs(t-sec)) for sec in [30,35,38,40,42,44,46,50,54,60,65]]])
fig.tight_layout();fig.savefig(BASE/'direction_material_comparison.png',dpi=150);plt.close(fig)
# The final decision is a review of full paths and actual side frames, not lower-first alone.
out['review']=dict(U03_direction_success=True,scope='Native original 20-joint horizontal L simulation; upward target acquisition followed by full transfer.',evidence='M0-M3 rise from below to axis height at near-coil radii while entering +Y target. M0 stays near the coil band as it reaches the top; no free over-top flight precedes descent. Front span and contact become sustained at53.8s. Side frames at40/50/60s and continuous replay corroborate.',N10='M0 free approach is above target (40s Z1.303m, radius0.332m); subsequent descending axis-height crossing is near target. Its upper-route acquisition remains the baseline.',limits='Not proof of every material point following a fixed global curve, nor validation on author T-pipe hardware.')
(BASE/'direction_review.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['review'],indent=2))
for k in ['U03','N10']:print(k,json.dumps(out[k]['events'],indent=2))
