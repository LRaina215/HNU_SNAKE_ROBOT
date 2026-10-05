"""Plot recorded-response estimator diagnostics, not new crossing claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segmented_pole_experiment import ROOT


root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
files=[('Horizontal','dynamic_filter_horizontal_derivative_bounded_local_cov_topology_140s.json'),
       ('Vertical','dynamic_filter_vertical_derivative_bounded_local_cov_aligned_topology_149s.json')]
fig,axes=plt.subplots(2,3,figsize=(14,7),sharex='row')
for row,(label,name) in enumerate(files):
    data=json.loads((root/name).read_text(encoding='utf8'));samples=data['rows'];t=[r['time'] for r in samples]
    for ax,key,ylabel in zip(axes[row],['target_error_deg','target_offset_mm','innovation_rms_deg'],
            ['Target axis error (deg)','Target line offset (mm)','Joint prediction RMS (deg)']):
        values=np.maximum([r[key] for r in samples],1e-5)
        ax.semilogy(t,values,color='#1768ac',linewidth=1.3)
        ax.set_ylabel(ylabel);ax.grid(True,which='major',alpha=.25);ax.set_xlabel('Recorded simulation time (s)')
        for i in range(1,len(samples)):
            if samples[i]['topology']!=samples[i-1]['topology']:
                ax.axvline(t[i],color='#cd7a19',alpha=.65,linewidth=.8)
    axes[row,0].set_title(label+' | estimates scored after inference',loc='left',fontsize=10)
    axes[row,1].set_title('Orange lines: response-selected topology changes',fontsize=9)
    audit=json.loads((root/(Path(name).stem+'_support_audit.json')).read_text(encoding='utf8'))['summary']
    axes[row,2].set_title('Model support precision %.1f%% | recall %.1f%%'%(100*audit['precision'],100*audit['recall']),fontsize=9)
fig.suptitle('Encoder / command driven private-model tracking\nOffline replay with calibrated nominal dynamics; closed-loop transfer remains unverified',fontsize=12)
fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(root/'dynamic_tracking_review.png',dpi=160);plt.close(fig)
print(root/'dynamic_tracking_review.png')
