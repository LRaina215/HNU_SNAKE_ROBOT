"""Plot held-out geometric scores and fixed-recording frame-prediction probes."""
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,names):
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for name in names:
        rows=json.loads((root/(name+'_estimate_review.json')).read_text(encoding='utf8'))['rows']
        t=[r['time'] for r in rows]
        values=[[r[key] for r in rows] for key in ('target_axis_error_deg','target_line_offset_mm','measured_target_region')]
        for ax,value in zip(axes,values):ax.plot(t,value,label=name.replace('horizontal_encoder_live_',''),lw=1)
    for ax,label in zip(axes,['Target direction error (deg)','Target line offset (mm)','Modules actually in target region / 21']):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Physical seconds after the saved horizontal drape')
    fig.suptitle('Causal encoder-only control: geometry truth is used solely for scoring')
    fig.tight_layout();fig.savefig(root/'reviews/live_estimator_comparison.png',dpi=140);plt.close(fig)
    probe=json.loads((root/'single_target_registration_probe.json').read_text(encoding='utf8'))
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for run in probe['runs']:
        for ax,key in zip(axes,['target_error_deg','target_offset_mm']):
            ax.plot([r['time'] for r in run['rows']],[r[key] for r in run['rows']],label=run['registration'])
    for ax,label in zip(axes,['Target direction error (deg)','Target line offset (mm)']):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
    axes[-1].set_xlabel('Physical time in the SAME recorded trajectory (s)')
    fig.suptitle('Source released: isolate frame prediction from closed-loop motion changes\nOffline replay only; not a new crossing result')
    fig.tight_layout();fig.savefig(root/'reviews/registration_probe.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('names',nargs='+')
    a=p.parse_args();plot(a.root,a.names)
