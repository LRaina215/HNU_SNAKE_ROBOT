"""Compare saved encoder-only runs with the accepted known-information pair."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame


def run():
    root=ROOT/'experiments/segmented_pole_20261005'
    fig,axes=plt.subplots(4,2,figsize=(12,11),sharex='col')
    for col,(name,baseline,title) in enumerate([
        ('horizontal_encoder_live_whole_coil','radial_front_relative_braking','Horizontal T'),
        ('vertical_encoder_live_whole_coil','vertical_initial_front_reference','Turn vertically downward')]):
        folder=root/'feedback_takeover'/name
        _,z,m,_,_=context(folder)
        t=z['time']-z['time'][0];points=z['points_mm'];centers=(points[:,:-1]+points[:,1:])/2
        speed=np.linalg.norm(np.diff(centers,axis=0),axis=2).max(axis=1)/np.diff(t)/1000
        _,b,_,_,_=context(root/'support_transfer'/baseline)
        bt=b['time']-b['time'][0];bc=(b['points_mm'][:,:-1]+b['points_mm'][:,1:])/2
        bs=np.linalg.norm(np.diff(bc,axis=0),axis=2).max(axis=1)/np.diff(bt)/1000
        axes[0,col].plot(t[1:],speed,lw=.8,label='Encoder-only candidate')
        axes[0,col].plot(bt[1:],bs,lw=.8,alpha=.7,label='Accepted known-information baseline')
        axes[0,col].axhline(.5,color='red',ls='--',lw=.8,label='0.5 m/s gate')
        review=json.loads((folder.parent/(name+'_estimate_review.json')).read_text(encoding='utf8'))['rows']
        axes[1,col].plot([r['time'] for r in review],[r['target_axis_error_deg'] for r in review],lw=1,label='Held-out target-axis error')
        c=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        axes[2,col].plot(t,z['region'].sum(axis=1),lw=1,label='Actual radial region / 21 modules')
        axes[2,col].plot([r['time'] for r in c],[r['formation_progress'] for r in c],lw=1,label='Formation command / 20 joints')
        corner,_,target,_=pipe_frame(m);progress=(points-corner)@target
        end=float(np.max((m.numeric('pipe_centerline').data.reshape(-1,4)[:,:3]*1000-corner)@target))
        axes[3,col].plot(t,progress.min(axis=1),label='Minimum body-point progress')
        axes[3,col].plot(t,progress.max(axis=1),label='Maximum body-point progress')
        axes[3,col].axhline(end,color='red',ls='--',label='Actual pipe end')
        accepted=json.loads((folder/'whole_crossing_audit.json').read_text(encoding='utf8'))['accepted_window']
        if accepted:
            for ax in axes[:,col]:ax.axvspan(accepted['start_seconds'],accepted['end_seconds'],color='green',alpha=.08)
        axes[0,col].set_title(title)
        axes[3,col].set_xlabel('Physical time after saved drape (s)')
    for row,label in enumerate(['Maximum module speed (m/s)','Target-axis error (deg)','Count (different denominators)','Progress along true target (mm)']):
        axes[row,0].set_ylabel(label)
        for ax in axes[row]:ax.grid(alpha=.25);ax.legend(fontsize=7,loc='best')
    fig.suptitle('Stage 2: functional progress does not establish reliable autonomous transfer\nTruth used for scoring only; green = accepted full-cycle window, not quality pass',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(root/'feedback_takeover/reviews/feasibility_comparison.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':run()
