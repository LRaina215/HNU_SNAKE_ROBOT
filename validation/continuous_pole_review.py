"""Continuous-motion milestones and support transfer; no stopping-hold score.

Checks are limited to real command application, phase/time budgets and matching
the previous first-eight-cycle trajectory. No source hashes or broad retests.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from segmented_pole_experiment import BASE, HORIZONTAL, load_checkpoint, pipe_frame, write_json

OUT=BASE/'continuous_v1'
MODES=['synchronous','front_first','rear_first']


def longest_gap(mask):
    longest=current=0
    for value in mask:
        current=current+1 if value else 0
        longest=max(longest,current)
    return longest*.01


def assess_continuous(folder):
    result=json.loads((folder/'result.json').read_text())
    z=np.load(folder/'trace.npz');contacts=np.load(folder/'contact_motion.npz')['contacts']
    commands=json.loads((folder/'commands.json').read_text())
    controls=json.loads((folder/'controls.json').read_text())
    count=result['steps']
    assert len(z['time'])==count*20+1 and len(commands)==count
    assert all(row['applied'] for row in commands)
    assert np.max(abs(np.diff(z['time'])-.01))<1e-10
    assert np.all(z['boundary_links']==6) and not np.any(z['stage']=='hold')
    assert np.isfinite(z['qpos']).all() and np.isfinite(z['ctrl']).all()
    for k,x in enumerate(commands):
        assert np.max(abs(z['ctrl'][k*20+1]-x['target']))<1e-12
    if count%2==0:
        assert np.max(abs(z['local_phase_total'][-1]+(count//2)*.15))<1e-12
    old=np.load(BASE/'alternating_v1'/f"{result['condition']}_c8"/'trace.npz')
    length=min(count*20+1,321)
    prefix_error=float(np.max(abs(z['qpos'][:length]-old['qpos'][:length])))
    assert prefix_error<1e-9,'first eight cycles changed; not a continuous extension'
    m,d,meta,c=load_checkpoint(HORIZONTAL);corner,source,target,radius=pipe_frame(m)
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    front=(centers[:,:6]@target).mean(axis=1);rear=(centers[:,10:]@source).mean(axis=1)
    head=z['points_mm'][:,0]@target
    normal=z['normal_force'];front_contact=np.any(normal[:,1,:6]>.1,axis=1)
    any_target=np.any(normal[:,1]>.1,axis=1);any_source=np.any(normal[:,0]>.1,axis=1)
    any_pipe=np.any(normal>.1,axis=(1,2))
    eligible=(np.arange(21)>=6)&~z['region'][0]
    endpoints=np.arange(0,len(z['time']),20)
    milestones=[];previous=0
    requested=[8,16,32]
    if count%2==0 and count//2 not in requested:requested.append(count//2)
    for cycle in sorted(requested):
        index=cycle*40
        if index>=len(z['time']):continue
        recent=endpoints[endpoints<=index][-3:]
        window_start=max(0,index-60)
        region=np.all(z['region'][recent],axis=0)
        touched=np.any(normal[window_start:index+1,1]>.1,axis=0)
        supported=region&touched
        new=supported&eligible
        interval=slice(previous,index+1)
        active_contacts=contacts[(contacts[:,0]>=z['time'][previous]-1e-8)&
            (contacts[:,0]<=z['time'][index]+1e-8)&(contacts[:,1]==1)&(contacts[:,2]<6)]
        chord=np.linalg.norm(centers[interval,9]-centers[interval,3],axis=1)
        force_max=m.actuator_forcerange[:,1]
        milestones.append(dict(cycle=cycle,elapsed_s=float(z['time'][index]-z['time'][0]),
            head_progress_mm=float(head[index]-head[0]),front_progress_mm=float(front[index]-front[0]),
            rear_progress_mm=float(rear[index]-rear[0]),interval_front_progress_mm=float(front[index]-front[previous]),
            interval_rear_progress_mm=float(rear[index]-rear[previous]),
            supported_target_modules=np.flatnonzero(supported).tolist(),new_supported_target_modules=np.flatnonzero(new).tolist(),
            target_region_modules=np.flatnonzero(region).tolist(),new_target_region_modules=np.flatnonzero(region&eligible).tolist(),
            target_contact_modules_at_mark=np.flatnonzero(normal[index,1]>.1).tolist(),
            source_contact_modules_at_mark=np.flatnonzero(normal[index,0]>.1).tolist(),
            target_contact_duty_last_0_6s=np.mean(normal[window_start:index+1,1]>.1,axis=0).tolist(),
            interval_front_contact_fraction=float(front_contact[interval].mean()),
            interval_any_target_contact_fraction=float(any_target[interval].mean()),
            interval_source_contact_fraction=float(any_source[interval].mean()),
            interval_max_tracking_error_deg=float(abs(z['error_deg'][interval]).max()),
            interval_actuator_saturation_fraction=float(np.mean(abs(z['torque'][interval])>=.99*force_max)),
            interval_clipped_joint_steps=sum(x['clipped_count'] for x in controls[previous//20:index//20]),
            interval_front_slip_median_mm_s=float(np.median(active_contacts[:,4])) if len(active_contacts) else None,
            bridge_chord_mm=float(chord[-1]),interval_bridge_chord_range_mm=[float(chord.min()),float(chord.max())]))
        previous=index
    out=dict(condition=result['condition'],outcome=result['outcome'],completed_cycles=count/2,
        elapsed_s=float(z['time'][-1]-z['time'][0]),first_8_cycles_qpos_max_difference=prefix_error,
        commands_applied=len(commands),ground=bool(z['ground'].any()),
        longest_front_contact_loss_s=longest_gap(~front_contact),longest_any_target_contact_loss_s=longest_gap(~any_target),
        longest_all_pipe_contact_loss_s=longest_gap(~any_pipe),milestones=milestones,
        final_head_progress_mm=float(head[-1]-head[0]),final_front_progress_mm=float(front[-1]-front[0]),
        final_rear_progress_mm=float(rear[-1]-rear[0]))
    write_json(folder/'analysis.json',out)
    return out


def main(root):
    reports=[assess_continuous(root/f'{mode}_c32') for mode in MODES]
    write_json(root/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    fig2,contact_axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for mode,ca in zip(MODES,contact_axes):
        z=np.load(root/f'{mode}_c32'/'trace.npz');t=z['time']-z['time'][0]
        centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        axes[0,0].plot(t,np.mean(centers[:,:6,1]-centers[0,:6,1],axis=1),label=mode)
        axes[0,1].plot(t,np.mean(centers[:,10:,0]-centers[0,10:,0],axis=1),label=mode)
        axes[1,0].plot(t,np.sum(z['normal_force'][:,1]>.1,axis=1),label=mode)
        axes[1,1].plot(t,np.linalg.norm(centers[:,9]-centers[:,3],axis=1),label=mode)
        ca.imshow((z['normal_force'][:,1]>.1).T,aspect='auto',origin='lower',interpolation='nearest',
            extent=[0,t[-1],-.5,20.5],cmap='Blues',vmin=0,vmax=1)
        ca.set_ylabel(mode+'\nModule index');ca.set_yticks(range(0,21,2))
    for ax,title in zip(axes.flat,['Front mean target-axis progress (mm)','Rear mean source-axis progress (mm)',
                                  'Modules contacting target pipe (>0.1 N)','Bridge endpoint separation (mm)']):
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        for time in [3.2,6.4,12.8]:ax.axvline(time,color='gray',ls=':',lw=.8)
    for ax in axes[1]:ax.set_xlabel('Continuous physical time (s)')
    contact_axes[-1].set_xlabel('Continuous physical time (s)')
    fig.suptitle('Continuous rolling: checkpoints do not pause motion')
    fig2.suptitle('Target-pipe contact history: geometric region is evaluated separately')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    fig2.tight_layout();fig2.savefig(root/'target_contacts.png',dpi=150);plt.close(fig2)
    for r in reports:
        print(r['condition'],r['outcome'],[(x['cycle'],round(x['front_progress_mm'],1),x['new_supported_target_modules']) for x in r['milestones']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,default=OUT)
    args=parser.parse_args();main(args.output_root.resolve())
