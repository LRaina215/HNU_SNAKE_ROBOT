"""Paired shape-feedback evaluation from recorded states; no hash checks."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from segmented_pole_experiment import BASE,HORIZONTAL,load_checkpoint,pipe_frame,write_json
from snake_control.supported_helix_field import SupportedHelixCommand

OUT=BASE/'supported_helix_v1'
MODES=['original','helix','connector','combined']


def coverage(points,corner,target):
    r=points[:,:7]-corner;r-=np.expand_dims(r@target,-1)*target
    return abs(np.arctan2(np.cross(r[:,:-1],r[:,1:])@target,
        np.sum(r[:,:-1]*r[:,1:],axis=-1)).sum(axis=1))/(2*np.pi)


def assess(folder):
    result=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz')
    controls=json.loads((folder/'controls.json').read_text());commands=json.loads((folder/'commands.json').read_text())
    n=result['rolling_samples'];steps=result['steps'];end=n-1;held=result['hold_seconds']>0
    assert len(commands)==steps and n==steps*20+1 and all(x['applied'] for x in commands)
    assert np.isfinite(z['qpos']).all() and np.max(abs(np.diff(z['time'])-.01))<1e-9
    for k,row in enumerate(commands):assert np.max(abs(z['ctrl'][20*k+1]-row['target']))<1e-12
    m,d,meta,c=load_checkpoint(HORIZONTAL,verify_hashes=False);corner,source,target,radius=pipe_frame(m)
    links=np.array(meta['settings']['links']);bounds=np.r_[0.,np.cumsum(links)-links/2,links.sum()]
    integration_error=0.
    for i in sorted(set([0,min(7,steps-1),steps-1])):
        c=controls[i]
        cmd=SupportedHelixCommand(bounds,c['observed_with_compliance'],c['preload'],c['total'],c['delta'],
            6*links.mean(),6*links.mean(),c['helix_reference'],c['axis_bend'],c['front_weights'],c['shape_gain'],c['separate'])
        integration_error=max(integration_error,float(abs(cmd.joint_angles(links,order=24)-c['raw']).max()))
    assert integration_error<1e-8
    prefix=None
    if result['condition']=='original':
        old=np.load(BASE/'migration_v1/synchronous_fixed/trace.npz')
        prefix=float(abs(z['qpos'][:n]-old['qpos'][:n]).max());assert prefix<1e-9
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    chord=np.linalg.norm(centers[:,9]-centers[:,3],axis=1)
    front=np.mean(centers[:,:6]@target,axis=1);head=z['points_mm'][:,0]@target
    wraps=coverage(z['points_mm'],corner,target);nf=z['normal_force']
    contact=np.any(nf[:,1,:6]>.1,axis=1)
    gaps=np.flatnonzero(np.convolve((~contact[:n]).astype(int),np.ones(20,dtype=int),'valid')==20)
    retreat=float(max(0.,head[end]-head[end:].min())) if held else None
    root=float(np.linalg.norm(z['qpos'][end:,:3]-z['qpos'][end,:3],axis=1).max()*1000) if held else None
    persistent=np.all(z['region'][[end,end-20,end-40]],axis=0)
    new=np.flatnonzero(persistent&(np.arange(21)>=6))
    hold_ok=bool(held and contact[end:].all() and retreat<=10 and root<=20 and not z['ground'][end:].any())
    gate=bool(steps==128 and front[end]-front[0]>=64.5 and wraps[:n].min()>=.8*wraps[0]
        and contact[:n].all() and len(new)>=2 and not z['ground'].any() and hold_ok)
    milestones=[]
    for cycle in (8,16,32,48,64):
        i=cycle*40
        if i<n:milestones.append(dict(cycle=cycle,front_progress_mm=float(front[i]-front[0]),
            front_coverage_turns=float(wraps[i]),target_region_modules=np.flatnonzero(z['region'][i]).tolist(),
            front_contact_force_N=float(nf[i,1,:6].sum())))
    report=dict(condition=result['condition'],outcome=result['outcome'],cycles=steps/2,gate=gate,
        commands=steps,integration_error_rad=integration_error,original_prefix_qpos_error=prefix,
        final_front_progress_mm=float(front[end]-front[0]),peak_front_progress_mm=float(front[:n].max()-front[0]),
        initial_coverage_turns=float(wraps[0]),minimum_rolling_coverage_turns=float(wraps[:n].min()),
        final_coverage_turns=float(wraps[end]),first_front_loss_0_2s_at=float(z['time'][gaps[0]]-z['time'][0]) if len(gaps) else None,
        front_contact_fraction=float(contact[:n].mean()),persistent_new_region_modules=new.tolist(),
        final_target_contact_modules=np.flatnonzero(nf[end,1]>.1).tolist(),
        rolling_floor=bool(z['ground'][:n].any()),hold_gate=hold_ok,hold_retreat_mm=retreat,hold_root_mm=root,
        first_command_change_deg=float(np.degrees(abs(z['ctrl'][1]-z['ctrl'][0])).max()),
        max_tracking_error_deg=float(abs(z['error_deg'][:n]).max()),
        force_saturation_fraction=float(np.mean(abs(z['torque'][:n])>=.99*m.actuator_forcerange[:,1])),
        clipped_joint_steps=result['clipped_joint_steps'],milestones=milestones)
    report.update(bridge_chord_initial_mm=float(chord[0]),bridge_chord_max_mm=float(chord[:n].max()),
        bridge_chord_final_mm=float(chord[end]),
        last_reference_front_tangent_errors_deg=controls[-1]['reference']['front_tangent_error_deg'])
    write_json(folder/'analysis.json',report);return report


def main(root,render):
    folders=[root/m for m in MODES];reports=[assess(f) for f in folders]
    initial=[np.load(f/'initial_state.npz')['state'] for f in folders]
    assert all(np.array_equal(initial[0],x) for x in initial[1:])
    write_json(root/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,d,meta,c=load_checkpoint(HORIZONTAL,verify_hashes=False);corner,source,target,radius=pipe_frame(m)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for f in folders:
        z=np.load(f/'trace.npz');t=z['time']-z['time'][0]
        centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        axes[0,0].plot(t,np.mean(centers[:,:6]@target,axis=1)-np.mean(centers[0,:6]@target),label=f.name)
        axes[0,1].plot(t,coverage(z['points_mm'],corner,target),label=f.name)
        axes[1,0].plot(t,z['normal_force'][:,1,:6].sum(axis=1),label=f.name)
        axes[1,1].plot(t,z['region'].sum(axis=1),label=f.name)
    for ax,title in zip(axes.flat,['Front mean progress (mm)','Front net angular coverage (turns)',
        'Front target-pipe normal force (N)','Modules in target region']):
        ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.axvline(25.6,color='gray',ls=':')
    for ax in axes[1]:ax.set_xlabel('Time (s), final 3 s fixed-target hold')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    if render:
        import segmented_pole_review as review
        review.load_checkpoint=lambda p:load_checkpoint(p,verify_hashes=False)
        for f in folders:review.render(f);print('rendered',f.name,flush=True)
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--render',action='store_true');a=p.parse_args();main(a.output_root,a.render)
