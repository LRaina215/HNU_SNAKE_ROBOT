"""Necessary execution checks and physical transfer metrics, without hashes."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from segmented_pole_experiment import BASE,HORIZONTAL,load_checkpoint,pipe_frame,write_json
from snake_control.alternating_pole_phase import LocalPhaseCommand
import numpy as np

OUT=BASE/'migration_v1'
MODES=['synchronous','front_first','rear_first']


def assess(folder):
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    z=np.load(folder/'trace.npz');commands=json.loads((folder/'commands.json').read_text())
    controls=json.loads((folder/'controls.json').read_text());n=result['rolling_samples'];steps=result['steps']
    assert n==20*steps+1 and len(commands)==steps and all(c['applied'] for c in commands)
    assert np.isfinite(z['qpos']).all() and np.max(abs(np.diff(z['time'])-.01))<1e-9
    b=np.array([c['boundary_links'] for c in controls]);changes=np.r_[0.,np.diff(b)]
    assert np.max(abs(changes))<=.25+1e-10
    assert np.all(abs(changes[1::2])<1e-12),'boundary changed within a phase cycle'
    for k,c in enumerate(commands):assert np.max(abs(z['ctrl'][k*20+1]-c['target']))<1e-12
    prefix_error=None
    if not result['migrating']:
        old=np.load(BASE/'continuous_v1'/f"{result['condition']}_c32"/'trace.npz')
        size=min(n,len(old['qpos']));prefix_error=float(abs(old['qpos'][:size]-z['qpos'][:size]).max())
        assert prefix_error<1e-9 and np.all(b==6.)
    m,d,meta,c=load_checkpoint(HORIZONTAL,verify_hashes=False);corner,source,target,radius=pipe_frame(m)
    links=np.array(meta['settings']['links']);nodes=np.r_[0.,np.cumsum(links)]
    bounds=np.r_[0.,(nodes[:-1]+nodes[1:])/2,nodes[-1]]
    # Reintegrate a representative changed-boundary command at higher order.
    idx=np.flatnonzero(changes)
    check_index=int(idx[len(idx)//2]) if len(idx) else 0
    row=controls[check_index]
    field=LocalPhaseCommand(bounds,row['observed_with_compliance'],row['preload'],row['total'],row['delta'],
        row['boundary_links']*links.mean(),6*links.mean())
    integral_error=float(np.max(abs(field.joint_angles(links,order=24)-row['raw'])))
    assert integral_error<1e-8
    remap_error=0.
    if len(idx):
        closed=np.asarray(row['total'])-row['delta']
        assert abs(closed[0]-closed[1])<1e-12
        old_center=controls[check_index-1]['boundary_links']*links.mean()
        old=LocalPhaseCommand(bounds,row['observed_with_compliance'],row['preload'],closed,[0,0],old_center,6*links.mean())
        new=LocalPhaseCommand(bounds,row['observed_with_compliance'],row['preload'],closed,[0,0],row['boundary_links']*links.mean(),6*links.mean())
        remap_error=float(abs(new.joint_angles(links)-old.joint_angles(links)).max())
        assert remap_error<1e-10
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    front=(centers[:,:6]@target).mean(axis=1);rear=(centers[:,10:]@source).mean(axis=1)
    head=z['points_mm'][:,0]@target;nf=z['normal_force']
    end=n-1;hold=slice(end,len(z['time']))
    front_contact=np.any(nf[:n,1,:6]>.1,axis=1)
    no_front=np.convolve((~front_contact).astype(int),np.ones(20,dtype=int),mode='valid')
    gaps=np.flatnonzero(no_front==20)
    peak=int(np.argmax(front[:n]))
    retreat=float(max(0.,head[end]-head[hold].min()))
    rootmotion=float(np.linalg.norm(z['qpos'][hold,:3]-z['qpos'][end,:3],axis=1).max()*1000)
    hold_gate=bool(result['hold_seconds'] and np.all(z['region'][hold]) and
        np.all(np.any(nf[hold,1]>.1,axis=1)) and not np.any(nf[hold,0]>.1)
        and not np.any(z['ground'][hold]) and retreat<=10 and rootmotion<=20)
    milestones=[]
    for cycle in (8,16,32,64,96,128,160,192):
        i=cycle*40
        if i>=n:continue
        milestones.append(dict(cycle=cycle,boundary=float(z['boundary_links'][i]),
            front_mm=float(front[i]-front[0]),rear_mm=float(rear[i]-rear[0]),
            region_modules=np.flatnonzero(z['region'][i]).tolist(),
            target_contact_modules=np.flatnonzero(nf[i,1]>.1).tolist()))
    out=dict(condition=result['condition'],migrating=result['migrating'],outcome=result['outcome'],
        cycles=steps/2,rolling_seconds=float(z['time'][end]-z['time'][0]),commands_applied=steps,
        fixed_32cycle_prefix_qpos_error=prefix_error,reintegration_error_rad=integral_error,
        closed_phase_remap_error_rad=remap_error,
        final_boundary=float(b[-1]),boundary_range=[float(b.min()),float(b.max())],
        first_migration_cycle=int(idx[0]//2+1) if len(idx) else None,
        first_migration_at_s=float(commands[idx[0]]['time']-z['time'][0]) if len(idx) else None,
        final_front_progress_mm=float(front[end]-front[0]),final_rear_progress_mm=float(rear[end]-rear[0]),
        final_head_progress_mm=float(head[end]-head[0]),
        final_target_region_modules=np.flatnonzero(z['region'][end]).tolist(),
        max_target_region_count=int(z['region'][:n].sum(axis=1).max()),
        final_target_contact_modules=np.flatnonzero(nf[end,1]>.1).tolist(),
        final_source_contact_modules=np.flatnonzero(nf[end,0]>.1).tolist(),
        rolling_front_contact_fraction=float(np.any(nf[:n,1,:6]>.1,axis=1).mean()),
        first_front_contact_loss_0_2s_at=float(z['time'][gaps[0]]-z['time'][0]) if len(gaps) else None,
        peak_front_progress_mm=float(front[peak]-front[0]),peak_front_progress_at_s=float(z['time'][peak]-z['time'][0]),
        final_unused_branch_contact_modules=np.flatnonzero(nf[end,2]>.1).tolist(),
        rolling_floor=bool(z['ground'][:n].any()),hold_floor=bool(z['ground'][hold].any()) if result['hold_seconds'] else None,
        hold_head_retreat_mm=retreat if result['hold_seconds'] else None,
        hold_root_displacement_mm=rootmotion if result['hold_seconds'] else None,hold_gate=hold_gate,
        complete_crossing=bool(result['outcome']=='full_transfer' and hold_gate),
        max_tracking_error_deg=float(abs(z['error_deg'][:n]).max()),
        actuator_saturation_fraction=float(np.mean(abs(z['torque'][:n])>=.99*m.actuator_forcerange[:,1])),
        clipped_joint_steps=result['clipped_joint_steps'],milestones=milestones)
    write_json(folder/'analysis.json',out);return out


def main(root,render=False):
    folders=[root/f'{mode}_{boundary}' for mode in MODES for boundary in ('fixed','migrating')]
    reports=[assess(f) for f in folders]
    states=[np.load(f/'initial_state.npz')['state'] for f in folders]
    assert all(np.array_equal(states[0],s) for s in states[1:])
    write_json(root/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for k,mode in enumerate(MODES):
        for boundary in ('fixed','migrating'):
            f=root/f'{mode}_{boundary}';z=np.load(f/'trace.npz')
            t=z['time']-z['time'][0]
            axes[0,k].plot(t,z['boundary_links'],label=boundary)
            axes[1,k].plot(t,z['region'].sum(axis=1),label=boundary)
            axes[2,k].plot(t,z['points_mm'][:,0,1]-z['points_mm'][0,0,1],label=boundary)
        axes[0,k].set_title(mode)
        for ax in axes[:,k]:ax.grid(alpha=.2);ax.legend()
        axes[2,k].set_xlabel('Physical time (s), includes final hold if reached')
    axes[0,0].set_ylabel('Boundary (link units)');axes[1,0].set_ylabel('Modules in target region (0-21)')
    axes[2,0].set_ylabel('Head target-axis progress (mm)')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150)
    # Focused view of the initial transfer and reversal, with the same data.
    for ax in axes.flat:ax.set_xlim(0,24)
    fig.savefig(root/'comparison_first24s.png',dpi=150);plt.close(fig)
    if render:
        import segmented_pole_review as review
        # Renderer only: use the same loader without redundant hash checking.
        review.load_checkpoint=lambda p:load_checkpoint(p,verify_hashes=False)
        for f in folders:review.render(f);print('rendered',f.name,flush=True)
    for r in reports:print(json.dumps(r),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUT);parser.add_argument('--render',action='store_true')
    args=parser.parse_args();main(args.output_root.resolve(),args.render)
