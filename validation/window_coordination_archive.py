"""Preserve the bounded window experiments, including unsuccessful results.

Checks actual relay delivery against planned interpolation; no new physics or
hash checks. Extract each ZIP into OUT to replay its saved trajectory.
"""
import json
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from segmented_pole_experiment import write_json
from window_coordination_experiment import OUT
from three_segment_review import context


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def delivery(folder):
    plan=read(folder/'plan.json')['arguments']
    controls=read(folder/'controls.json')
    commands=read(folder/'commands.json')
    assessment=read(folder/'assessment.json')
    assert assessment['initial_pose_exact'] and assessment['EXE_delivery_checked']
    count=round(plan['dt']/plan['delivery_period'])
    assert len(commands)==len(controls)*count
    with np.load(folder/'trace.npz') as z:previous=z['ctrl'][0].copy()
    largest=0.
    for i,control in enumerate(controls):
        target=np.asarray(control['target'])
        actual=np.asarray([c['target'] for c in commands[i*count:(i+1)*count]])
        fractions=np.arange(1,count+1)/count
        assert np.allclose(actual,previous+(target-previous)*fractions[:,None],atol=1e-12,rtol=0)
        largest=max(largest,float(np.degrees(abs(np.diff(np.vstack([previous,actual]),axis=0))).max()))
        previous=target
    assert largest<=90*plan['delivery_period']+1e-8
    return dict(command_count=len(commands),interpolation_matches=True,
        maximum_command_increment_deg=largest,initial_pose_and_EXE_delivery_checked=True)


def coordination_plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    controls=read(folder/'controls.json')
    t=np.array([c['time'] for c in controls])
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for key,label in [('formation_progress','Formation command'),('actual_prefix','Transfer prefix proxy'),('rear','Rear boundary')]:
        axes[0].plot(t,[c[key] for c in controls],label=label)
    axes[1].plot(t,[c['front_wrap'] for c in controls],label='Measured front angular wrap')
    axes[1].axhline(.5,color='grey',ls='--',lw=.8)
    axes[2].plot(t,[c['phase_difference'] for c in controls],label='Rear minus front phase')
    for key,label in [('feed','Rear supply'),('follow','Command following'),('grip','Wrap/contact proxy')]:
        axes[3].plot(t,[c[key] for c in controls],label=label)
    for ax,label in zip(axes,['Joint index / count','Turns','Radians','Factor']):
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Physical seconds after checkpoint')
    fig.suptitle(folder.name+'\nTransfer prefix starts at the captured front count; not a support certificate.')
    fig.tight_layout()
    out=folder.parent/'reviews';out.mkdir(exist_ok=True)
    fig.savefig(out/(folder.name+'_coordination.png'),dpi=130);plt.close(fig)


def diagnose(folder):
    result,z,m,d,meta=context(folder)
    t=z['time']-z['time'][0]
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    speed=np.r_[np.zeros((1,21)),np.linalg.norm(np.diff(centers,axis=0),axis=2)/np.diff(t)[:,None]]
    report=read(folder/'process_metrics.json')
    valid=(t>=1)&(t<=report['metrics_end_seconds']+1e-6)
    row,module=np.unravel_index(np.where(valid[:,None],speed,-1).argmax(),speed.shape)
    controls=read(folder/'controls.json')
    ci=max(0,np.searchsorted([c['time'] for c in controls],t[row],side='left'))
    control=controls[min(ci,len(controls)-1)]
    peak=dict(time_seconds=float(t[row]),module=int(module),speed_mm_s=float(speed[row,module]),
        source_contact_modules=np.flatnonzero(z['normal_force'][row,0]>.1).tolist(),
        target_contact_modules=np.flatnonzero(z['normal_force'][row,1]>.1).tolist(),
        target_region_modules=int(z['region'][row].sum()),ground=bool(z['ground'][row]),
        commanded_partition=z['partition'][row].tolist(),
        formation_progress=control['formation_progress'],front_wrap_proxy=control['front_wrap'])
    source=np.any(z['normal_force'][:,0]>.1,axis=1)
    absent=np.convolve((~source).astype(int),np.ones(60,dtype=int),mode='full')[:len(t)]>=60
    ids=np.flatnonzero(absent)
    out=dict(peak_module_speed=peak,
        first_source_absence_0_6s_end=float(t[ids[0]]) if len(ids) else None,
        assessment=read(folder/'assessment.json'),
        limitation='Angular wrap and contact presence do not establish force closure; prefix has a captured-front floor.')
    write_json(folder/'failure_diagnosis.json',out)
    return out


def package(root=OUT,names=None):
    destination=root/'archives';destination.mkdir(exist_ok=True)
    index={row['run']:row for row in read(root/'runs.json')} if names and (root/'runs.json').exists() else {}
    for folder in sorted(root.iterdir()):
        if names and folder.name not in names:continue
        if not (folder/'stage1_gate.json').is_file():continue
        check=delivery(folder);write_json(folder/'delivery_check.json',check)
        result=read(folder/'result.json');plan=read(folder/'plan.json')
        controls=read(folder/'controls.json')
        diagnosis=diagnose(folder)
        coordination_plot(folder)
        output=destination/(folder.name+'.zip')
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.png','.mp4','.pyc'))
        with ZipFile(output,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
            for path in files:archive.write(path,path.relative_to(root).as_posix())
        index[folder.name]=dict(run=folder.name,archive=output.relative_to(root).as_posix(),
            hypothesis=result['hypothesis'],settings=plan['arguments'],outcome=result['outcome'],
            metrics=read(folder/'process_metrics.json'),acceptance=read(folder/'whole_crossing_audit.json'),
            stage1=read(folder/'stage1_gate.json'),delivery=check,diagnosis=diagnosis,
            coordination=dict(maximum_phase_difference=float(max(abs(c['phase_difference']) for c in controls)),
                initial_front_wrap=controls[0]['front_wrap'],final_front_wrap=controls[-1]['front_wrap'],
                maximum_formation=max(c['formation_progress'] for c in controls)),
            files=len(files),bytes=output.stat().st_size)
        print(folder.name,round(output.stat().st_size/1024**2,2),'MiB',flush=True)
    write_json(root/'runs.json',[index[key] for key in sorted(index)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args();package(args.root,args.names)
