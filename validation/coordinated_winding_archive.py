"""Check recorded delivery and preserve this bounded iteration, including failures.

No physics runs or hashes. Extract bundles into OUT to restore each run folder.
"""
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from coordinated_winding_experiment import OUT
from segmented_pole_experiment import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def delivery_check(folder):
    plan=read(folder/'plan.json')['arguments']
    controls=read(folder/'controls.json')
    commands=read(folder/'commands.json')
    assessment=read(folder/'assessment.json')
    assert assessment['initial_pose_exact'] and assessment['EXE_delivery_checked']
    # The first lead-only trial predates substep delivery and sends once/dt.
    delivery_period=plan.get('delivery_period',plan['dt'])
    count=round(plan['dt']/delivery_period)
    assert len(commands)==len(controls)*count
    with np.load(folder/'trace.npz') as z:
        previous=z['ctrl'][0].copy()
    largest=0.
    for i,control in enumerate(controls):
        target=np.asarray(control['target'])
        actual=np.asarray([c['target'] for c in commands[i*count:(i+1)*count]])
        fractions=np.arange(1,count+1)/count
        assert np.allclose(actual,previous+(target-previous)*fractions[:,None],atol=1e-12,rtol=0)
        delta=np.diff(np.vstack([previous,actual]),axis=0)
        largest=max(largest,float(np.degrees(abs(delta)).max()))
        previous=target
    assert largest<=plan['speed_limit']*delivery_period+1e-8
    return dict(command_count=len(commands),interpolation_matches=True,
                maximum_command_increment_deg=largest,
                initial_pose_and_EXE_delivery_checked=True)


def package():
    destination=OUT/'archives';destination.mkdir(exist_ok=True)
    index=[]
    for folder in sorted(OUT.iterdir()):
        if not (folder/'result.json').is_file():continue
        check=delivery_check(folder)
        write_json(folder/'delivery_check.json',check)
        result=read(folder/'result.json')
        metrics=read(folder/'process_metrics.json')
        gate=read(folder/'whole_crossing_audit.json')
        output=destination/(folder.name+'.zip')
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.png','.mp4','.pyc'))
        with ZipFile(output,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
            for path in files:archive.write(path,path.relative_to(OUT).as_posix())
        index.append(dict(run=folder.name,archive=output.relative_to(OUT).as_posix(),
            hypothesis=result['hypothesis'],settings=read(folder/'plan.json')['arguments'],
            outcome=result['outcome'],metrics=metrics,acceptance=gate,delivery=check,
            files=len(files),bytes=output.stat().st_size))
        print(folder.name,round(output.stat().st_size/1024**2,2),'MiB',flush=True)
    # The longer feedback run only changes the time limit. Its common prefix
    # should reproduce the earlier 160 s trajectory, not silently retune it.
    with np.load(OUT/'feedback_horizontal/trace.npz') as a, np.load(OUT/'feedback_extended_horizontal/trace.npz') as b:
        n=len(a['time'])
        differences={k:float(np.max(abs(a[k]-b[k][:n]))) for k in ('qpos','qvel','ctrl','phase')}
        assert all(value==0 for value in differences.values())
    write_json(OUT/'extension_check.json',dict(common_samples=n,maximum_absolute_difference=differences))
    write_json(OUT/'runs.json',index)


if __name__=='__main__':package()
