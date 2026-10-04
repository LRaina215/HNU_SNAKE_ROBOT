"""Lossless research data bundles; videos/plots are stored separately.

The unpacked working copies remain local. Archive relative names include the
run directory, so extracting into traveling_winding restores review inputs.
"""
import json
from zipfile import ZipFile,ZIP_DEFLATED
from segmented_pole_experiment import BASE,write_json
from traveling_winding_audit import audit


if __name__=='__main__':
    root=BASE/'traveling_winding'
    destination=root/'archives'
    destination.mkdir(exist_ok=True)
    index=[]
    for folder in sorted(root.glob('v*')):
        if not folder.is_dir() or not (folder/'result.json').exists():continue
        result=json.loads((folder/'result.json').read_text(encoding='utf8'))
        report=audit(folder)
        output=destination/(folder.name+'.zip')
        files=[p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.png','.mp4','.pyc')]
        with ZipFile(output,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
            for path in files:archive.write(path,path.relative_to(root).as_posix())
        index.append(dict(run=folder.name,archive=output.relative_to(root).as_posix(),
            hypothesis=result.get('hypothesis'),steps=result['steps'],outcome=result['outcome'],
            full_crossing=report['full_crossing_and_continued_rolling'],
            maximum_region_modules=report['maximum_target_region_modules'],
            final_region_modules=report['final_target_region_modules'],files=len(files),bytes=output.stat().st_size))
        print(folder.name,round(output.stat().st_size/1024**2,2),'MiB',flush=True)
    write_json(root/'runs.json',index)
