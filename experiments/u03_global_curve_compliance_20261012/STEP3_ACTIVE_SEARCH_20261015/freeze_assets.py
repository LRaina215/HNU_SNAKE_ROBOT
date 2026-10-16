"""Copy local dependencies once. No dynamics and no content hashes."""
from pathlib import Path
import shutil, json, xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
files=['contact_sensor.py','estimated_observer.py','estimated_curve.py','from_source_curve.py',
       'native_mapping.py','source_mode.py','geometry.py','global_update.py','evaluation.py',
       'source_controller.py','visual_overlay.py','static_equilibrium.py','source_constants.json']
for name in files:
    destination=BASE/name
    if not destination.exists():shutil.copy2(BASE/'FROZEN_BASE'/name,destination)
assets=BASE/'assets';assets.mkdir(exist_ok=True)
original=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
shutil.copy2(original,assets/'model.xml')
records=[]
for mesh in ET.parse(original).findall('./asset/mesh'):
    if mesh.get('file'):
        path=Path(mesh.get('file'));dest=assets/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original.parent/path,dest)
        records.append(dict(source=str(original.parent/path),destination=str(dest.relative_to(BASE)),bytes=dest.stat().st_size))
shutil.copy2(BASE/'FROZEN_BASE/SOURCE_INITIAL.npz',assets/'inherited_initial.npz')
(BASE/'ASSET_MANIFEST.json').write_text(json.dumps(dict(model=str(original),mesh_files=records,
    dependency_source='FROZEN_BASE at snapshot time; no later STEP2 reads',hashes_used=False),indent=2))
