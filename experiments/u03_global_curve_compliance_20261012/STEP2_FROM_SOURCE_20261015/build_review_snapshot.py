"""Preserve the exact reviewable implementation; no approval and no dynamics."""
from pathlib import Path
import shutil,json,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;root=next(p for p in P.parents if (p/'mujoco_model').is_dir());model=root/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
dest=P/'REVIEW_SNAPSHOT';dest.mkdir(exist_ok=False)
names=[f.name for f in P.glob('*.py')]+['REGISTERED.json','SOURCE_INITIAL.npz','SOURCE_INITIAL_PROOF.json','SOURCE_INITIAL_21_RODS.csv','source_constants.json','PRE_DYNAMICS_AUDIT.json','DESIGN.md','SOURCE_INITIAL_DUAL.png']
for name in names:shutil.copy2(P/name,dest/name)
shutil.copy2(model,dest/'model.xml')
for mesh in ET.parse(model).findall('./asset/mesh'):
    if mesh.get('file'):
        target=dest/mesh.get('file');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(model.parent/mesh.get('file'),target)
(dest/'SNAPSHOT_STATUS.json').write_text(json.dumps(dict(files=names,independent_review_passed=False,dynamics_authorized=False,new_dynamics_starts=0,
    future_check='Exact byte equality for source/initial/model/meshes; numeric registration equality excluding only two approval flags; no hash checks'),indent=2)+'\n')
print(json.dumps(dict(review_snapshot=str(dest),independent_review_passed=False,files=len(names))))
