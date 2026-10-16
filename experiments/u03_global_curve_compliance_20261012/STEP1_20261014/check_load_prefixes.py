"""Confirm pre-load trajectories and lower-acquisition lineage, no dynamics."""
from physics import *
import json
path=BASE/'FINAL_REVIEW.json'; review=json.loads(path.read_text()); checks=[]
for condition,prefix,onset in [('RECEIVE_LAG','L',96.),('SOURCE_DRAG','D',108.)]:
    for variant in ['OPEN','SPEED','SHAPE','JOINT']:
        name=prefix+'_'+variant; nominal='N_SPEED' if variant in ['OPEN','SPEED'] else 'N_SHAPE'
        a=np.load(BASE/name/'trace.npz'); b=np.load(BASE/nominal/'trace.npz'); mask=a['t']<onset-1e-6
        diff={k:float(np.max(abs(a[k][mask]-b[k][mask]))) for k in ['qpos','qvel','ctrl','raw_command','centers','S']}
        checks.append(dict(case=name,nominal_reference=nominal,until_s=onset,max_differences=diff,all_equal=all(v==0. for v in diff.values())))
assert all(x['all_equal'] for x in checks)
review['development_before_load_equality']=checks
review['direction_review']='Executor inspected current-reference actual dual-view montages at110s and230s. Every development trajectory before96s/108s exactly matches its independently verified nominal trajectory, preserving the early lower-up acquisition. Full target progression and same-direction axial spin checked separately.'
path.write_text(json.dumps(review,indent=2))
print(json.dumps(checks,indent=2))
