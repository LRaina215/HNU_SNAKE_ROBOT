"""Reproduce only-change audit using saved states; no physical steps or SHA."""
from physics import *
import json
a=np.load(BASE/'A13/trace.npz');c=np.load(BASE/'A14/trace.npz');ix=c['t']<=32.;j=a['t']<=32.;rows={}
for key in ['t','qpos','qvel','ctrl','points','forces','raw_command']:
    rows[key]=dict(exact_equal=bool(np.array_equal(a[key][j],c[key][ix])),max_abs_delta=float(np.max(abs(a[key][j]-c[key][ix]))))
out=dict(initial_bytes_equal=(BASE/'A13/initial.npz').read_bytes()==(BASE/'A14/initial.npz').read_bytes(),model_bytes_equal=(BASE/'A13/model.xml').read_bytes()==(BASE/'A14/model.xml').read_bytes(),unchanged_through_t32_samples=int(ix.sum()),unchanged_prefix=rows,actual_trial_count=1)
(BASE/'A14/isolation_audit.json').write_text(json.dumps(out,indent=2));print(out)
