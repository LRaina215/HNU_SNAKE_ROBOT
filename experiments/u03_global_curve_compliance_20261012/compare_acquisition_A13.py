"""Identical actual front/source/full geometry criteria including U03."""
from physics import *
import json
def stable(mask,t,seconds=2):
    start=None
    for k,v in enumerate(mask):
        if not v:start=None
        elif start is None:start=k
        elif t[k]-t[start]>=seconds:return float(t[start])
    return None
rows=[]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A11',BASE/'A11'),('A12',BASE/'A12'),('A13',BASE/'A13')]:
    z=np.load(path/'trace.npz');t=z['t'];c=z['centers'];f=z['forces'];r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);yrange=(c[:,:,1]>.15)&(c[:,:,1]<1.2);target=(r>.07)&(r<.12)&yrange
    frontgeo=(r[:,:6]>.065)&(r[:,:6]<.13)&yrange[:,:6];span=np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1);front=frontgeo.all(1)&(span>np.pi)&(f[:,1,:6].sum(1)>.2);full=target.all(1)&(f[:,0].sum(1)<.1)&(f[:,1].sum(1)>.2)
    source_r=np.linalg.norm(c[:,:,1:]-[0,1],axis=2);sourcegeo=(c[:,:,0]>-.2)&(c[:,:,0]<.95)&(source_r>.07)&(source_r<.12);source_force=f[:,0].sum(1);active=np.flatnonzero(source_force>.1);geom=np.flatnonzero(sourcegeo.any(1));suffix=np.flatnonzero(~full);suffix=int(suffix[-1]+1) if len(suffix) else 0
    contacts=[]
    for v in range(6):
        k=np.flatnonzero(f[:,1,v]>.1);contacts.append(dict(M=v,first_s=float(t[k[0]]) if len(k) else None,last_s=float(t[k[-1]]) if len(k) else None))
    samples=[]
    for sec in [35,39,45,53.8,60,80,99.2,133.2,160,float(t[-1])]:
        if sec>t[-1]:continue
        k=int(np.argmin(abs(t-sec)));samples.append(dict(t=float(t[k]),head_center=c[k,0].tolist(),tail_center=c[k,-1].tolist(),front_target_geometry=frontgeo[k].tolist(),front_span_deg=float(np.rad2deg(span[k])),source_geometry_modules=np.flatnonzero(sourcegeo[k]).tolist(),source_contact_modules=np.flatnonzero(f[k,0]>.1).tolist(),source_N=float(source_force[k]),front_target_N=float(f[k,1,:6].sum()),target_straight_rods=int(target[k].sum()),target_straight_modules=np.flatnonzero(target[k]).tolist()))
    rows.append(dict(case=name,front_stable_s=stable(front,t),full_stable_s=stable(full,t),post_full_suffix_s=float(t[-1]-t[suffix]) if suffix<len(t) else 0,last_source_contact_s=float(t[active[-1]]) if len(active) else None,last_source_straight_geometry_s=float(t[geom[-1]]) if len(geom) else None,max_target_straight_rods=int(target.sum(1).max()),front_contacts=contacts,samples=samples))
out=dict(criteria='original front6 radius65..130mm, Y.15..1.2m, halfturn/contact sum>.2N,2s; full21 radius70..120mm plus source<.1N,target>.2N,2s. Contact classification includes original elbow.',source='saved actual traces only; no nominal-window success',cases=rows)
(BASE/'A13_ACQUISITION_COMPARISON.json').write_text(json.dumps(out,indent=2));(BASE/'A13/acquisition_comparison.json').write_text(json.dumps(out,indent=2))
for a in rows:print({k:v for k,v in a.items() if k not in ['samples','front_contacts']})
