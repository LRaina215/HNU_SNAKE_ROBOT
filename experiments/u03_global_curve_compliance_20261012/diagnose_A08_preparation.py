from physics import *
from two_window_reference import TwoWindowReference,LINK
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=TwoWindowReference(nodes(d),d.qpos[7:]);out=dict(physical_steps=0,spec=ref.spec)
for t in np.arange(.6,20,.2):
    S=ref.feed_per_cycle*.02*(t-.5);ref.set_nominal_family(S);loc=ref.start_arc+LINK*S;p=ref.spline(loc)
    for n in range(21):
        ds=np.linspace(.1*LINK,2.5*LINK,70);dist=np.linalg.norm(ref.spline(loc-ds)-p,axis=1);ids=np.flatnonzero(dist>=LINK)
        if not len(ids):
            grid=np.linspace(loc-.2,loc,500);v=ref.spline(grid,1);acc=ref.spline(grid,2);kap=np.linalg.norm(np.cross(v,acc),axis=1)/np.linalg.norm(v,axis=1)**3
            out['failure']=dict(t=float(t),S_cells=float(S),node=n+1,arc=float(loc),connector_start=float(ref.start_arc),connector_end=float(ref.target_arc),max_distance_mm=float(dist.max()*1000),max_curvature_per_m=float(kap.max()),local_min_radius_mm=float(1000/kap.max()));break
        j=int(ids[0]);lo=loc-ds[j];hi=loc if j==0 else loc-ds[j-1]
        for _ in range(12):
            mid=(lo+hi)/2
            if np.linalg.norm(ref.spline(mid)-p)>=LINK:lo=mid
            else:hi=mid
        loc=(lo+hi)/2;p=ref.spline(loc)
    if 'failure' in out:break
(BASE/'A08_preparation_failure.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
