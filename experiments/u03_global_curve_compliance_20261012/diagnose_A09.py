from physics import *
from two_window_reference import TwoWindowReference,LINK
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);r=TwoWindowReference(nodes(d),d.qpos[7:],clearance_connector=True)
e=json.loads((BASE/'A09/reference_error.json').read_text());S=e['S'];r.set_nominal_family(S);loc=r.start_arc+LINK*S;p=r.spline(loc)
out=dict(physical_steps=0,actual_reference_exception_time=e['t'],S_cells=S)
for n in range(21):
    ds=np.linspace(.1*LINK,2.5*LINK,70);dist=np.linalg.norm(r.spline(loc-ds)-p,axis=1);ids=np.flatnonzero(dist>=LINK)
    if not len(ids):
        ss=np.linspace(loc-.2,loc,1000);v=r.spline(ss,1);acc=r.spline(ss,2);kap=np.linalg.norm(np.cross(v,acc),axis=1)/np.linalg.norm(v,axis=1)**3
        xx=r.spline(ss);phi=np.unwrap(np.arctan2(xx[:,2]-1,xx[:,0]-1.1))
        out.update(failed_next_node=n+1,at_arc_m=loc,connector_bounds_m=[r.start_arc,r.target_arc],local_2_5_link_max_distance_mm=float(dist.max()*1000),max_local_curvature_per_m=float(kap.max()),local_min_radius_mm=float(1000/kap.max()),local_target_polar_span_deg=float(np.rad2deg(np.ptp(phi))),local_Y_range_m=[float(xx[:,1].min()),float(xx[:,1].max())]);break
    j=int(ids[0]);lo=loc-ds[j];hi=loc if j==0 else loc-ds[j-1]
    for _ in range(12):
        mid=(lo+hi)/2
        if np.linalg.norm(r.spline(mid)-p)>=LINK:lo=mid
        else:hi=mid
    loc=(lo+hi)/2;p=r.spline(loc)
z=np.load(BASE/'A09/trace.npz');rows=[]
for sec in [35,38,40,42,45,47.8]:
    i=np.argmin(abs(z['t']-sec));c=z['centers'][i,:6];span=np.ptp(np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1)));rows.append(dict(t=float(z['t'][i]),M0_m=c[0].tolist(),front_target_N=float(z['forces'][i,1,:6].sum()),target_forces_N=z['forces'][i,1,:6].tolist(),front_span_deg=float(np.rad2deg(span)),front_tracking_deg=float(np.rad2deg(np.max(abs(z['ctrl'][i,:6]-z['qpos'][i,7:13]))))))
out['actual_samples']=rows;(BASE/'A09_exception_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
