"""Locate nominal chord search failure without re-running lost dynamics."""
from physics import *
from curve_reference import GlobalCurveReference,LINK
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);ref=GlobalCurveReference(nodes(d),d.qpos[7:],moving_family=True)
failure=None
for t in np.arange(40,50,.025):
 S=ref.feed_per_cycle*.02*(t-.5);ref.set_nominal_family(S);s=ref.start_arc+S;point=ref.spline(s)
 for n in range(21):
  ds=np.linspace(.1*LINK,2.5*LINK,70);trial=s-ds;dist=np.linalg.norm(ref.spline(trial)-point,axis=1);ids=np.flatnonzero(dist>=LINK)
  if not len(ids):
   ds2=np.linspace(.1*LINK,10*LINK,1000);d2=np.linalg.norm(ref.spline(s-ds2)-point,axis=1);ids2=np.flatnonzero(d2>=LINK)
   failure=dict(nominal_geometry_t=float(t),not_actual_end_time=True,S=float(S),node=n,maximum_distance_in_original_search_m=float(dist.max()),physical_rod_length_m=LINK,extended_search_first_arc_gap_m=None if not len(ids2) else float(ds2[ids2[0]]),reference_curve_min_curvature_radius_m=float(np.min(np.linalg.norm(ref.spline(s-ds2,1),axis=1)**3/np.maximum(np.linalg.norm(np.cross(ref.spline(s-ds2,1),ref.spline(s-ds2,2)),axis=1),1e-12))),diagnosis='Original local search range insufficient; existence outside local range does not establish physically feasible bridge or hinge projection. Tight nominal bends change chord/arc mapping.');break
  j=ids[0];lo=trial[j];hi=s if j==0 else trial[j-1]
  for _ in range(12):
   mid=(lo+hi)/2
   if np.linalg.norm(ref.spline(mid)-point)>=LINK:lo=mid
   else:hi=mid
  s=(lo+hi)/2;point=ref.spline(s)
 if failure:break
(BASE/'A04_nominal_geometry_diagnosis.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2))
