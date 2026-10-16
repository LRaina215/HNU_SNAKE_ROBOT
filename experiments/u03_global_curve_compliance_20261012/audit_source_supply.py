"""Existing trajectories only; distinguish material polar and carrier phase."""
from pathlib import Path
import numpy as np,json
base=Path(__file__).resolve().parent;root=base.parents[1];rows=[]
for name,path in [('U03',root/'experiments/cmu_native_underup_20261012/U03'),('A10',base/'A10')]:
    z=np.load(path/'trace.npz');t=z['t'];c=z['centers'];polar=np.unwrap(np.arctan2(c[:,:,2]-1,c[:,:,1]),axis=0)
    dx=(c[:,10:,0]-c[0,10:,0]).mean(1);dp=np.rad2deg((polar[:,10:]-polar[0,10:]).mean(1))
    for sec in [0,10,20,30,40]:
        i=int(np.argmin(abs(t-sec)));r=dict(case=name,t=float(t[i]),rear10_20_dx_mm=float(dx[i]*1000),rear10_20_material_polar_delta_deg=float(dp[i]))
        if name=='A10':
            C=z['nominal_curve_points'];cc=(C[:,:-1]+C[:,1:])/2;pp=np.unwrap(np.arctan2(cc[:,:,2]-1,cc[:,:,1]),axis=0)
            r.update(reference_dx_mm=float((cc[i,10:,0]-cc[0,10:,0]).mean()*1000),reference_material_polar_delta_deg=float(np.rad2deg((pp[i,10:]-pp[0,10:]).mean())),S=float(z['S'][i]),nominal_connector_arc_m=float(z['connector_arc_m'][i]),source_material_tail_station_m=float(.5*(z['reference_locations'][i,-1]+z['reference_locations'][i,-2])),carrier_roll_phase_deg=float(np.rad2deg(z['roll_phase'][i])))
        rows.append(r)
out=dict(source='saved actual and nominal arrays only; no dynamics or fitting of per-frame poses',material_polar_definition='atan2(z-1,y), unwrap per actual material module over time; average modules10..20',samples=rows)
(base/'A11_SOURCE_RELATION_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
