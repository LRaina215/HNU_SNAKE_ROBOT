"""Source material comparison from saved trajectories, no simulation."""
from physics import *
import json
rows=[]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A10',BASE/'A10'),('A11',BASE/'A11')]:
    z=np.load(path/'trace.npz');t=z['t'];c=z['centers'];phi=np.unwrap(np.arctan2(c[:,:,2]-1,c[:,:,1]),axis=0)
    C=z['nominal_curve_points'] if 'nominal_curve_points' in z else None
    cc=(C[:,:-1]+C[:,1:])/2 if C is not None else None
    pphi=np.unwrap(np.arctan2(cc[:,:,2]-1,cc[:,:,1]),axis=0) if cc is not None else None
    for sec in [0,10,20,30,40]:
        if sec>t[-1]:continue
        i=int(np.argmin(abs(t-sec)));r=dict(case=name,t=float(t[i]),actual_rear_dx_mm=float((c[i,10:,0]-c[0,10:,0]).mean()*1000),actual_material_polar_delta_deg=float(np.rad2deg((phi[i,10:]-phi[0,10:]).mean())),head_center_m=c[i,0].tolist(),max_ctrl_actual_deg=float(np.rad2deg(max(abs(z['ctrl'][i]-z['qpos'][i,7:])))))
        if cc is not None:r.update(reference_rear_dx_mm=float((cc[i,10:,0]-cc[0,10:,0]).mean()*1000),reference_material_polar_delta_deg=float(np.rad2deg((pphi[i,10:]-pphi[0,10:]).mean())),rear_mean_world_x_error_mm=float((c[i,10:,0]-cc[i,10:,0]).mean()*1000),head_world_error_mm=float(np.linalg.norm(c[i,0]-cc[i,0])*1000))
        rows.append(r)
out=dict(source='saved raw actual and nominal traces; same rear material modules10..20; no refit or controller correction',samples=rows)
(BASE/'A11_source_comparison.json').write_text(json.dumps(out,indent=2));(BASE/'A11/source_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps([r for r in rows if r['case']=='A11'],indent=2))
