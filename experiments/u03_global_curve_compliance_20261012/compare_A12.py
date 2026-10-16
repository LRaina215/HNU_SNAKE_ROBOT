"""Same saved-trace criteria and paired A11/A12 handover diagnostics."""
from physics import *
import json
out=dict(source='saved actual traces only; no new physical steps',cases=[])
for name in ['A11','A12']:
    path=BASE/name;z=np.load(path/'trace.npz');t=z['t'];c=z['centers'];q=z['qpos'][:,7:];ctrl=z['ctrl'];raw=z['raw_command'];f=z['forces'];C=z['nominal_curve_points'];cc=.5*(C[:,:-1]+C[:,1:]);phi=np.unwrap(np.arctan2(c[:,:,2]-1,c[:,:,1]),axis=0);cp=np.unwrap(np.arctan2(cc[:,:,2]-1,cc[:,:,1]),axis=0)
    r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);target=(r>.07)&(r<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2);span=np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1)
    front_geometry=(r[:,:6]>.065)&(r[:,:6]<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2)
    front=front_geometry.all(1)&(span>np.pi)&(f[:,1,:6].sum(1)>.2)
    samples=[]
    for sec in [10,20,30,38,40,45,50,55,60,65,70,75,80,94.925,100,120,160,200,float(t[-1])]:
        if sec>t[-1]:continue
        k=int(np.argmin(abs(t-sec)));err=np.rad2deg(ctrl[k]-q[k]);j=int(np.argmax(abs(err)));jrows=[]
        for v in [10,11]:jrows.append(dict(J=v+1,raw_deg=float(np.rad2deg(raw[k,v])),ctrl_deg=float(np.rad2deg(ctrl[k,v])),actual_deg=float(np.rad2deg(q[k,v])),error_deg=float(err[v]),torque_Nm=float(z['torque'][k,v]),adjacent_source_N=float(f[k,0,v:v+2].sum()),adjacent_target_N=float(f[k,1,v:v+2].sum())))
        samples.append(dict(t=float(t[k]),rear_dx_mm=float((c[k,10:,0]-c[0,10:,0]).mean()*1000),rear_reference_dx_mm=float((cc[k,10:,0]-cc[0,10:,0]).mean()*1000),rear_world_x_error_mm=float((c[k,10:,0]-cc[k,10:,0]).mean()*1000),rear_polar_delta_deg=float(np.rad2deg((phi[k,10:]-phi[0,10:]).mean())),rear_reference_polar_delta_deg=float(np.rad2deg((cp[k,10:]-cp[0,10:]).mean())),head_center_m=c[k,0].tolist(),source_N=float(f[k,0].sum()),target_N=float(f[k,1].sum()),front_target_N=float(f[k,1,:6].sum()),front_span_deg=float(np.rad2deg(span[k])),target_straight_modules=int(target[k].sum()),max_tracking_deg=float(abs(err).max()),worst_joint=j+1,max_raw_ctrl_deg=float(np.rad2deg(abs(raw[k]-ctrl[k]).max())),joints=jrows))
    contacts=[]
    for v in range(6):
        k=np.flatnonzero(f[:,1,v]>.1);contacts.append(dict(M=v,first_s=float(t[k[0]]) if len(k) else None,last_s=float(t[k[-1]]) if len(k) else None,peak_N=float(f[:,1,v].max())))
    local=(t>=49.95)&(t<=80.05);stats=[]
    for v in [10,11]:
        sat=local&(abs(z['torque'][:,v])>=2.355-1e-6);e=np.rad2deg(abs(ctrl[:,v]-q[:,v]));stats.append(dict(J=v+1,window='50..80s',saturation_sample_seconds=float(sat.sum()*.1),max_tracking_deg=float(e[local].max()),mean_tracking_deg=float(e[local].mean())))
    row=dict(case=name,result=json.loads((path/'result.json').read_text()),audit=json.loads((path/'audit.json').read_text()),max_target_straight_modules=int(target.sum(1).max()),any_front6_geometry_halfturn_contact=bool(front.any()),front_contacts=contacts,handover_stats=stats,samples=samples)
    out['cases'].append(row)
(BASE/'A12_comparison.json').write_text(json.dumps(out,indent=2));(BASE/'A12/comparison.json').write_text(json.dumps(out,indent=2))
for a in out['cases']:
    print(a['case'],a['result'],a['handover_stats'],'max target',a['max_target_straight_modules'],'contacts',a['front_contacts'])
