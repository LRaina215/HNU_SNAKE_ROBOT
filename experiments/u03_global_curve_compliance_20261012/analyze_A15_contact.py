"""Actual receiving contact vs template/world-pose gap; no physical steps."""
from physics import *
import json

def segment_radius(P):
    x=P[:-1,[0,2]]-[1.1,1];v=P[1:,[0,2]]-P[:-1,[0,2]];u=np.clip(-(x*v).sum(1)/np.maximum((v*v).sum(1),1e-20),0,1)
    return np.linalg.norm(x+u[:,None]*v,axis=1)

rows=[]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A14',BASE/'A14'),('A15',BASE/'A15')]:
    z=np.load(path/'trace.npz');t=z['t'];p=z['points'];c=z['centers'];f=z['forces'];C=z['nominal_curve_points'] if 'nominal_curve_points' in z else None
    m=mujoco.MjModel.from_xml_path(str(path/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;tg=[m.geom('pipe_straight_04').id,m.geom('pipe_sharp_out_03').id]
    rad=np.linalg.norm(c[:,:6,[0,2]]-[1.1,1],axis=2);geo=(rad>.065)&(rad<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2);span=np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1)
    ix=np.flatnonzero(geo.all(1));best=int(ix[np.argmax(span[ix])]) if ix.size else int(np.argmax(geo.sum(1)))
    times=sorted(set([35,40,45,50,52,53.8,53.9,56.4,60,70,float(t[best]),float(t[-1])]))
    for sec in times:
        if sec>t[-1]+.01:continue
        k=int(np.argmin(abs(t-sec)));d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);gaps=[];dist=[];status=[]
        for b in range(h,h+6):
            gs=np.flatnonzero(m.geom_bodyid==b);hit=[float(ct.dist) for ct in d.contact if (ct.geom1 in tg and ct.geom2 in gs) or (ct.geom2 in tg and ct.geom1 in gs)]
            cd=float(min(hit)*1000) if hit else None;dist.append(cd)
            if cd is not None:gaps.append(cd);status.append('actual_original_contact')
            else:
                val=min(float(mujoco.mj_geomDistance(m,d,int(g),int(pipe),.005,None)) for g in gs for pipe in tg)
                if val<0:gaps.append(None);status.append('inconsistent_negative_query_rejected')
                elif val>=.005-1e-9:gaps.append(None);status.append('outside5mm_query')
                else:gaps.append(val*1000);status.append('positive_sub5mm_query')
        cc=(C[k,:-1]+C[k,1:])/2 if C is not None else None
        rows.append(dict(case=name,t=float(t[k]),front6_target_N=f[k,1,:6].tolist(),front6_geo=geo[k].tolist(),front6_span_deg=float(np.rad2deg(span[k])),actual_simple_front6_gap_mm=((segment_radius(p[k,:7])-.0855)*1000).tolist(),nominal_simple_front6_gap_mm=((segment_radius(C[k,:7])-.0855)*1000).tolist() if C is not None else None,actual_original_front6_target_contact_dist_mm=dist,original_front6_gap_mm=gaps,query_status=status,head_actual_center=c[k,0].tolist(),head_C_center=cc[0].tolist() if cc is not None else None,head_world_error_mm=((c[k,0]-cc[0])*1000).tolist() if cc is not None else None,source_N=float(f[k,0].sum()),target_N=float(f[k,1].sum())))
out=dict(physical_steps=0,method='Actual saved states in original model; true contact.dist and saved normal force first. Positive5mm bounded geometry query auxiliary; bound censored/negative contradictions rejected. Simple infinite-axis envelope diagnostic only, omits finite ends/elbow/native mesh. World head error distinct from template radial gap.',samples=rows)
(BASE/'A15/contact_comparison.json').write_text(json.dumps(out,indent=2));(BASE/'A15_CONTACT_COMPARISON.json').write_text(json.dumps(out,indent=2))
for r in rows:
    if r['case']=='A15' and abs(r['t']-53.9)<.02:print(r,flush=True)
