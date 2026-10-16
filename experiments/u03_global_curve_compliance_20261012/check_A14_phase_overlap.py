"""One diagnostic-only extension of M slope decay; no dynamics/no grid."""
from physics import *
from overlap_window_reference import OverlapWindowReference
from snake_control.pipe_clearance import PipeClearance
from inspect_A10_native import fitted
import json

class PhaseOverlapCandidate(OverlapWindowReference):
    @staticmethod
    def phase_cells(S):
        x=np.clip((S-5)/4,0,1)
        return min(S,5)+4*(x-2.5*x**4+3*x**5-x**6)

m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d)
orig=OverlapWindowReference(nodes(d),d.qpos[7:]);candidate=PhaseOverlapCandidate(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d)
times=np.arange(30,60.001,.1);rows=[]
for t in times:
    S=orig.feed_per_cycle*.02*max(0,t-.5);a={}
    for label,r in [('A13',orig),('M_decay_5_to_9',candidate)]:
        q,i=r.command(S);C=i['nominal_curve_points'];cc=(C[:-1]+C[1:])/2
        d.qpos[:]=z['qpos'];d.qpos[7:]=q;mujoco.mj_forward(m,d);shape=fitted(nodes(d),C)
        gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128
        rad=np.linalg.norm(cc[:6,[0,2]]-[1.1,1],axis=1);geo=(rad>.065)&(rad<.13)&(cc[:6,1]>.15)&(cc[:6,1]<1.2)
        span=np.rad2deg(np.ptp(np.unwrap(np.arctan2(cc[:6,2]-1,cc[:6,0]-1.1))))
        a[label]=dict(F_m=r.feed_distance(S),M_m=r.source_chart_shift,source_theta_delta_deg=float(np.rad2deg(-r.beta*(r.feed_distance(S)-r.source_chart_shift))),
            source_interface_x_m=float(r.source_exit[0]),connector_m=r.connector_arc,front_span_deg=float(span),front_geo=geo.tolist(),front_Y_m=cc[:6,1].tolist(),
            whole_rod_bound_mm=float(gap.min()*1000),raw_max_deg=float(np.rad2deg(abs(q).max())),raw_native_front_mm=shape['front_max_mm'],raw_q=q.tolist(),
            nominal_source_nodes=int(round(i['source_reference']*22)),nominal_target_nodes=int(round(i['target_reference']*22)))
    a['t']=float(t);a['max_source_tail_q_difference_deg']=float(np.rad2deg(max(abs(np.array(a['A13']['raw_q'])[14:]-np.array(a['M_decay_5_to_9']['raw_q'])[14:]))))
    rows.append(a)
out=dict(physical_steps=0,changed='Only M=.054*mu slope decay extends from S5..7 to S5..9, final mu6->7. F/G controls/common psi and actual initialization otherwise identical. This changes source-interface X and material window coherently; not solely a polar world correction.',status='diagnostic candidate only; no authorized conclusion that phase mismatch causes all pose drift',samples=rows,
    candidate_min_bound_mm=min(r['M_decay_5_to_9']['whole_rod_bound_mm'] for r in rows),candidate_max_raw_deg=max(r['M_decay_5_to_9']['raw_max_deg'] for r in rows),
    max_source_tail_q_difference_deg=max(r['max_source_tail_q_difference_deg'] for r in rows))
(BASE/'A14_DIAGNOSIS/M_overlap_local_check.json').write_text(json.dumps(out,indent=2))
print({k:v for k,v in out.items() if k!='samples'},flush=True)
for sec in [35,40,45,48,50,51,52,53.8,54,55,60]:
    r=min(rows,key=lambda x:abs(x['t']-sec));print({k:({kk:vv for kk,vv in v.items() if kk!='raw_q'} if isinstance(v,dict) else v) for k,v in r.items()},flush=True)
