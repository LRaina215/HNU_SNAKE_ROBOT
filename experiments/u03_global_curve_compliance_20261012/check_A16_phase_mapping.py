"""One spatial phase proposal mapped through real curvature/native FK."""
from physics import *
from target_fit_reference import TargetFitReference
from receiver_phase_reference import ReceiverPhaseReference
from snake_control.pipe_clearance import PipeClearance
import json

m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL)['qpos'];d.qpos[:]=initial;mujoco.mj_forward(m,d);p0=nodes(d);q0=d.qpos[7:].copy();refs=[('A15',TargetFitReference(p0,q0)),('A16_proposal',ReceiverPhaseReference(p0,q0))];clear=PipeClearance(m,d);z=np.load(BASE/'A15/trace.npz');actual=z['points'];tt=z['t'];rows=[]
def place(P,Q):
    x=P[15:];y=Q[15:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V;return (P-cx)@R+cy
n=np.arange(4);basis=-.18388917*n+(n%2)*np.pi/2;X=np.column_stack([np.sin(basis),np.cos(basis)])
for sec in [47,50,50.7,52,53,53.6,53.9,54.9,56.4,60,70]:
    k=int(np.argmin(abs(tt-sec)));P=actual[k]
    for name,r in refs:
        S=r.feed_per_cycle*.02*max(0,sec-.5);q,i=r.command(S);C=i['nominal_curve_points'];d.qpos[:]=initial;d.qpos[7:]=q;mujoco.mj_forward(m,d);pred=place(nodes(d),P);pc=(pred[:-1]+pred[1:])/2;cc=(C[:-1]+C[1:])/2
        gap=clear.link_minimum(C,samples=65)-.029-max(np.linalg.norm(np.diff(C,axis=0),axis=1))/128;coef=np.linalg.lstsq(X,q[:4],rcond=None)[0];phase=np.arctan2(coef[1],coef[0]);rss=np.sqrt(np.mean((X@coef-q[:4])**2))
        rows.append(dict(case=name,t=sec,entry_phase_deg=float(np.rad2deg(r.target_phi)),connector_m=r.connector_arc,target_nodes=int(round(i['target_reference']*22)),C_front_Y_m=cc[:7,1].tolist(),C_head_polar_deg=float(np.rad2deg(np.arctan2(cc[0,2]-1,cc[0,0]-1.1))),predicted_native_M0_to_M6_centers_m=pc[:7].tolist(),predicted_native_M0_to_M6_polar_deg=np.rad2deg(np.arctan2(pc[:7,2]-1,pc[:7,0]-1.1)).tolist(),raw_first4_intrinsic_phase_deg=float(np.rad2deg(phase)),raw_first4_wave_rms_deg=float(np.rad2deg(rss)),raw_max_deg=float(np.rad2deg(abs(q).max())),C_min_whole_rod_bound_mm=float(gap.min()*1000),common_psi=float(i['roll_phase']),F_m=float(i['supply_distance_m']),M_m=float(i['source_chart_shift_m'])))
out=dict(physical_steps=0,method='Single receiving-phase proposal, no grid. Native raw FK placed in existing A15 actual posterior15..21 pose only for predictive diagnostic, never a runtime input. Predictions assume source pose unchanged and are not dynamics evidence. Fixed first4 intrinsic waveform basis/RMS, not physical target polar or arbitrary root rotation.',samples=rows)
(BASE/'A16_DIAGNOSIS/phase_mapping_proposal.json').write_text(json.dumps(out,indent=2))
for x in rows:
    print(dict(case=x['case'],t=x['t'],entry=x['entry_phase_deg'],conn=x['connector_m'],target_nodes=x['target_nodes'],pred_head=x['predicted_native_M0_to_M6_centers_m'][0],pred_M1=x['predicted_native_M0_to_M6_centers_m'][1],native_phase=x['raw_first4_intrinsic_phase_deg'],rms=x['raw_first4_wave_rms_deg'],raw=x['raw_max_deg'],gap=x['C_min_whole_rod_bound_mm']),flush=True)
