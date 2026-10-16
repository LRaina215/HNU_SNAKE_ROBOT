"""Radial extremes/finite original contacts after original driver rule, no dynamics."""
from physics import *
from receiver_phase_reference import ReceiverPhaseReference
from target_fit_reference import TargetFitReference
import json

m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL)['qpos'];d.qpos[:]=initial;mujoco.mj_forward(m,d);p=nodes(d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];h=m.body('head').id;r=ReceiverPhaseReference(p,d.qpos[qa]);old=TargetFitReference(p,d.qpos[qa]);z=np.load(BASE/'A15/trace.npz');tt=z['t'];points=z['points'];oldctrl=z['ctrl'];k0=int(np.argmin(abs(tt-47.9)));last=oldctrl[k0].copy();rate=np.radians(20);q=last.copy();info=None;snap={};wanted=[50,50.7,52,53,53.6,53.9,54.9,56.4,60,70]
for step in range(int(round((70-47.9)/.001))+1):
    t=47.9+step*.001
    if step%25==0:q,info=r.command(r.feed_per_cycle*.02*max(0,t-.5))
    last=np.clip(np.clip(q,last-rate*.001,last+rate*.001),-np.radians(88),np.radians(88))
    for sec in wanted:
        if abs(t-sec)<.0001:snap[sec]=(q.copy(),last.copy(),info['nominal_curve_points'].copy())
    if step%5000==0:print('driver checked',round(t,3),flush=True)

def placed(q,P):
    d.qpos[:]=initial;d.qpos[qa]=q;mujoco.mj_forward(m,d);N=nodes(d);x=N[15:];y=P[15:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V
    Q=np.empty(9);mujoco.mju_quat2Mat(Q,d.qpos[3:7]);quat=np.empty(4);mujoco.mju_mat2Quat(quat,(R.T@Q.reshape(3,3)).ravel());d.qpos[:3]=(d.qpos[:3]-cx)@R+cy;d.qpos[3:7]=quat;mujoco.mj_forward(m,d);return nodes(d)
target=[m.geom('pipe_straight_04').id,m.geom('pipe_sharp_out_03').id];rows=[]
for t in wanted:
    k=int(np.argmin(abs(tt-t)));P=points[k];q,limited,C=snap[t];baseline,ii=old.command(old.feed_per_cycle*.02*max(0,t-.5))
    for label,command in [('A15_baseline_raw',baseline),('A15_baseline_actual_ctrl',oldctrl[k]),('A16_raw',q),('A16_original_driver_limited',limited)]:
        N=placed(command,P);c=(N[:-1]+N[1:])/2;rad=np.linalg.norm(c[:7,[0,2]]-[1.1,1],axis=1)*1000;mods=[]
        for v in range(7):
            gs=np.flatnonzero(m.geom_bodyid==h+v);hits=[dict(pipe=m.geom(ct.geom1).name if ct.geom1 in target else m.geom(ct.geom2).name,dist_mm=float(ct.dist*1000)) for ct in d.contact if (ct.geom1 in target and ct.geom2 in gs) or (ct.geom2 in target and ct.geom1 in gs)];minimum=min((x['dist_mm'] for x in hits),default=None)
            simple_straight=bool(.15<c[v,1]<1.2);capsule_Y_overlap=bool(max(N[v,1],N[v+1,1])+.029>=.15 and min(N[v,1],N[v+1,1])-.029<=1.2)
            mods.append(dict(module=v,center_m=c[v].tolist(),axis_r_mm=float(rad[v]),center_in_straight_Y=simple_straight,simplified_rod_Y_overlaps_straight=capsule_Y_overlap,original_contact_pairs=hits,original_min_contact_dist_mm=minimum))
        rows.append(dict(t=t,case=label,M0_to_M6=mods,min_radial_module=int(rad.argmin()),max_radial_module=int(rad.argmax()),driver_raw_to_limited_max_deg=float(np.rad2deg(abs(q-limited).max())) if label.startswith('A16') else None))
out=dict(physical_steps=0,method='Hypothetical native FK registered to fixed existing A15 posterior15..21 world pose. Original compiled mj_forward contact.dist and pipe identity used; no anomalous distance query. Centers/infinite-axis radii and simplified rod Y overlap do not replace finite original contacts. Driver1ms20deg/s and88deg clip, ideal25ms updates seeded at unchanged A15 t47.9 control; command time jitter/servo/dynamics not represented.',samples=rows)
(BASE/'A16_DIAGNOSIS/radial_limits_contacts.json').write_text(json.dumps(out,indent=2))
for row in rows:
    if row['case']=='A16_original_driver_limited' and row['t'] in [52,53.9,56.4]:print(row,flush=True)
