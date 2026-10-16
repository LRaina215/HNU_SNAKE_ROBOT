"""Original compiled geometry and native/nominal receiving gap, no dynamics."""
from physics import *
from phase_overlap_reference import PhaseOverlapReference
from target_fit_reference import TargetFitReference
import json

m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d);p=nodes(d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];qinit=d.qpos[qa].copy();h=m.body('head').id
refs=[('A14',PhaseOverlapReference(p,qinit)),('A15',TargetFitReference(p,qinit))];tg=[m.geom('pipe_straight_04').id,m.geom('pipe_sharp_out_03').id]
def segment_radius(P):
    x=P[:-1,[0,2]]-[1.1,1];v=P[1:,[0,2]]-P[:-1,[0,2]];u=np.clip(-(x*v).sum(1)/np.maximum((v*v).sum(1),1e-20),0,1)
    return np.linalg.norm(x+u[:,None]*v,axis=1)
rows=[]
for t in [0,35,40,45,50,52,53.9,56.4,60,70,100,160,235]:
    for name,r in refs:
        q,i=r.command(r.feed_per_cycle*.02*max(0,t-.5));C=i['nominal_curve_points'];d.qpos[:]=z['qpos'];d.qpos[qa]=q;mujoco.mj_forward(m,d);native=nodes(d)
        # Head/target registration for distance diagnostics only, not a controller.
        x=native[:7];y=C[:7];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V
        Q=np.empty(9);mujoco.mju_quat2Mat(Q,d.qpos[3:7]);Q=R.T@Q.reshape(3,3);quat=np.empty(4);mujoco.mju_mat2Quat(quat,Q.ravel());d.qpos[:3]=(d.qpos[:3]-cx)@R+cy;d.qpos[3:7]=quat;mujoco.mj_forward(m,d);placed=nodes(d)
        gap=[];pairs=[]
        for b in range(h,h+6):
            gs=np.flatnonzero(m.geom_bodyid==b);dd=[(float(mujoco.mj_geomDistance(m,d,int(g),int(pipe),.05,None)),int(g),int(pipe)) for g in gs for pipe in tg];v,g,pipe=min(dd);gap.append(v*1000);pairs.append([m.geom(g).name or str(g),m.geom(pipe).name])
        rows.append(dict(case=name,t=t,nominal_front6_axis_min_r_mm=(segment_radius(C[:7])*1000).tolist(),nominal_front6_simplified_gap_mm=((segment_radius(C[:7])-.0855)*1000).tolist(),target_registered_native_front6_axis_min_r_mm=(segment_radius(placed[:7])*1000).tolist(),target_registered_original_front6_geom_gap_mm=gap,closest_pairs=pairs,front_native_shape_rms_mm=float(np.sqrt(np.mean(np.sum((placed[:7]-C[:7])**2,axis=1)))*1000),raw_max_deg=float(np.rad2deg(abs(q).max())),common_psi=float(i['roll_phase']),source_M_m=float(i['source_chart_shift_m'])))
out=dict(physical_steps=0,method='Original compiled cylinder/mesh mj_geomDistance, native raw-command FK rigidly registered with first7 worldC nodes. Root pose registration diagnostic only; no physics/control feedback. Nominal line gaps are simplified, infinite axis omits finite cylinder/elbow. Native shape RMS and true finite-geometry distances distinguish mapping from simple envelope. Signed overlaps here are hypothetical placed geometry, not actual penetration.',samples=rows)
(BASE/'A15_CONTACT_PREPARATION.json').write_text(json.dumps(out,indent=2))
for x in rows:
    if x['t'] in [0,50,53.9,60,160]:print(x,flush=True)
