"""Necessary saved-state observation and curve-direction checks, no dynamics."""
from physics import *
from receiver_phase_reference import ReceiverPhaseReference
from feedback_curve import FeedbackCurve
from material_observer import MaterialObserver,cached_truth
from visual_overlay import overlay
import json,cv2

OLD=BASE.parent
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL)
d.qpos[:]=initial['qpos'];d.qvel[:]=initial['qvel'];d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(m,d)
qa=m.jnt_qposadr[m.actuator_trnid[:,0]];p=nodes(d)
ref=FeedbackCurve(p,d.qpos[qa]);baseline=ReceiverPhaseReference(p,d.qpos[qa]);zero=[]
for S in [0.,8.,11.,15.,18.,23.,35.]:
    q,info=ref.command(S);q0,_=baseline.command(S);zero.append(dict(S=S,max_difference_rad=float(np.max(abs(q-q0)))))
assert all(r['max_difference_rad']==0 for r in zero)
saved=np.load(OLD/'B1_NOM_V2/trace.npz');arrays={k:saved[k] for k in ['t','qpos','qvel','ctrl','points','centers','forces','S']}
obs=MaterialObserver(ref);samples=[];selected=[50.,70.,94.2,110.,113.7,120.,135.5,180.,234.9]
indices={int(np.argmin(abs(arrays['t']-t))) for t in selected}
history=[]
for i,t in enumerate(arrays['t']):
    truth=dict(centers=arrays['centers'][i],forces=arrays['forces'][i],body_depth=np.zeros((2,21)),joint_actual=arrays['qpos'][i,qa],joint_ctrl=arrays['ctrl'][i],joint_torque=np.zeros(20))
    info=obs.update(truth,float(arrays['S'][i]),.1)
    history.append(info['material_progress_observed_m'].copy())
    if i in indices:
        samples.append(dict(t=float(t),M20_actual_role=int(info['actual_material_role'][20]),
            M20_progress_m=float(info['material_progress_observed_m'][20]),M20_source_N=float(truth['forces'][0,20]),
            front_confirmed=info['front_confirmed'],target_lead_m=info['target_progress_lead_m'],
            source_rate_ratio=info['source_progress_rate_ratio'],target_rate_ratio=info['target_progress_rate_ratio'],
            maximum_unobserved_target_prediction_m=float(np.max(info['material_target_expected_unobserved_m'])),
            mean_unobserved_target_prediction_m=float(np.mean(info['material_target_expected_unobserved_m'])),
            maximum_all_chart_unobserved_time_s=float(np.max(info['material_unobserved_time_s'])),
            maximum_all_chart_unobserved_nominal_feed_m=float(np.max(info['material_unobserved_nominal_feed_m'])),
            target_anchor_S_M20=float(info['target_anchor_S'][20])))
np.savez(BASE/'saved_nominal_observer_history.npz',t=arrays['t'],material_progress_m=np.asarray(history))
direction=[]
for sec in [110.,113.7]:
    i=int(np.argmin(abs(arrays['t']-sec)));d.qpos[:]=arrays['qpos'][i];d.qvel[:]=arrays['qvel'][i];d.ctrl[:]=arrays['ctrl'][i];mujoco.mj_forward(m,d)
    truth=cached_truth(m,d);truth['forces']=arrays['forces'][i]
    S=float(arrays['S'][i]);ref.delta=0.;ref.source_gamma=0.;q0,info=ref.command(S)
    roles=dict(source_loaded_material=20,source_late_filtered_N=30.,front_confirmed=True,source_late_depth_m=truth['body_depth'][0,20])
    ref.observe_all(m,d,truth,roles,info,.5,True);ref.delta=0.;unit=ref.source_direction.copy()
    result=[]
    for gamma in [0.,.1,-.1]:
        # Negative gamma is a private direction diagnostic, never a live command.
        field=info['curvature_field'].copy();bounds=info['curvature_bounds_mm'];mat=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:]);v=np.clip((mid-mat[15])/(mat[-1]-mat[15]),0,1);w=np.sin(np.pi*v)**2;w[(bounds[:-1]<mat[15])|(bounds[1:]>mat[-1])]=0
        field+=gamma*w[:,None]*unit/1000;b=ref.joint_bending(bounds,field,mat);q=-np.where(np.arange(20)%2,b[:,0],b[:,1]);delta_q=q-q0
        private=mujoco.MjData(m);private.qpos[:]=arrays['qpos'][i];private.qpos[qa]+=delta_q;mujoco.mj_forward(m,private)
        deepest=0.
        for ct in private.contact:
            gs=[ct.geom1,ct.geom2];names=[m.geom(g).name for g in gs];bs=[int(m.geom_bodyid[g]) for g in gs]
            if any(n in ['pipe_straight_01','pipe_sharp_in_02'] for n in names) and m.body('tail').id in bs:deepest=max(deepest,-ct.dist)
        result.append(dict(gamma_per_m=gamma,max_joint_delta_deg=float(np.rad2deg(np.max(abs(delta_q)))),first14_delta_rad=float(np.max(abs(delta_q[:14]))),source_private_geometry_depth_mm=1000*deepest))
        assert np.max(abs(delta_q[:14]))==0
    direction.append(dict(t=sec,direction=unit.tolist(),samples=result,scope='private fixed-head FK, no dynamics; contact response not predicted'))
    ref.source_gamma=0.
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1];cam.azimuth=140;cam.elevation=-26
roles=obs.update(dict(centers=arrays['centers'][i],forces=arrays['forces'][i],body_depth=truth['body_depth'],joint_actual=arrays['qpos'][i,qa],joint_ctrl=arrays['ctrl'][i],joint_torque=np.zeros(20)),S,.1)
q,info=ref.command(S);info.update(roles);ren.update_scene(d,cam);overlay(ren.scene,m,info);cv2.imwrite(str(BASE/'overlay_saved_state.png'),cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR));ren.close()
(BASE/'PREPARATION.json').write_text(json.dumps(dict(zero_feedback_command_equality=zero,material_key_states=samples,source_direction_checks=direction,scope='saved traces and private FK only; no new dynamics'),indent=2))
print(json.dumps(dict(material_key_states=samples,source_direction_checks=direction),indent=2))
