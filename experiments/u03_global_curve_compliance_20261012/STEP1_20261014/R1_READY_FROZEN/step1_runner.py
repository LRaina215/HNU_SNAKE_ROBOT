"""Stage B paired runner: frozen A16 nominal route and one local curve coordinate."""
from physics import *
from curve_reference import GlobalCurveReference,LINK
from curve_projector import CurveProjector
from two_window_reference import TwoWindowReference
from continuous_window_reference import ContinuousWindowReference
from supply_window_reference import SupplyWindowReference
from handover_window_reference import HandoverWindowReference
from overlap_window_reference import OverlapWindowReference
from phase_overlap_reference import PhaseOverlapReference
from target_fit_reference import TargetFitReference
from receiver_phase_reference import ReceiverPhaseReference
from feedback_curve import FeedbackCurve
from material_observer import MaterialObserver,SharedSpeed,cached_truth
from visual_overlay import overlay
import argparse,json,shutil,time,pickle,traceback,xml.etree.ElementTree as ET
import cv2,os,scipy

ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--duration',type=float,default=235);ap.add_argument('--depth',type=float,default=.76);ap.add_argument('--feed_turns',type=float,default=1);ap.add_argument('--roll_sign',type=float,default=1);ap.add_argument('--radius',type=float,default=.0895);ap.add_argument('--pitch',type=float,default=.183);ap.add_argument('--prepare',action='store_true');ap.add_argument('--projection',action='store_true');ap.add_argument('--moving_family',action='store_true');ap.add_argument('--anchor_frame',action='store_true');ap.add_argument('--bezier_bridge',action='store_true');ap.add_argument('--local_material',action='store_true');ap.add_argument('--smooth_source',action='store_true');ap.add_argument('--discrete_transport',action='store_true');ap.add_argument('--axial_family',action='store_true');ap.add_argument('--release_curve',action='store_true');ap.add_argument('--two_windows',action='store_true');ap.add_argument('--clearance_connector',action='store_true');ap.add_argument('--continuous_windows',action='store_true');ap.add_argument('--supply_windows',action='store_true');ap.add_argument('--handover_windows',action='store_true');ap.add_argument('--overlap_windows',action='store_true');ap.add_argument('--phase_overlap_windows',action='store_true');ap.add_argument('--target_fit_windows',action='store_true');ap.add_argument('--receiver_phase_windows',action='store_true');ap.add_argument('--compliance',action='store_true');ap.add_argument('--speed-feedback',action='store_true');ap.add_argument('--condition',choices=['NOM','RECEIVE_LAG','SOURCE_DRAG'],default='NOM');a=ap.parse_args()
if a.receiver_phase_windows and not (a.target_fit_windows and a.phase_overlap_windows and a.overlap_windows):ap.error('--receiver_phase_windows requires target_fit/phase_overlap/overlap windows')
if a.target_fit_windows and not (a.phase_overlap_windows and a.overlap_windows):ap.error('--target_fit_windows requires --phase_overlap_windows and --overlap_windows')
if a.phase_overlap_windows and not a.overlap_windows:ap.error('--phase_overlap_windows requires --overlap_windows')
ReferenceClass=FeedbackCurve
shift_unit='cells' if (a.two_windows or a.continuous_windows or a.supply_windows or a.handover_windows or a.overlap_windows) else 'm'
case=BASE/a.case;case.mkdir(exist_ok=False)
for src in ['curve_reference.py','curve_projector.py','physics.py','geometry_backbone.py','two_window_reference.py','polar_window_reference.py','continuous_window_reference.py','supply_window_reference.py','handover_window_reference.py','overlap_window_reference.py','phase_overlap_reference.py','target_fit_reference.py','receiver_phase_reference.py','local_curve_compliance.py','step1_runner.py','feedback_curve.py','material_observer.py','visual_overlay.py','DESIGN.md','REGISTERED.json']:shutil.copy2(BASE/src,case/src)
shutil.copy2(MODEL,case/'model.xml');shutil.copy2(INITIAL,case/'initial.npz')
for mesh in ET.parse(MODEL).findall('./asset/mesh'):
    if mesh.get('file'):
        src=MODEL.parent/mesh.get('file');dest=case/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d);initial=snapshot(m,d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
ref=ReferenceClass(initial['points'],d.qpos[qa]) if a.overlap_windows else HandoverWindowReference(initial['points'],d.qpos[qa]) if a.handover_windows else SupplyWindowReference(initial['points'],d.qpos[qa]) if a.supply_windows else ContinuousWindowReference(initial['points'],d.qpos[qa]) if a.continuous_windows else TwoWindowReference(initial['points'],d.qpos[qa],clearance_connector=a.clearance_connector) if a.two_windows else GlobalCurveReference(initial['points'],d.qpos[qa],radius=a.radius,pitch=a.pitch,bridge_depth=a.depth,roll_sign=a.roll_sign,feed_turns=a.feed_turns,moving_family=a.moving_family,anchor_frame=a.anchor_frame,bezier_bridge=a.bezier_bridge,local_material=a.local_material,smooth_source=a.smooth_source,discrete_transport=a.discrete_transport,axial_family=a.axial_family,release_curve=a.release_curve);q0,i0=ref.command(0)
projector=CurveProjector(m,d.qpos.copy(),i0['reference_frame']) if a.projection else None
if projector:q0,extra=projector.command(i0['reference_points'],i0['reference_frame']);i0.update(extra)
observer=MaterialObserver(ref);speed=SharedSpeed();truth=cached_truth(m,d);roles=observer.update(truth,0.,.025);source_info=ref.observe_all(m,d,truth,roles,i0,.025,False);speed_info=speed.update(roles,.025,False);i0.update(**roles,**source_info,**speed_info)
plan=dict(runtime=dict(python=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,thread_settings={key:os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"]}),parameters=vars(a),global_route=dict(ref.spec,high_level_feedback=bool(a.compliance or a.speed_feedback)),initial_joint_generation_error_deg=np.rad2deg(q0-d.qpos[qa]).tolist(),original_model=str(MODEL),initial_state_source=str(INITIAL),model_unchanged=True,high_level_feedback=bool(a.compliance or a.speed_feedback),disturbance=json.loads((BASE/"REGISTERED.json").read_text())["conditions"][a.condition],nominal_frequency_hz=.02,settle_s=.5,slew_deg_s=20,command_limit_deg=88,root_changed_after_initial=False,acceptance=dict(duration_s=235,post_full_transfer_s=50,all_center_radius_m=[.07,.12],target_Y_m=[.15,1.2],axis_angle_deg=5,minimum_each_module_post_y_m=.04,front_span_deg=180,front_stable_s=2,full_stable_s=2,max_actual_joint_deg=90.5,max_torque_nm=2.355,no_floor=True,no_self_penetration_over_m=.003))
(case/'plan.json').write_text(json.dumps(plan,indent=2));np.savez(case/'global_route.npz',points=ref.global_points());print('PREPARED initial max geometric error mm',ref.initial_error_m*1000,'joint error deg',np.rad2deg(q0-d.qpos[qa]),flush=True)
if a.prepare:raise SystemExit
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1];wr=cv2.VideoWriter(str(case/'actual_side_oblique_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520))
rows=[];last=d.ctrl.copy();next_command=0.;next_sample=0.;deep_age=0.;unsupported_age=0.;info=i0;raw=last.copy();stop='duration';load_steps=0;load_impulse=np.zeros(3);lost_S=0.;start=time.time()
while d.time<a.duration:
    t=float(d.time)
    if t>=next_command:
        S=ref.feed_per_cycle*.02*max(0,t-.5)-lost_S
        try:
            truth=cached_truth(m,d);roles=observer.update(truth,S,.025);speed_info=speed.update(roles,.025,a.speed_feedback and t>=.5);source_info=ref.observe_all(m,d,truth,roles,info,.025,a.compliance and t>=.5)
            raw,info=ref.command(S) if t>=.5 else (z['ctrl'].copy(),i0.copy())
            info.update(**roles,**source_info,**speed_info,nominal_time_S=ref.feed_per_cycle*.02*max(0,t-.5),lost_nominal_S=lost_S)
            if projector and t>=.5:raw,extra=projector.command(info['reference_points'],info['reference_frame']);info.update(extra)
        except Exception as exc:
            stop='reference_or_projection_failure';print('REFERENCE_FAILURE t',t,'S',S,'reason',str(exc),flush=True);(case/'reference_error.json').write_text(json.dumps(dict(t=t,S=S,error=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc()),indent=2));break
        next_command+=.025
    ctrl=np.clip(raw,last-np.radians(20)*m.opt.timestep,last+np.radians(20)*m.opt.timestep);ctrl=np.clip(ctrl,-np.radians(88),np.radians(88));d.ctrl[:]=ctrl;last=ctrl.copy();d.xfrc_applied[:]=0.;load=plan["disturbance"];force=np.array(load["world_force_N"]) if load["interval_s"][0]<=t<load["interval_s"][1] else np.zeros(3);d.xfrc_applied[m.body(load["body"]).id,:3]=force;load_steps+=int(np.linalg.norm(force)>0.);load_impulse+=force*m.opt.timestep;mujoco.mj_step(m,d)
    if t>=.5:lost_S+=ref.feed_per_cycle*.02*(1-speed.ratio)*m.opt.timestep
    if t>=next_sample:
        row=snapshot(m,d);row.update(t=t,applied_head_force_N=d.xfrc_applied[m.body('head').id,:3].copy(),applied_load_force_N=force.copy(),raw_command=raw.copy(),tracking=float(np.max(abs(ctrl-d.qpos[qa]))),**info);rows.append(row);next_sample+=.1
        deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
        if int(round(t*10))%100==0:
            progress=dict(t=t,S=info['S'],chi=info['chi'],head=row['points'][0].tolist(),forces=row['forces'].sum(1).tolist(),depth_mm=(row['depth']*1000).tolist(),reference_head=info['reference_points'][0].tolist());print(json.dumps(progress),flush=True);(case/'progress.json').write_text(json.dumps(progress,indent=2))
        ims=[]
        for az,el,label in [(0,0,'SIDE'),(140,-26,'OBLIQUE')]:
            cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);overlay(ren.scene,m,info);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,f'{a.case} ACTUAL {label} t={t:.1f}s S={info["S"]:.3f}{shift_unit}',(10,27),0,.55,(255,255,255),1);cv2.putText(im,f'G cyan / C magenta / actual source orange target green',(10,49),0,.39,(255,255,255),1);cv2.putText(im,f'u={speed.ratio:.2f} delta={ref.delta*1000:.2f}mm sourceK={ref.source_gamma:.3f}/m M20role={int(roles["actual_material_role"][20])}',(10,68),0,.40,(255,255,255),1);ims.append(im)
        im=np.concatenate(ims,axis=1);wr.write(im)
        if int(round(t*10))%100==0:cv2.imwrite(str(case/f'actual_{int(round(t)):03d}s.png'),im)
        if deep_age>=.3:stop='sustained_pipe_penetration';break
        if row['depth'][2]>.003:stop='self_penetration';break
        if row['depth'][3]>0:stop='floor_contact';break
        if unsupported_age>=.5 and t>1.5:stop='loss_all_pipe_contact';break
        if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
        if (case/'stop_and_save.request').exists():stop='requested_stop';break
wr.release();ren.close();np.savez_compressed(case/'trace.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]});state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'integration_state.npz',state=state,ctrl=d.ctrl,last_command=last,S=info['S'],next_command=next_command,raw=raw);np.savez(case/'final_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
with (case/'controller_checkpoint.pkl').open('wb') as fp:pickle.dump(dict(reference=ref,last_command=last,raw_command=raw,last_valid_info=info,next_command=next_command,next_sample=next_sample,physical_time=d.time,lost_S=lost_S,observer=observer,speed=speed,projector=None if projector is None else dict(last=projector.last,initial_frame=projector.initial_frame,initial_rotation=projector.initial_rotation,reference_qpos=projector.d.qpos.copy())),fp)
result=dict(applied_load_steps=load_steps,applied_load_impulse_N_s=load_impulse.tolist(),stop=stop,sim_time=float(d.time),wall_s=time.time()-start,success=None,acceptance_status="pending_saved_trace_audit",maximum_depth_mm=(np.max([r['depth'] for r in rows],axis=0)*1000).tolist(),max_actual_joint_deg=float(np.rad2deg(np.max(abs(np.array([r['qpos'][qa] for r in rows]))))),max_torque_nm=float(np.max(abs(np.array([r['torque'] for r in rows])))))
(case/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)



cap=cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
for fi in [0,frames//2,frames-1]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,fi);checks.append(bool(cap.read()[0]))
cap.release();(case/'video_metadata.json').write_text(json.dumps(dict(fps=fps,frames=frames,duration=frames/fps,readback=checks,source='same actual dynamics run, current nominalG cyan/localC magenta/actual material roles;0.1s10fps1x; no replay',camera_only_adapts_to_actual_height=False),indent=2))
