"""Four preregistered given-contact unknown-target comparisons."""
from physics import *
from evaluation import snapshot
from contact_sensor import read_sensor,classified_observation,known_source
from geometry import Geometry
from estimated_curve import EstimatedCurve
from estimated_observer import EstimatedObserver
from cylinder_estimator import CylinderEstimator
from global_update import GlobalUpdater,probe_command
from visual_overlay import overlay
import argparse,json,pickle,time,shutil,traceback,cv2,scipy,os,xml.etree.ElementTree as ET

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(str(type(value)))

ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args()
registration=json.loads((BASE/'REGISTERED.json').read_text());config=next(c for c in registration['cases'] if c['case']==a.case)
assert mujoco.__version__=='3.2.7' and np.__version__=='1.23.5' and scipy.__version__=='1.13.1'
assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'])
case=BASE/a.case;case.mkdir(exist_ok=False)
files=['physics.py','native_mapping.py','source_mode.py','geometry.py','contact_sensor.py','cylinder_estimator.py','estimated_curve.py','estimated_observer.py','global_update.py','visual_overlay.py','evaluation.py','step2_runner.py','run_step2.ps1','REGISTERED.json','DESIGN.md','source_constants.json','PREPARATION.json','OBSERVATION_PREPARATION.json']
for name in files:
    assert (BASE/name).read_bytes()==(BASE/registration['freeze_directory']/name).read_bytes(),name
    shutil.copy2(BASE/name,case/name)
shutil.copy2(BASE/'GIVEN_INITIAL/integration_state.npz',case/'given_initial.npz');shutil.copy2(MODEL,case/'model.xml')
for mesh in ET.parse(MODEL).findall('./asset/mesh'):
    if mesh.get('file'):
        dest=case/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),dest)
m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);initial=np.load(case/'given_initial.npz')
mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
constants=json.loads((case/'source_constants.json').read_text());prior=Geometry.from_dict(registration['conditions'][config['condition']]);ref=EstimatedCurve(constants,prior,float(initial['receiving_delta_m']))
observer=EstimatedObserver(ref,registration['source_known']);estimator=CylinderEstimator(prior,registration['source_known'],registration['estimator']);updater=GlobalUpdater(registration['global_update'],config['write_global'])
S0=float(initial['S']);raw,info=ref.command(S0);sensor=read_sensor(m,d);roles=observer.update(classified_observation(sensor,prior,registration['source_known']),S0,.025,prior)
ref.receiving_qualified_mask=roles['estimated_receiver_qualified']
source_info=ref.observe_all(m,d,classified_observation(sensor,prior,registration['source_known']),roles,info,.025,True);info.update(**roles,**source_info)
plan=dict(configuration=config,initial_prior=prior.as_dict(),registration=registration,initial_reference_S=S0,
    input_permissions=registration['unknown_input_boundary'],original_model=str(MODEL),model_unchanged=True,initial_is_given=True,initial_is_historical_complete_resume=False,
    runtime=dict(python=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,threads={key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']}),
    distinction='Only GlobalUpdater write permission differs within pair. Same estimator/probe/feedback/reference implementation, prior, full initial, physics, frequency and budget.')
(case/'plan.json').write_text(json.dumps(plan,indent=2));print('PREPARED',a.case,'S0',S0,'prior',prior.as_dict(),flush=True)
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1.]
wr=cv2.VideoWriter(str(case/'actual_side_oblique_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520))
contact_log=(case/'contact_evidence.jsonl').open('w');fit_log=(case/'geometry_hypotheses.jsonl').open('w');update_log=(case/'global_updates.jsonl').open('w')
rows=[];last=d.ctrl.copy();raw=last.copy();next_command=0.;next_sample=0.;deep_age=0.;unsupported_age=0.;stop='duration';start=time.time();fit_count=0;update_count=0
while d.time<registration['duration_s']:
    t=float(d.time);S=S0+ref.feed_per_cycle*registration['nominal_frequency_hz']*max(0,t-registration['initial_settle_s'])
    try:
        if t>=next_command:
            sensor=read_sensor(m,d);probe=probe_command(t,estimator.first_contact,registration['probe']);estimator.observe(sensor,t,probe)
            ref.sensor_geometry=estimator.geometry.copy();ref.probe=probe
            observed=classified_observation(sensor,estimator.geometry,registration['source_known']);roles=observer.update(observed,S,.025,estimator.geometry,estimator.last.get('reliable',False))
            ref.receiving_qualified_mask=roles['estimated_receiver_qualified']
            source_info=ref.observe_all(m,d,observed,roles,info,.025,True)
            updater.update(ref,estimator,S,t)
            raw,info=ref.command(S) if t>=registration['initial_settle_s'] else (initial['ctrl'].copy(),ref.command(S0)[1])
            info.update(**roles,**source_info,shared_speed_ratio=1.,estimated_entry_m=estimator.geometry.entry.copy(),estimated_axis=estimator.geometry.axis.copy(),estimated_radius_m=estimator.geometry.radius,
                applied_global_entry_m=ref.geometry.entry.copy(),applied_global_axis=ref.geometry.axis.copy(),applied_global_radius_m=ref.geometry.radius,
                geometry_reliable=bool(estimator.last.get('reliable',False)),geometry_confidence=float(estimator.last.get('confidence',0.)),
                probe_world_directions=info['reference_frame'][:,1:].copy(),global_update_count=sum(x['written'] for x in updater.history))
            contact_log.write(json.dumps(dict(t=t,contacts=[dict(c,**classification) for c,classification in zip(sensor['contacts'],observed['contact_estimated_classes'])],probe=probe),default=serial)+'\n')
            for entry in estimator.fit_history[fit_count:]:fit_log.write(json.dumps(entry,default=serial)+'\n')
            fit_count=len(estimator.fit_history)
            for entry in updater.history[update_count:]:update_log.write(json.dumps(entry,default=serial)+'\n')
            update_count=len(updater.history);next_command+=.025
    except Exception as exc:
        stop='controller_failure';(case/'controller_error.json').write_text(json.dumps(dict(t=t,S=S,error=str(exc),traceback=traceback.format_exc()),indent=2));print('CONTROLLER_FAILURE',t,str(exc),flush=True);break
    ctrl=np.clip(raw,last-np.radians(20)*m.opt.timestep,last+np.radians(20)*m.opt.timestep);ctrl=np.clip(ctrl,-np.radians(88),np.radians(88));d.ctrl[:]=ctrl;last=ctrl.copy();d.xfrc_applied[:]=0.;mujoco.mj_step(m,d)
    if t>=next_sample:
        row=snapshot(m,d);row.update(t=t,raw_command=raw.copy(),tracking=float(np.max(abs(ctrl-d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]))),**info);rows.append(row);next_sample+=.1
        deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
        if int(round(t*10))%100==0:
            progress=dict(t=t,S=S,forces=row['forces'].sum(1).tolist(),depth_mm=(row['depth']*1000).tolist(),estimate_reliable=estimator.last.get('reliable',False),estimate=estimator.geometry.as_dict(),applied_global=ref.geometry.as_dict(),G_updates=sum(x['written'] for x in updater.history))
            print(json.dumps(progress),flush=True);(case/'progress.json').write_text(json.dumps(progress,indent=2));contact_log.flush();fit_log.flush();update_log.flush()
        images=[]
        for az,el,label in [(0,0,'SIDE'),(140,-26,'OBLIQUE')]:
            cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);overlay(ren.scene,m,info,estimator.geometry);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
            cv2.putText(im,f'{a.case} ACTUAL {label} t={t:.1f}s S={S:.3f}',(10,27),0,.55,(255,255,255),1)
            cv2.putText(im,'G cyan / C magenta / estimate axis yellow / ESTIMATED receiver green bridge purple',(10,49),0,.34,(255,255,255),1)
            cv2.putText(im,f'fit={info["geometry_confidence"]:.2f} writes={info["global_update_count"]} probe={np.linalg.norm(ref.probe):.3f}/m delta={ref.delta*1000:.2f}mm',(10,68),0,.40,(255,255,255),1);images.append(im)
        im=np.concatenate(images,axis=1);wr.write(im)
        if int(round(t*10))%100==0:cv2.imwrite(str(case/f'actual_{int(round(t)):03d}s.png'),im)
        if deep_age>=.3:stop='sustained_pipe_penetration';break
        if row['depth'][2]>.003:stop='self_penetration';break
        if row['depth'][3]>0:stop='floor_contact';break
        if unsupported_age>=.5 and t>1.5:stop='loss_all_pipe_contact';break
        if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
        if (case/'stop_and_save.request').exists():stop='requested_stop';break
wr.release();ren.close();contact_log.close();fit_log.close();update_log.close()
np.savez_compressed(case/'trace.npz',**{key:np.array([r[key] for r in rows]) for key in rows[0]})
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez(case/'integration_state.npz',state=state,last_command=last,raw_command=raw,S=S,next_command=next_command,next_sample=next_sample)
np.savez(case/'final_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
with (case/'controller_checkpoint.pkl').open('wb') as fp:pickle.dump(dict(reference=ref,observer=observer,estimator=estimator,updater=updater,last_command=last,raw_command=raw,last_valid_info=info,next_command=next_command,next_sample=next_sample),fp)
result=dict(stop=stop,sim_time=float(d.time),wall_s=time.time()-start,success=None,acceptance_status='pending saved-state evaluation',first_eligible_contact_t=estimator.first_contact,first_reliable_geometry_t=estimator.first_reliable,first_G_write_t=updater.first_update,
    G_write_count=sum(x['written'] for x in updater.history),hypothesis_count=len(estimator.fit_history),maximum_depth_mm=(np.max([r['depth'] for r in rows],axis=0)*1000).tolist())
(case/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
cap=cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
for fi in [0,frames//2,frames-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,fi);checks.append(bool(cap.read()[0]))
cap.release();(case/'video_metadata.json').write_text(json.dumps(dict(frames=frames,fps=fps,duration_s=frames/fps,readback=checks,source='one continuous actual dynamics run; currentG/currentC/estimated roles; no pose playback'),indent=2))

