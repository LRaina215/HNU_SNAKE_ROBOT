"""Step2 from successful Step1 physical t0, original coordinated progression."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import json,time,shutil,argparse,xml.etree.ElementTree as ET
import cv2,scipy
from physics import *
from rolling_controller import RollingController
from evaluation import snapshot
from visual_overlay import overlay
from action_stop_evaluator import ActionStopEvaluator

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def require_freeze(r):
    F=BASE/'LOCAL_FREEZE';check=json.loads((BASE/'SELF_REVIEW.json').read_text())
    if not r['dynamics_authorized'] or not check['passed'] or check['version']!=r['version']:raise RuntimeError('Local necessary self-check/freeze absent')
    for file in list(BASE.glob('*.py'))+[BASE/name for name in ['REGISTERED.json','initial.npz','STEP1_SOURCE_CONSTANTS.json']]:
        if file.read_bytes()!=(F/file.name).read_bytes():raise RuntimeError('Frozen file changed: '+file.name)
    if MODEL.read_bytes()!=(F/'model.xml').read_bytes():raise RuntimeError('Original physical model changed')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file') and (MODEL.parent/mesh.get('file')).read_bytes()!=(F/mesh.get('file')).read_bytes():raise RuntimeError('Original physical mesh changed')

def run(case_name):
    r=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));require_freeze(r);assert mujoco.__version__=='3.2.7' and np.__version__=='1.23.5' and scipy.__version__=='1.13.1'
    case=next(c for c in r['cases'] if c['case']==case_name);out=BASE/case_name;out.mkdir(exist_ok=False)
    for file in BASE.glob('*.py'):shutil.copy2(file,out/file.name)
    for name in ['REGISTERED.json','initial.npz','STEP1_SOURCE_CONSTANTS.json','SELF_REVIEW.json','CASE_REGISTRATION.json','IMPLEMENTATION.md']:shutil.copy2(BASE/name,out/name)
    shutil.copy2(MODEL,out/'model.xml')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file'):
            dest=out/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),dest)
    m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));d=mujoco.MjData(m);z=np.load(out/'initial.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d)
    controller=RollingController(m,d,r,case['condition'],case['write_global']);G0=controller.initial_G.copy();np.savez_compressed(out/'initial_G_family.npz',points=G0)
    last=d.ctrl.copy();raw=last.copy();next_command=0.;next_sample=0.;rows=[];fit_n=0;write_n=0;stop='duration';deep_age=0.;unsupported_age=0.;started=False;error=None;wall=time.time();goal=ActionStopEvaluator()
    evidence=(out/'contact_evidence.jsonl').open('w');fits=(out/'geometry_hypotheses.jsonl').open('w');writes=(out/'global_updates.jsonl').open('w')
    renderer=mujoco.Renderer(m,width=700,height=520);camera=mujoco.MjvCamera();camera.distance=1.7;camera.lookat[:]=[.95,.34,1.]
    video=cv2.VideoWriter(str(out/'actual_dual_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520));assert video.isOpened()
    (out/'plan.json').write_text(json.dumps(dict(version=r['version'],case=case,physical_t0_included=True,initial_source=r['initial_source'],no_root_reset_or_trajectory_playback=True,original_S_F_M_roll_release=True,normal_precontact_advance_allowed=True,model_unchanged=True),indent=2))
    try:
        while True:
            t=float(d.time)
            if t>=next_command-1e-10:
                raw,info=controller.command(d,t);sensor=__import__('contact_sensor').read_sensor(m,d)
                evidence.write(json.dumps(dict(t=t,contacts=sensor['contacts'],probe_requested=controller.ref.probe,probe_transmitted=controller.applied_probe,probe_evidence_input=info['probe_evidence_input_per_m']),default=serial)+'\n')
                for entry in controller.estimator.fit_history[fit_n:]:fits.write(json.dumps(entry,default=serial)+'\n')
                for entry in controller.updater.history[write_n:]:writes.write(json.dumps(entry,default=serial)+'\n')
                fit_n=len(controller.estimator.fit_history);write_n=len(controller.updater.history);next_command+=r['command_interval_s']
                if controller.first_capture_t is not None and abs(t-controller.first_capture_t)<1e-8:
                    capture=snapshot(m,d);capture.update(t=t,**info);(out/'CAPTURE_SNAPSHOT.json').write_text(json.dumps(capture,default=serial))
            if t>=next_sample-1e-10:
                row=snapshot(m,d);row.update(t=t,initial_global_family_dense=G0,raw_command=raw.copy(),body_xmat=d.xmat[m.body('head').id:m.body('head').id+21].copy(),**info);rows.append(row);next_sample+=.1
                images=[]
                for az,el,label in [(0,0,'SIDE'),(140,-26,'OBLIQUE')]:
                    camera.azimuth=az;camera.elevation=el;renderer.update_scene(d,camera);overlay(renderer.scene,m,dict(info,initial_global_family_dense=G0),controller.estimator.geometry);im=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                    cv2.putText(im,f'{case_name} ACTUAL {label} t={t:.1f}s S={controller.S:.3f}',(10,24),0,.48,(255,255,255),1)
                    cv2.putText(im,'G0 blue / Gt cyan / C pink / actual body contact roles',(10,45),0,.35,(255,255,255),1)
                    cv2.putText(im,f'G writes={info["global_update_count"]} C delta={controller.ref.delta*1000:.2f}mm sourceK={controller.ref.source_gamma:.3f}/m',(10,65),0,.37,(255,255,255),1);images.append(im)
                frame=np.concatenate(images,axis=1);video.write(frame)
                if int(round(t*10))%100==0:
                    cv2.imwrite(str(out/f'actual_{int(round(t)):03d}s.png'),frame);progress=dict(t=t,phase=controller.phase,S=controller.S,head=row['points'][0].tolist(),source_N=info['known_source_vertical_N'],receiver_N=info['receiver_vertical_N'],G_writes=info['global_update_count'],reliable=info['geometry_reliable'],C_delta_mm=controller.ref.delta*1000,source_gamma=controller.ref.source_gamma)
                    (out/'LIVE_STATE.json').write_text(json.dumps(dict(progress,points=row['points'],G=info['global_reference_points'],C=info['nominal_curve_points'],qpos=row['qpos'],raw=raw,torque=row['torque']),default=serial));evidence.flush();fits.flush();writes.flush();print(json.dumps(progress),flush=True)
                deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
                if deep_age>=.3:stop='sustained_pipe_penetration';break
                if row['depth'][2]>.003:stop='self_penetration';break
                if row['depth'][3]>0.:stop='floor_contact';break
                if unsupported_age>=.5 and t>1.5:stop='loss_all_pipe_contact';break
                if abs(d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]).max()>np.radians(90.5):stop='joint_limit';break
                if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
                if goal.update(row):stop='registered_complete_action_window';break
            if (out/'stop_and_save.request').exists():stop='requested_stop';break
            if t>=r['duration_s']:break
            rate=np.radians(20)*m.opt.timestep;last=np.clip(np.clip(raw,last-rate,last+rate),-np.radians(88),np.radians(88));d.ctrl[:]=last;d.xfrc_applied[:]=0.
            if not started:(out/'DYNAMICS_START.json').write_text(json.dumps(dict(first_integrated_time_s=t,physical_t0_logged=True,initial_source=r['initial_source'])));started=True
            mujoco.mj_step(m,d)
    except Exception as exc:stop='controller_or_runner_failure';error=repr(exc);(out/'error.txt').write_text(error)
    finally:evidence.close();fits.close();writes.close();video.release();renderer.close()
    if rows:np.savez_compressed(out/'trace.npz',**{k:np.array([row[k] for row in rows]) for k in rows[0]})
    state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'integration_state.npz',state=state,last_command=last,S=controller.S)
    result=dict(stop=stop,sim_time=float(d.time),wall_s=time.time()-wall,error=error,success=None,action_chain_complete=False,role='Step2 original-Step1-t0 development; final four still separate',first_release_t=controller.first_release_t,first_raw_front_contact_t=controller.estimator.first_front_contact,first_reliable_t=controller.estimator.first_reliable,first_G_write_t=controller.updater.first_update,first_capture_t=controller.first_capture_t)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='T_UPDATE');run(parser.parse_args().case)
