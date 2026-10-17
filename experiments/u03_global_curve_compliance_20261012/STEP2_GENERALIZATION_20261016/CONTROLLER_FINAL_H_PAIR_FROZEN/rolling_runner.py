"""Actual new-scene Step2 dynamics, copied per case before integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,datetime,shutil,xml.etree.ElementTree as ET
import cv2,scipy
from physics import *
from rolling_controller import RollingController
from evaluation import snapshot
from action_stop_evaluator import ActionStopEvaluator
from visual_overlay import overlay
from geometry import Geometry
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(file,v):file.write_text(json.dumps(v,default=serial,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def check_freeze(r):
    F=BASE/'LOCAL_FREEZE'
    for file in list(BASE.glob('*.py'))+[BASE/'REGISTERED.json',BASE/'STEP1_SOURCE_CONSTANTS.json']:
        assert file.read_bytes()==(F/file.name).read_bytes(),file
    for name,scene in r['scenes'].items():
        folder=BASE/Path(scene['model_file']).parent
        for file in folder.rglob('*'):
            if file.is_file():assert file.read_bytes()==(F/file.relative_to(BASE)).read_bytes(),file
def run(name):
    r=json.loads((BASE/'REGISTERED.json').read_text());check_freeze(r);case=next(x for x in r['cases'] if x['case']==name);scene=r['scenes'][case['scene']];assert mujoco.__version__=='3.2.7' and np.__version__=='1.23.5' and scipy.__version__=='1.13.1'
    out=BASE/name;out.mkdir(exist_ok=False)
    for file in BASE.glob('*.py'):shutil.copy2(file,out/file.name)
    shutil.copy2(BASE/'REGISTERED.json',out/'REGISTERED.json');shutil.copy2(BASE/'STEP1_SOURCE_CONSTANTS.json',out/'STEP1_SOURCE_CONSTANTS.json')
    shutil.copy2(BASE/scene['model_file'],out/'model.xml');shutil.copy2(BASE/scene['initial_file'],out/'initial.npz');shutil.copy2(BASE/scene['evaluation_file'],out/'evaluation_truth.json')
    for mesh in ET.parse(out/'model.xml').findall('./asset/mesh'):
        if mesh.get('file'):
            dst=out/mesh.get('file');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2((BASE/scene['model_file']).parent/mesh.get('file'),dst)
    save(out/'scene_config.json',scene);set_control_frame(scene['source_frame_rotation'],scene['source_frame_origin_m']);settings=dict(r['controller'],source_known=scene['source_known'],conditions={'PRIOR':scene['prior']})
    m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));d=mujoco.MjData(m);z=np.load(out/'initial.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d);controller=RollingController(m,d,settings,'PRIOR',case['write_global']);G0=to_world_points(controller.initial_G);truth=json.loads((out/'evaluation_truth.json').read_text());goal=ActionStopEvaluator(truth,r['acceptance'])
    last=d.ctrl.copy();raw=last.copy();next_command=0.;next_sample=0.;rows=[];fit_n=0;write_n=0;stop='duration';deep_age=0.;unsupported_age=0.;started=False;error=None;wall=time.time()
    evidence=(out/'contact_evidence.jsonl').open('w');fits=(out/'geometry_hypotheses.jsonl').open('w');writes=(out/'global_updates.jsonl').open('w')
    renderer=mujoco.Renderer(m,width=700,height=520);camera=mujoco.MjvCamera();camera.distance=scene['camera']['distance'];camera.lookat[:]=scene['camera']['lookat'];video=cv2.VideoWriter(str(out/'actual_dual_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520));assert video.isOpened()
    save(out/'plan.json',dict(version=r['version'],case=case,physical_t0=True,normal_precontact_rolling=True,gravity_world=m.opt.gravity,control_information='Prior/source/robot/unlabeled contacts only; evaluation truth never passed to controller',spin_method=r['acceptance']['spin_method']))
    try:
        while True:
            t=float(d.time)
            if t>=next_command-1e-10:
                raw,local=controller.command(d,t);info=world_info(local);sensor=__import__('contact_sensor').read_sensor(m,d);evidence.write(json.dumps(dict(t=t,contacts=sensor['contacts'],probe_requested=controller.ref.probe,probe_transmitted=controller.applied_probe,probe_evidence_input=info['probe_evidence_input_per_m']),default=serial)+'\n')
                for row in controller.estimator.fit_history[fit_n:]:fits.write(json.dumps(row,default=serial)+'\n')
                for row in controller.updater.history[write_n:]:writes.write(json.dumps(row,default=serial)+'\n')
                fit_n=len(controller.estimator.fit_history);write_n=len(controller.updater.history);next_command+=settings['command_interval_s']
                if controller.first_capture_t is not None and abs(t-controller.first_capture_t)<1e-8:
                    capture=snapshot(m,d);capture.update(t=t,**info);save(out/'CAPTURE_SNAPSHOT.json',capture)
            if t>=next_sample-1e-10:
                row=snapshot(m,d);row.update(t=t,initial_global_family_dense=G0,raw_command=raw.copy(),body_xmat=d.xmat[m.body('head').id:m.body('head').id+21].copy(),**info);complete=goal.update(row);rows.append(row);next_sample+=.1;images=[]
                estimate=Geometry(to_world_points(controller.estimator.geometry.entry),to_world_vectors(controller.estimator.geometry.axis),controller.estimator.geometry.radius)
                for (az,el),label in zip(scene['camera']['views'],['SIDE','OBLIQUE']):
                    camera.azimuth=az;camera.elevation=el;renderer.update_scene(d,camera);overlay(renderer.scene,m,dict(info,initial_global_family_dense=G0),estimate);im=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,f'{name} ACTUAL {label} t={t:.1f}s S={controller.S:.3f}',(10,24),0,.48,(255,255,255),1);cv2.putText(im,'G0 blue / Gt cyan / C pink / actual estimated roles',(10,45),0,.35,(255,255,255),1);cv2.putText(im,f'G writes={info["global_update_count"]} C delta={controller.ref.delta*1000:.2f}mm sourceK={controller.ref.source_gamma:.3f}/m',(10,65),0,.37,(255,255,255),1);images.append(im)
                frame=np.concatenate(images,axis=1);video.write(frame)
                if int(round(t*10))%100==0:
                    cv2.imwrite(str(out/f'actual_{int(round(t)):03d}s.png'),frame);progress=dict(t=t,phase=controller.phase,S=controller.S,head=row['points'][0],source_Fz_N=info['known_source_vertical_N'],receiver_Fz_N=info['receiver_vertical_N'],G_writes=info['global_update_count'],reliable=info['geometry_reliable'],C_delta_mm=controller.ref.delta*1000,source_gamma=controller.ref.source_gamma,full21=bool(row['evaluation_full21']));save(out/'LIVE_STATE.json',dict(progress,points=row['points']));evidence.flush();fits.flush();writes.flush();print(json.dumps(progress,default=serial),flush=True)
                deep_age=deep_age+.1 if max(row['depth'][:2])>r['acceptance']['pipe_depth_m'] else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
                if deep_age>=r['acceptance']['pipe_depth_persistence_s']:stop='sustained_pipe_penetration';break
                if row['depth'][2]>r['acceptance']['self_depth_m']:stop='self_penetration';break
                if row['depth'][3]>0:stop='floor_contact';break
                if unsupported_age>=.5 and t>1.5:stop='loss_all_pipe_contact';break
                if abs(d.qpos[7:]).max()>np.radians(90.5):stop='joint_limit';break
                if not np.isfinite(d.qpos).all():stop='nonfinite';break
                if complete:stop='registered_complete_action_window';break
            if (out/'stop_and_save.request').exists():stop='requested_stop';break
            if t>=r['duration_s']:break
            rate=np.radians(settings['servo_slew_deg_s'])*m.opt.timestep;last=np.clip(np.clip(raw,last-rate,last+rate),-np.radians(settings['servo_limit_deg']),np.radians(settings['servo_limit_deg']));d.ctrl[:]=last;d.xfrc_applied[:]=0.
            if not started:save(out/'DYNAMICS_START.json',dict(at=datetime.datetime.now().astimezone().isoformat(),first_integrated_time_s=t,physical_t0_logged=True,case=case,version=r['version']));started=True
            mujoco.mj_step(m,d)
    except Exception as exc:stop='controller_or_runner_failure';error=repr(exc);(out/'error.txt').write_text(error)
    finally:evidence.close();fits.close();writes.close();video.release();renderer.close()
    if rows:np.savez_compressed(out/'trace.npz',**{k:np.asarray([row[k] for row in rows]) for k in rows[0]})
    result=dict(stop=stop,sim_time=float(d.time),wall_s=time.time()-wall,error=error,case=case,version=r['version'],action_window=goal.complete,first_release_t=controller.first_release_t,first_raw_front_contact_t=controller.estimator.first_front_contact,first_reliable_t=controller.estimator.first_reliable,first_G_write_t=controller.updater.first_update,first_capture_t=controller.first_capture_t,success_pending_audit=True);save(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);run(ap.parse_args().case)
