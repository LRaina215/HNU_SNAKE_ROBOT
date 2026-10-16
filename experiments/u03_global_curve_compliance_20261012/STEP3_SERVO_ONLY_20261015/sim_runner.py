"""Plant, virtual servo-register adapter, and independent truth evaluator.

The controller gets only ServoObservation. Evaluation can abort a physically
unsafe run but never select a search direction, step phase, G, or local shape.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,time,pickle,shutil,traceback
import numpy as np
import mujoco
import cv2
from servo_observer import ServoObservation
from servo_controller import ServoSearchController
from evaluation import snapshot

BASE=Path(__file__).resolve().parent


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class VirtualServoRegisters:
    def __init__(self,model):
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].copy()
        self.limits=np.max(abs(model.actuator_forcerange),axis=1).copy()
        if np.any(self.limits<=0):raise ValueError('load register requires declared actuator force limits')
    def sample(self,data):
        # Explicit idealized virtual sensor model. No root, contacts or body poses exported.
        q=np.round(data.qpos[self.qadr]/np.deg2rad(.01))*np.deg2rad(.01)
        load=np.round((data.actuator_force/self.limits)/.005)*.005
        return ServoObservation(float(data.time),q,data.ctrl.copy(),load)


def reference_panel(info):
    image=np.full((520,320,3),25,np.uint8)
    cv2.putText(image,'G / C VIRTUAL CHART',(8,22),0,.5,(255,255,255),1)
    cv2.putText(image,'NO WORLD REGISTRATION',(8,43),0,.42,(190,190,190),1)
    for panel,(axes,label) in enumerate([((0,1),'X-Y'),((0,2),'X-Z')]):
        g=info.get('global_reference_dense');c=info.get('nominal_curve_dense')
        if g is None or c is None:continue
        allp=np.vstack([g,c])[:,axes];lo=allp.min(axis=0)-.025;span=np.maximum(allp.max(axis=0)-lo+.025,.2)
        scale=min(295/span[0],170/span[1]);origin=np.array([10,90+panel*210+170])
        for points,color in [(g,(255,205,40)),(c,(190,65,235))]:
            xy=origin+(points[:,axes]-lo)*[scale,-scale]
            cv2.polylines(image,[np.asarray(xy,np.int32)],False,color,1)
        cv2.putText(image,label,(8,80+panel*210),0,.4,(240,240,240),1)
    return image


def run(name):
    if (BASE/'STOP_REQUESTED.json').exists():raise RuntimeError('Stop boundary is active; no new run')
    reg=json.loads((BASE/'REGISTERED_DEVELOPMENT.json').read_text(encoding='utf-8-sig'))
    case=next(c for c in reg['development_runs'] if c['id']==name)
    out=BASE/'runs'/name;out.mkdir(parents=True,exist_ok=False)
    for file in BASE.glob('*.py'):shutil.copy2(file,out/file.name)
    for file in ['REGISTERED_DEVELOPMENT.json','source_constants.json']:shutil.copy2(BASE/file,out/file)
    shutil.copytree(BASE/'assets',out/'assets')
    m=mujoco.MjModel.from_xml_path(str(out/'assets/model.xml'));d=mujoco.MjData(m)
    initial=np.load(out/'assets/inherited_initial.npz',allow_pickle=False)
    mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    d.qpos[0]+=case['source_translation_x_m'];d.time=0.;mujoco.mj_forward(m,d)
    initial_state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m,d,initial_state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(out/'INITIAL_STATE.npz',state=initial_state,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
    registers=VirtualServoRegisters(m)
    constants=json.loads((BASE/'source_constants.json').read_text(encoding='utf-8-sig'))
    # Controller settings contain neither case identity/translation nor eval geometry.
    control_settings={k:reg[k] for k in ['control','nominal_frequency_hz']}
    controller=ServoSearchController(constants,d.ctrl.copy(),control_settings,case['active_search'],case['hold_only'])
    last=d.ctrl.copy();raw=last.copy();info={};rows=[];events_count=0;next_control=0.;next_sample=0.
    wall=time.monotonic();next_stop_check=wall;next_progress=0.;stop='duration';deep_age=0.;no_support_age=0.
    video=None;renderer=None;sensors=None;events=None
    log_entry=dict(id=name,version=reg['version'],kind='development',started_at=time.time(),duration_s=case['duration_s'],source_translation_x_m=case['source_translation_x_m'])
    with (BASE/'RUN_LEDGER.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log_entry)+'\n')
    try:
        renderer=mujoco.Renderer(m,width=700,height=520)
        camera=mujoco.MjvCamera();camera.distance=1.85;camera.lookat[:]=[.8,.22,1.]
        video=cv2.VideoWriter(str(out/'actual_dual_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1720,520))
        if not video.isOpened():raise RuntimeError('video output unavailable')
        sensors=(out/'servo_telemetry.jsonl').open('w',encoding='utf-8');events=(out/'controller_events.jsonl').open('w',encoding='utf-8')
        while True:
            t=float(d.time)
            if time.monotonic()>=next_stop_check:
                next_stop_check=time.monotonic()+1.
                if (BASE/'STOP_REQUESTED.json').exists():stop='quota_or_user_stop';break
            if t>=next_control-1e-8:
                packet=registers.sample(d)
                sensors.write(json.dumps(dict(t=packet.t,q=packet.q,sent=packet.sent,load_fraction=packet.load_fraction),default=serial)+'\n')
                raw,info=controller.command(packet)
                for event in controller.events[events_count:]:events.write(json.dumps(event,default=serial)+'\n')
                events_count=len(controller.events);next_control+=reg['command_interval_s']
            if t>=next_sample-1e-8:
                row=snapshot(m,d);row.update(t=t,raw_command=raw.copy(),body_xmat=d.xmat[m.body('head').id:m.body('head').id+21].copy(),**info)
                rows.append(row);next_sample+=.1
                pictures=[]
                for azimuth,elevation,label in [(90,-8,'SOURCE SIDE'),(140,-26,'OBLIQUE')]:
                    camera.azimuth=azimuth;camera.elevation=elevation;renderer.update_scene(d,camera)
                    picture=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                    cv2.putText(picture,f'{name} ACTUAL {label}',(10,24),0,.43,(255,255,255),1)
                    cv2.putText(picture,f't={t:.1f}s S={controller.S:.3f} {controller.mode}',(10,46),0,.43,(255,255,255),1)
                    pictures.append(picture)
                frame=np.concatenate(pictures+[reference_panel(info)],axis=1);video.write(frame)
                if int(round(t*10))%100==0:cv2.imwrite(str(out/f'actual_{round(t):03d}s.png'),frame)
                ev=reg['evaluation_only']
                deep_age=deep_age+.1 if max(row['depth'][:2])>ev['max_pipe_depth_m'] else 0.
                no_support_age=no_support_age+.1 if row['forces'].sum()<.2 else 0.
                if t==0:
                    centers=row['centers'];rho=np.linalg.norm(centers[:,1:]-[0,1],axis=1)
                    valid=bool(np.all((centers[:,0]>-.2)&(centers[:,0]<.95)&(rho>.065)&(rho<.13)) and row['forces'][1].sum()==0.)
                    (out/'INITIAL_EVALUATION.json').write_text(json.dumps(dict(all21_on_source=valid,target_contact_N=float(row['forces'][1].sum()),source_contact_N=float(row['forces'][0].sum()),controller_received_initial_position=False),indent=2))
                    if not valid:stop='invalid_full_source_initial';break
                if deep_age>=ev['sustained_depth_s']:stop='evaluation_abort_sustained_pipe_penetration';break
                if row['depth'][2]>.003:stop='evaluation_abort_self_penetration';break
                if row['depth'][3]>0:stop='evaluation_abort_floor_contact';break
                if no_support_age>=ev['no_pipe_contact_s']:stop='evaluation_abort_no_pipe_support';break
                if np.max(abs(d.qpos[registers.qadr]))>np.deg2rad(90.5):stop='evaluation_abort_joint_limit';break
                if t>=next_progress:
                    print(json.dumps(dict(t=t,mode=controller.mode,S=controller.S,release=controller.release,attempt=controller.attempt)),flush=True)
                    sensors.flush();events.flush();next_progress+=10.
            if controller.failure:stop=controller.failure;break
            if t>=case['duration_s']-1e-8:break
            rate=np.deg2rad(reg['servo_slew_deg_s'])*m.opt.timestep
            last=np.clip(np.clip(raw,last-rate,last+rate),-np.deg2rad(reg['servo_limit_deg']),np.deg2rad(reg['servo_limit_deg']))
            d.ctrl[:]=last;d.xfrc_applied[:]=0.;mujoco.mj_step(m,d)
    except Exception:
        stop='implementation_error';(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        if sensors:sensors.close()
        if events:events.close()
        if video:video.release()
        if renderer:renderer.close()
        if rows:
            np.savez_compressed(out/'trace.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(out/'FINAL_STATE.npz',state=state,last_sent=last,raw=raw,next_control=next_control,next_sample=next_sample)
        with (out/'controller_checkpoint.pkl').open('wb') as f:pickle.dump(controller,f)
        result=dict(id=name,version=reg['version'],stop=stop,sim_time_s=float(d.time),wall_s=time.monotonic()-wall,
            final_mode=controller.mode,attempts=controller.attempt,action_chain_complete=False,success=None,
            first_servo_interaction_hypothesis=controller.first_interaction,first_capture_hypothesis=controller.first_capture_hypothesis,
            actual_outcome='pending independent evaluation',true_target_geometry_received=False,virtual_servo_sensor_model=True)
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
        with (BASE/'RUN_LEDGER.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=name,finished_at=time.time(),stop=stop,sim_time_s=float(d.time)))+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);run(ap.parse_args().run)
