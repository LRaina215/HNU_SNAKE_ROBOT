"""Continuous physical runner. Intentionally blocked pending independent review."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from physics import *
from source_controller import FromSourceController
from evaluation import snapshot
import json,argparse,pickle,time,shutil,cv2,scipy,xml.etree.ElementTree as ET
from visual_overlay import overlay

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def require_review(reg):
    review=BASE/'INDEPENDENT_REVIEW.json'
    if not reg['dynamics_authorized'] or not reg['independent_initial_and_chain_review_passed'] or not review.exists():
        raise RuntimeError('Dynamics disabled: independent full-source initial and actual controller-chain review pending')
    approved=json.loads(review.read_text())
    if approved.get('approved') is not True or approved.get('version')!=reg['version']:raise RuntimeError('Independent review version/approval mismatch')
    freeze=BASE/'REVIEW_SNAPSHOT'
    for file in BASE.glob('*.py'):
        if not (freeze/file.name).exists() or file.read_bytes()!=(freeze/file.name).read_bytes():raise RuntimeError('Code changed since review snapshot: '+file.name)
    for name in ['SOURCE_INITIAL.npz','source_constants.json']:
        if (BASE/name).read_bytes()!=(freeze/name).read_bytes():raise RuntimeError('Initial/constants changed since review')
    frozen_reg=json.loads((freeze/'REGISTERED.json').read_text());current=dict(reg)
    for key in ['dynamics_authorized','independent_initial_and_chain_review_passed']:frozen_reg.pop(key);current.pop(key)
    if current!=frozen_reg:raise RuntimeError('Scientific preregistration changed since review snapshot')
    if MODEL.read_bytes()!=(freeze/'model.xml').read_bytes():raise RuntimeError('Original physics model changed')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file') and (MODEL.parent/mesh.get('file')).read_bytes()!=(freeze/mesh.get('file')).read_bytes():raise RuntimeError('Original mesh changed')

def run(case_name):
    reg=json.loads((BASE/'REGISTERED.json').read_text());require_review(reg)
    assert mujoco.__version__=='3.2.7' and np.__version__=='1.23.5' and scipy.__version__=='1.13.1'
    assert all(os.environ[k]=='1' for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'])
    case=next(c for c in reg['cases'] if c['case']==case_name);out=BASE/case_name;out.mkdir(exist_ok=False)
    for file in BASE.glob('*.py'):shutil.copy2(file,out/file.name)
    for name in ['REGISTERED.json','SOURCE_INITIAL.npz','SOURCE_INITIAL_PROOF.json','source_constants.json','INDEPENDENT_REVIEW.json']:shutil.copy2(BASE/name,out/name)
    shutil.copy2(MODEL,out/'model.xml')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file'):
            dest=out/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),dest)
    m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));d=mujoco.MjData(m);initial=np.load(out/'SOURCE_INITIAL.npz')
    mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    control=FromSourceController(m,d,reg,case['condition'],case['write_global']);last=d.ctrl.copy();raw=last.copy();rows=[]
    (out/'plan.json').write_text(json.dumps(dict(configuration=case,version=reg['version'],runtime=dict(python=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__),
        physical_model_unchanged=True,initial_is_full_source=True,initial_is_historical_resume=False,only_global_write_permission_differs_within_pair=True),indent=2))
    next_command=0.;next_sample=0.;stop='duration';deep_age=0.;unsupported_age=0.;fit_n=0;write_n=0;wall=time.time()
    evidence=(out/'contact_evidence.jsonl').open('w');fits=(out/'geometry_hypotheses.jsonl').open('w');writes=(out/'global_updates.jsonl').open('w')
    ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.9;cam.lookat[:]=[.85,.25,1.]
    video=cv2.VideoWriter(str(out/'actual_dual_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520))
    if not video.isOpened():raise RuntimeError('Video setup failed before any dynamics')
    started=False
    # t=0 is logged before the first physics step. Stabilization is included.
    while True:
        t=float(d.time)
        if t>=next_command-1e-10:
            try:raw,info=control.command(d,t)
            except Exception as exc:stop='controller_failure';(out/'error.txt').write_text(repr(exc));break
            from contact_sensor import read_sensor
            evidence.write(json.dumps(dict(t=t,contacts=read_sensor(m,d)['contacts'],probe=control.ref.probe),default=serial)+'\n')
            for entry in control.estimator.fit_history[fit_n:]:fits.write(json.dumps(entry,default=serial)+'\n')
            for entry in control.updater.history[write_n:]:writes.write(json.dumps(entry,default=serial)+'\n')
            fit_n=len(control.estimator.fit_history);write_n=len(control.updater.history);next_command+=reg['command_interval_s']
        if t>=next_sample-1e-10:
            row=snapshot(m,d);head=m.body('head').id
            row.update(t=t,raw_command=raw.copy(),body_xmat=d.xmat[head:head+21].copy(),**info);rows.append(row);next_sample+=.1
            images=[]
            for az,el,label in [(90,-8,'SOURCE SIDE'),(140,-26,'OBLIQUE')]:
                cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);overlay(ren.scene,m,info,control.estimator.geometry)
                im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
                cv2.putText(im,f'{case_name} ACTUAL {label} t={t:.1f}s S={control.S:.3f}',(10,25),0,.48,(255,255,255),1)
                images.append(im)
            frame=np.concatenate(images,axis=1);video.write(frame)
            if int(round(t*10))%100==0:cv2.imwrite(str(out/f'actual_{int(round(t)):03d}s.png'),frame)
            deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
            if t==0.:
                c=row['centers'];r=np.linalg.norm(c[:,1:]-[0,1],axis=1)
                if not np.all((c[:,0]>-.2)&(c[:,0]<.95)&(r>=.065)&(r<=.13)) or row['forces'][1].sum()!=0.:stop='invalid_full_source_initial';break
            if deep_age>=.3:stop='sustained_pipe_penetration';break
            if row['depth'][2]>.003:stop='self_penetration';break
            if row['depth'][3]>0.:stop='floor_contact';break
            if unsupported_age>=.5:stop='loss_all_pipe_contact';break
            if np.max(abs(d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]))>np.radians(90.5):stop='joint_limit';break
            if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
            if int(round(t*10))%100==0:
                evidence.flush();fits.flush();writes.flush();print(json.dumps(dict(t=t,S=control.S,source_ready=control.source_ready,front_confirmed=control.observer.front_confirmed)),flush=True)
        if control.failure is not None:stop=control.failure;break
        if (out/'stop_and_save.request').exists():stop='requested_stop';break
        if t>=reg['duration_s']:break
        # Servo integrates toward native curve commands; never write qpos/root.
        rate=np.radians(reg['servo_slew_deg_s'])*m.opt.timestep
        last=np.clip(np.clip(raw,last-rate,last+rate),-np.radians(reg['servo_limit_deg']),np.radians(reg['servo_limit_deg']))
        d.ctrl[:]=last;d.xfrc_applied[:]=0.
        if not started:
            (out/'DYNAMICS_START.json').write_text(json.dumps(dict(first_integrated_time_s=t,full_source_t0_logged=bool(rows and rows[0]['t']==0.))))
            started=True
        mujoco.mj_step(m,d)
    evidence.close();fits.close();writes.close();video.release();ren.close()
    if rows:np.savez_compressed(out/'trace.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(out/'integration_state.npz',state=state,S=control.S,last_command=last,next_command=next_command,next_sample=next_sample)
    checkpoint=control.__dict__.copy();checkpoint.pop('m')
    with (out/'controller_checkpoint.pkl').open('wb') as f:pickle.dump(checkpoint,f)
    result=dict(stop=stop,sim_time=float(d.time),wall_s=time.time()-wall,success=None,action_chain_complete=False,acceptance_status='pending independent full-chain evaluation',
        first_release_t=control.first_release_t,first_raw_front_contact_t=control.estimator.first_front_contact,first_reliable_t=control.estimator.first_reliable,first_G_write_t=control.updater.first_update)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    cap=cv2.VideoCapture(str(out/'actual_dual_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);readback=[]
    for frame_index in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);readback.append(bool(cap.read()[0]))
    cap.release();(out/'video_metadata.json').write_text(json.dumps(dict(frames=n,fps=fps,readback=readback,physical_t0_included=True,continuous_actual_dynamics=True),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);run(ap.parse_args().case)
