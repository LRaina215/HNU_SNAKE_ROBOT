"""One reviewed diagnostic, untrimmed physical t0; original MuJoCo model."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,shutil,xml.etree.ElementTree as ET
import cv2,scipy
from physics import *
from direct_controller import DirectController
from evaluation import snapshot
from visual_overlay import overlay

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def require_review(reg):
    review=BASE/'SELF_REVIEW.json';freeze=BASE/'LOCAL_FREEZE'
    if reg.get('requirements')!='STEP2_FINAL_20261016' or not reg['dynamics_authorized'] or not review.exists():raise RuntimeError('Final-contract registration/self-check absent')
    checked=json.loads(review.read_text(encoding='utf-8'))
    if checked.get('passed') is not True or checked.get('version')!=reg['version']:raise RuntimeError('Current local self-check failed')
    initial_review=json.loads((BASE/reg['initial_review']).resolve().read_text(encoding='utf-8'))
    if not initial_review.get('limited_geometry_and_static_support_review_passed'):raise RuntimeError('Given initial limited review is absent')
    reviewed_initial=initial_review.get('initial_file')
    if not reviewed_initial:raise RuntimeError('Independent initial review must identify its exact initial file')
    initial_review_file=(BASE/reg['initial_review']).resolve();reviewed_path=Path(reviewed_initial)
    if not reviewed_path.is_absolute():reviewed_path=(initial_review_file.parent/reviewed_path).resolve()
    if reviewed_path.read_bytes()!=(BASE/'GIVEN_INITIAL.npz').read_bytes():raise RuntimeError('Independent initial review belongs to a different given pose')
    for file in BASE.glob('*.py'):
        if not (freeze/file.name).exists() or file.read_bytes()!=(freeze/file.name).read_bytes():raise RuntimeError('Code changed since review: '+file.name)
    for file in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz']:
        if (BASE/file).read_bytes()!=(freeze/file).read_bytes():raise RuntimeError('Reviewed initial changed')
    saved=json.loads((freeze/'REGISTERED.json').read_text(encoding='utf-8'));current=dict(reg)
    for key in ['dynamics_authorized','independent_initial_and_chain_review_passed','current_code_independently_reviewed']:saved.pop(key,None);current.pop(key,None)
    if current!=saved:raise RuntimeError('Registration changed since review')
    if MODEL.read_bytes()!=(freeze/'model.xml').read_bytes():raise RuntimeError('Original model changed')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file') and (MODEL.parent/mesh.get('file')).read_bytes()!=(freeze/mesh.get('file')).read_bytes():raise RuntimeError('Original mesh changed: '+mesh.get('file'))

def run(case_name):
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));require_review(reg)
    assert mujoco.__version__=='3.2.7' and np.__version__=='1.23.5' and scipy.__version__=='1.13.1'
    case=next(c for c in reg['cases'] if c['case']==case_name);out=BASE/case_name;out.mkdir(exist_ok=False)
    for file in BASE.glob('*.py'):shutil.copy2(file,out/file.name)
    for name in ['REGISTERED.json','GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json','SELF_REVIEW.json']:shutil.copy2(BASE/name,out/name)
    shutil.copy2(MODEL,out/'model.xml')
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file'):
            target=out/mesh.get('file');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),target)
    m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));d=mujoco.MjData(m);z=np.load(out/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    control=DirectController(m,d,reg,case['condition'],case['write_global']);last=d.ctrl.copy();raw=last.copy();rows=[];next_command=0.;next_sample=0.;stop='duration';deep_age=0.;unsupported_age=0.;fit_n=0;write_n=0;wall=time.time()
    # G0 is the initial prescribed family head route, sampled without observation.
    G0=np.vstack([control.planner.given[::-1],np.array([control.planner.goals(a,0.,0.)[0] for a in np.linspace(0,1,31)])[1:],np.array([control.planner.wrap_guide(b,0.)[1] for b in np.linspace(.001,1.,41)])])
    np.savez_compressed(out/'initial_G_family.npz',points=G0)
    evidence=(out/'contact_evidence.jsonl').open('w');fits=(out/'geometry_hypotheses.jsonl').open('w');writes=(out/'global_updates.jsonl').open('w')
    ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.9;cam.lookat[:]=[.94,.25,1.]
    video=cv2.VideoWriter(str(out/'actual_dual_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520))
    if not video.isOpened():raise RuntimeError('Video initialization failed before any integration')
    (out/'plan.json').write_text(json.dumps(dict(version=reg['version'],role='single development diagnostic',case=case,physical_t0_included=True,actual_root_write_after_initialization=False),indent=2))
    started=False;error=None
    try:
        while True:
            t=float(d.time)
            if t>=next_command-1e-10:
                raw,info=control.command(d,t);evidence.write(json.dumps(dict(t=t,contacts=__import__('contact_sensor').read_sensor(m,d)['contacts'],probe_requested=control.ref.probe,probe_transmitted=control.applied_probe,probe_evidence_input=info['probe_evidence_input_per_m']),default=serial)+'\n')
                for entry in control.estimator.fit_history[fit_n:]:fits.write(json.dumps(entry,default=serial)+'\n')
                for entry in control.updater.history[write_n:]:writes.write(json.dumps(entry,default=serial)+'\n')
                fit_n=len(control.estimator.fit_history);write_n=len(control.updater.history);next_command+=reg['command_interval_s']
            if t>=next_sample-1e-10:
                row=snapshot(m,d);head=m.body('head').id;row.update(t=t,initial_global_family_dense=G0,raw_command=raw.copy(),body_xmat=d.xmat[head:head+21].copy(),**info);rows.append(row);next_sample+=.1
                images=[]
                for az,el,label in [(90,-8,'SOURCE SIDE'),(140,-26,'OBLIQUE')]:
                    cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);overlay(ren.scene,m,dict(info,initial_global_family_dense=G0),control.estimator.geometry);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
                    cv2.putText(im,f'{case_name} ACTUAL {label} t={t:.1f}s {control.phase} a={control.alpha:.2f} b={control.beta:.2f} S={control.S:.3f}',(10,25),0,.39,(255,255,255),1);cv2.putText(im,'G0 blue / Gt cyan / C pink / actual source orange-target green',(10,45),0,.33,(255,255,255),1);images.append(im)
                frame=np.concatenate(images,axis=1);video.write(frame)
                if int(round(t*10))%100==0:cv2.imwrite(str(out/f'actual_{int(round(t)):03d}s.png'),frame)
                deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
                if deep_age>=.3:stop='sustained_pipe_penetration';break
                if row['depth'][2]>.003:stop='self_penetration';break
                if row['depth'][3]>0.:stop='floor_contact';break
                if unsupported_age>=.5:stop='loss_all_pipe_contact';break
                if abs(d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]).max()>np.radians(90.5):stop='joint_limit';break
                if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
                if int(round(t*10))%100==0:
                    evidence.flush();fits.flush();writes.flush();print(json.dumps(dict(t=t,phase=control.phase,alpha=control.alpha,beta=control.beta,S=control.S,source_N=info['known_source_vertical_N'],receiver_N=info['receiver_vertical_N'],source_drift_m=info['actual_source_boundary_motion_m'],G_writes=info['global_update_count'],reliable=info['geometry_reliable'])),flush=True)
            if control.failure is not None:stop=control.failure;break
            if (out/'stop_and_save.request').exists():stop='requested_stop';break
            if t>=reg['duration_s']:break
            rate=np.radians(reg['servo_slew_deg_s'])*m.opt.timestep;last=np.clip(np.clip(raw,last-rate,last+rate),-np.radians(reg['servo_limit_deg']),np.radians(reg['servo_limit_deg']));d.ctrl[:]=last;d.xfrc_applied[:]=0.
            if not started:(out/'DYNAMICS_START.json').write_text(json.dumps(dict(first_integrated_time_s=t,physical_t0_logged=bool(rows and rows[0]['t']==0.))));started=True
            mujoco.mj_step(m,d)
    except Exception as exc:stop='controller_or_runner_failure';error=repr(exc);(out/'error.txt').write_text(error)
    finally:evidence.close();fits.close();writes.close();video.release();ren.close()
    if rows:np.savez_compressed(out/'trace.npz',**{key:np.array([row[key] for row in rows]) for key in rows[0]})
    state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'integration_state.npz',state=state,last_command=last,alpha=control.alpha,beta=control.beta,S=control.S)
    result=dict(stop=stop,sim_time=float(d.time),wall_s=time.time()-wall,error=error,success=None,action_chain_complete=False,role='development diagnostic; separate final four-case comparison required',first_release_t=control.first_release_t,first_raw_front_contact_t=control.estimator.first_front_contact,first_reliable_t=control.estimator.first_reliable,first_G_write_t=control.updater.first_update,first_capture_t=control.first_capture_t)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',default='T_UPDATE');run(ap.parse_args().case)
