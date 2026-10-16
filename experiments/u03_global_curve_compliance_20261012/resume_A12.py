"""Stage A: nominal geometric reference only; original joint servo remains."""
from physics import *
from curve_reference import GlobalCurveReference,LINK
from curve_projector import CurveProjector
from two_window_reference import TwoWindowReference
from continuous_window_reference import ContinuousWindowReference
from supply_window_reference import SupplyWindowReference
from handover_window_reference import HandoverWindowReference
import argparse,json,shutil,time,pickle,traceback,xml.etree.ElementTree as ET
import cv2

ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--duration',type=float,default=235);ap.add_argument('--depth',type=float,default=.76);ap.add_argument('--feed_turns',type=float,default=1);ap.add_argument('--roll_sign',type=float,default=1);ap.add_argument('--radius',type=float,default=.0895);ap.add_argument('--pitch',type=float,default=.183);ap.add_argument('--prepare',action='store_true');ap.add_argument('--projection',action='store_true');ap.add_argument('--moving_family',action='store_true');ap.add_argument('--anchor_frame',action='store_true');ap.add_argument('--bezier_bridge',action='store_true');ap.add_argument('--local_material',action='store_true');ap.add_argument('--smooth_source',action='store_true');ap.add_argument('--discrete_transport',action='store_true');ap.add_argument('--axial_family',action='store_true');ap.add_argument('--release_curve',action='store_true');ap.add_argument('--two_windows',action='store_true');ap.add_argument('--clearance_connector',action='store_true');ap.add_argument('--continuous_windows',action='store_true');ap.add_argument('--supply_windows',action='store_true');ap.add_argument('--handover_windows',action='store_true');a=ap.parse_args()
shift_unit='cells' if (a.two_windows or a.continuous_windows or a.supply_windows or a.handover_windows) else 'm'
case=BASE/a.case
plan=json.loads((case/'plan.json').read_text());old_result=json.loads((case/'pause_result.json').read_text())
m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
state=np.load(case/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);d.ctrl[:]=state['ctrl'];mujoco.mj_forward(m,d)
with (case/'controller_checkpoint.pkl').open('rb') as fp:checkpoint=pickle.load(fp)
ref=checkpoint['reference'];projector=None;z=np.load(case/'initial.npz');old=np.load(case/'trace.npz')
prior_rows=[{k:old[k][j].copy() for k in old.files} for j in range(len(old['t']))]
for src in ['integration_state.npz','controller_checkpoint.pkl','final_state.npz']:
    shutil.copy2(case/src,case/('pause_'+src))
shutil.copy2(BASE/'resume_A12.py',case/'resume_A12.py')
(case/'resume.json').write_text(json.dumps(dict(pause_reason='supervision registration requirement subsequently withdrawn; not a dynamics failure',pause_time_s=float(d.time),full_integration_state_restored=True,controller_checkpoint_restored=True,initial_state_replayed=False,controller_version='original frozen longer-source/last6 registration',target_total_physical_time_s=235),indent=2))
print('RESUME exact integration/controller state at',d.time,'prior samples',len(prior_rows),flush=True)
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1];wr=cv2.VideoWriter(str(case/'actual_side_oblique_2x_part2.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1400,520))
rows=prior_rows;last=checkpoint['last_command'];next_command=checkpoint['next_command'];next_sample=checkpoint['next_sample'];deep_age=0.;unsupported_age=0.;info=checkpoint['last_valid_info'];raw=checkpoint['raw_command'];stop='duration';start=time.time()
while d.time<a.duration:
    t=float(d.time)
    if t>=next_command:
        S=ref.feed_per_cycle*.02*max(0,t-.5)
        try:
            raw,info=ref.command(S) if t>=.5 else (z['ctrl'].copy(),i0)
            if projector and t>=.5:raw,extra=projector.command(info['reference_points'],info['reference_frame']);info.update(extra)
        except Exception as exc:
            stop='reference_or_projection_failure';print('REFERENCE_FAILURE t',t,'S',S,'reason',str(exc),flush=True);(case/'reference_error.json').write_text(json.dumps(dict(t=t,S=S,error=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc()),indent=2));break
        next_command+=.025
    ctrl=np.clip(raw,last-np.radians(20)*m.opt.timestep,last+np.radians(20)*m.opt.timestep);ctrl=np.clip(ctrl,-np.radians(88),np.radians(88));d.ctrl[:]=ctrl;last=ctrl.copy();mujoco.mj_step(m,d)
    if t>=next_sample:
        row=snapshot(m,d);row.update(t=t,raw_command=raw.copy(),tracking=float(np.max(abs(ctrl-d.qpos[qa]))),**info);rows.append(row);next_sample+=.1
        deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.;unsupported_age=unsupported_age+.1 if row['forces'].sum()<.2 else 0.
        if int(round(t*10))%100==0:
            progress=dict(t=t,S=info['S'],chi=info['chi'],head=row['points'][0].tolist(),forces=row['forces'].sum(1).tolist(),depth_mm=(row['depth']*1000).tolist(),reference_head=info['reference_points'][0].tolist());print(json.dumps(progress),flush=True);(case/'progress.json').write_text(json.dumps(progress,indent=2))
        ims=[]
        for az,el,label in [(0,0,'SIDE'),(140,-26,'OBLIQUE')]:
            cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,f'{a.case} ACTUAL {label} t={t:.1f}s S={info["S"]:.3f}{shift_unit}',(10,27),0,.55,(255,255,255),1);ims.append(im)
        im=np.concatenate(ims,axis=1);wr.write(im)
        if int(round(t*10))%100==0:cv2.imwrite(str(case/f'actual_{int(round(t)):03d}s.png'),im)
        if deep_age>=.3:stop='sustained_pipe_penetration';break
        if row['depth'][2]>.003:stop='self_penetration';break
        if row['depth'][3]>0:stop='floor_contact';break
        if unsupported_age>=.5 and t>1.5:stop='loss_all_pipe_contact';break
        if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
        if (case/'stop_and_save.request').exists():stop='requested_stop';break
wr.release();ren.close();np.savez_compressed(case/'trace.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]});state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'integration_state.npz',state=state,ctrl=d.ctrl,last_command=last,S=info['S'],next_command=next_command,raw=raw);np.savez(case/'final_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
with (case/'controller_checkpoint.pkl').open('wb') as fp:pickle.dump(dict(reference=ref,last_command=last,raw_command=raw,last_valid_info=info,next_command=next_command,next_sample=next_sample,physical_time=d.time,projector=None if projector is None else dict(last=projector.last,initial_frame=projector.initial_frame,initial_rotation=projector.initial_rotation,reference_qpos=projector.d.qpos.copy())),fp)
result=dict(stop=stop,sim_time=float(d.time),wall_s=old_result['wall_s']+time.time()-start,success=False,maximum_depth_mm=(np.max([r['depth'] for r in rows],axis=0)*1000).tolist(),max_actual_joint_deg=float(np.rad2deg(np.max(abs(np.array([r['qpos'][qa] for r in rows]))))),max_torque_nm=float(np.max(abs(np.array([r['torque'] for r in rows])))))
(case/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)


