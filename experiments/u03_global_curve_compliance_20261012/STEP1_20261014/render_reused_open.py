"""Existing A16 trajectory, current reference overlay, no dynamics."""
from physics import *
from receiver_phase_reference import ReceiverPhaseReference
from material_observer import MaterialObserver
from visual_overlay import overlay
import json,cv2

case=BASE/'N_OPEN_REUSE';case.mkdir(exist_ok=True);old=BASE.parent/'A16';z=np.load(old/'trace.npz')
t=z['t'];points=z['points'];centers=z['centers'];forces=z['forces'];qpos=z['qpos'];qvel=z['qvel'];ctrl=z['ctrl'];S=z['S']
m=mujoco.MjModel.from_xml_path(str(old/'model.xml'));d=mujoco.MjData(m);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
initial=np.load(old/'initial.npz');d.qpos[:]=initial['qpos'];d.qvel[:]=initial['qvel'];mujoco.mj_forward(m,d)
ref=ReceiverPhaseReference(nodes(d),d.qpos[qa]);obs=MaterialObserver(ref)
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1]
writer=cv2.VideoWriter(str(case/'actual_side_oblique_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1400,520))
gp=z['reference_points'];dense=z['nominal_curve_dense'];history=[]
for i,sec in enumerate(t):
    d.qpos[:]=qpos[i];d.qvel[:]=qvel[i];mujoco.mj_forward(m,d)
    truth=dict(centers=centers[i],forces=forces[i],body_depth=np.zeros((2,21)),joint_actual=qpos[i,qa],joint_ctrl=ctrl[i],joint_torque=np.zeros(20))
    roles=obs.update(truth,float(S[i]),.1);history.append(roles['material_progress_observed_m'].copy())
    info=dict(global_reference_points=gp[i],reference_points=gp[i],nominal_curve_dense=dense[i],**roles);images=[]
    for az,el,label in [(0,0,'SIDE'),(140,-26,'OBLIQUE')]:
        cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);overlay(ren.scene,m,info)
        image=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(image,f'N_OPEN reused A16 ACTUAL {label} t={sec:.1f}s S={S[i]:.3f}',(10,27),0,.49,(255,255,255),1);cv2.putText(image,'currentG cyan / C magenta / actual source orange target green',(10,49),0,.40,(255,255,255),1);images.append(image)
    image=np.concatenate(images,axis=1);writer.write(image)
    if int(round(sec*10))%100==0:cv2.imwrite(str(case/f'actual_{int(round(sec)):03d}s.png'),image)
writer.release();ren.close();cap=cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
for idx in [0,frames//2,frames-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,idx);checks.append(bool(cap.read()[0]))
cap.release();np.savez(case/'observer_replay.npz',t=t,progress=np.asarray(history))
(case/'REUSE.json').write_text(json.dumps(dict(physical_source='../A16',trace='../A16/trace.npz',model='../A16/model.xml',initial='../A16/initial.npz',verdict='../A16/b_metrics.json',zero_feedback_commands_checked='PREPARATION.json',new_dynamics=False,video_source='saved actual states,0.1s10fps; observer role overlay postprocessed; no pose-driven replay'),indent=2))
(case/'video_metadata.json').write_text(json.dumps(dict(frames=frames,fps=fps,duration=frames/fps,readback=checks,new_dynamics=False),indent=2))
print('Reused A16 overlay video complete',frames,fps,checks)
