"""Summarize recorded dynamics and concatenate their continuous 1x replay.
Does not call mj_step or regenerate any physical trial.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/alternating_helix_transfer_20261007'
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    rows=[]
    for case in sorted(BASE.glob('A[0-9][0-9]')):
        result=json.loads((case/'result.json').read_text(encoding='utf8'))
        plan=json.loads((case/'plan.json').read_text(encoding='utf8'))
        with np.load(case/'trace.npz') as z:
            control=z['ctrl'];first=control[0]
            with np.load(plan['start']) as initial:entry_error=float(abs(first-initial['ctrl']).max())
            row=dict(trial=case.name,change=plan['change'],initial=plan['start'],outcome=result['outcome'],seconds=result['physical_seconds'],front_delta_y_mm=result['front_mean_y_advance_mm'],rear_delta_x_mm=result['rear_mean_x_advance_mm'],tail_center_mm=result['tail_center_mm'],max_command_deg=result['max_command_deg'],max_actual_joint_deg=result['max_joint_deg'],max_torque_Nm=result['max_torque_Nm'],first_command_difference_rad=entry_error,max_dispatch_step_deg=float(np.degrees(abs(np.diff(control,axis=0))).max()),target_modules=result['final_target_modules'],source_modules=result['final_source_modules'])
        rows.append(row)
    write(BASE/'summary.json',dict(complete=False,trials=rows,hardware_prepared=False,important='Forward x/y displacement alone is not a success criterion: A11 suspended tail, A12 fell to floor.',contact_review=dict(source='postprocessing saved A03/A06 qpos every 50 ms; no dynamics rerun',A03_max_pipe_penetration_mm=2.6715742432204292,A03_nonadjacent_self_penetration_mm=0.,A06_max_pipe_penetration_mm=3.677712671802525,A06_nonadjacent_self_penetration_mm=1.3920651043492507,A06_worst_pair=[10,20],A06_time_s=158.2)))
    chain=['A06','A09','A10','A11','A12'];boundaries=[];offset=0.;last=None
    supported=cv2.VideoWriter(str(BASE/'continuous_supported_420s_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1020,480))
    full=cv2.VideoWriter(str(BASE/'continuous_to_failure_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1020,480))
    if not supported.isOpened() or not full.isOpened():raise RuntimeError('video writer failed')
    frames=0;support_frames=0
    model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    render_data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=480,width=720)
    camera=mujoco.MjvCamera();camera.lookat[:]=[1.08,.2,.65];camera.distance=2.1;camera.azimuth=150;camera.elevation=-20
    for case_name in chain:
        case=BASE/case_name
        with np.load(case/'trace.npz') as z:
            qpos=z['qpos'];qvel=z['qvel'];ctrl=z['ctrl'];stages=z['stage'];times=z['sim_time'];duration=float(times[-1]-times[0])
            difference=None if last is None else dict(qpos=float(abs(qpos[0]-last[0]).max()),qvel=float(abs(qvel[0]-last[1]).max()),ctrl=float(abs(ctrl[0]-last[2]).max()),time=float(abs(times[0]-last[3])))
            if difference and max(difference.values())>1e-8:raise RuntimeError('discontinuous physical branch')
            last=(qpos[-1].copy(),qvel[-1].copy(),ctrl[-1].copy(),float(times[-1]))
            boundaries.append(dict(case=case_name,start_s=offset,end_s=offset+duration,entry_difference=difference))
            cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));i=0
            if not cap.isOpened():raise RuntimeError('missing segment video')
            while True:
                ok,frame=cap.read()
                if not ok:break
                if i==0 and case_name!=chain[0]:i+=1;continue
                k=min(i*5,len(stages)-1);global_t=offset+float(times[k]-times[0])
                if case_name in ['A11','A12']:
                    # Wider view includes the physically hanging/falling tail.
                    render_data.qpos[:]=qpos[k];mujoco.mj_forward(model,render_data)
                    renderer.update_scene(render_data,camera=camera)
                    frame[:,:720]=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                    cv2.putText(frame,'saved physics, 1x replay; wider view',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.rectangle(frame,(0,0),(720,28),(20,20,20),-1)
                cv2.putText(frame,f'{case_name} total t={global_t:.2f}s  {stages[k]}',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                full.write(frame);frames+=1
                if case_name in ['A06','A09']:supported.write(frame);support_frames+=1
                i+=1
            cap.release();offset+=duration
    supported.release();full.release();renderer.close()
    write(BASE/'continuous_video.json',dict(chain=boundaries,total_physical_seconds=offset,frames=frames,fps=20,supported_frames=support_frames,supported_physical_seconds=420.,speed=1.,initial_frame=True,duplicate_boundary_frames_removed=True,physical_state_continuity_checked=True,source='actual saved dynamics videos, no nominal root or animation poses'))
    # Read a small set of deliverable frames to verify the muxed artifacts.
    cap=cv2.VideoCapture(str(BASE/'continuous_to_failure_1x.mp4'));images=[]
    for sec in [0,180,419,489,607.75]:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,f=cap.read()
        if ok:images.append(f)
    cap.release()
    cv2.imwrite(str(BASE/'continuous_keyframes.jpg'),np.vstack(images))
    print(json.dumps(dict(trials=len(rows),frames=frames,physical_seconds=offset,boundaries=boundaries),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
