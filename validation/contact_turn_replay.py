"""Replay saved qpos/ctrl; rendering never advances physics or contacts hardware."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    case=args.case.resolve()
    r=np.load(case/'trace.npz')
    model_path=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml' if case.parent.name=='stage1' else case.parents[1]/'models/horizontal_straight.xml'
    m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
    t=r['sim_time']
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    im=axes[0].imshow(r['torque'].T,aspect='auto',origin='lower',extent=[t[0],t[-1],.5,20.5],cmap='coolwarm',vmin=-2.355,vmax=2.355)
    axes[0].set_ylabel('Joint ID');fig.colorbar(im,ax=axes[0],label='Actuator torque (N m)')
    axes[1].plot(t,np.degrees(abs(r['ctrl']-r['q'])).max(1),lw=1)
    axes[1].set_ylabel('Max tracking error (deg)')
    axes[2].plot(t,r['contact_modules'].sum(1),label='Pipe contact modules')
    axes[2].plot(t,r['ground'].astype(int),label='Ground contact')
    axes[2].set_xlabel('Physical time (s)');axes[2].set_ylabel('Count');axes[2].legend()
    fig.suptitle(case.name)
    fig.savefig(case/'diagnostics.png',dpi=160);plt.close(fig)
    renderer=mujoco.Renderer(m,height=480,width=720)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[.85,.2,1.];camera.distance=2.6;camera.azimuth=135;camera.elevation=-25
    writer=None;frame_count=0
    try:
        if args.video:
            writer=cv2.VideoWriter(str(case/'replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20.,(720,480))
            if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
        selected=set([0,len(t)//2,len(t)-1])
        for i in range(len(t)):
            if i not in selected and (not args.video or i%5):continue
            d.qpos[:]=r['qpos'][i];d.qvel[:]=r['qvel'][i];d.ctrl[:]=r['ctrl'][i]
            mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=camera)
            rgb=renderer.render().copy()
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.putText(bgr,f"t={t[i]:.2f}s   {r['stage'][i]}   cmd={r['applied_command_id'][i]}",
                        (12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
            if i in selected:cv2.imwrite(str(case/f'frame_{i:05d}.png'),bgr)
            if writer is not None and i%5==0:writer.write(bgr);frame_count+=1
    finally:
        renderer.close()
        if writer:writer.release()
    print(json.dumps(dict(case=str(case),rendered_video_frames=frame_count,physics_advanced=False)))


if __name__=='__main__':main()
