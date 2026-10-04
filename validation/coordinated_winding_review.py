"""Normal-speed replay and process plots, including unsuccessful trials."""
import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import pipe_frame
from front_support_experiment import coverage


def review(folder, video=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result,z,m,d,meta=context(folder)
    t=z['time']-z['time'][0]
    corner,source,axis,_=pipe_frame(m)
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    audit=json.loads((folder/'whole_crossing_audit.json').read_text(encoding='utf8'))
    end=audit['accepted_window']['end_seconds'] if audit['accepted_window'] else t[-1]
    keep=t<=end+1e-6
    out=folder.parent/'reviews';out.mkdir(exist_ok=True)
    velocity=np.max(np.abs(np.degrees(z['qvel'][:,m.jnt_dofadr[m.actuator_trnid[:,0]]])),axis=1)
    wrap,_=coverage(z['points_mm'],corner,axis)
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    axes[0].plot(t[keep],abs(z['error_deg'][keep]).max(axis=1),label='Actual maximum tracking error')
    axes[1].plot(t[keep],velocity[keep],label='Actual peak joint speed')
    for i,label in enumerate(('Source pipe','Target pipe')):
        axes[2].plot(t[keep],z['normal_force'][keep,i].sum(axis=1),label=label)
    axes[3].plot(t[keep],z['region'][keep].sum(axis=1),label='Actual modules in target band (21 total)')
    ct=np.array([c.get('time',(i+1)*.2) for i,c in enumerate(controls)])
    ok=ct<=end
    axes[3].plot(ct[ok],np.array([c['formation_progress'] for c in controls])[ok],label='Formation command (20 joints)')
    for ax,label in zip(axes,['Error (deg)','Speed (deg/s)','Normal force (N)','Modules / joints']):
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend(fontsize=8,loc='upper left')
    axes[-1].set_xlabel('Physical seconds after saved checkpoint')
    fig.suptitle(folder.name);fig.tight_layout();fig.savefig(out/(folder.name+'_process.png'),dpi=130);plt.close(fig)
    if not video:return
    import cv2
    from PIL import Image,ImageDraw
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=corner/1000+axis*.3-source*.08
    camera.distance=2.1;camera.azimuth=125;camera.elevation=-50 if abs(axis[2])<.5 else -25
    colors=np.array([[.15,.7,.3,1.],[1.,.63,.1,1.],[.2,.5,1.,1.]])
    modules={}
    for b in range(m.nbody):
        name=m.body(b).name
        idx=0 if name=='head' else 20 if name=='tail' else int(name[4:]) if name.startswith('link') else None
        if idx is not None:modules[b]=idx
    renderer=mujoco.Renderer(m,height=720,width=960)
    writer=cv2.VideoWriter(str(out/(folder.name+'_1x.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,720))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    peak=int(np.argmax(np.where(t>1,velocity,-1)))
    indices=list(range(0,np.flatnonzero(keep)[-1]+1,10))
    try:
        for i in sorted(set(indices+[peak])):
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
            f,r=z['partition'][i]
            for geom in range(m.ngeom):
                module=modules.get(int(m.geom_bodyid[geom]))
                if module is not None:
                    m.geom_rgba[geom]=colors[0 if module<=f else 1 if module<=r else 2];m.geom_matid[geom]=-1
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,960,65),fill='black')
            draw.text((8,5),f'{folder.name} | t={t[i]:.1f}s | REAL TIME 1x | wrap={wrap[i]:.2f} turns',fill='white')
            draw.text((8,24),f'Actual error={abs(z["error_deg"][i]).max():.1f}deg | peak joint speed={velocity[i]:.1f}deg/s | target band={z["region"][i].sum()}/21',fill='white')
            draw.text((8,43),'Green/orange/blue: commanded front/middle/rear membership, NOT support.',fill='white')
            if i in indices:writer.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
            if i==peak:im.save(out/(folder.name+'_peak_speed.png'))
            if i==indices[-1]:im.save(out/(folder.name+'_end.png'))
    finally:renderer.close();writer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--video',action='store_true');args=p.parse_args();review(args.folder,args.video)
