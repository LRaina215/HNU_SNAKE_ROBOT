"""Render the accepted interval and plots; source data remain unmodified."""
import argparse
import json
import numpy as np
import mujoco

from segmented_pole_experiment import BASE,pipe_frame
from three_segment_review import context
from front_support_experiment import coverage


def review(name,video=True):
    import cv2
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=BASE/'traveling_winding'/name
    result,z,m,d,meta=context(folder)
    report=json.loads((folder/'whole_crossing_audit.json').read_text())
    controls=json.loads((folder/'controls.json').read_text())
    output=folder.parent/'reviews'
    output.mkdir(exist_ok=True)
    corner,source,axis,_=pipe_frame(m)
    times=z['time']-z['time'][0]
    end=report['accepted_window']['end_seconds']
    keep=times<=end+1e-8
    wrap,_=coverage(z['points_mm'],corner,axis)
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    advance=(centers.mean(axis=1)-centers[0].mean(axis=0))@axis
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    axes[0].plot(times[keep],z['region'][keep].sum(axis=1),label='Measured modules in target band (21 total)')
    ct=np.arange(1,len(controls)+1)*.2
    cm=ct<=end
    axes[0].plot(ct[cm],np.array([r['formation_progress'] for r in controls])[cm],label='Commanded formation progress (20 joints)')
    axes[1].plot(times[keep],wrap[keep],label='Actual net wrapping around target')
    axes[2].plot(times[keep],np.sum(z['normal_force'][keep,0],axis=1),label='Source pipe')
    axes[2].plot(times[keep],np.sum(z['normal_force'][keep,1],axis=1),label='Target pipe')
    axes[3].plot(times[keep],advance[keep],label='Whole-body mean advance along target')
    for ax,label in zip(axes,['Modules / joints','Turns','Normal contact force (N)','Advance (mm)']):
        ax.set_ylabel(label)
        ax.axvspan(report['accepted_window']['start_seconds'],end,color='green',alpha=.12)
        ax.grid(alpha=.25);ax.legend(loc='upper left',fontsize=8)
    axes[-1].set_xlabel('Seconds from the user checkpoint; green = accepted full rolling cycle')
    fig.tight_layout();fig.savefig(output/f'{name}_metrics.png',dpi=140);plt.close(fig)
    if not video:return
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=corner/1000+axis*.35-source*.05
    camera.distance=2.0
    camera.azimuth=125
    camera.elevation=-50 if abs(axis[2])<.5 else -25
    colors=np.array([[.15,.7,.3,1.],[1.,.63,.1,1.],[.2,.5,1.,1.]])
    modules={}
    for b in range(m.nbody):
        nameb=m.body(b).name
        idx=0 if nameb=='head' else 20 if nameb=='tail' else int(nameb[4:]) if nameb.startswith('link') else None
        if idx is not None:modules[b]=idx
    renderer=mujoco.Renderer(m,height=720,width=960)
    writer=cv2.VideoWriter(str(output/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,720))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    try:
        indices=list(range(0,int(np.flatnonzero(keep)[-1])+1,10))
        for i in indices:
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
            f,r=z['partition'][i]
            for geom in range(m.ngeom):
                module=modules.get(int(m.geom_bodyid[geom]))
                if module is not None:
                    m.geom_rgba[geom]=colors[0 if module<=f else 1 if module<=r else 2]
                    m.geom_matid[geom]=-1
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,960,65),fill='black')
            draw.text((9,5),f'{name} | t={times[i]:.1f}s | playback 2x | wrap={wrap[i]:.2f} turns',fill='white')
            source_count=int(np.count_nonzero(z['normal_force'][i,0]>.1))
            target_count=int(np.count_nonzero(z['normal_force'][i,1]>.1))
            draw.text((9,24),f'Measured target band={z["region"][i].sum()}/21 | contact modules source={source_count} target={target_count}',fill='white')
            draw.text((9,43),'Green/orange/blue = front/middle/rear commands; color alone is NOT proof of transfer.',fill='white')
            writer.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
            if i==indices[-1]:im.save(output/f'{name}_accepted.png')
    finally:
        renderer.close();writer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name');p.add_argument('--plots-only',action='store_true')
    args=p.parse_args();review(args.name,not args.plots_only)
