"""Normal-speed continuous replay and physical plots; saved truth is display only."""
import argparse
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from three_segment_review import context
from segmented_pole_experiment import pipe_frame


def render(root, transfer):
    folder=root/transfer
    _,child,m,d,_=context(folder)
    keys=('time','qpos','qvel','ctrl','points_mm','normal_force','torque','error_deg','phase')
    with np.load(root/'approach/trace.npz') as parent,child:
        z={key:np.concatenate((parent[key],child[key][1:])) for key in keys}
        takeover=float(parent['time'][-1]-parent['time'][0])
    t=z['time']-z['time'][0]
    audit=json.loads((folder/'whole_crossing_audit.json').read_text(encoding='utf8'))
    out=root.parent/'reviews';out.mkdir(exist_ok=True)
    stem=root.name+'_'+transfer
    corner,source,target,_=pipe_frame(m)
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    axial=(centers-corner)@target
    speed=np.r_[0.,np.max(np.linalg.norm(np.diff(centers,axis=0)/np.diff(t)[:,None,None],axis=2),axis=1)/1000]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for label,indices in [('Head',0),('Mean',None),('Tail',20)]:
        axes[0].plot(t,axial.mean(axis=1) if indices is None else axial[:,indices],label=label)
    axes[1].plot(t,np.count_nonzero(z['normal_force'][:,0]>.1,axis=1),label='Actual source contacts')
    axes[1].plot(t,np.count_nonzero(z['normal_force'][:,1]>.1,axis=1),label='Actual target contacts')
    axes[2].plot(t,speed,label='Maximum module speed')
    axes[2].axhline(.5,color='grey',ls='--',label='Declared speed gate')
    saturation=np.mean(abs(z['torque'])>=.99*m.actuator_forcerange[:,1],axis=1)
    axes[3].plot(t,np.convolve(saturation,np.ones(100)/100,mode='same'),label='1s mean saturated-joint fraction')
    for ax,label in zip(axes,('Target axial position (mm)','Contact module count','Speed (m/s)','Fraction')):
        ax.axvline(takeover,color='purple',ls='--',label='Automatic takeover')
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend(fontsize=8)
        if audit['accepted_window']:
            window=audit['accepted_window'];ax.axvspan(takeover+window['start_seconds'],takeover+window['end_seconds'],color='green',alpha=.12)
    axes[-1].set_xlabel('Continuous physical mission time (s)')
    fig.tight_layout();fig.savefig(out/(stem+'_physical.png'),dpi=140);plt.close(fig)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=corner/1000-source*.1+target*.2
    camera.distance=2.;camera.azimuth=125;camera.elevation=-35
    modules={m.body(name).id:i for i,name in enumerate(['head']+['link'+str(i) for i in range(1,20)]+['tail'])}
    renderer=mujoco.Renderer(m,height=640,width=960)
    path=out/(stem+'_1x.mp4')
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,640))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    try:
        for clock in np.arange(0,t[-1]+1e-8,.05):
            i=int(np.argmin(abs(t-clock)))
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
            for geom in range(m.ngeom):
                module=modules.get(int(m.geom_bodyid[geom]))
                if module is None:continue
                force=z['normal_force'][i,:,module]
                color=[.1,.85,.25,1] if force[1]>.1 else [.2,.45,1,1] if force[0]>.1 else [.7,.2,.8,1] if force[2]>.1 else [1,.55,.1,1]
                m.geom_matid[geom]=-1;m.geom_rgba[geom]=color
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,960,52),fill='black')
            stage='natural approach / response probe' if t[i]<takeover else 'automatic local acquisition / transfer / rolling'
            draw.text((8,5),f'1x | physical time {t[i]:.2f}s | {stage}',fill='white')
            draw.text((8,24),'Actual contacts: green=target, blue=source, purple=other branch, orange=no pipe contact',fill='white')
            writer.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
            if abs(clock-takeover)<.026:im.save(out/(stem+'_takeover.png'))
            if clock+.05>t[-1]:im.save(out/(stem+'_final.png'))
    finally:
        renderer.close();writer.release()
    print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--transfer',default='transfer');args=p.parse_args();render(args.root,args.transfer)
