"""Offline localization of support loss; no additional physics experiment."""
import json
import numpy as np
from spatial_migration_review import OUT,MODES
from segmented_pole_experiment import HORIZONTAL,load_checkpoint,pipe_frame,write_json


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,d,meta,c=load_checkpoint(HORIZONTAL,verify_hashes=False)
    corner,source,target,radius=pipe_frame(m)
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True);rows=[]
    for column,mode in enumerate(MODES):
        for boundary in ('fixed','migrating'):
            folder=OUT/f'{mode}_{boundary}';z=np.load(folder/'trace.npz')
            report=json.loads((folder/'analysis.json').read_text());n=int(report['cycles']*40+1)
            t=z['time'][:n]-z['time'][0];points=z['points_mm'][:n,:7]-corner
            radial=points-(points@target)[:,:,None]*target
            cross=np.cross(radial[:,:-1],radial[:,1:])@target
            dot=np.sum(radial[:,:-1]*radial[:,1:],axis=-1)
            # Signed net angular coverage of the ORIGINAL six front modules;
            # an open turn is not counted as a complete load-bearing wrap.
            coverage=np.abs(np.arctan2(cross,dot).sum(axis=1))/(2*np.pi)
            forces=z['normal_force'][:n,1,:6].sum(axis=1)
            axes[0,column].plot(t,coverage,label=boundary)
            axes[1,column].plot(t,forces,label=boundary)
            axes[2,column].plot(t,np.max(abs(z['error_deg'][:n,3:10]),axis=1),label=boundary)
            loss=report['first_front_contact_loss_0_2s_at'];samples=[]
            for ts in [0.,12.8,16.,loss-.4,loss,loss+.4]:
                k=int(np.argmin(abs(t-ts)));lo=max(0,k-40)
                samples.append(dict(time_s=float(t[k]),front_net_wrap_turns=float(coverage[k]),
                    front_target_normal_force_N=float(forces[k]),
                    front_target_normal_force_previous_0_4s_mean_N=float(forces[lo:k+1].mean()),
                    front_contact_modules=np.flatnonzero(z['normal_force'][k,1,:6]>.1).tolist(),
                    boundary_links=float(z['boundary_links'][k]),
                    bridge_max_tracking_error_deg=float(abs(z['error_deg'][k,3:10]).max())))
            rows.append(dict(condition=mode,boundary=boundary,samples=samples))
        axes[0,column].set_title(mode)
        for ax in axes[:,column]:ax.set_xlim(0,24);ax.grid(alpha=.2);ax.legend()
        axes[2,column].set_xlabel('Physical time (s)')
    axes[0,0].set_ylabel('Front 6 modules net angular coverage (turns)')
    axes[1,0].set_ylabel('Front target-pipe normal force sum (N)')
    axes[2,0].set_ylabel('J4-J10 max tracking error (deg)')
    fig.tight_layout();fig.savefig(OUT/'support_loss.png',dpi=150);plt.close(fig)
    write_json(OUT/'support_loss.json',rows)
    # Render a compact three-frame view around support loss for one migration
    # case, using saved physical poses, not a newly simulated trajectory.
    import mujoco
    from PIL import Image,ImageDraw
    folder=OUT/'rear_first_migrating';z=np.load(folder/'trace.npz')
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=corner/1000-source*.12+target*.12
    camera.distance=1.6;camera.azimuth=125;camera.elevation=-55
    renderer=mujoco.Renderer(m,height=480,width=640);sheet=Image.new('RGB',(1920,480))
    try:
        for j,ts in enumerate((12.8,18.,19.4)):
            k=int(np.argmin(abs(z['time']-z['time'][0]-ts)))
            d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.ctrl[:]=z['ctrl'][k]
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,28),fill='black')
            draw.text((8,8),f'rear_first migrating | +{ts:.1f}s | boundary={z["boundary_links"][k]:.2f}',fill='white')
            sheet.paste(im,(j*640,0))
        sheet.save(OUT/'support_loss_frames.png')
    finally:renderer.close()


if __name__=='__main__':main()
