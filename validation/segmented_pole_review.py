"""Audit raw paired traces; render replay without running any controller."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from segmented_pole_experiment import BASE, ROOT, load_checkpoint, pipe_frame, sha, write_json, POLICY


def input_path(result):
    # Use the archived relative input so replay works after cloning elsewhere.
    name=Path(result['checkpoint']).name
    return BASE/'inputs'/name


def audit(folder):
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    z=np.load(folder/'trace.npz');commands=json.loads((folder/'commands.json').read_text(encoding='utf8'))
    m,d,meta,controller=load_checkpoint(input_path(result))
    corner,source,target,radius=pipe_frame(m)
    assert all(sha(folder/'sources'/name)==digest for name,digest in result['source_hashes'].items())
    initial=np.load(folder/'initial_state.npz')['state']
    stored=np.load(input_path(result)/'state.npz')['integration_state']
    assert np.array_equal(initial,stored),'different initial integration state'
    assert np.array_equal(z['qpos'][0],d.qpos)
    assert np.array_equal(z['qvel'][0],d.qvel)
    assert np.array_equal(z['ctrl'][0],d.ctrl)
    assert len(commands)==result['steps'] and all(c['applied'] for c in commands)
    assert np.allclose(np.diff(z['time']),.01,atol=1e-10)
    ids=m.actuator_trnid[:,0];qadr=m.jnt_qposadr[ids]
    fk_error=0.;contacts_error=0.
    forces=[];regions=[]
    for i in range(len(z['time'])):
        d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
        d.time=z['time'][i];mujoco.mj_forward(m,d)
        pts=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{j}').xanchor for j in range(1,21)],d.site('s_tail').xpos])*1000
        fk_error=max(fk_error,float(np.max(abs(pts-z['points_mm'][i]))))
        centers=(pts[:-1]+pts[1:])/2;rel=centers-corner
        ds=np.linalg.norm(np.cross(rel,source),axis=1)
        dt=np.linalg.norm(np.cross(rel,target),axis=1)
        region=(dt+POLICY['region_margin_mm']<ds)&(rel@target>radius)&(dt<radius+29+POLICY['region_clearance_mm'])
        assert np.array_equal(region,z['region'][i]),'region does not match physical geometry'
        regions.append(region)
        assert np.allclose(np.degrees(d.ctrl-d.qpos[qadr]),z['error_deg'][i],atol=1e-8)
        # Geometry/contact membership independently reconstructed; force values
        # can differ without the original solver warmstart and are not compared.
        contact=np.zeros((3,21),bool)
        names=['pipe_straight_01','pipe_straight_03','pipe_unused_branch_02']
        for ct in d.contact:
            for side in (0,1):
                name=m.geom(int(ct.geom[side])).name
                body=m.body(int(m.geom_bodyid[ct.geom[1-side]])).name
                module=0 if body=='head' else int(body[4:]) if body.startswith('link') else None
                if name in names and module is not None:contact[names.index(name),module]=True
        assert not np.any((z['normal_force'][i]>.1)&~contact),'force assigned to wrong pipe/module'
    for command in commands:
        mask=z['applied_id']==command['id']
        assert mask.any() and np.allclose(z['ctrl'][mask],command['target'],atol=1e-12)
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf8'))
    geometry_error=0.
    for command,g in zip(commands,geometry):
        index=int(np.argmin(abs(z['time']-command['time'])))
        assert abs(z['time'][index]-g['sim_time'])<1e-8,'feedback simulation time mismatch'
        geometry_error=max(geometry_error,float(np.max(abs(np.array(g['points_mm'])-z['points_mm'][index]))))
    assert geometry_error<1e-6,'feedback pose differs from physics'
    movement=np.flatnonzero(z['stage']=='rolling')
    progress=(z['points_mm'][:,0]-z['points_mm'][0,0])@target
    eligible=np.array(result['transfer_eligible'])&~z['region'][0]
    runs=np.zeros(21,int);maximum_new=0
    for command in commands:
        index=np.flatnonzero(z['applied_id']==command['id'])[19] # end of 0.2 s rolling step
        runs=np.where(z['region'][index]&eligible,runs+1,0)
        maximum_new=max(maximum_new,int(np.sum(runs>=3)))
    report=dict(folder=folder.name,initial_state_exact=True,command_delivery_checked=True,
        fk_max_error_mm=fk_error,contact_pipe_assignment_checked=True,geometric_regions_checked=True,
        feedback_pose_max_error_mm=geometry_error,
        maximum_sustained_new_modules=maximum_new,peak_head_progress_mm=float(progress[movement].max()),
        final_rolling_progress_mm=float(progress[movement[-1]]),
        passed=result['passed'],outcome=result['outcome'])
    write_json(folder/'audit.json',report)
    return report


def render(folder):
    import cv2
    from PIL import Image,ImageDraw
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    m,d,meta,c=load_checkpoint(input_path(result));z=np.load(folder/'trace.npz')
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    corner,source,target,radius=pipe_frame(m)
    camera.lookat[:]=corner/1000-source*.12+target*.12
    camera.distance=1.6;camera.azimuth=125;camera.elevation=-55
    renderer=mujoco.Renderer(m,height=640,width=960)
    writer=cv2.VideoWriter(str(folder/'replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,640))
    if not writer.isOpened():raise RuntimeError('video writer unavailable')
    try:
        for index in range(0,len(z['time']),5):
            d.qpos[:]=z['qpos'][index];d.qvel[:]=z['qvel'][index];d.ctrl[:]=z['ctrl'][index]
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            text=f"{folder.name} | +{z['time'][index]-z['time'][0]:.2f}s | {z['stage'][index]} | phase {z['phase'][index]:.2f}"
            draw.rectangle((0,0,960,28),fill='black');draw.text((8,8),text,fill='white')
            writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
            if index==0:im.save(folder/'initial.png')
            if index>=len(z['time'])-5:im.save(folder/'final.png')
    finally:writer.release();renderer.close()


def plot(folders,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for folder in folders:
        a=json.loads((folder/'result.json').read_text(encoding='utf8'));z=np.load(folder/'trace.npz')
        m,d,meta,c=load_checkpoint(input_path(a));corner,source,target,radius=pipe_frame(m)
        t=z['time']-z['time'][0];head=(z['points_mm'][:,0]-z['points_mm'][0,0])@target
        label=a['method']+(f" w={a['options']['transition_links']:g}" if a['method']=='segmented' else '')
        axes[0].plot(t,head,label=label)
        axes[1].plot(t,np.sum(z['region']&np.array(a['transfer_eligible'])&~z['region'][0],axis=1),label=label)
        axes[2].plot(t,np.max(abs(z['error_deg']),axis=1),label=label)
    axes[0].axhline(64.5,color='gray',ls='--');axes[1].axhline(2,color='gray',ls='--')
    for ax in axes:ax.grid(alpha=.25);ax.legend(loc='best')
    axes[0].set_ylabel('Head advance along target (mm)')
    axes[1].set_ylabel('New modules in target region\n(instantaneous, not persistence gate)')
    axes[2].set_ylabel('Maximum tracking error (deg)');axes[2].set_xlabel('Physical time since checkpoint (s)')
    fig.suptitle('Post-contact rolling: includes initial switch and final 3 s fixed-target hold')
    fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',type=Path,nargs='+')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    for folder in args.folders:
        print(json.dumps(audit(folder)),flush=True)
        if args.render:render(folder)
    if args.plot:plot(args.folders,args.plot)
