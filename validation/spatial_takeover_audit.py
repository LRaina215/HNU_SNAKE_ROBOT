"""Audit saved relay/physics data and plot spatial-takeover controls."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse

import mujoco
import numpy as np

from spatial_takeover_experiment import OUT, HORIZONTAL, load_checkpoint, write_json, sha
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone


def main():
    reports=[]
    saved=np.load(HORIZONTAL/'state.npz')['integration_state']
    for folder in sorted(OUT.iterdir()):
        if not (folder/'result.json').exists():continue
        r=json.loads((folder/'result.json').read_text(encoding='utf8'))
        a=json.loads((folder/'analysis.json').read_text(encoding='utf8'))
        z=np.load(folder/'trace.npz');commands=json.loads((folder/'commands.json').read_text(encoding='utf8'))
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        feedback=json.loads((folder/'geometry.json').read_text(encoding='utf8'))
        assert np.array_equal(np.load(folder/'initial_state.npz')['state'],saved)
        assert len(z['time'])==r['steps']*20+301 and len(commands)==r['steps']
        assert all(x['applied'] for x in commands)
        assert np.max(abs(np.diff(z['time'])-.01))<1e-10
        assert np.all(z['boundary_links']==6)
        assert all(sha(folder/'sources'/p)==h for p,h in r['source_hashes'].items())
        m,d,meta,c=load_checkpoint(HORIZONTAL);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        links=meta['settings']['links'];signs=np.where(np.arange(20)%2,-1.,1.)
        fk_error=0.;feedback_error=0.;command_error=0.;field_error=0.;tail_floor=False
        # All physical states, including hold, independently inspected for floor contact.
        for i in range(len(z['time'])):
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
            mujoco.mj_forward(m,d)
            points=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{j}').xanchor for j in range(1,21)],d.site('s_tail').xpos])*1000
            fk_error=max(fk_error,float(abs(points-z['points_mm'][i]).max()))
            for ct in d.contact:
                names=[m.geom(int(x)).name for x in ct.geom]
                bodies=[m.body(int(m.geom_bodyid[x])).name for x in ct.geom]
                if 'floor' in names and 'tail' in bodies:tail_floor=True
        for k,(command,control,g) in enumerate(zip(commands,controls,feedback)):
            command_error=max(command_error,float(abs(z['ctrl'][k*20+1]-command['target']).max()))
            feedback_error=max(feedback_error,float(abs(np.radians(g['angles_deg'])*signs-z['qpos'][k*20,qadr]).max()))
            assert np.max(abs(np.asarray(g['points_mm'])-z['points_mm'][k*20]))<1e-6
            # Only the spans are needed for independent integration of saved field.
            bone=SpatialBishopBackbone(np.asarray(g['points_mm']),links)
            phase=control['phase']-meta['settings']['phase'] if folder.name.startswith('captured') else 0.
            field_error=max(field_error,float(abs(bone.joint_angles(phase,control['field'])-control['raw']).max()))
        assert fk_error<1e-8 and feedback_error<1e-10 and command_error<1e-12 and field_error<1e-10
        assert not tail_floor,'tail floor contact omitted by historical base logger'
        a.update(condition=folder.name,source_hashes=r['source_hashes'],fk_error_mm=fk_error,
            feedback_error_rad=feedback_error,command_error_rad=command_error,field_error_rad=field_error,
            initial_state_identical=True,tail_floor_contact=tail_floor,clipped_joint_steps=r['clipped_joint_steps'])
        write_json(folder/'audit.json',a);reports.append(a)
    write_json(OUT/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for mode in ['captured_hold','feedback_hold','captured_roll','feedback_roll','force_roll']:
        if not (OUT/mode/'trace.npz').exists():continue
        z=np.load(OUT/mode/'trace.npz');t=z['time']-z['time'][0]
        col=int(mode.endswith('roll'));centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        axes[0,col].plot(t,np.mean(centers[:,:6,1]-centers[0,:6,1],axis=1),label=mode)
        axes[1,col].plot(t,z['normal_force'][:,1,:6].sum(axis=1),label=mode)
    for col in range(2):
        for row in range(2):
            axes[row,col].axvline(1.6 if col else 6.4,color='gray',ls='--')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
        axes[0,col].set_title('8 rolling steps + hold' if col else 'Fixed phase + hold')
        axes[1,col].set_xlabel('Physical time (s)')
    axes[0,0].set_ylabel('Front mean target-axis progress (mm)')
    axes[1,0].set_ylabel('Front target normal force sum (N)')
    fig.suptitle('Spatial takeover: contact retained, rolling hold gate still fails')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print('Audited '+str(len(reports))+' runs; all relay/FK/field/source checks passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_root.resolve()
    main()
