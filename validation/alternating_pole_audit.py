"""Audit matched phase budgets, physical feedback, target field and holding motion."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from alternating_pole_experiment import OUT, MODES, HORIZONTAL, load_checkpoint, write_json, sha
from alternating_pole_phase import LocalPhaseCommand, schedule
from spatial_bishop_backbone import SpatialBishopBackbone
from segmented_pole_experiment import pipe_frame


def audit(folder):
    r=json.loads((folder/'result.json').read_text());a=json.loads((folder/'analysis.json').read_text())
    z=np.load(folder/'trace.npz');controls=json.loads((folder/'controls.json').read_text())
    feedback=json.loads((folder/'geometry.json').read_text());commands=json.loads((folder/'commands.json').read_text())
    stored=np.load(HORIZONTAL/'state.npz')['integration_state']
    assert np.array_equal(stored,np.load(folder/'initial_state.npz')['state'])
    assert all(sha(folder/'sources'/name)==value for name,value in r['source_hashes'].items())
    assert len(z['time'])==r['steps']*20+301 and len(commands)==r['steps']
    assert all(x['applied'] for x in commands)
    assert np.max(abs(np.diff(z['time'])-.01))<1e-10
    assert np.all(z['boundary_links']==6)
    m,d,meta,c=load_checkpoint(HORIZONTAL);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    links=np.asarray(meta['settings']['links']);signs=np.where(np.arange(20)%2,-1.,1.)
    fk_error=feedback_error=field_error=delivery_error=0.;ground=False
    for i in range(len(z['time'])):
        d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
        mujoco.mj_forward(m,d)
        p=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{j}').xanchor for j in range(1,21)],d.site('s_tail').xpos])*1000
        fk_error=max(fk_error,float(abs(p-z['points_mm'][i]).max()))
        for ct in d.contact:
            names=[m.geom(int(g)).name for g in ct.geom]
            bodies=[m.body(int(m.geom_bodyid[g])).name for g in ct.geom]
            ground |= 'floor' in names and any(x in ('head','tail') or x.startswith('link') for x in bodies)
    previous=np.zeros(2)
    for k,(x,g,command) in enumerate(zip(controls,feedback,commands)):
        total,delta=schedule(r['condition'],k,meta['settings']['phase_step'])
        assert np.max(abs(np.asarray(x['total'])-total))<1e-12 and np.max(abs(np.asarray(x['delta'])-delta))<1e-12
        assert np.max(abs(total-previous-delta))<1e-12;previous=total
        assert np.max(abs(z['local_phase_total'][k*20+1:(k+1)*20+1]-total))<1e-12
        assert abs(total[0]-total[1])<=.15+1e-12
        if k%2:assert abs(total[0]-total[1])<1e-12
        feedback_error=max(feedback_error,float(abs(np.radians(g['angles_deg'])*signs-z['qpos'][k*20,qadr]).max()))
        assert np.max(abs(np.asarray(g['points_mm'])-z['points_mm'][k*20]))<1e-6
        bone=SpatialBishopBackbone(np.asarray(g['points_mm']),links)
        field=LocalPhaseCommand(bone.bounds,x['observed_with_compliance'],x['preload'],total,delta,6*links.mean(),6*links.mean())
        field_error=max(field_error,float(abs(field.joint_angles(links,order=24)-x['raw']).max()))
        expected=np.clip(x['raw'],m.jnt_range[m.actuator_trnid[:,0],0]+.01,m.jnt_range[m.actuator_trnid[:,0],1]-.01)
        assert np.max(abs(expected-x['target']))<1e-12
        delivery_error=max(delivery_error,float(abs(z['ctrl'][k*20+1]-command['target']).max()))
    assert np.max(abs(previous+r['cycles']*.15))<1e-12
    assert fk_error<1e-8 and feedback_error<1e-10 and field_error<1e-10 and delivery_error<1e-12
    assert ground==a['ground']
    stop=r['steps']*20;target=pipe_frame(m)[2]
    head=z['points_mm'][:,0];hproj=head@target
    worst=stop+np.argmin(hproj[stop:]);centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    c0=centers[stop,:6].mean(axis=0);c1=centers[worst,:6].mean(axis=0)
    aa=centers[stop,:6]-c0;bb=centers[worst,:6]-c1
    u,s,vt=np.linalg.svd(aa.T@bb);diag=np.eye(3);diag[-1,-1]=np.linalg.det(u@vt);rot=u@diag@vt
    predicted=(head[stop]-c0)@rot+c1
    rigid=dict(rotation_deg=float(np.degrees(Rotation.from_matrix(rot).magnitude())),
        head_prediction_error_mm=float(np.linalg.norm(head[worst]-predicted)),
        head_nonrigid_axial_retreat_mm=float((predicted-head[worst])@target),
        head_rigid_rotation_axial_retreat_mm=float(((head[stop]-c0)-(head[stop]-c0)@rot)@target),
        centroid_axial_retreat_mm=float((c0-c1)@target))
    assert abs(rigid['head_nonrigid_axial_retreat_mm']+rigid['head_rigid_rotation_axial_retreat_mm']+
               rigid['centroid_axial_retreat_mm']-a['hold_head_retreat_mm'])<1e-8
    a.update(fk_error_mm=fk_error,feedback_error_rad=feedback_error,field_quadrature_error_rad=field_error,
        delivered_target_error_rad=delivery_error,initial_state_identical=True,matched_phase_budget=True,
        clipped_joint_steps=r['clipped_joint_steps'],hold_rigid_fit=rigid,
        audit_source_sha256=sha(Path(__file__)))
    write_json(folder/'audit.json',a)
    return a


def main():
    reports=[audit(OUT/f'{mode}_c8') for mode in MODES]
    extension=any(r['short_gate'] for r in reports[1:])
    write_json(OUT/'summary.json',reports)
    write_json(OUT/'extension_gate.json',dict(allowed=extension,
        reason='an alternating short gate passed' if extension else 'both alternating conditions failed holding; no 16-cycle or migration trials'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for mode in MODES:
        z=np.load(OUT/f'{mode}_c8'/'trace.npz');t=z['time']-z['time'][0]
        centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        axes[0,0].plot(t,np.mean(centers[:,:6,1]-centers[0,:6,1],axis=1),label=mode)
        axes[0,1].plot(t,z['points_mm'][:,0,1]-z['points_mm'][0,0,1],label=mode)
        axes[1,0].plot(t,z['normal_force'][:,1,:6].sum(axis=1),label=mode)
        axes[1,1].plot(t,np.linalg.norm(centers[:,9]-centers[:,3],axis=1),label=mode)
    for ax,title in zip(axes.flat,['Front mean axial progress (mm)','Head axial progress (mm)',
                                  'Front target normal force sum (N)','Bridge endpoint separation (mm)']):
        ax.set_title(title);ax.axvline(3.2,color='gray',ls='--');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Physical seconds from identical checkpoint')
    fig.suptitle('8 cycles, matched time and phase: all final hold gates fail')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps([{k:r[k] for k in ['condition','short_gate','hold_rigid_fit']} for r in reports]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_root.resolve();main()
