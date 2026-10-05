"""Read-only geometry/actuator audit of the progressive trial; no mj_step."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco

from progressive_winding_experiment import OUT, BASELINE, HORIZONTAL, metrics
from segmented_pole_experiment import write_json, pipe_frame
from spatial_takeover_experiment import points_for_angles
from snake_control.mujoco_checkpoint import load_checkpoint


def diagnose(root=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,d,_,_=load_checkpoint(HORIZONTAL,verify_hashes=False)
    corner,source,axis,radius=pipe_frame(m)
    qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    controls=json.loads((root/'rolling/controls.json').read_text(encoding='utf8'))
    geometry=json.loads((root/'rolling/geometry.json').read_text(encoding='utf8'))
    z=np.load(root/'rolling/trace.npz')
    cfg=geometry[0]['config']
    signs=np.where(np.arange(20)%2,-1.,1.)
    low,high=np.sort(np.radians([[j['minimum_deg'],j['maximum_deg']] for j in cfg['joints']])*signs[:,None],axis=1).T
    rows=[]
    for k,c in enumerate(controls):
        i=k*20
        d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i]
        ids=np.array(c['forming_joint_ids'])
        correction=np.radians(c['forming_delta_deg'])
        raw=np.array(c['raw']);raw[ids]-=correction
        previous=z['ctrl'][i]
        without=np.clip(previous+np.clip(raw-previous,-np.radians(18),np.radians(18)),low+.01,high-.01)
        with_guide=np.array(c['target'])
        forces=[]
        for command in [without,with_guide]:
            d.ctrl[:]=command
            mujoco.mj_forward(m,d)
            forces.append(d.actuator_force.copy())
        q=np.array(c['measured'])
        before=points_for_angles(m,d,q)
        perturbed=q.copy();perturbed[ids]+=correction
        after=points_for_angles(m,d,perturbed)
        ref=c['forming_reference'];ends=np.array(ref['endpoints']);anchor=ref['anchor']
        endpoint_ids=np.arange(anchor+1,anchor+4)
        norm_before=np.linalg.norm(ends-before[endpoint_ids])
        norm_after=np.linalg.norm(ends-after[endpoint_ids])
        delta_force=(forces[1]-forces[0])[ids]
        row=dict(seconds=k*.2, joint_ids=ids, command_change_deg=np.degrees((with_guide-without)[ids]),
            without_force_Nm=forces[0][ids], with_force_Nm=forces[1][ids], delta_force_Nm=delta_force,
            geometry_error_before_mm=float(norm_before), geometry_error_after_mm=float(norm_after),
            geometry_improvement_mm=float(norm_before-norm_after),
            predicted_connector_shift_mm=float(np.linalg.norm(np.array(c['forming_connector_shift_mm']),axis=1).max()),
            source_curve_clearance_mm=float(np.linalg.norm(np.cross(np.array(ref['curve'])-corner,source),axis=1).min()-radius-29))
        if 'forming_raw_before_conditioning' in c:
            conditioning_force=[]
            for key in ['forming_raw_before_conditioning','forming_raw_after_conditioning']:
                d.ctrl[:]=c[key];mujoco.mj_forward(m,d)
                conditioning_force.append(d.actuator_force.copy())
            row['conditioning_only_instantaneous_force_error_Nm']=float(abs(conditioning_force[1]-conditioning_force[0]).max())
        rows.append(row)
    meaningful=[(abs(np.array(row['command_change_deg']))>.05) for row in rows]
    unchanged=[(abs(row['delta_force_Nm'])<1e-8) & mask for row,mask in zip(rows,meaningful)]
    summary=dict(
        frame_count=len(rows),
        meaningful_joint_command_changes=int(sum(mask.sum() for mask in meaningful)),
        changed_commands_with_identical_instantaneous_force=int(sum(mask.sum() for mask in unchanged)),
        force_unchanged_fraction=float(sum(mask.sum() for mask in unchanged)/sum(mask.sum() for mask in meaningful)),
        unchanged_force_opposed_by_guidance=int(sum(np.sum(mask & (np.array(row['command_change_deg'])*row['without_force_Nm']<0)) for mask,row in zip(unchanged,rows))),
        geometry_improvement_fraction=float(np.mean([row['geometry_improvement_mm']>0 for row in rows])),
        mean_predicted_geometry_improvement_mm=float(np.mean([row['geometry_improvement_mm'] for row in rows])),
        min_reference_source_clearance_mm=min(row['source_curve_clearance_mm'] for row in rows),
        limitation='fixed observed qpos/qvel one-frame actuator counterfactual, not a second dynamic run; shell cylinder clearance is an approximation')
    if 'conditioning_only_instantaneous_force_error_Nm' in rows[0]:
        summary['conditioning_only_max_force_error_Nm']=max(row['conditioning_only_instantaneous_force_error_Nm'] for row in rows)
        assert summary['conditioning_only_max_force_error_Nm']<1e-9
    write_json(root/'diagnosis.json',dict(summary=summary,frames=rows))
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,k in zip(axes,[0,len(controls)-1]):
        c=controls[k];p=z['points_mm'][k*20]-corner
        curve=np.array(c['forming_reference']['curve'])-corner
        ends=np.array(c['forming_reference']['endpoints'])-corner
        # Looking along the source axis: target axis runs horizontally here.
        ax.axhspan(-radius,radius,color='gray',alpha=.2,label='target pipe silhouette')
        ax.plot(p[:,1],p[:,2],'o-',color='gray',alpha=.6,label='actual body')
        ax.plot(p[5:9,1],p[5:9,2],'o-',color='darkorange',lw=3,label='existing edge + forming modules')
        ax.plot(curve[:,1],curve[:,2],'--',color='purple',label='helix continuation objective')
        ax.scatter(ends[:,1],ends[:,2],marker='x',c='purple',s=80)
        for j in range(5,9): ax.annotate(f'P{j}',(p[j,1],p[j,2]))
        ax.set_title(f'{k*.2:.1f}s: side projection (depth omitted)')
        ax.set_xlabel('target axis (mm)');ax.set_ylabel('height relative to corner (mm)')
        ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'formation_geometry.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUT)
    diagnose(parser.parse_args().output_root)
