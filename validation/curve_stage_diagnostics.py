"""Actual stage records and conditional-axis diagnostics; no physical steps."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=ROOT/'experiments/curve_guided_transfer_20261006/stage_continuation_20261006'
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'


def main():
    summary={}
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    for case in ['R1_C0','R2_C0']:
        b=BASE/case;z=np.load(b/'trace.npz')
        controls=json.loads((b/'control.json').read_text(encoding='utf8'))
        commands=json.loads((b/'commands.json').read_text(encoding='utf8'))
        t=z['sim_time'];p=z['centers_mm'];speed=np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None,None],axis=2)
        counts=np.count_nonzero(z['contact_force_N']>.1,axis=2)
        ct=np.array([r['time_s'] for r in controls]);h=np.degrees([r['retained_shape_rad'] for r in controls])
        delta=np.degrees([r['compliance_delta_rad'] for r in controls]);nominal=np.array([r['nominal_rad'] for r in controls])
        ack=np.array([r['target_rad'] for r in commands]);cloud=np.array([[r['cloud_rms_mm'] for r in c['material_axes']] for c in controls])
        centroid=p.mean(axis=1);drift=np.linalg.norm(centroid-centroid[0],axis=1)
        axes=[]
        for moment in [0.,5.,20.,float(ct[-1])]:
            ci=int(abs(ct-moment).argmin());i=int(abs(t-ct[ci]).argmin())
            data.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(model,data)
            R=data.body('head').xmat.reshape(3,3)
            for a, world, side in zip(controls[ci]['material_axes'],[np.array([1.,0,0]),np.array([0.,1,0])],['source','target']):
                angle=float(np.degrees(np.arccos(np.clip(np.dot(a['direction_head'],R.T@world),-1,1))))
                axes.append(dict(time_s=float(ct[ci]),side=side,axis_error_deg=angle,cloud_rms_mm=a['cloud_rms_mm'],
                    truth_use='OFFLINE DIAGNOSTIC ONLY; not an online controller field'))
        metrics=dict(actual_seconds=float(t[-1]),actual_frames=len(t),actual_commands=len(commands),
            all_commands_applied=all(c['applied'] for c in commands),
            state_ack_identity_error_rad=float(abs(ack-nominal-np.radians(h)).max()),
            final_body_centroid_displacement_mm=float(drift[-1]),
            final_root_rotation_deg=float(np.degrees((Rotation.from_quat(z['qpos'][-1,[4,5,6,3]])*
                Rotation.from_quat(z['qpos'][0,[4,5,6,3]]).inv()).magnitude())),
            after5_minimum_contacts_per_pipe=counts[t>=5-1e-8].min(axis=0).tolist(),
            maximum_retained_state_deg=float(abs(h).max()),
            last_retained_state_deg=h[-1].tolist(),late_angle_constraints=sum(c['angle_constrained_joints']>0 for c in controls if c['time_s']>=35.25),
            late_rate_constraints=sum(c['rate_limited_joints']>0 for c in controls if c['time_s']>=35.25),
            ground_frames=int(np.count_nonzero(z['ground'])),self_contact_frames=int(np.count_nonzero(z['self_contact'])),
            added_external_force_maximum=float(z['external_applied_force_maximum'].max()),
            added_generalized_force_maximum=float(z['generalized_applied_force_maximum'].max()),
            conditional_axis_diagnostic=axes,physics_steps_added=0)
        (b/'diagnostic_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf8')
        summary[case]=metrics
        fig,ax=plt.subplots(3,2,figsize=(11,9),sharex=True)
        ax[0,0].plot(t,counts[:,0],label='Source');ax[0,0].plot(t,counts[:,1],label='Target')
        ax[0,0].axhline(2,color='k',ls=':');ax[0,0].set_ylabel('Contact modules (>0.1 N)');ax[0,0].legend()
        ax[0,1].plot(t[1:],speed.max(axis=1));ax[0,1].set_ylabel('Maximum module speed (mm/s)')
        ax[1,0].plot(t,drift,label='Geometric body centroid')
        ax[1,0].plot(t,np.linalg.norm(z['qpos'][:,:3]-z['qpos'][0,:3],axis=1)*1000,label='Head root')
        ax[1,0].set_ylabel('Displacement (mm)');ax[1,0].legend()
        ax[1,1].plot(t,np.degrees(abs(z['q']-z['ctrl'])).max(axis=1));ax[1,1].set_ylabel('Maximum tracking error (deg)')
        ax[2,0].plot(ct,abs(h).max(axis=1),label='Executable retained state')
        ax[2,0].plot(ct,abs(delta).max(axis=1),label='A-D difference');ax[2,0].set_ylabel('Maximum adjustment (deg)');ax[2,0].legend()
        ax[2,1].plot(ct,cloud[:,0],label='Source cloud RMS');ax[2,1].plot(ct,cloud[:,1],label='Target cloud RMS')
        ax[2,1].axhline(16.125,color='r',ls=':');ax[2,1].set_ylabel('Conditional material gauge RMS (mm)');ax[2,1].legend()
        for a in ax.flat:a.grid(alpha=.2);a.axvline(5,color='.5',ls='--')
        ax[2,0].set_xlabel('Physical time (s)');ax[2,1].set_xlabel('Physical time (s)')
        fig.suptitle(case+' actual physics; geometry-derived axes remain conditional')
        fig.tight_layout();fig.savefig(b/'actual_diagnostics.png',dpi=140);plt.close(fig)
    (BASE/'holding_diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{f:v[f] for f in ['actual_seconds','final_body_centroid_displacement_mm','after5_minimum_contacts_per_pipe','late_angle_constraints','late_rate_constraints']} for k,v in summary.items()}),flush=True)


if __name__=='__main__':main()
