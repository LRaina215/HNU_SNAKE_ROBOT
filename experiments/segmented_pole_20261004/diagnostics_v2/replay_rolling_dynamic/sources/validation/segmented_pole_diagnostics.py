"""Causal diagnostics for post-contact support loss; no gait/physics retuning.

Paired controls: original target hold, frozen geometric reconstruction at fixed
phase, live geometry at fixed phase, live rolling with fixed boundary, and exact
command replay of the previous rolling/dynamic-boundary run. Same horizontal
checkpoint, width=6 (the video discussed), 32 x 0.2 s then 3 s target hold.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from segmented_pole_experiment import (BASE, ROOT, HORIZONTAL, Trial, MovingAxisBackbone,
    SwitchContinuity, smoothstep, write_json, sha)

OUT=BASE/'diagnostics_v2'
REFERENCE=BASE/'horizontal/final_segmented_w6'
CONDITIONS=['target_hold','frozen_shape_fixed_phase','live_shape_fixed_phase',
            'rolling_fixed_boundary','replay_rolling_dynamic']


class DiagnosticTrial(Trial):
    def __init__(self,folder):
        self.contact_rows=[];self.spins=[];self.jacobian_error=None
        super().__init__(HORIZONTAL,folder)
        dest=self.folder/'sources/validation/segmented_pole_diagnostics.py'
        shutil.copy2(Path(__file__),dest)

    def snapshot(self):
        row=super().snapshot()
        m,d=self.m,self.scratch
        assert m.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC
        jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp)
        spins=[]
        for module in range(21):
            body=m.body('head' if module==0 else 'tail' if module==20 else f'link{module}').id
            mujoco.mj_jac(m,d,None,jacr,d.xpos[body],body)
            omega=jacr@d.qvel
            spins.append(float(omega@d.xmat[body].reshape(3,3)[:,2]))
        self.spins.append(spins)
        names=['pipe_straight_01','pipe_straight_03','pipe_unused_branch_02']
        for k,ct in enumerate(d.contact):
            for side in (0,1):
                name=m.geom(int(ct.geom[side])).name
                body=int(m.geom_bodyid[ct.geom[1-side]])
                label=m.body(body).name
                module=0 if label=='head' else 20 if label=='tail' else int(label[4:]) if label.startswith('link') else None
                if name not in names or module is None:continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
                if force[0]<=.1:continue
                pipe=names.index(name);axis=self.target if pipe==1 else self.source
                if module==20:row['normal_force'][pipe,module]+=force[0]
                # Pipes are static. World-point Jacobian gives robot surface
                # velocity, not joint speed or motion of contact-index labels.
                mujoco.mj_jac(m,d,jacp,None,ct.pos,body)
                velocity=jacp@d.qvel
                frame=ct.frame.reshape(3,3)
                local=frame@velocity
                radius=ct.pos-self.corner/1000
                radius-=axis*(radius@axis)
                circum=np.cross(axis,radius/np.linalg.norm(radius))
                ratio=np.linalg.norm(force[1:ct.dim]/ct.friction[:ct.dim-1])/force[0]
                self.contact_rows.append([d.time,pipe,module,force[0],
                    np.linalg.norm(local[1:])*1000,velocity@axis*1000,velocity@circum*1000,ratio])
                if self.jacobian_error is None:
                    self.jacobian_error=self.check_jacobian(body,ct.pos,velocity)
        return row

    def check_jacobian(self,body,point,expected):
        m,d=self.m,self.scratch;local=d.xmat[body].reshape(3,3).T@(point-d.xpos[body])
        positions=[];eps=1e-6
        for sign in (-1,1):
            check=mujoco.MjData(m);check.qpos[:]=d.qpos
            mujoco.mj_integratePos(m,check.qpos,d.qvel,sign*eps)
            mujoco.mj_kinematics(m,check)
            positions.append(check.xpos[body]+check.xmat[body].reshape(3,3)@local)
        error=float(np.max(abs((positions[1]-positions[0])/(2*eps)-expected)))
        assert error<1e-6,error
        return error

    def close(self):
        super().close()
        np.savez_compressed(self.folder/'contact_motion.npz',
            contacts=np.asarray(self.contact_rows).reshape(-1,8),body_spin_rad_s=np.asarray(self.spins))


def run_diagnostic(condition):
    folder=OUT/condition;sim=None
    try:
        sim=DiagnosticTrial(folder)
        m,d=sim.m,sim.d;settings=sim.meta['settings'];links=np.asarray(settings['links'])
        phase0=sim.phase;initial_ctrl=d.ctrl.copy()
        initial_g=copy.deepcopy(sim.meta['geometry_at_capture'])
        frozen=MovingAxisBackbone(np.asarray(initial_g['points_mm']),links,sim.original.parameters,
                                  sim.source,sim.target,6*links.mean(),6*links.mean(),initial_g['curve']['gamma_min_rad'])
        switch=SwitchContinuity(frozen,initial_ctrl,phase0)
        commands=json.loads((REFERENCE/'commands.json').read_text(encoding='utf8'))
        for step in range(32):
            sim.stage='rolling' # retained trace schema; result identifies fixed-phase controls
            g=sim.fresh()
            cg=initial_g if step==0 else g
            phase=phase0-(step+1)*settings['phase_step'] if condition.startswith('rolling') else phase0
            weight=float(smoothstep((step+1)/6))
            if condition=='target_hold':raw=initial_ctrl.copy()
            elif condition=='replay_rolling_dynamic':
                raw=np.asarray(commands[step]['target']);phase=commands[step]['phase']
            else:
                bone=frozen if condition=='frozen_shape_fixed_phase' else MovingAxisBackbone(
                    np.asarray(cg['points_mm']),links,sim.original.parameters,sim.source,sim.target,
                    6*links.mean(),6*links.mean(),cg['curve']['gamma_min_rad'])
                raw=switch.angles(bone,phase,weight)
            # Exact original ctrl is retained in hold/replay; same original gait
            # angle policy for freshly reconstructed commands.
            target=raw if condition in ('target_hold','replay_rolling_dynamic') else np.clip(
                raw,m.jnt_range[sim.jids,0]+.01,m.jnt_range[sim.jids,1]-.01)
            sim.phase=phase;sim.geometry_log.append(cg)
            sim.controls.append(dict(step=step+1,condition=condition,phase=phase,weight=weight,
                                     raw=raw.tolist(),target=target.tolist()))
            sim.command(target,g);sim.advance(.2)
        sim.stage='hold';sim.advance(3.)
        result=dict(condition=condition,method='segmented',options=dict(transition_links=6),
                    checkpoint=str(HORIZONTAL),steps=32,phase0=phase0,final_phase=sim.phase,
                    boundary_links=6.,settings=settings,jacobian_error_m_s=sim.jacobian_error,
                    source_sha256=sha(Path(__file__)),reference_sha256=sha(REFERENCE/'commands.json'),
                    initial_state_sha256=hashlib.sha256(sim.initial_state.tobytes()).hexdigest(),
                    contacts_columns=['sim_time','pipe_index','module_index','normal_N','tangent_speed_mm_s',
                                      'axial_velocity_mm_s','circumferential_velocity_mm_s','full_cone_utilization'],
                    checkpoint_sha256=sha(HORIZONTAL/'checkpoint.json'))
        if condition=='replay_rolling_dynamic':
            old=np.load(REFERENCE/'trace.npz')
            errors={k:float(np.max(abs(np.asarray([r[k] for r in sim.records])-old[k]))) for k in ['time','qpos','qvel','ctrl']}
            errors['non_tail_normal_force']=float(np.max(abs(np.array([r['normal_force'] for r in sim.records])[:,:,:20]-old['normal_force'][:,:,:20])))
            result['tail_contact_max_N']=float(np.max([r['normal_force'][:,20].max() for r in sim.records]))
            result['replay_errors']=errors
            assert max(errors.values())<1e-9,errors
        write_json(folder/'result.json',result)
    finally:
        if sim:sim.close()


def analyze():
    reports=[]
    for condition in CONDITIONS:
        folder=OUT/condition;r=json.loads((folder/'result.json').read_text(encoding='utf8'))
        z=np.load(folder/'trace.npz');v=np.load(folder/'contact_motion.npz');contacts=v['contacts']
        centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        t=z['time']-z['time'][0]
        # Source +X, target +Y in this specific horizontal checkpoint; assert via model.
        from segmented_pole_experiment import load_checkpoint,pipe_frame
        m,d,meta,c=load_checkpoint(HORIZONTAL);corner,source,target,radius=pipe_frame(m)
        intervals=[]
        for lo,hi in [(0,1.2),(1.2,3.),(3.,6.4),(6.4,9.4)]:
            mask=(t>=lo-1e-8)&(t<=hi+1e-8);ids=np.flatnonzero(mask);first,last=ids[0],ids[-1]
            row=dict(start_s=lo,end_s=hi)
            for label,mods,axis,pipe in [('front',np.arange(6),target,1),('rear',np.arange(10,21),source,0)]:
                normal=z['normal_force'][mask,pipe][:,mods]
                relevant=contacts[(contacts[:,0]>=z['time'][first]-1e-8)&(contacts[:,0]<=z['time'][last]+1e-8)&
                                  (contacts[:,1]==pipe)&np.isin(contacts[:,2],mods)]
                delta=centers[:,mods]-corner
                # Angular displacement of module centres around the pipe axis.
                e1=np.array([0.,0.,1.]);e2=np.cross(axis,e1)
                angles=np.unwrap(np.arctan2(delta@e2,delta@e1),axis=0)
                row[label]=dict(axial_net_mm=float(np.mean((centers[last,mods]-centers[first,mods])@axis)),
                    orbital_angle_net_deg=np.degrees(angles[last]-angles[first]).tolist(),
                    body_spin_net_deg=np.degrees(np.trapz(v['body_spin_rad_s'][mask][:,mods],t[mask],axis=0)).tolist(),
                    contact_count_mean=float(np.mean((normal>.1).sum(axis=1))),
                    no_contact_fraction=float(np.mean(~np.any(normal>.1,axis=1))),
                    normal_sum_mean_N=float(normal.sum(axis=1).mean()),
                    tangent_slip_median_mm_s=float(np.median(relevant[:,4])) if len(relevant) else None,
                    cone_utilization_median=float(np.median(relevant[:,7])) if len(relevant) else None,
                    near_cone_limit_fraction=float(np.mean(relevant[:,7]>.9)) if len(relevant) else None)
            intervals.append(row)
        root_move=np.linalg.norm(z['qpos'][:,:3]-z['qpos'][0,:3],axis=1)*1000
        result=dict(condition=condition,intervals=intervals,root_max_motion_mm=float(root_move.max()),
            initial_state_sha256=r['initial_state_sha256'],jacobian_error_m_s=r['jacobian_error_m_s'],
            final_source_contact_modules=np.flatnonzero(z['normal_force'][-1,0]>.1).tolist(),
            final_target_contact_modules=np.flatnonzero(z['normal_force'][-1,1]>.1).tolist(),
            replay_errors=r.get('replay_errors'))
        write_json(folder/'analysis.json',result);reports.append(result)
    assert len({r['initial_state_sha256'] for r in reports})==1
    write_json(OUT/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for condition in CONDITIONS:
        z=np.load(OUT/condition/'trace.npz');t=z['time']-z['time'][0]
        axes[0].plot(t,z['normal_force'][:,1,:6].sum(axis=1),label=condition)
        centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
        axes[1].plot(t,np.mean((centers[:,:6]-centers[0,:6])@target,axis=1))
        axes[2].plot(t,np.linalg.norm(z['qpos'][:,:3]-z['qpos'][0,:3],axis=1)*1000)
    axes[0].set_ylabel('Front target normal force sum (N)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Front mean axial displacement (mm)');axes[2].set_ylabel('Root displacement (mm)')
    axes[2].set_xlabel('Physical seconds from saved checkpoint')
    for ax in axes:ax.axvline(1.2,color='gray',ls=':');ax.axvline(6.4,color='gray',ls='--');ax.grid(alpha=.2)
    fig.suptitle('Width 6 diagnostic controls: switch ends at 1.2 s; final hold starts at 6.4 s')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=CONDITIONS)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.analyze:analyze()
    elif args.condition:run_diagnostic(args.condition)
    else:parser.error('choose --condition or --analyze')
