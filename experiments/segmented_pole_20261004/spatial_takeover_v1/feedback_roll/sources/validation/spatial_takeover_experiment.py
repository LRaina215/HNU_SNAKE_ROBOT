"""Bounded horizontal-checkpoint validation of spatial shape takeover.

Fixed boundary, same physics/K/phase step. First run hold controls (6.4+3 s).
Only after feedback hold passes, run eight rolling steps and a 3 s hold.
Frozen captured-field rolling is a mechanism control, not adaptive locomotion.
"""
import argparse
import json
import shutil

import mujoco
import numpy as np

from segmented_pole_experiment import (BASE, ROOT, HORIZONTAL, load_checkpoint,
    pipe_frame, local_sections, write_json, sha, MovingAxisBackbone)
from segmented_pole_backbone import AxisField
from segmented_pole_diagnostics import DiagnosticTrial
from spatial_bishop_backbone import SpatialBishopBackbone, SpatialShapeController

OUT = BASE/'spatial_takeover_v1'
EXTRA = ['codes/spatial_bishop_backbone.py', 'validation/spatial_takeover_experiment.py']


def points_for_angles(m, d, angles):
    scratch = mujoco.MjData(m);scratch.qpos[:] = d.qpos
    scratch.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]] = angles
    mujoco.mj_kinematics(m, scratch)
    return np.vstack([scratch.site('s_head').xpos,
        *[scratch.joint(f'joint{i}').xanchor for i in range(1, 21)], scratch.site('s_tail').xpos])*1000


def checks():
    m, d, meta, original = load_checkpoint(HORIZONTAL)
    links = np.asarray(meta['settings']['links']);adr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    rng = np.random.default_rng(104)
    errors=[];gauge=[];quadrature=[]
    for q in [d.ctrl, d.qpos[adr], np.zeros(20), *rng.uniform(-1.4, 1.4, (12, 20))]:
        bone = SpatialBishopBackbone(points_for_angles(m, d, q), links)
        field, rotation = bone.material_field(q)
        errors.append(float(np.max(abs(bone.joint_angles(field=field)-q))))
        a=np.asarray(bone.sample()['points_mm'])
        b=np.asarray(bone.sample(field=field, material_rotation=rotation)['points_mm'])
        gauge.append(float(np.max(abs(a-b))))
        # Independent interval quadrature at a nonzero material phase.
        phase=.37;nodes=np.r_[0., np.cumsum(links)];manual=[]
        for j,(lo,hi) in enumerate(zip(nodes[:-2], nodes[2:])):
            split=np.unique(np.r_[lo,bone.bounds[(bone.bounds>lo)&(bone.bounds<hi)],hi])
            total=np.zeros(2)
            for left,right in zip(split[:-1],split[1:]):
                index=np.searchsorted(bone.bounds,(left+right)/2)-1
                total+=field[index]*(right-left)
            co,sn=np.cos(phase),np.sin(phase)
            manual.append(-(total[0]*co-total[1]*sn if j%2 else total[0]*sn+total[1]*co))
        quadrature.append(float(np.max(abs(manual-bone.joint_angles(phase,field)))))
    assert max(errors)<1e-10 and max(gauge)<1e-8 and max(quadrature)<1e-12
    g=meta['geometry_at_capture'];p=np.array(g['points_mm']);corner,source,target,_=pipe_frame(m)
    old=MovingAxisBackbone(p,links,original.parameters,source,target,6*links.mean(),6*links.mean())
    phase=meta['settings']['phase'];off=old.align_phase(d.ctrl,phase)
    axes=old.axis_field.frames(np.cumsum(links)-links/2)[0][:,:,2]
    tangents=np.diff(p,axis=0)/links[:,None]
    report=dict(fk_to_yamada_error_rad=max(errors),gauge_reconstruction_error_mm=max(gauge),
        independent_quadrature_error_rad=max(quadrature),cases=len(errors),
        initial_old_target_gap_deg=np.degrees(old.joint_angles(phase+off)-d.ctrl),
        measured_link_pitch_deg=np.degrees(np.arcsin(np.clip((axes*tangents).sum(axis=1),-1,1))),
        nominal_pitch_deg=np.degrees(original.parameters.pitch_angle))
    write_json(OUT/'geometry_checks.json',report)
    return report


def assess(folder, rolling):
    z=np.load(folder/'trace.npz');v=np.load(folder/'contact_motion.npz')
    m,d,meta,original=load_checkpoint(HORIZONTAL);corner,source,target,_=pipe_frame(m)
    t=z['time']-z['time'][0];end=1.6 if rolling else 6.4
    hold=np.flatnonzero(t>=end-1e-8);move=np.flatnonzero(t<=end+1e-8)
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    front=(centers[:,:6]@target).mean(axis=1);rear=(centers[:,10:]@source).mean(axis=1)
    head=z['points_mm'][:,0]@target
    front_contact=np.any(z['normal_force'][:,1,:6]>.1,axis=1)
    source_contact=np.any(z['normal_force'][:,0]>.1,axis=1)
    root=np.linalg.norm(z['qpos'][:,:3]-z['qpos'][0,:3],axis=1)*1000
    hold_root=np.linalg.norm(z['qpos'][hold,:3]-z['qpos'][hold[0],:3],axis=1)*1000
    retreat=float(max(0.,head[hold[0]]-head[hold].min()))
    static_pass=bool(root.max()<=20 and max(0.,head[0]-head.min())<=10 and
                     front_contact.all() and source_contact.all() and not z['ground'].any())
    final_hold_pass=bool(hold_root.max()<=20 and retreat<=10 and front_contact[hold].all() and
                         source_contact[hold].all() and not z['ground'].any())
    result=dict(root_max_motion_mm=float(root.max()),front_move_mm=float(front[move[-1]]-front[0]),
        rear_move_mm=float(rear[move[-1]]-rear[0]),hold_root_motion_mm=float(hold_root.max()),
        hold_head_retreat_mm=retreat,front_contact_fraction=float(front_contact.mean()),
        hold_front_contact_fraction=float(front_contact[hold].mean()),static_gate=static_pass,
        final_hold_gate=final_hold_pass,ground=bool(z['ground'].any()),
        max_tracking_error_deg=float(abs(z['error_deg']).max()),
        rolling_gate=bool(rolling and front[move[-1]]-front[0]>=10 and final_hold_pass),
        max_step_target_deg=float(np.degrees(abs(np.diff(z['ctrl'],axis=0))).max()))
    contacts=v['contacts'];mask=(contacts[:,0]<=z['time'][move[-1]]+1e-8)&(contacts[:,1]==1)&(contacts[:,2]<6)
    result['front_slip_median_mm_s']=float(np.median(contacts[mask,4])) if mask.any() else None
    write_json(folder/'analysis.json',result)
    return result


def run(mode):
    rolling=mode.endswith('roll');feedback=mode.startswith('feedback')
    if feedback and rolling:
        gate=json.loads((OUT/'feedback_hold/analysis.json').read_text())
        if not gate['static_gate']:raise RuntimeError('feedback static hold gate failed; rolling not enabled')
    sim=None;folder=OUT/mode
    try:
        sim=DiagnosticTrial(folder)
        for name in EXTRA:
            dest=folder/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        links=np.asarray(sim.meta['settings']['links']);initial_q=sim.d.qpos[sim.qadr].copy()
        nominal=SpatialBishopBackbone(points_for_angles(sim.m,sim.d,sim.d.ctrl),links)
        actual=SpatialBishopBackbone(points_for_angles(sim.m,sim.d,initial_q),links)
        controller=SpatialShapeController(nominal,sim.d.ctrl,actual,initial_q)
        initial_ctrl=sim.d.ctrl.copy();phase0=sim.phase;steps=8 if rolling else 32
        joints=np.cumsum(links)-links/2
        initial_axis=AxisField(sim.target,sim.source,sim.records[0]['points_mm'][1]-sim.records[0]['points_mm'][0],
                               6*links.mean(),6*links.mean())
        # Fixed world directions and boundary; head-frame changes must not move the axis field.
        axes=initial_axis.frames(joints)[0][:,:,2]
        gains=sim.original.parameters.gains(20)*(-1 if sim.original.parameters.inside else 1)
        for step in range(steps):
            g=sim.fresh();p=np.array(g['points_mm']);angles=np.radians(g['angles_deg'])*sim.bridge.signs
            delta=-sim.meta['settings']['phase_step'] if rolling else 0.
            total=(step+1)*delta
            if feedback:
                current=SpatialBishopBackbone(p,links)
                rows=local_sections(p,axes,g['curve']['gamma_min_rad'])
                field=controller.feedback_field(current,angles,rows,gains,total,delta)
                raw=current.joint_angles(field=field)
                reconstruct=current
            else:
                field=controller.nominal;raw=nominal.joint_angles(total,field)
                reconstruct=nominal
            target=np.clip(raw,sim.m.jnt_range[sim.jids,0]+.01,sim.m.jnt_range[sim.jids,1]-.01)
            sim.stage=mode;sim.phase=phase0+total
            sim.geometry_log.append(g)
            sim.controls.append(dict(step=step+1,phase=sim.phase,boundary_links=6.,target=target,
                raw=raw,clipped_count=int(np.count_nonzero(abs(target-raw)>1e-10)),
                field=field,initial_target_gap_deg=np.degrees(target-initial_ctrl) if step==0 else None))
            sim.command(target,g);sim.advance(.2)
        sim.stage='hold';sim.advance(3.)
        result=dict(condition=mode,method='spatial_bishop',options=dict(transition_links=6),checkpoint=str(HORIZONTAL),
            steps=steps,settings=sim.meta['settings'],boundary_links=6.,mujoco=mujoco.__version__,
            source_hashes={name:sha(ROOT/name) for name in EXTRA},
            phase0_target_error_rad=float(abs(nominal.joint_angles(field=controller.nominal)-initial_ctrl).max()),
            preload_joint_error_deg=np.degrees(initial_ctrl-initial_q),
            jacobian_error_m_s=sim.jacobian_error,
            clipped_joint_steps=sum(x['clipped_count'] for x in sim.controls))
        write_json(folder/'result.json',result)
        write_json(folder/'captured_nominal_curve.json',nominal.sample(field=controller.nominal,material_rotation=controller.rotation))
    finally:
        if sim:sim.close()
    print(json.dumps(assess(folder,rolling)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['captured_hold','feedback_hold','captured_roll','feedback_roll'])
    parser.add_argument('--checks',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.checks:print(json.dumps(checks(),default=lambda x:x.tolist()))
    elif args.mode:run(args.mode)
    else:parser.error('choose --checks or --mode')
