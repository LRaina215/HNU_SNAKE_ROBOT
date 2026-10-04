"""Matched-time synchronous/front-first/rear-first rolling, fixed boundary.

Eight cycles: 16 x 0.2 s, both end phases advance -1.2 rad, then 3 s
fixed-target hold. Only if an alternating group passes, compare all three at
16 cycles from the same capture. No parameter search or boundary migration.
"""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.integrate import quad

from spatial_takeover_experiment import points_for_angles
from segmented_pole_experiment import BASE, ROOT, HORIZONTAL, load_checkpoint, write_json, sha, local_sections, pipe_frame
from segmented_pole_diagnostics import DiagnosticTrial
from segmented_pole_backbone import AxisField
from spatial_bishop_backbone import SpatialBishopBackbone, SpatialShapeController
from alternating_pole_phase import LocalPhaseCommand, schedule

OUT=BASE/'alternating_v1'
MODES=['synchronous','front_first','rear_first']
EXTRA=['codes/spatial_bishop_backbone.py','codes/alternating_pole_phase.py',
       'validation/spatial_takeover_experiment.py','validation/alternating_pole_experiment.py']


class AlternatingTrial(DiagnosticTrial):
    def __init__(self,folder):
        self.local_total=np.zeros(2)
        super().__init__(folder)
        for path in EXTRA:
            dest=self.folder/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)

    def snapshot(self):
        row=super().snapshot()
        row['local_phase_total']=self.local_total.copy()
        # Historical base logger omits tail/floor; cover it here as well.
        for contact in self.scratch.contact:
            names=[self.m.geom(int(g)).name for g in contact.geom]
            bodies=[self.m.body(int(self.m.geom_bodyid[g])).name for g in contact.geom]
            row['ground'] |= 'floor' in names and 'tail' in bodies
        return row


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(modes=MODES,primary_cycles=8,extension_cycles=16,
        extension_rule='run all three at 16 cycles only if front_first or rear_first passes the 8-cycle short gate',
        half_step_seconds=.2,cycle_phase_rad=-.15,hold_seconds=3.,boundary_links=6.,transition_links=6.,
        front_modules=list(range(6)),rear_modules=list(range(10,21)),
        short_gate=dict(front_mean_progress_mm=10.,hold_head_retreat_mm=10.,hold_root_motion_mm=20.,
                        front_target_and_source_contacts_throughout_hold=True,no_floor=True),
        no_friction_gain_or_joint_count_tuning=True,
        note='local coordination screen, not the original complete-transfer gate or a full crossing')
    write_json(OUT/'plan.json',plan)
    m,d,meta,c=load_checkpoint(HORIZONTAL);links=np.array(meta['settings']['links']);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    b=SpatialBishopBackbone(points_for_angles(m,d,d.qpos[qadr]),links)
    f,R=b.material_field(d.qpos[qadr]);preload=f*.17
    equal=LocalPhaseCommand(b.bounds,f,preload,[-.6,-.6],[-.075,-.075],6*links.mean(),6*links.mean())
    co,sn=np.cos(-.075),np.sin(-.075);pc,ps=np.cos(-.6),np.sin(-.6)
    expected=b.joint_angles(field=f@np.array([[co,sn],[-sn,co]])+preload@np.array([[pc,ps],[-ps,pc]]))
    uniform_error=float(abs(equal.joint_angles(links)-expected).max())
    phased=LocalPhaseCommand(b.bounds,f,preload,[-.6,-.45],[-.15,0.],6*links.mean(),6*links.mean())
    nodes=np.r_[0.,np.cumsum(links)];independent=[]
    for j,(lo,hi) in enumerate(zip(nodes[:-2],nodes[2:])):
        component=0 if j%2 else 1
        splits=phased.breaks();inside=splits[(splits>lo)&(splits<hi)]
        val=quad(lambda s:-phased.evaluate([s])[0,component],lo,hi,points=inside,epsabs=1e-11,epsrel=1e-11)[0]
        independent.append(val)
    quadrature_error=float(abs(phased.joint_angles(links)-independent).max())
    for mode in MODES:
        cumulative=np.zeros(2)
        for half in range(32):
            total,delta=schedule(mode,half);cumulative+=delta
            assert np.max(abs(cumulative-total))<1e-12
            assert abs(total[0]-total[1])<=.15+1e-12
            if half%2:assert abs(total[0]-total[1])<1e-12
    sample=phased.sample(b,R)
    assert uniform_error<1e-12 and quadrature_error<1e-10 and sample['frame_error']<1e-6
    report=dict(uniform_phase_error_rad=uniform_error,adaptive_quadrature_error_rad=quadrature_error,
                reconstruction_frame_error=sample['frame_error'],phase_budget_and_cycle_closure=True)
    write_json(OUT/'geometry_checks.json',report);write_json(OUT/'geometry_sample.json',sample)
    print(json.dumps(report),flush=True)


def run(mode,cycles,continuous=False):
    if cycles==16 and not continuous:
        outcomes=[json.loads((OUT/f'{name}_c8'/'analysis.json').read_text()) for name in MODES[1:]]
        if not any(x['short_gate'] for x in outcomes):raise RuntimeError('no alternating short gate passed')
    folder=OUT/f'{mode}_c{cycles}';sim=None
    try:
        sim=AlternatingTrial(folder);links=np.array(sim.meta['settings']['links']);phase0=sim.phase
        q=sim.d.qpos[sim.qadr].copy()
        target_bone=SpatialBishopBackbone(points_for_angles(sim.m,sim.d,sim.d.ctrl),links)
        measured_bone=SpatialBishopBackbone(points_for_angles(sim.m,sim.d,q),links)
        controller=SpatialShapeController(target_bone,sim.d.ctrl,measured_bone,q)
        p=sim.records[0]['points_mm'];axis=AxisField(sim.target,sim.source,p[1]-p[0],6*links.mean(),6*links.mean())
        axes=axis.frames(np.cumsum(links)-links/2)[0][:,:,2]
        gains=sim.original.parameters.gains(20)*(-1 if sim.original.parameters.inside else 1)
        outcome='completed';lost_support_samples=0
        for half in range(2*cycles):
            g=sim.fresh();points=np.asarray(g['points_mm']);angles=np.radians(g['angles_deg'])*sim.bridge.signs
            bone=SpatialBishopBackbone(points,links);field,rotation=bone.material_field(angles)
            rows=local_sections(points,axes,g['curve']['gamma_min_rad'])
            correction=bone.section_delta(rows,gains,rotation)
            total,delta=schedule(mode,half,sim.meta['settings']['phase_step'])
            command=LocalPhaseCommand(bone.bounds,field+correction,controller.preload,total,delta,6*links.mean(),6*links.mean())
            raw=command.joint_angles(links)
            targets=np.clip(raw,sim.m.jnt_range[sim.jids,0]+.01,sim.m.jnt_range[sim.jids,1]-.01)
            sim.local_total=total;sim.phase=phase0+total.mean();sim.stage=mode
            sim.geometry_log.append(g)
            sim.controls.append(dict(half_step=half+1,total=total,delta=delta,
                observed_with_compliance=field+correction,preload=controller.preload,
                raw=raw,target=targets,clipped_count=int(np.count_nonzero(abs(raw-targets)>1e-10))))
            sim.command(targets,g);sim.advance(.2)
            if continuous:
                # Sampling checkpoints do not interrupt physical advancement.
                if (half+1) in (16,32,64):
                    write_json(folder/f'cycle_{(half+1)//2}.json',dict(cycle=(half+1)//2,
                        sim_time=float(sim.d.time),qpos=sim.d.qpos.copy(),total_phase=total))
                for row in sim.records[-20:]:
                    lost_support_samples=lost_support_samples+1 if not np.any(row['normal_force']>.1) else 0
                    if row['ground']:outcome='floor_contact';break
                    if lost_support_samples>=20:outcome='whole_body_support_lost_0.2s';break
                if outcome!='completed':break
        if not continuous:
            sim.stage='hold';sim.advance(3.)
        result=dict(condition=mode,cycles=cycles,steps=len(sim.commands),method='alternating_spatial_bishop',
            options=dict(transition_links=6),checkpoint=str(HORIZONTAL),settings=sim.meta['settings'],
            mujoco=mujoco.__version__,boundary_links=6.,continuous=continuous,outcome=outcome,
            final_total_phases=sim.local_total,phase_trace_mean_only=True,
            clipped_joint_steps=sum(x['clipped_count'] for x in sim.controls))
        if not continuous:
            result['source_hashes']={str(p.relative_to(folder/'sources')).replace('\\','/'):sha(p)
                for p in (folder/'sources').rglob('*') if p.is_file()}
        write_json(folder/'result.json',result)
        write_json(folder/'last_command_curve.json',command.sample(bone,rotation))
    finally:
        if sim:sim.close()
    if continuous:
        from continuous_pole_review import assess_continuous
        print(json.dumps(assess_continuous(folder)),flush=True)
    else:print(json.dumps(assess(folder)),flush=True)


def assess(folder):
    result=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz');v=np.load(folder/'contact_motion.npz')
    control=json.loads((folder/'controls.json').read_text());cycles=result['cycles'];end=cycles*.4
    m,d,meta,c=load_checkpoint(HORIZONTAL);corner,source,target,radius=pipe_frame(m)
    t=z['time']-z['time'][0];h=np.flatnonzero(t>=end-1e-8);last=h[0]
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    front=(centers[:,:6]@target).mean(axis=1);rear=(centers[:,10:]@source).mean(axis=1)
    head=z['points_mm'][:,0]@target
    fc=np.any(z['normal_force'][:,1,:6]>.1,axis=1);sc=np.any(z['normal_force'][:,0]>.1,axis=1)
    hold_root=np.linalg.norm(z['qpos'][h,:3]-z['qpos'][last,:3],axis=1)*1000
    worst=h[np.argmin(head[h])];retreat=float(max(0.,head[last]-head[worst]))
    front_component=float(front[last]-front[worst]);local_component=float((head[last]-front[last])-(head[worst]-front[worst]))
    a=centers[last,:6]-centers[last,:6].mean(axis=0);b=centers[worst,:6]-centers[worst,:6].mean(axis=0)
    u,s,vt=np.linalg.svd(a.T@b);D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt)
    shape_rms=float(np.sqrt(np.mean(np.sum((a@(u@D@vt)-b)**2,axis=1))))
    halves=[]
    for k,x in enumerate(control):
        first,stop=k*20,(k+1)*20
        row=dict(half_step=k+1,delta=x['delta'],front_mean_mm=float(front[stop]-front[first]),rear_mean_mm=float(rear[stop]-rear[first]))
        for side,mods,pipe,axis in [('front',np.arange(3),1,target),('rear',np.arange(10,21),0,source)]:
            displacement=centers[stop,mods]-centers[first,mods]
            row[side+'_pure_axial_mm']=float(np.mean(displacement@axis))
            row[side+'_pure_motion_mm']=float(np.mean(np.linalg.norm(displacement,axis=1)))
        halves.append(row)
    hold_ok=bool(retreat<=10 and hold_root.max()<=20 and fc[h].all() and sc[h].all() and not z['ground'].any())
    rows=v['contacts'];contact_rows=rows[(rows[:,0]<=z['time'][last]+1e-8)&(rows[:,1]==1)&(rows[:,2]<6)]
    pauses=[]
    if result['condition']!='synchronous':
        for side,index in [('front',0),('rear',1)]:
            paused=[x for x in halves if x['delta'][index]==0]
            pauses.append(dict(side=side,mean_absolute_axial_mm=float(np.mean([abs(x[side+'_pure_axial_mm']) for x in paused])),
                mean_spatial_motion_mm=float(np.mean([x[side+'_pure_motion_mm'] for x in paused]))))
    bridge_chord=np.linalg.norm(centers[:,9]-centers[:,3],axis=1)
    out=dict(condition=result['condition'],cycles=cycles,front_progress_mm=float(front[last]-front[0]),rear_progress_mm=float(rear[last]-rear[0]),
        head_progress_mm=float(head[last]-head[0]),hold_head_retreat_mm=retreat,hold_root_motion_mm=float(hold_root.max()),
        hold_front_centroid_retreat_at_worst_head_mm=front_component,
        hold_head_relative_to_centroid_retreat_mm=local_component,front_nonrigid_rms_at_worst_head_mm=shape_rms,
        front_contact_fraction=float(fc.mean()),hold_front_contact_fraction=float(fc[h].mean()),
        source_contact_fraction=float(sc.mean()),hold_gate=hold_ok,
        short_gate=bool(front[last]-front[0]>=10 and hold_ok),ground=bool(z['ground'].any()),
        front_slip_median_mm_s=float(np.median(contact_rows[:,4])) if len(contact_rows) else None,
        max_tracking_error_deg=float(abs(z['error_deg']).max()),
        max_step_target_deg=float(np.degrees(abs(np.diff(z['ctrl'],axis=0))).max()),
        bridge_chord_range_mm=[float(bridge_chord.min()),float(bridge_chord.max())],
        paused_segment_motion=pauses,half_steps=halves)
    write_json(folder/'analysis.json',out)
    return {k:v for k,v in out.items() if k!='half_steps'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--cycles',type=int,choices=[8,16,32],default=8)
    parser.add_argument('--continuous',action='store_true',help='no intermediate or final hold; a separate continuous-motion experiment')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();OUT=(args.output_root or (BASE/'continuous_v1' if args.continuous else OUT)).resolve()
    if args.cycles==32 and not args.continuous:parser.error('32 cycles require --continuous')
    if args.prepare and args.continuous:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'plan.json',dict(modes=MODES,cycles=args.cycles,record_cycles=[8,16,32],
            half_step_seconds=.2,cycle_phase_rad=-.15,hold_seconds=0,boundary_links=6.,transition_links=6.,
            stop_conditions=['floor contact','whole body has no pipe force >0.1 N for 0.2 s'],
            support_measure='target-region membership at three consecutive command endpoints, plus any same-module target contact during that 0.6 s window',
            note='new continuous-motion question; previous stopping-hold failures remain unchanged; no SHA checks'))
    elif args.prepare:prepare()
    elif args.mode:run(args.mode,args.cycles,args.continuous)
    else:parser.error('choose --prepare or --mode')
