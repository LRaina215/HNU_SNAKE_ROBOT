"""Four fixed-boundary ablations: original, helix, connector, combined."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import mujoco
from spatial_migration_experiment import MigrationTrial
from segmented_pole_experiment import BASE,ROOT,HORIZONTAL,write_json,local_sections
from spatial_takeover_experiment import points_for_angles
from spatial_bishop_backbone import SpatialBishopBackbone,SpatialShapeController
from segmented_pole_backbone import AxisField,smoothstep
from alternating_pole_phase import schedule
from supported_helix_field import SupportedHelixCommand,reference_fields

OUT=BASE/'supported_helix_v1'
MODES=['original','helix','connector','combined']
CYCLES=64


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',dict(modes=MODES,cycles=CYCLES,half_step_seconds=.2,
        cycle_phase_rad=-.15,rolling='synchronous',boundary_links=6.,transition_links=6.,
        helix=dict(full_weight_through_link=6.,taper_ends_link=9.,shape_gain=.1,
            ramp_seconds=1.6,tangent_relaxation_links=2.,pitch_mm=220.,
            radius='median front joint distance to known target axis / (1 + mean front K)'),
        connector='subtract world-axis-turn derivative before rolling; add it back afterward',
        stop_conditions=['floor','no whole-body pipe contact >0.1 N for 0.2 s'],
        hold_seconds=3.,gate=dict(front_progress_mm=64.5,
            minimum_front_coverage_fraction_of_initial=.8,front_contact_throughout=True,
            final_new_region_modules=2,hold_head_retreat_mm=10.,hold_root_motion_mm=20.,no_floor=True),
        scope='known-axis local shape feedback; no exact connector endpoint/arc-length constraint',
        follow_up='no migration until combined fixed-boundary gate passes',
        no_parameter_search=True,no_SHA_checks=True))


def run(root,mode):
    folder=root/mode;sim=None
    try:
        sim=MigrationTrial(folder)
        for name in ['codes/supported_helix_field.py','validation/supported_helix_experiment.py','validation/supported_helix_review.py']:
            dst=folder/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        links=np.array(sim.meta['settings']['links']);link=links.mean();phase0=sim.phase
        q=sim.d.qpos[sim.qadr].copy();p=sim.records[0]['points_mm']
        controller=SpatialShapeController(SpatialBishopBackbone(points_for_angles(sim.m,sim.d,sim.d.ctrl),links),
            sim.d.ctrl,SpatialBishopBackbone(points_for_angles(sim.m,sim.d,q),links),q)
        axis=AxisField(sim.target,sim.source,p[1]-p[0],6*link,6*link)
        axes=axis.frames(np.cumsum(links)-links/2)[0][:,:,2]
        gains=sim.original.parameters.gains(20)*(-1 if sim.original.parameters.inside else 1)
        r=p[:7]-sim.corner;r-=np.outer(r@axis.target,axis.target)
        winding=float(np.sign(np.sum(np.cross(r[:-1],r[1:])@axis.target)))
        if winding==0:raise ValueError('initial front winding undefined')
        outcome='time_limit';lost=0
        for half in range(2*CYCLES):
            g=sim.fresh();points=np.asarray(g['points_mm']);angles=np.radians(g['angles_deg'])*sim.bridge.signs
            bone=SpatialBishopBackbone(points,links);field,rotation=bone.material_field(angles)
            rows=local_sections(points,axes,g['curve']['gamma_min_rad'])
            correction=bone.section_delta(rows,gains,rotation)
            desired,bend,weights,info=reference_fields(bone,points,rotation,axis,sim.corner,
                sim.meta['settings']['pitch'],winding,gains)
            gain=.1*float(smoothstep((half+1)*.2/1.6)) if mode in ('helix','combined') else 0.
            total,delta=schedule('synchronous',half,sim.meta['settings']['phase_step'])
            command=SupportedHelixCommand(bone.bounds,field+correction,controller.preload,total,delta,6*link,6*link,
                desired,bend,weights,gain,mode in ('connector','combined'))
            raw=command.joint_angles(links)
            targets=np.clip(raw,sim.m.jnt_range[sim.jids,0]+.01,sim.m.jnt_range[sim.jids,1]-.01)
            sim.local_total=total;sim.phase=phase0+total.mean();sim.stage='rolling'
            sim.geometry_log.append(g)
            sim.controls.append(dict(half_step=half+1,total=total,delta=delta,boundary_links=6.,
                observed_with_compliance=field+correction,preload=controller.preload,helix_reference=desired,
                axis_bend=bend,front_weights=weights,shape_gain=gain,separate=mode in ('connector','combined'),
                reference=info,raw=raw,target=targets,clipped_count=int(np.count_nonzero(abs(raw-targets)>1e-10))))
            sim.command(targets,g);sim.advance(.2)
            for row in sim.records[-20:]:
                lost=lost+1 if not np.any(row['normal_force']>.1) else 0
                if row['ground']:outcome='floor_contact';break
                if lost>=20:outcome='whole_body_support_lost';break
            if outcome!='time_limit':break
            if (half+1)%32==0:print(f'{mode}: cycle {(half+1)//2}, radius={info["target_radius_mm"]:.1f}mm',flush=True)
        rolling_samples=len(sim.records);held=outcome=='time_limit'
        if held:sim.stage='hold';sim.advance(3.)
        write_json(folder/'result.json',dict(condition=mode,method='supported_helix_field',
            checkpoint=str(HORIZONTAL),settings=sim.meta['settings'],options=dict(transition_links=6),
            mujoco=mujoco.__version__,outcome=outcome,steps=len(sim.commands),rolling_samples=rolling_samples,
            hold_seconds=3. if held else 0.,winding=winding,
            clipped_joint_steps=sum(x['clipped_count'] for x in sim.controls)))
        write_json(folder/'last_command_curve.json',command.sample(bone,rotation))
    finally:
        if sim:sim.close()
    from supported_helix_review import assess
    print(assess(folder),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--mode',choices=MODES);p.add_argument('--output-root',type=Path,default=OUT)
    a=p.parse_args()
    if a.prepare:prepare(a.output_root)
    elif a.mode:run(a.output_root,a.mode)
    else:p.error('choose --prepare or --mode')
