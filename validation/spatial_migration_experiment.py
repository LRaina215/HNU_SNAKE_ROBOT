"""Matched fixed/migrating boundary trials, continuous rolling then fixed hold."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil

from segmented_pole_experiment import BASE, ROOT, HORIZONTAL, Trial, write_json, local_sections
from alternating_pole_experiment import AlternatingTrial, EXTRA, MODES
from spatial_takeover_experiment import points_for_angles
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone, SpatialShapeController
from snake_control.alternating_pole_phase import LocalPhaseCommand, schedule
from snake_control.segmented_pole_backbone import AxisField
from snake_control.spatial_boundary_migration import SpatialBoundaryMigration
import mujoco
import numpy as np

OUT=BASE/'migration_v1'
MAX_CYCLES=192


class MigrationTrial(AlternatingTrial):
    def __init__(self,folder):
        self.local_total=np.zeros(2);self.contact_rows=[];self.spins=[]
        # The inherited contact logger's Jacobian formula was already checked.
        self.jacobian_error=0.
        Trial.__init__(self,HORIZONTAL,folder,verify_hashes=False)
        for path in EXTRA+['validation/segmented_pole_diagnostics.py',
                'snake_control/spatial_boundary_migration.py','validation/spatial_migration_experiment.py',
                'validation/spatial_migration_review.py']:
            dest=self.folder/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,dest)


def all_target_region(records):
    return len(records)>=41 and all(np.all(records[-1-offset]['region']) for offset in (0,20,40))


def transfer_ready(records):
    # Body geometry must persist at three control endpoints; no original-pipe
    # support for 0.6 s. Boundary labels are deliberately not used for success.
    if len(records)<61:return False
    recent=records[-61:]
    return (all_target_region(records)
            and not any(np.any(r['normal_force'][0]>.1) for r in recent)
            and all(np.any(r['normal_force'][1]>.1) for r in recent)
            and not any(r['ground'] for r in recent))


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',dict(modes=MODES,boundaries=['fixed','migrating'],
        max_cycles=MAX_CYCLES,half_step_seconds=.2,cycle_phase_rad=-.15,hold_seconds=3.,
        initial_boundary_links=6.,transition_links=6.,boundary_update='cycle start, closed endpoint phases',
        migration=dict(persistence_cycles=3,max_links_per_cycle=.25,
            candidate='minimum known-axis prefix distance, target contact in nearby modules in last 0.6 s',
            terminal='all modules in target region at three endpoints; transition may move beyond tail if supported'),
        stop_conditions=['floor contact','no pipe contact >0.1 N for 0.2 s','full transfer','192 cycles'],
        full_transfer='all 21 modules in target region at three control endpoints, source contact absent 0.6 s, target contact present throughout',
        hold_gate=dict(all_modules_target_region=True,target_contact_throughout=True,no_source_contact=True,
            no_floor=True,max_head_retreat_mm=10.,max_root_displacement_mm=20.),
        attribution='known geometry and simulation contact truth; not autonomous perception or the full 2023 A-D controller',
        no_parameter_search=True,no_SHA_checks=True))


def run(root,mode,migrating):
    folder=root/f'{mode}_{"migrating" if migrating else "fixed"}';sim=None
    try:
        sim=MigrationTrial(folder);links=np.array(sim.meta['settings']['links']);link=links.mean();phase0=sim.phase
        q=sim.d.qpos[sim.qadr].copy()
        controller=SpatialShapeController(SpatialBishopBackbone(points_for_angles(sim.m,sim.d,sim.d.ctrl),links),
            sim.d.ctrl,SpatialBishopBackbone(points_for_angles(sim.m,sim.d,q),links),q)
        p=sim.records[0]['points_mm'];axis=AxisField(sim.target,sim.source,p[1]-p[0],6*link,6*link)
        axes=axis.frames(np.cumsum(links)-links/2)[0][:,:,2]
        tracker=SpatialBoundaryMigration();gains=sim.original.parameters.gains(20)*(-1 if sim.original.parameters.inside else 1)
        outcome='time_limit';lost=0;ready=False
        for half in range(2*MAX_CYCLES):
            # Migrate only when the two endpoint phases have closed. This
            # avoids remapping an existing front/rear phase offset abruptly.
            if migrating and half%2==0 and half:
                row=sim.records[-1];centers=(row['points_mm'][:-1]+row['points_mm'][1:])/2
                touched=np.any([r['normal_force'][1]>.1 for r in sim.records[-61:]],axis=0)
                sim.boundary=tracker.update(centers,sim.corner,sim.source,sim.target,touched,all_target_region(sim.records))
                axis.center=sim.boundary*link
                axes=axis.frames(np.cumsum(links)-links/2)[0][:,:,2]
            g=sim.fresh();points=np.asarray(g['points_mm']);angles=np.radians(g['angles_deg'])*sim.bridge.signs
            bone=SpatialBishopBackbone(points,links);field,rotation=bone.material_field(angles)
            rows=local_sections(points,axes,g['curve']['gamma_min_rad'])
            correction=bone.section_delta(rows,gains,rotation)
            total,delta=schedule(mode,half,sim.meta['settings']['phase_step'])
            command=LocalPhaseCommand(bone.bounds,field+correction,controller.preload,total,delta,sim.boundary*link,6*link)
            raw=command.joint_angles(links)
            targets=np.clip(raw,sim.m.jnt_range[sim.jids,0]+.01,sim.m.jnt_range[sim.jids,1]-.01)
            sim.local_total=total;sim.phase=phase0+total.mean();sim.stage='rolling'
            sim.geometry_log.append(g)
            sim.controls.append(dict(half_step=half+1,total=total,delta=delta,boundary_links=sim.boundary,
                boundary_candidate=tracker.candidate,streak=tracker.streak,
                observed_with_compliance=field+correction,preload=controller.preload,
                raw=raw,target=targets,clipped_count=int(np.count_nonzero(abs(raw-targets)>1e-10))))
            sim.command(targets,g);sim.advance(.2)
            for row in sim.records[-20:]:
                lost=lost+1 if not np.any(row['normal_force']>.1) else 0
                if row['ground']:outcome='floor_contact';break
                if lost>=20:outcome='whole_body_support_lost';break
            if outcome!='time_limit':break
            ready=transfer_ready(sim.records)
            if ready and half%2:
                outcome='full_transfer';break
            if (half+1)%32==0:
                print(f'{folder.name}: cycle {(half+1)//2}, boundary={sim.boundary:.2f}, target_modules={int(sim.records[-1]["region"].sum())}',flush=True)
        rolling_samples=len(sim.records)
        # Fixed-target stopping check is separate from continued rolling. It
        # never changes the preceding trajectory or rescues a failed transfer.
        held=outcome in ('time_limit','full_transfer')
        if held:sim.stage='hold';sim.advance(3.)
        write_json(folder/'result.json',dict(condition=mode,migrating=migrating,method='spatial_boundary_migration',
            checkpoint=str(HORIZONTAL),options=dict(transition_links=6),settings=sim.meta['settings'],
            mujoco=mujoco.__version__,outcome=outcome,steps=len(sim.commands),rolling_samples=rolling_samples,
            hold_seconds=3. if held else 0.,final_boundary_links=sim.boundary,
            clipped_joint_steps=sum(x['clipped_count'] for x in sim.controls)))
        write_json(folder/'last_command_curve.json',command.sample(bone,rotation))
    finally:
        if sim:sim.close()
    from spatial_migration_review import assess
    print(json.dumps(assess(folder)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--boundary',choices=['fixed','migrating']);parser.add_argument('--output-root',type=Path,default=OUT)
    args=parser.parse_args();root=args.output_root.resolve()
    if args.prepare:prepare(root)
    elif args.mode and args.boundary:run(root,args.mode,args.boundary=='migrating')
    else:parser.error('choose --prepare or both --mode and --boundary')
