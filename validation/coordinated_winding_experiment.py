"""Bounded crossing experiment; original physics, checkpoint and EXE relay."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import numpy as np
import mujoco

from three_segment_experiment import ThreeTrial
from segmented_pole_experiment import ROOT, HORIZONTAL, VERTICAL, write_json
from spatial_takeover_experiment import points_for_angles
from snake_control.coordinated_compliant_winding import CoordinatedCompliantWinding, CoordinationOptions
from snake_control.traveling_compliant_winding import FormationOptions
from snake_control.three_segment_compliance import ThreeSegmentOptions
from three_segment_review import assess
from traveling_winding_audit import audit, evaluate

OUT = ROOT/'experiments/segmented_pole_20261005/coordinated_winding'


def run(args):
    folder=OUT/args.name
    checkpoint=HORIZONTAL if args.scene=='horizontal' else VERTICAL
    sim=None
    result=dict(method='coordinated_compliant_winding',checkpoint=str(checkpoint),
        outcome='engineering_failure',steps=0,hypothesis=args.hypothesis,
        mujoco=mujoco.__version__,step_seconds=args.dt)
    try:
        sim=ThreeTrial(checkpoint,folder)
        for name in ('snake_control/coordinated_compliant_winding.py','snake_control/traveling_compliant_winding.py',
                     'snake_control/three_segment_incremental.py','snake_control/spatial_bishop_backbone.py',
                     'validation/coordinated_winding_experiment.py','validation/traveling_winding_audit.py',
                     'validation/coordinated_winding_metrics.py','validation/spatial_takeover_experiment.py'):
            dst=folder/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dst)
        g=sim.fresh()
        controller=CoordinatedCompliantWinding(g,sim.original.parameters,sim.d.ctrl,sim.phase,
            sim.corner,sim.source,sim.target,sim.radius,measured=sim.d.qpos[sim.qadr].copy(),
            forward_points=lambda q:points_for_angles(sim.m,sim.d,q),pitch_mm=sim.meta['settings']['pitch'],
            options=ThreeSegmentOptions(follow_seconds=.06,maximum_speed_rad_s=np.radians(args.speed_limit)),
            formation=FormationOptions(recovery=args.recovery),
            coordination=CoordinationOptions(args.lead,args.phase_feedback))
        result['configuration']=controller.configuration()
        result['settings']=sim.meta['settings']
        write_json(folder/'plan.json',dict(**result,arguments=vars(args),physics_changed=False,
            stop='floor or 0.2s whole-body support loss; full crossing plus full rolling cycle',
            compare='saved v9 horizontal; process quality includes error, actual velocity and saturation',
            limitations='known axes/contact; no real robot commands; no SHA checks'))
        sim.partition[:]=[controller.front,controller.rear]
        sim.records[0]['partition']=sim.partition.copy()
        phase0=sim.phase;lost=0
        result['outcome']='time_limit'
        for k in range(round(args.seconds/args.dt)):
            g=sim.fresh();measured=np.radians(g['angles_deg'])*sim.bridge.signs
            phase=phase0-(k+1)*args.dt*.1875
            q,info=controller.compute(g,measured,phase,args.dt,
                                     [r['normal_force'] for r in sim.records[-60:]])
            sim.phase=controller.phase;sim.stage='coordinated_forming'
            info.update(target=q,measured=measured,phase=sim.phase,step=k+1,time=(k+1)*args.dt)
            sim.controls.append(info);sim.geometry_log.append(g)
            sim.partition[:]=[controller.front,controller.rear]
            first=len(sim.records)
            # All interpolated references go through the same EXE relay. The
            # physical gain, timestep and torque limit are never changed.
            origin=sim.d.ctrl.copy()
            count=round(args.dt/args.delivery_period)
            for substep in range(1,count+1):
                command=origin+(q-origin)*(substep/count)
                sim.command(command,g);sim.advance(args.dt/count)
            result['steps']=k+1
            for row in sim.records[first:]:
                lost=0 if np.any(row['normal_force']>.1) else lost+1
                if row['ground'] or lost>=20:
                    result['outcome']='support_lost';break
            if result['outcome']=='support_lost':break
            if (k+1)%16==0:
                if controller.progress>=controller.n:
                    keys=('time','points_mm','normal_force','ground','phase','region','error_deg')
                    trace={key:np.asarray([r[key] for r in sim.records]) for key in keys}
                    if evaluate(trace,sim.m)['full_crossing_and_continued_rolling']:
                        result['outcome']='full_crossing_validated';break
                row=sim.records[-1]
                print(f'{args.name}: t={(k+1)*args.dt:.1f} formation={controller.progress:.2f} '
                      f'partition={sim.partition.tolist()} region={int(row["region"].sum())} '
                      f'error={np.max(abs(row["error_deg"])):.1f} rate={controller.progress_scale:.2f}',flush=True)
        result['events']=controller.events
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if sim is not None:
            write_json(folder/'result.json',result);sim.close()
    assess(folder);audit(folder)
    from coordinated_winding_metrics import metrics
    print(json.dumps(metrics(folder),ensure_ascii=False,indent=2),flush=True)
    if args.video:
        from coordinated_winding_review import review
        review(folder,video=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name');p.add_argument('--hypothesis',required=True)
    p.add_argument('--scene',choices=['horizontal','vertical_down'],default='horizontal')
    p.add_argument('--seconds',type=float,default=110.)
    p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--delivery-period',type=float,default=.02)
    p.add_argument('--lead',type=float,default=8.)
    p.add_argument('--speed-limit',type=float,default=90.)
    p.add_argument('--recovery',type=float,default=.35)
    p.add_argument('--phase-feedback',action='store_true')
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if args.delivery_period<=0 or args.delivery_period>args.dt or not np.isclose(
            args.dt/args.delivery_period,round(args.dt/args.delivery_period)):
        p.error('delivery-period must divide dt and be no greater than dt')
    run(args)
