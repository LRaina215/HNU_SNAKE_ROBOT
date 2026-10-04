"""Adaptive research runner: each revision freezes its source and full trace."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import mujoco

from three_segment_experiment import ThreeTrial, DT
from segmented_pole_experiment import ROOT, BASE, HORIZONTAL, VERTICAL, write_json
from spatial_takeover_experiment import points_for_angles
from traveling_compliant_winding import TravelingCompliantWinding, FormationOptions
from three_segment_compliance import ThreeSegmentOptions
from three_segment_review import assess, render
from traveling_winding_audit import audit, evaluate


def run(args):
    checkpoint = HORIZONTAL if args.scene == 'horizontal' else VERTICAL
    folder = BASE/'traveling_winding'/args.name
    sim = None
    result = dict(method='traveling_compliant_winding', checkpoint=str(checkpoint), steps=0,
                  outcome='engineering_failure', hypothesis=args.hypothesis, mujoco=mujoco.__version__)
    try:
        sim = ThreeTrial(checkpoint, folder)
        for name in ('codes/traveling_compliant_winding.py','codes/three_segment_incremental.py',
                     'codes/spatial_bishop_backbone.py','validation/traveling_winding_experiment.py',
                     'validation/spatial_takeover_experiment.py','validation/traveling_winding_audit.py'):
            dst = folder/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name,dst)
        g = sim.fresh()
        controller = TravelingCompliantWinding(g,sim.original.parameters,sim.d.ctrl,sim.phase,
            sim.corner,sim.source,sim.target,sim.radius,measured=sim.d.qpos[sim.qadr].copy(),
            forward_points=lambda q: points_for_angles(sim.m,sim.d,q),pitch_mm=sim.meta['settings']['pitch'],
            options=ThreeSegmentOptions(follow_seconds=args.follow),
            formation=FormationOptions(args.speed,args.lead,args.recovery,args.release,args.end_recovery))
        result['configuration'] = dict(controller.configuration(),controller_class=type(controller).__name__)
        result['settings'] = sim.meta['settings']
        write_json(folder/'plan.json',dict(**result,maximum_steps=args.steps,phase_increment=args.phase_step,
            stop='floor or whole body pipe contact absent 0.2s; audited full crossing plus an entire rolling cycle',
            success='whole body on target, source released, wrapping retained and continued rolling',
            command='Python -> isolated EXE -> MuJoCo',physics_changed=False,
            reverse_after_seconds=args.reverse_after))
        sim.partition[:] = [controller.front,controller.rear]
        sim.records[0]['partition'] = sim.partition.copy()
        phase0 = sim.phase
        lost = 0
        result['outcome'] = 'time_limit'
        for k in range(args.steps):
            g = sim.fresh()
            measured = np.radians(g['angles_deg'])*sim.bridge.signs
            signed_step = k+1
            if args.reverse_after is not None and (k+1)*DT > args.reverse_after:
                signed_step = 2*round(args.reverse_after/DT)-(k+1)
            phase = phase0+signed_step*args.phase_step
            q, info = controller.compute(g,measured,phase,DT,[r['normal_force'] for r in sim.records[-60:]])
            info.update(target=q.copy(),measured=measured.copy(),phase=phase,step=k+1)
            sim.controls.append(info)
            sim.geometry_log.append(g)
            sim.partition[:] = [controller.front,controller.rear]
            sim.phase,sim.stage = phase,'rolling_and_forming'
            begin = len(sim.records)
            sim.command(q,g)
            sim.advance(DT)
            result['steps'] = k+1
            for row in sim.records[begin:]:
                lost = 0 if np.any(row['normal_force']>.1) else lost+1
                if row['ground'] or lost >= 20:
                    result['outcome'] = 'support_lost'
                    break
            if result['outcome'] == 'support_lost': break
            if (k+1)%16 == 0 and controller.progress >= controller.n and args.reverse_after is None:
                keys=('time','points_mm','normal_force','ground','phase','region','error_deg')
                trace={key:np.asarray([r[key] for r in sim.records]) for key in keys}
                if evaluate(trace,sim.m)['full_crossing_and_continued_rolling']:
                    result['outcome']='full_crossing_validated'
                    break
            if (k+1)%16 == 0:
                row=sim.records[-1]
                print(f'{args.name}: step={k+1} formation={controller.progress:.2f} '
                      f'partition={sim.partition.tolist()} target_region={int(row["region"].sum())} '
                      f'contact={np.flatnonzero(row["normal_force"][1]>.1).tolist()}',flush=True)
        result['events'] = controller.events
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',outcome='engineering_failure')
        raise
    finally:
        if sim is not None:
            write_json(folder/'result.json',result)
            sim.close()
    report=assess(folder)
    print(json.dumps({k:v for k,v in report.items() if k!='boundary_events'},ensure_ascii=False,indent=2),flush=True)
    print(json.dumps(audit(folder),ensure_ascii=False,indent=2),flush=True)
    if args.video: render(folder)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name')
    p.add_argument('--hypothesis',default='Frozen travelling formation and compliant handoff validation.')
    p.add_argument('--scene',choices=['horizontal','vertical_down'],default='horizontal')
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--speed',type=float,default=.4)
    p.add_argument('--lead',type=float,default=4.)
    p.add_argument('--recovery',type=float,default=.35)
    p.add_argument('--release',type=float,default=3.)
    p.add_argument('--follow',type=float,default=.06)
    p.add_argument('--phase-step',type=float,default=-.0375)
    p.add_argument('--video',action='store_true')
    p.add_argument('--end-recovery',type=float,default=.25)
    p.add_argument('--reverse-after',type=float,help='Independent direction-control check after crossing (seconds).')
    run(p.parse_args())
