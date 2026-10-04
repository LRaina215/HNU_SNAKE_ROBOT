"""Stage-1 window coordination through the unchanged EXE/MuJoCo relay."""
import argparse
import json
import shutil
import numpy as np
import mujoco
from three_segment_experiment import ThreeTrial
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json
from spatial_takeover_experiment import points_for_angles
from window_coordinated_winding import WindowCoordinatedWinding,WindowOptions
from traveling_compliant_winding import FormationOptions
from three_segment_compliance import ThreeSegmentOptions
from three_segment_review import assess
from traveling_winding_audit import audit,evaluate
from coordinated_winding_metrics import metrics

OUT=ROOT/'experiments/segmented_pole_20261005/window_coordination'


def run(args):
    folder=OUT/args.name;sim=None
    checkpoint=HORIZONTAL if args.scene=='horizontal' else VERTICAL
    result=dict(method='window_coordinated_winding',checkpoint=str(checkpoint),steps=0,
        outcome='engineering_failure',hypothesis=args.hypothesis,mujoco=mujoco.__version__,step_seconds=args.dt)
    try:
        sim=ThreeTrial(checkpoint,folder)
        for name in ('codes/window_coordinated_winding.py','codes/traveling_compliant_winding.py',
                'codes/three_segment_incremental.py','codes/spatial_bishop_backbone.py',
                'validation/window_coordination_experiment.py','validation/traveling_winding_audit.py',
                'validation/coordinated_winding_metrics.py','validation/spatial_takeover_experiment.py'):
            dest=folder/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        g=sim.fresh()
        controller=WindowCoordinatedWinding(g,sim.original.parameters,sim.d.ctrl,sim.phase,
            sim.corner,sim.source,sim.target,sim.radius,measured=sim.d.qpos[sim.qadr].copy(),
            forward_points=lambda q:points_for_angles(sim.m,sim.d,q),pitch_mm=sim.meta['settings']['pitch'],
            options=ThreeSegmentOptions(follow_seconds=args.middle_follow,maximum_speed_rad_s=np.radians(90)),
            formation=FormationOptions(),
            window=WindowOptions(args.joints_per_rad,args.maximum_lead,8.,args.tail_mode,args.finish_mode,
                                 args.supported_recovery))
        result.update(configuration=controller.configuration(),settings=sim.meta['settings'])
        write_json(folder/'plan.json',dict(**result,arguments=vars(args),physics_changed=False,
            stop='floor or 0.2 s no pipe support; full crossing and continued rolling',
            process_gate='after 1 s: joint peak <=90 deg/s, module peak <=500 mm/s, error<=15 deg, saturation<=0.297',
            limitations='known axes/contact; only stage 1; no real robot commands; no SHA checks'))
        sim.partition[:]=[controller.front,controller.rear];sim.records[0]['partition']=sim.partition.copy()
        phase0=sim.phase;lost=0;result['outcome']='time_limit'
        for k in range(round(args.seconds/args.dt)):
            g=sim.fresh();measured=np.radians(g['angles_deg'])*sim.bridge.signs
            phase=phase0-(k+1)*args.dt*.1875
            q,info=controller.compute(g,measured,phase,args.dt,[r['normal_force'] for r in sim.records[-60:]])
            sim.phase=controller.phase;sim.stage='window_coordination';sim.partition[:]=[controller.front,controller.rear]
            info.update(target=q,measured=measured,phase=sim.phase,step=k+1,time=(k+1)*args.dt)
            sim.controls.append(info);sim.geometry_log.append(g)
            first=len(sim.records);origin=sim.d.ctrl.copy();count=round(args.dt/args.delivery_period)
            for sub in range(1,count+1):sim.command(origin+(q-origin)*(sub/count),g);sim.advance(args.dt/count)
            result['steps']=k+1
            for row in sim.records[first:]:
                lost=0 if np.any(row['normal_force']>.1) else lost+1
                if row['ground'] or lost>=20:result['outcome']='support_lost';break
            if result['outcome']=='support_lost':break
            if (k+1)%16==0:
                if controller.progress>=controller.n:
                    keys=('time','points_mm','normal_force','ground','phase','region','error_deg')
                    trace={key:np.asarray([r[key] for r in sim.records]) for key in keys}
                    if evaluate(trace,sim.m)['full_crossing_and_continued_rolling']:
                        result['outcome']='full_crossing_validated';break
                row=sim.records[-1]
                print(f'{args.name} t={(k+1)*args.dt:.1f} form={controller.progress:.2f} '
                    f'prefix={info["actual_prefix"]} rear={controller.rear} region={int(row["region"].sum())} '
                    f'wrap={info["front_wrap"]:.2f} grip={info["grip"]:.2f} feed={info["feed"]:.2f} '
                    f'lag={info["phase_difference"]:.2f} error={np.max(abs(row["error_deg"])):.1f}',flush=True)
        result['events']=controller.events
    except Exception as e:
        result['error']=f'{type(e).__name__}: {e}';raise
    finally:
        if sim is not None:write_json(folder/'result.json',result);sim.close()
    assess(folder);audit(folder);report=metrics(folder)
    w=report['windows']['after_capture']
    quality=dict(joint_speed=w['maximum_joint_speed_deg_s']<=90,
        body_speed=w['maximum_module_speed_mm_s']<=500,tracking=w['maximum_error_deg']<=15,
        saturation=w['saturated_joint_sample_fraction']<=.297)
    write_json(folder/'stage1_gate.json',dict(complete=report['complete'],quality=quality,
        passed=report['complete'] and all(quality.values())))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name');p.add_argument('--hypothesis',required=True)
    p.add_argument('--scene',choices=['horizontal','vertical_down'],default='horizontal')
    p.add_argument('--seconds',type=float,default=120.);p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--delivery-period',type=float,default=.02)
    p.add_argument('--tail-mode',choices=['support','same','hold','coupled'],default='support')
    p.add_argument('--finish-mode',choices=['original','continuous'],default='original')
    p.add_argument('--maximum-lead',type=float,default=2.)
    p.add_argument('--joints-per-rad',type=float,default=.4/.1875)
    p.add_argument('--supported-recovery',type=float,default=.1)
    p.add_argument('--middle-follow',type=float,default=.06)
    args=p.parse_args()
    if args.delivery_period<=0 or not np.isclose(args.dt/args.delivery_period,round(args.dt/args.delivery_period)):
        p.error('delivery-period must divide dt')
    run(args)
