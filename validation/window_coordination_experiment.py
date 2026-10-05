"""Stage-1 window coordination through the unchanged EXE/MuJoCo relay."""
import argparse
import json
import shutil
from pathlib import Path
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
from support_transfer_winding import SupportTransferWinding,RateMatchedMiddleWinding,CoherentHelixWinding,ReferenceMemoryWinding,SupportHoldWinding,AnchoredHelixWinding,TurningFormationWinding,SmoothTurningFormationWinding,AcquisitionRateWinding,AcquiredShapeWinding,BothEndsShapeWinding,ContactReferenceWinding,CapacityPreloadWinding,RadialAcquisitionWinding
from contact_support_margin import StaticSupportBound
from velocity_feedback_delivery import VelocityFeedbackDelivery
from encoder_shape_kinematics import EncoderShapeKinematics

OUT=ROOT/'experiments/segmented_pole_20261005/window_coordination'


def run(args):
    folder=(args.output_root or OUT)/args.name;sim=None
    checkpoint=HORIZONTAL if args.scene=='horizontal' else VERTICAL
    result=dict(method='window_coordinated_winding',checkpoint=str(checkpoint),steps=0,
        outcome='engineering_failure',hypothesis=args.hypothesis,mujoco=mujoco.__version__,step_seconds=args.dt)
    try:
        sim=ThreeTrial(checkpoint,folder)
        for name in ('codes/window_coordinated_winding.py','codes/traveling_compliant_winding.py','codes/support_transfer_winding.py',
                'codes/three_segment_incremental.py','codes/spatial_bishop_backbone.py',
                'validation/window_coordination_experiment.py','validation/traveling_winding_audit.py',
                'validation/coordinated_winding_metrics.py','validation/spatial_takeover_experiment.py',
                'validation/contact_support_margin.py','codes/velocity_feedback_delivery.py','codes/encoder_shape_kinematics.py',
                'codes/encoder_pipe_geometry.py'):
            dest=folder/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        g=sim.fresh()
        choices=((args.support_memory,SupportTransferWinding),
            (args.middle_rate_match,RateMatchedMiddleWinding),
            (args.smooth_turning,SmoothTurningFormationWinding),
            (args.turning_formation,TurningFormationWinding),
            (args.anchored_helix,AnchoredHelixWinding),
            (args.coherent_helix,CoherentHelixWinding),
            (args.reference_memory,ReferenceMemoryWinding),
            (args.support_hold,SupportHoldWinding),
            (args.couple_acquisition,AcquisitionRateWinding),
            (args.acquired_shape,AcquiredShapeWinding),
            (args.both_ends_shape,BothEndsShapeWinding),
            (args.contact_reference,ContactReferenceWinding),
            (args.capacity_preload,CapacityPreloadWinding),
            (args.radial_acquisition,RadialAcquisitionWinding))
        cls=next((kind for selected,kind in choices if selected),WindowCoordinatedWinding)
        extra=dict(support_error_limit=sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0]) if args.support_memory or args.capacity_preload or args.radial_acquisition else {}
        if args.radial_acquisition:extra['coordinate_acquisition']=args.coordinate_acquisition
        if args.contact_reference or args.capacity_preload or args.radial_acquisition:
            extra['local_contact_projection']=args.local_contact_projection
            extra['dual_contact_projection']=args.dual_contact_projection
        if args.support_hold:
            bound=StaticSupportBound(sim.m)
            extra['support_bound']=lambda:bound.evaluate(sim.d.qpos,sim.m.geom('pipe_straight_03').id)
            extra['hold_bound']=args.hold_bound
            extra['bound_held_phase']=args.bound_held_phase
        corner,source,target,radius=sim.corner,sim.source,sim.target,sim.radius
        if args.encoder_initial_axes:
            from encoder_pipe_geometry import right_angle_hypothesis
            estimate=right_angle_hypothesis(np.asarray(g['points_mm']),sim.meta['settings']['radius'])
            write_json(folder/'initial_geometry_estimate.json',estimate)
            if not estimate['geometric_candidate']:
                raise RuntimeError('Encoder shape does not support a right-angle branch hypothesis')
            corner,source,target,radius=(estimate[k] for k in ('corner_mm','source','target','pipe_radius_mm'))
            result['geometry_input']='Encoder-reconstructed points, nominal coil radius, intersecting right-angle prior; contacts and global frame remain stage-2 test inputs.'
        controller=cls(g,sim.original.parameters,sim.d.ctrl,sim.phase,
            corner,source,target,radius,measured=sim.d.qpos[sim.qadr].copy(),
            forward_points=lambda q:points_for_angles(sim.m,sim.d,q),pitch_mm=sim.meta['settings']['pitch'],
            options=ThreeSegmentOptions(follow_seconds=args.middle_follow,maximum_speed_rad_s=np.radians(args.shape_speed)),
            formation=FormationOptions(complete_end_spans=args.complete_end_spans),
            window=WindowOptions(args.joints_per_rad,args.maximum_lead,args.target_lead_deg,args.tail_mode,args.finish_mode,
                                 args.supported_recovery,args.rear_phase_lead,args.support_centers,args.rolling_lookahead),**extra)
        result.update(configuration=controller.configuration(),settings=sim.meta['settings'])
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        write_json(folder/'plan.json',dict(**result,arguments=arguments,physics_changed=False,
            stop='floor or 0.2 s no pipe support; full crossing and continued rolling',
            process_gate='after 1 s: joint peak <=90 deg/s, module peak <=500 mm/s, error<=15 deg, saturation<=0.297',
            limitations='known axes/contact; only stage 1; no real robot commands; no SHA checks'))
        sim.partition[:]=[controller.front,controller.rear];sim.records[0]['partition']=sim.partition.copy()
        phase0=sim.phase;lost=0;result['outcome']='time_limit'
        delivery=None
        if args.velocity_feedback:
            # -1 protects the whole captured front, not a horizontal-specific
            # fixed count. It resolves to 5/7 for the two archived checkpoints.
            reference_module=controller.initial_front if args.body_reference_module==-1 else args.body_reference_module
            delivery=VelocityFeedbackDelivery(np.radians(g['angles_deg'])*sim.bridge.signs,
                args.velocity_feedback,args.delivery_period,sim.m.jnt_range[sim.jids],
                sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0] if args.delivery_torque_equivalent else None,
                args.velocity_filter,
                EncoderShapeKinematics(sim.m,reference_module).jacobians if args.body_damping else None,
                args.body_damping,args.body_speed_limit,sim.m.actuator_gainprm[:,0])
        for k in range(round(args.seconds/args.dt)):
            g=sim.fresh();measured=np.radians(g['angles_deg'])*sim.bridge.signs
            phase=phase0-(k+1)*args.dt*.1875
            q,info=controller.compute(g,measured,phase,args.dt,[r['normal_force'] for r in sim.records[-60:]])
            sim.phase=controller.phase;sim.stage='window_coordination';sim.partition[:]=[controller.front,controller.rear]
            info.update(target=q,measured=measured,phase=sim.phase,step=k+1,time=(k+1)*args.dt)
            sim.controls.append(info);sim.geometry_log.append(g)
            first=len(sim.records);origin=sim.d.ctrl.copy();count=round(args.dt/args.delivery_period)
            info['delivery_feedback']=[]
            velocity_weight=1-np.asarray(info['acquired_weight']) if args.velocity_scope=='unacquired' else np.ones(controller.n)
            for sub in range(1,count+1):
                target=origin+(q-origin)*(sub/count)
                if delivery is not None:
                    # Use actual relay encoder samples, not simulator qvel or
                    # contact forces, for the faster velocity-error feedback.
                    inner_g=g if sub==1 else sim.fresh()
                    inner_measured=np.radians(inner_g['angles_deg'])*sim.bridge.signs
                    target,inner_info=delivery.compute(target,(q-origin)/args.dt,
                        inner_measured,sim.d.ctrl.copy(),velocity_weight)
                    info['delivery_feedback'].append(inner_info)
                    sim.command(target,inner_g)
                else:sim.command(target,g)
                sim.advance(args.dt/count)
            if delivery is not None:
                controller.last_target=sim.d.ctrl.copy()
                info['delivered_target']=sim.d.ctrl.copy()
            result['steps']=k+1
            for row in sim.records[first:]:
                lost=0 if np.any(row['normal_force']>.1) else lost+1
                if row['ground'] or lost>=20:result['outcome']='support_lost';break
            if result['outcome']=='support_lost':break
            if (k+1)%max(1,round(3.2/args.dt))==0:
                if controller.progress>=controller.n:
                    keys=('time','points_mm','normal_force','ground','phase','region','error_deg')
                    trace={key:np.asarray([r[key] for r in sim.records]) for key in keys}
                    if evaluate(trace,sim.m)['full_crossing_and_continued_rolling']:
                        result['outcome']='full_crossing_validated';break
                row=sim.records[-1]
                print(f'{args.name} t={(k+1)*args.dt:.1f} form={controller.progress:.2f} '
                    f'prefix={info["actual_prefix"]} rear={controller.rear} region={int(row["region"].sum())} '
                    f'wrap={info["front_wrap"]:.2f} grip={info["grip"]:.2f} feed={info["feed"]:.2f} '
                    f'lag={info["phase_difference"]:.2f} error={np.max(abs(row["error_deg"])):.1f} '
                    f'held={info.get("rear_reference_held",False)}',flush=True)
        result['events']=controller.events
    except Exception as e:
        result['outcome']='controller_error'
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
    p.add_argument('--encoder-initial-axes',action='store_true',
        help='Replace initial pipe geometry and split with encoder-shape hypothesis; contact truth remains.')
    p.add_argument('--seconds',type=float,default=120.);p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--delivery-period',type=float,default=.02)
    p.add_argument('--velocity-feedback',type=float,default=0.,
        help='Encoder velocity-error position correction, gain in seconds; zero preserves original delivery')
    p.add_argument('--velocity-scope',choices=['all','unacquired'],default='all')
    p.add_argument('--delivery-torque-equivalent',action='store_true')
    p.add_argument('--velocity-filter',type=float,default=0.,help='Encoder velocity low-pass time constant in seconds')
    p.add_argument('--body-damping',type=float,default=0.,help='Relative module-speed damping through position targets, Ns/m')
    p.add_argument('--body-speed-limit',type=float,default=.2,help='Relative module speed above which coupled damping acts, m/s')
    p.add_argument('--body-reference-module',type=int,default=0,help='Encoder-relative velocity frame; -1 uses the captured front boundary')
    p.add_argument('--coordinate-acquisition',action='store_true',help='Radial acquisition: slow both phase clocks when measured recruitment lags')
    p.add_argument('--tail-mode',choices=['support','same','hold','coupled'],default='support')
    p.add_argument('--finish-mode',choices=['original','continuous'],default='original')
    p.add_argument('--maximum-lead',type=float,default=2.)
    p.add_argument('--target-lead-deg',type=float,default=8.)
    p.add_argument('--hold-bound',type=float,default=1.)
    p.add_argument('--bound-held-phase',action='store_true')
    p.add_argument('--rear-phase-lead',type=float,default=0.)
    p.add_argument('--support-centers',action='store_true')
    p.add_argument('--complete-end-spans',action='store_true')
    p.add_argument('--local-contact-projection',action='store_true')
    p.add_argument('--dual-contact-projection',action='store_true')
    p.add_argument('--rolling-lookahead',type=float)
    p.add_argument('--joints-per-rad',type=float,default=.4/.1875)
    p.add_argument('--supported-recovery',type=float,default=.1)
    p.add_argument('--middle-follow',type=float,default=.06)
    p.add_argument('--shape-speed',type=float,default=90.,help='Outer shape-reference slew in deg/s; inner delivery retains 90 deg/s limit')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--support-memory',action='store_true')
    modes.add_argument('--middle-rate-match',action='store_true')
    modes.add_argument('--coherent-helix',action='store_true')
    modes.add_argument('--reference-memory',action='store_true')
    modes.add_argument('--support-hold',action='store_true')
    modes.add_argument('--anchored-helix',action='store_true')
    modes.add_argument('--turning-formation',action='store_true')
    modes.add_argument('--smooth-turning',action='store_true')
    modes.add_argument('--couple-acquisition',action='store_true')
    modes.add_argument('--acquired-shape',action='store_true')
    modes.add_argument('--both-ends-shape',action='store_true')
    modes.add_argument('--contact-reference',action='store_true')
    modes.add_argument('--capacity-preload',action='store_true')
    modes.add_argument('--radial-acquisition',action='store_true')
    p.add_argument('--output-root',type=Path)
    args=p.parse_args()
    if args.delivery_period<=0 or not np.isclose(args.dt/args.delivery_period,round(args.dt/args.delivery_period)):
        p.error('delivery-period must divide dt')
    run(args)
