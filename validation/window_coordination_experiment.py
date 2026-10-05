"""Window coordination and staged feedback substitution via the EXE relay."""
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
    checkpoint=args.checkpoint or (HORIZONTAL if args.scene=='horizontal' else VERTICAL)
    result=dict(method='window_coordinated_winding',checkpoint=str(checkpoint),steps=0,
        outcome='engineering_failure',hypothesis=args.hypothesis,mujoco=mujoco.__version__,step_seconds=args.dt)
    try:
        sim=ThreeTrial(checkpoint,folder)
        for name in ('codes/window_coordinated_winding.py','codes/traveling_compliant_winding.py','codes/support_transfer_winding.py',
                'codes/three_segment_incremental.py','codes/spatial_bishop_backbone.py',
                'validation/window_coordination_experiment.py','validation/traveling_winding_audit.py',
                'validation/coordinated_winding_metrics.py','validation/spatial_takeover_experiment.py',
                'validation/contact_support_margin.py','codes/velocity_feedback_delivery.py','codes/encoder_shape_kinematics.py',
                'codes/encoder_pipe_geometry.py','codes/encoder_pipe_proximity.py','codes/target_frame_acquisition.py'):
            dest=folder/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        g=sim.fresh()
        if args.section_domain:
            shutil.copy2(ROOT/'codes/section_compliance_domain.py',folder/'sources/codes/section_compliance_domain.py')
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
        if args.target_frame_acquisition:
            from target_frame_acquisition import TargetFrameAcquisition
            cls=TargetFrameAcquisition
        if args.observable_support:
            from observable_support_transfer import ObservableSupportTransfer
            cls=ObservableSupportTransfer
            shutil.copy2(ROOT/'codes/observable_support_transfer.py',folder/'sources/codes/observable_support_transfer.py')
        extra=dict(support_error_limit=sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0]) if args.support_memory or args.capacity_preload or args.radial_acquisition else {}
        if args.target_frame_acquisition:
            extra.update(conserve_middle=args.conserve_middle,distinguish_lines=args.distinguish_lines)
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
        tracker=None;tracking_clock=None;probe_motion_reference=None
        if args.dynamic_observer:
            from feedback_takeover_live_tracking import EncoderPipeTracker
            from feedback_takeover_collect_response import collect_response
            from feedback_takeover_response_calibration import calibrate,calibrate_network
            from feedback_takeover_dynamic_tracker import DynamicPipeTracker
            for name in ('feedback_takeover_live_tracking.py','feedback_takeover_encoder_tracking.py','feedback_takeover_cad_refinement.py',
                    'feedback_takeover_collect_response.py','feedback_takeover_response_calibration.py',
                    'feedback_takeover_dynamic_tracker.py','feedback_takeover_dynamic_filter.py','feedback_takeover_dynamic_observer_probe.py',
                    'feedback_takeover_response_gate.py'):
                shutil.copy2(ROOT/'validation'/name,folder/'sources/validation'/name)
            measured=np.radians(g['angles_deg'])*sim.bridge.signs
            load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-measured),sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
            if args.response_state:
                records=json.loads((args.response_state/'response_probe.json').read_text(encoding='utf8'))
                fit=json.loads((args.response_state/'response_calibration.json').read_text(encoding='utf8'))
                probe_motion_reference=np.array(json.loads((args.response_state/'motion_reference.json').read_text(encoding='utf8')))
                if not np.allclose(records[-1]['q'],measured,atol=1e-10,rtol=0):
                    raise RuntimeError('Automatic probe handoff does not match current encoders')
                result['automatic_response_handoff']=str(args.response_state)
                initial_estimate=fit['unrefined']
                for filename in ('response_probe.json','response_probe_plan.json','motion_reference.json'):
                    shutil.copy2(args.response_state/filename,folder/filename)
            else:
                initial_tracker=EncoderPipeTracker(sim.m,measured,load,sim.meta['settings']['radius'])
                initial_estimate=initial_tracker.estimate
                probe_motion_reference=sim.d.ctrl.copy()
                records=collect_response(sim,initial_tracker,folder)
                print(args.name,'calibrating collected relay responses; simulation waits during computation',flush=True)
            if args.response_state:pass
            elif args.response_calibration_cache:
                # A repeated deterministic probe may reuse its expensive fit.
                # Match every measured input, not physical truth or a file hash.
                cached=args.response_calibration_cache
                prior=json.loads((cached/'response_probe.json').read_text(encoding='utf8'))
                same=len(prior)==len(records) and all(np.allclose(a[key],b[key],atol=1e-10,rtol=0)
                    for a,b in zip(prior,records) for key in ('time','q','command'))
                if not same:raise RuntimeError('Calibration cache does not match the newly executed probe')
                fit=json.loads((cached/'response_calibration.json').read_text(encoding='utf8'))
                result['identical_probe_calibration_reused']=str(cached)
                print(args.name,'reused fit after matching all new probe measurements/commands',flush=True)
            else:fit=(calibrate_network if args.network_calibration else calibrate)(sim.m,sim.meta['source_model'],initial_tracker.estimate,records,args.junction_search)
            write_json(folder/'response_calibration.json',fit)
            if args.require_response_gate:
                from feedback_takeover_response_gate import judge
                gate=judge(sim.m,sim.meta['source_model'],fit,records)
                write_json(folder/'response_gate.json',gate)
                if not gate['passed']:raise RuntimeError('Automatic takeover response gate rejected this state')
            if fit['held_out_rms_deg']>.1:
                raise RuntimeError('Response calibration rejected: held-out joint RMS exceeds 0.1 deg')
            tracker=DynamicPipeTracker(sim.m,sim.meta['source_model'],fit,records)
            g=dict(sim.fresh(),points_mm=tracker.previous.copy());tracking_clock=sim.d.time
            estimate=tracker.estimate
            corner,source,target,radius=(estimate[k] for k in ('corner_mm','source','target','pipe_radius_mm'))
            write_json(folder/'initial_unrefined_geometry.json',initial_estimate)
            write_json(folder/'initial_geometry_estimate.json',estimate)
            result['geometry_input']='On-run relay probe, joint-response calibration and private dynamics/encoder filter. True axes, contacts, root pose and root velocity excluded.'
            result['response_probe_seconds']=0. if args.response_state else 1.2
            result['calibration_computation_pauses_simulated_time']=True
        elif args.encoder_tracking:
            from feedback_takeover_live_tracking import EncoderPipeTracker
            for name in ('feedback_takeover_live_tracking.py','feedback_takeover_encoder_tracking.py','feedback_takeover_cad_refinement.py'):
                shutil.copy2(ROOT/'validation'/name,folder/'sources/validation'/name)
            measured=np.radians(g['angles_deg'])*sim.bridge.signs
            load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-measured),
                sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
            tracker=EncoderPipeTracker(sim.m,measured,load,sim.meta['settings']['radius'],args.single_target_after_release,args.whole_coil_registration)
            g=dict(g,points_mm=tracker.previous.copy());tracking_clock=sim.d.time
            estimate=tracker.estimate
            corner,source,target,radius=(estimate[k] for k in ('corner_mm','source','target','pipe_radius_mm'))
            write_json(folder/'initial_unrefined_geometry.json',tracker.initial)
            write_json(folder/'initial_geometry_estimate.json',estimate)
            result['geometry_input']='Encoder-only identity-head FK; causal local registration and fixed-radius CAD/load refinement; true base pose and pipe geometry excluded.'
        elif args.encoder_initial_axes:
            from encoder_pipe_geometry import right_angle_hypothesis
            estimate=right_angle_hypothesis(np.asarray(g['points_mm']),sim.meta['settings']['radius'])
            if args.cad_refine_axes:
                from encoder_pipe_proximity import EncoderPipeProximity
                from feedback_takeover_cad_refinement import refine
                source_path=ROOT/'validation/feedback_takeover_cad_refinement.py'
                shutil.copy2(source_path,folder/'sources/validation'/source_path.name)
                measured=np.radians(g['angles_deg'])*sim.bridge.signs
                load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-measured),
                    sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
                write_json(folder/'initial_unrefined_geometry.json',estimate)
                estimate=refine(EncoderPipeProximity(sim.m),measured,np.asarray(g['points_mm']),estimate,load)
            write_json(folder/'initial_geometry_estimate.json',estimate)
            if not estimate['geometric_candidate']:
                raise RuntimeError('Encoder shape does not support a right-angle branch hypothesis')
            corner,source,target,radius=(estimate[k] for k in ('corner_mm','source','target','pipe_radius_mm'))
            result['geometry_input']='Encoder-reconstructed points, nominal coil radius, intersecting right-angle prior; contacts and global frame remain stage-2 test inputs.'
        motion_reference=probe_motion_reference if args.preserve_probe_intent else sim.d.ctrl
        controller=cls(g,sim.original.parameters,motion_reference,sim.phase,
            corner,source,target,radius,measured=np.radians(g['angles_deg'])*sim.bridge.signs,
            forward_points=tracker.points if tracker is not None else lambda q:points_for_angles(sim.m,sim.d,q),pitch_mm=sim.meta['settings']['pitch'],
            options=ThreeSegmentOptions(follow_seconds=args.middle_follow,maximum_speed_rad_s=np.radians(args.shape_speed)),
            formation=FormationOptions(complete_end_spans=args.complete_end_spans),
            window=WindowOptions(args.joints_per_rad,args.maximum_lead,args.target_lead_deg,args.tail_mode,args.finish_mode,
                                 args.supported_recovery,args.rear_phase_lead,args.support_centers,args.rolling_lookahead,args.section_domain,args.feedback_progress_guard),**extra)
        if args.preserve_probe_intent:
            # Probe targets hold/perturb the current posture; they are not the
            # stored motion intent. Retain the old torque-equivalent preload,
            # while slew limiting and middle yielding start at the command
            # that was actually executed at the end of the probe.
            controller.last_target=sim.d.ctrl.copy();controller.command_before_last=sim.d.ctrl.copy()
            controller.middle_reference=sim.d.ctrl.copy()
            write_json(folder/'probe_resume_reference.json',dict(motion_reference=motion_reference,applied_probe_reference=sim.d.ctrl.copy(),
                meaning='Internal nominal intent retained; no physical pose reset, torque increase or direct command jump.'))
        result.update(configuration=controller.configuration(),settings=sim.meta['settings'])
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        write_json(folder/'plan.json',dict(**result,arguments=arguments,physics_changed=False,
            stop='floor or 0.2 s no pipe support; full crossing and continued rolling',
            process_gate='after 1 s: joint peak <=90 deg/s, module peak <=500 mm/s, error<=15 deg, saturation<=0.297',
            limitations=dict(manual_drape_checkpoint=not bool(sim.meta.get('automatic_capture')),physical_pose_input=tracker is None,
                known_axes=not (args.encoder_initial_axes or args.encoder_tracking or args.dynamic_observer),
                contact_labels=not args.cad_support,right_angle_comparable_radius_prior=bool(args.encoder_initial_axes or args.encoder_tracking or args.dynamic_observer),
                horizontal_vertical_gravity_prior=args.dynamic_observer,
                natural_trigger_validated=False,natural_trigger_attempt=bool(args.response_state),hardware_validated=False)))
        sim.partition[:]=[controller.front,controller.rear];sim.records[0]['partition']=sim.partition.copy()
        phase0=sim.phase;lost=0;result['outcome']='time_limit'
        delivery=None
        if args.velocity_feedback:
            # -1 protects the whole captured front, not a horizontal-specific
            # fixed count. It resolves to 5/7 for the two archived checkpoints.
            reference_module=controller.initial_front if args.body_reference_module in (-1,-2) else args.body_reference_module
            shape_kinematics=EncoderShapeKinematics(sim.m,reference_module) if args.body_damping else None
            delivery=VelocityFeedbackDelivery(np.radians(g['angles_deg'])*sim.bridge.signs,
                args.velocity_feedback,args.delivery_period,sim.m.jnt_range[sim.jids],
                sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0] if args.delivery_torque_equivalent else None,
                args.velocity_filter,
                shape_kinematics.jacobians if shape_kinematics is not None else None,
                args.body_damping,args.body_speed_limit,sim.m.actuator_gainprm[:,0])
        support_observer=None;support_history=[];support_info={};support_clock=None
        if args.dynamic_observer:support_observer=tracker
        elif args.cad_support:
            from encoder_pipe_proximity import EncoderPipeProximity
            support_observer=EncoderPipeProximity(sim.m)
        def control_feedback(feedback,refine_now):
            nonlocal tracking_clock
            if tracker is None:return feedback
            angles=np.radians(feedback['angles_deg'])*sim.bridge.signs
            if tracking_clock!=sim.d.time:
                load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-angles),
                    sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
                if args.dynamic_observer:points=tracker.update(angles,sim.d.ctrl.copy(),sim.d.time-tracking_clock)
                else:points=tracker.update(angles,load,refine_now,bool(np.any(controller.weights[:,2]>1e-12)))
                tracking_clock=sim.d.time
            else:points=tracker.previous.copy()
            estimate=tracker.estimate
            controller.corner=estimate['corner_mm'];controller.source=estimate['source']
            controller.target=estimate['target'];controller.radius=estimate['pipe_radius_mm']
            controller.body_axis=-controller.target
            return dict(feedback,points_mm=points)
        def observe_support(feedback,angles):
            nonlocal support_info,support_clock
            if support_observer is None or support_clock==sim.d.time:return
            # Current MuJoCo position-servo load is algebraically this clipped
            # error. No qfrc/contact query, and not an independent force sensor.
            load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-angles),
                sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
            if args.dynamic_observer:
                proxy=tracker.support.copy();support_info=dict(model_support_N=proxy,not_measured_contact=True)
            else:
                proxy,support_info=support_observer.observe(angles,np.asarray(feedback['points_mm']),
                    controller.corner,controller.source,controller.target,controller.radius,load,
                    tracker.target_only if tracker is not None else False)
            # 50Hz observations held over the existing 100Hz Boolean-history
            # interface. These are hypotheses, not measured normal-force samples.
            support_history.extend([proxy.copy() for _ in range(round(args.delivery_period/.01))])
            del support_history[:-60]
            support_clock=sim.d.time
        for k in range(round(args.seconds/args.dt)):
            g=control_feedback(sim.fresh(),True);measured=np.radians(g['angles_deg'])*sim.bridge.signs
            observe_support(g,measured)
            phase=phase0-(k+1)*args.dt*.1875
            contact_input=support_history if support_observer is not None else [r['normal_force'] for r in sim.records[-60:]]
            try:q,info=controller.compute(g,measured,phase,args.dt,contact_input)
            except ValueError as error:
                write_json(folder/'failed_control_feedback.json',dict(error=str(error),geometry=g,
                    measured=measured,requested_phase=phase,source=controller.source,target=controller.target,
                    corner=controller.corner,weights=controller.weights,previous_target=controller.last_target,
                    observer=tracker.info if tracker is not None else None))
                raise
            if args.body_reference_module==-2 and delivery is not None and shape_kinematics is not None:
                # Protect the acquired, formed coil from a damper intended for
                # the still-transferring body. Admission already has the .6s
                # acquisition filter; no contact truth or manual cut is used.
                reference_module=controller.initial_front
                while reference_module<controller.n and info['acquired_weight'][reference_module]>.9:
                    reference_module+=1
                reference_module=min(reference_module,max(controller.initial_front,int(controller.progress)))
                shape_kinematics.reference_module=reference_module
                info['body_reference_module']=reference_module
            if support_observer is not None:info['support_observer']=support_info
            if tracker is not None:info['geometry_observer']=tracker.info
            sim.phase=controller.phase;sim.stage='window_coordination';sim.partition[:]=[controller.front,controller.rear]
            info.update(target=q,measured=measured,phase=sim.phase,step=k+1,time=(k+1)*args.dt+result.get('response_probe_seconds',0.))
            sim.controls.append(info);sim.geometry_log.append(g)
            first=len(sim.records);origin=sim.d.ctrl.copy();count=round(args.dt/args.delivery_period)
            info['delivery_feedback']=[]
            velocity_weight=1-np.asarray(info['acquired_weight']) if args.velocity_scope=='unacquired' else np.ones(controller.n)
            for sub in range(1,count+1):
                target=origin+(q-origin)*(sub/count)
                if delivery is not None:
                    # Use actual relay encoder samples, not simulator qvel or
                    # contact forces, for the faster velocity-error feedback.
                    inner_g=g if sub==1 else control_feedback(sim.fresh(),False)
                    inner_measured=np.radians(inner_g['angles_deg'])*sim.bridge.signs
                    observe_support(inner_g,inner_measured)
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
    p.add_argument('--checkpoint',type=Path,help='Automatically captured takeover state; defaults to the original manual fixture.')
    p.add_argument('--require-response-gate',action='store_true',help='Require held-out response and target-constraint evidence before takeover.')
    p.add_argument('--junction-search',action='store_true',help='Response-only coarse corner search for partially formed natural drapes.')
    p.add_argument('--network-calibration',action='store_true',help='Compare L and both T constraint hypotheses from probe prediction.')
    p.add_argument('--feedback-progress-guard',action='store_true',help='Slow both phase clocks and retain rear support when measured admission regresses beyond formation lead.')
    p.add_argument('--response-state',type=Path,help='Probe evidence from automatic approach; checkpoint must be its exact post-probe state.')
    p.add_argument('--encoder-initial-axes',action='store_true',
        help='Replace initial pipe geometry and split with encoder-shape hypothesis; contact truth remains.')
    p.add_argument('--encoder-tracking',action='store_true',
        help='Use only encoder-frame geometry and causal CAD/load tracking, excluding floating pose and true pipe geometry.')
    p.add_argument('--dynamic-observer',action='store_true',
        help='Collect a relay probe, calibrate virtual pipes from joint responses, then infer local geometry/support through private dynamics.')
    p.add_argument('--response-calibration-cache',type=Path,help='Reuse a fit only when the freshly executed probe angles and commands match completely.')
    p.add_argument('--preserve-probe-intent',action='store_true',
        help='Resume pre-probe nominal reference/preload; slew and middle references start at the actually applied probe command.')
    p.add_argument('--section-domain',action='store_true',help='Fade sectional A-D correction near degenerate projections; retain measured 3-D rolling and valid spans.')
    p.add_argument('--single-target-after-release',action='store_true',
        help='Live estimator: stop constraining the unobserved old junction once the rear controller is gone.')
    p.add_argument('--whole-coil-registration',action='store_true',
        help='Single-target tracking: predict its frame from the whole coil instead of the deforming free front.')
    p.add_argument('--cad-support',action='store_true',
        help='Replace contact labels by private CAD-distance and encoder-error/load hypotheses; no physical contact input.')
    p.add_argument('--cad-refine-axes',action='store_true',
        help='Refine encoder-inferred initial geometry against near-shell CAD/load evidence before control.')
    p.add_argument('--target-frame-acquisition',action='store_true',
        help='Recruit using target-coil geometry and support, without absolute junction progress.')
    p.add_argument('--conserve-middle',action='store_true',
        help='Target-frame variant: release one rear joint per newly measured acquired module.')
    p.add_argument('--distinguish-lines',action='store_true',
        help='Target-frame acquisition also requires closer proximity to target than source axis.')
    p.add_argument('--observable-support',action='store_true',
        help='Retain eight rear joints until a sustained full target coil and angular support coverage are observed.')
    p.add_argument('--seconds',type=float,default=120.);p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--delivery-period',type=float,default=.02)
    p.add_argument('--velocity-feedback',type=float,default=0.,
        help='Encoder velocity-error position correction, gain in seconds; zero preserves original delivery')
    p.add_argument('--velocity-scope',choices=['all','unacquired'],default='all')
    p.add_argument('--delivery-torque-equivalent',action='store_true')
    p.add_argument('--velocity-filter',type=float,default=0.,help='Encoder velocity low-pass time constant in seconds')
    p.add_argument('--body-damping',type=float,default=0.,help='Relative module-speed damping through position targets, Ns/m')
    p.add_argument('--body-speed-limit',type=float,default=.2,help='Relative module speed above which coupled damping acts, m/s')
    p.add_argument('--body-reference-module',type=int,default=0,help='Encoder-relative velocity frame; -1 initial front, -2 follows acquired and formed prefix')
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
    if args.observable_support and not (args.target_frame_acquisition and args.radial_acquisition):
        p.error('observable-support requires target-frame-acquisition and radial-acquisition')
    if args.encoder_tracking and not (args.cad_support and args.target_frame_acquisition and args.radial_acquisition):
        p.error('encoder-tracking requires cad-support, target-frame-acquisition and radial-acquisition')
    if args.dynamic_observer and (args.encoder_tracking or not (args.cad_support and args.radial_acquisition)):
        p.error('dynamic-observer requires cad-support and radial-acquisition; excludes encoder-tracking')
    if args.preserve_probe_intent and not args.dynamic_observer:p.error('preserve-probe-intent requires dynamic-observer')
    if args.delivery_period<=0 or not np.isclose(args.dt/args.delivery_period,round(args.dt/args.delivery_period)):
        p.error('delivery-period must divide dt')
    run(args)
