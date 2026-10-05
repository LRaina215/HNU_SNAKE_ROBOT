"""Natural compliant approach and automatic candidate capture through the EXE.

The scene is used only to construct the initial physical state and score it.
Control geometry comes from identity-head FK of returned encoder angles. A
geometric candidate triggers the existing response probe, never proves support.
This development entry preserves the original simulation-only force mode;
it does not enable force mode or change defaults in the hardware entry.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import shutil
import subprocess
from datetime import datetime
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,AxisReplay,write_json,pipe_frame
from three_segment_experiment import ThreeTrial
from snake_control.mujoco_checkpoint import load_checkpoint,controller_state
from snake_control.pole_compliance import ComplianceController
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
from snake_control.encoder_pipe_geometry import right_angle_hypothesis
from snake_control.velocity_feedback_delivery import VelocityFeedbackDelivery
from snake_control.encoder_shape_kinematics import EncoderShapeKinematics

OUT=ROOT/'experiments/segmented_pole_20261005/natural_takeover'


def save_state(folder,m,d,meta,controller,geometry):
    """Exact automatic state boundary; no hashes or controller-selected pose."""
    folder.mkdir(parents=True,exist_ok=False)
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(m,kind));mujoco.mj_getState(m,d,state,kind)
    mujoco.mj_saveModel(m,str(folder/'model.mjb'))
    np.savez_compressed(folder/'state.npz',integration_state=state,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
    saved=copy.deepcopy(meta)
    saved.update(format_version=1,mujoco_version=mujoco.__version__,state_type=int(kind),sim_time=float(d.time),
                 controller=controller_state(controller),geometry_at_capture=geometry,hashes={},source_hashes={},
                 automatic_capture=True,captured_at=datetime.now().astimezone().isoformat())
    write_json(folder/'checkpoint.json',saved)


class LocalEncoders:
    def __init__(self,model):
        self.m=copy.copy(model);self.d=mujoco.MjData(self.m)
        self.d.qpos[:7]=[0,0,0,1,0,0,0]
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]

    def points(self,q):
        self.d.qpos[self.qadr]=q;mujoco.mj_kinematics(self.m,self.d)
        return 1000*np.vstack([self.d.site('s_head').xpos,
            *[self.d.joint('joint'+str(i)).xanchor for i in range(1,21)],self.d.site('s_tail').xpos])


def run(args):
    root=OUT/args.name;root.mkdir(parents=True,exist_ok=False)
    fixture=(args.resume_auto/'candidate') if args.resume_auto else (HORIZONTAL if args.scene=='horizontal' else VERTICAL)
    m,d,meta,prior=load_checkpoint(fixture,verify_hashes=False)
    # This is mission initialization, not a mid-motion reset. The same shipped
    # prewrapped keyframe is used by the interactive Ctrl+U workflow.
    if args.resume_auto:
        parent=json.loads((args.resume_auto/'result.json').read_text(encoding='utf8'))
        if not meta.get('automatic_capture') or parent['outcome']!='geometric_candidate':
            raise ValueError('Only an automatically detected natural-approach boundary can be resumed here')
    else:
        key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'experiment_start')
        if key<0:raise ValueError('Initial experiment keyframe missing')
        mujoco.mj_resetDataKeyframe(m,d,key)
        # Test-fixture variation only, before mission time starts. The source
        # direction is used to place the robot, never sent to the controller.
        if args.approach_offset_mm:
            _,source,_,_=pipe_frame(m)
            d.qpos[:3]+=args.approach_offset_mm/1000*source
        mujoco.mj_forward(m,d)
    params=prior.parameters;links=prior.expected_lengths
    controller=prior if args.resume_auto else ComplianceController(params,links,ignore_feedback_risk=True)
    if not args.resume_auto:meta['settings'].update(phase=args.initial_phase,completed=0,initialized=False)
    save_state(root/'initial',m,d,meta,controller,None)
    sim=replay=None;hypotheses=[];attempts=[];response_state=None
    result=dict(method='natural_single_axis_compliance',scene=args.scene,checkpoint=str(root/'initial'),outcome='time_limit',
        automatic_prefix=str(args.resume_auto) if args.resume_auto else None)
    try:
        sim=ThreeTrial(root/'initial',root/'approach');replay=AxisReplay(smoothing=1.)
        shutil.copy2(__file__,sim.folder/'sources/validation/natural_takeover_experiment.py')
        write_json(sim.folder/'plan.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},controller_inputs='encoder angles and own commands; identity-head FK',
            physics_changes=False,candidate='geometry criterion persists 0.6 simulated seconds, then response gate',
            initial_state='shipped prewrapped keyframe plus one nominal application',force_mode='original simulation diagnostic only'))
        kinematics=LocalEncoders(sim.m)
        curve=HelixBackbone.from_radius_pitch(meta['settings']['radius'],meta['settings']['pitch'])
        sim.stage='initial_application'
        initial_target=curve.joint_angles(20,links,sim.phase)
        if args.resume_auto:
            fresh=sim.fresh();controller.key=(fresh['run_id'],fresh['config_id'],-1)
        elif args.smooth_start:
            start=np.radians(sim.fresh()['angles_deg'])*sim.bridge.signs
            # Send actual angles through the relay before the first integration;
            # then form the nominal coil over one second, with no pose reset.
            sim.command(start,sim.fresh())
            forming=VelocityFeedbackDelivery(start,.02,.02,sim.m.jnt_range[sim.jids],
                sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0],.02)
            for sub in range(1,51):
                fresh=sim.fresh();measured=np.radians(fresh['angles_deg'])*sim.bridge.signs
                command,_=forming.compute(start+(initial_target-start)*sub/50.,initial_target-start,
                    measured,sim.d.ctrl.copy(),np.ones(20))
                sim.command(command,fresh);sim.advance(.02)
        else:sim.command(initial_target,sim.fresh());sim.advance(.2)
        delivery=None
        if args.smooth:
            q=np.radians(sim.fresh()['angles_deg'])*sim.bridge.signs
            kinematics_for_speed=EncoderShapeKinematics(sim.m,0)
            delivery=VelocityFeedbackDelivery(q,.02,.02,sim.m.jnt_range[sim.jids],
                sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0],.02,
                kinematics_for_speed.jacobians,args.body_damping,.2,sim.m.actuator_gainprm[:,0])
        persistent=max(2,round(.6/args.dt))-1 if args.resume_auto else 0
        lost=0;last_probe_phase=None
        for step in range(round(args.seconds/args.dt)):
            if (root/'STOP').exists():
                result['outcome']='development_stopped';break
            g=sim.fresh();q=np.radians(g['angles_deg'])*sim.bridge.signs
            local=dict(g,points_mm=kinematics.points(q).tolist())
            hypothesis=right_angle_hypothesis(local['points_mm'],meta['settings']['radius'])
            persistent=persistent+1 if hypothesis['geometric_candidate'] else 0
            hypotheses.append(dict(time=float(sim.d.time),phase=sim.phase,persistent=persistent,estimate=hypothesis))
            retry_ready=last_probe_phase is None or abs(sim.phase-last_probe_phase)>=.3-1e-12
            if persistent>=max(2,round(.6/args.dt)) and retry_ready:
                accepted=True
                if args.transfer:
                    from feedback_takeover_live_tracking import EncoderPipeTracker
                    from feedback_takeover_collect_response import collect_response
                    from feedback_takeover_response_calibration import calibrate,calibrate_network
                    from feedback_takeover_response_gate import judge
                    probe=root/('probe_%02d'%(len(attempts)+1));probe.mkdir()
                    sources=probe/'sources';sources.mkdir()
                    for filename in ('feedback_takeover_response_calibration.py','feedback_takeover_response_gate.py',
                            'feedback_takeover_dynamic_observer_probe.py','feedback_takeover_collect_response.py'):
                        # Python retains the first imported implementation in
                        # this process. Later probes must archive that same
                        # source even if another development run edits disk.
                        source=(root/'probe_01/sources' if attempts else ROOT/'validation')/filename
                        shutil.copy2(source,sources/filename)
                    last_probe_phase=sim.phase
                    write_json(probe/'motion_reference.json',sim.d.ctrl.copy())
                    load=np.clip(sim.m.actuator_gainprm[:,0]*(sim.d.ctrl-q),sim.m.actuator_forcerange[:,0],sim.m.actuator_forcerange[:,1])
                    initial_tracker=EncoderPipeTracker(sim.m,q,load,meta['settings']['radius'])
                    records=collect_response(sim,initial_tracker,probe)
                    if args.response_cache:
                        cached=args.response_cache/('probe_%02d'%(len(attempts)+1))
                        prior_records=json.loads((cached/'response_probe.json').read_text(encoding='utf8'))
                        same=len(records)==len(prior_records) and all(np.allclose(a[key],b[key],atol=1e-10,rtol=0)
                            for a,b in zip(records,prior_records) for key in ('time','q','command'))
                        if not same:raise RuntimeError('Cached natural-probe measurements do not match this newly executed probe')
                        fit=json.loads((cached/'response_calibration.json').read_text(encoding='utf8'))
                        shutil.copytree(cached/'sources',probe/'cached_fit_sources')
                        write_json(probe/'cache_reuse.json',dict(source=str(cached),all_new_measurements_and_commands_matched=True))
                    else:
                        fit=(calibrate_network if args.network_calibration else calibrate)(sim.m,sim.meta['source_model'],initial_tracker.estimate,records,args.junction_search)
                    write_json(probe/'response_calibration.json',fit)
                    gate=judge(sim.m,sim.meta['source_model'],fit,records);write_json(probe/'response_gate.json',gate)
                    accepted=gate['geometry_response_accepted'] if args.acquire_verified_drape else gate['passed']
                    attempts.append(dict(folder=str(probe),time=float(sim.d.time),phase=sim.phase,passed=accepted,
                        permission='local_acquisition' if args.acquire_verified_drape else 'transfer'))
                    print(args.name,'probe',len(attempts),'accepted',accepted,'criteria',gate['criteria'],flush=True)
                    if (root/'STOP').exists():
                        result['outcome']='development_stopped';break
                    if delivery is not None:
                        delivery.previous_measured=np.array(records[-1]['q']);delivery.filtered_velocity[:]=0.
                    if not accepted:
                        persistent=0
                        continue
                    response_state=probe
                meta['settings'].update(phase=sim.phase,completed=step+1,initialized=True)
                fresh=sim.fresh();fresh=dict(fresh,points_mm=kinematics.points(np.radians(fresh['angles_deg'])*sim.bridge.signs).tolist())
                save_state(root/'candidate',sim.m,sim.d,meta,controller,fresh)
                outcome=('geometry_verified_acquisition' if args.acquire_verified_drape else 'response_accepted') if args.transfer else 'geometric_candidate'
                result.update(outcome=outcome,candidate_time=float(sim.d.time),candidate=str(root/'candidate'))
                print(args.name,'automatic candidate at',sim.d.time,flush=True);break
            projected=replay.geometry(local)
            next_phase=sim.phase-args.phase_step
            target=controller.compute(projected,next_phase,sim.d.ctrl.copy(),
                now_ms=projected['sample_unix_ms'],reference_phase=sim.phase)
            if target is None:
                result.update(outcome='invalid_feedback',error=controller.status);break
            sim.phase=next_phase;sim.stage='natural_compliance'
            if delivery is None:sim.command(target,g);sim.advance(args.dt)
            else:
                origin=sim.d.ctrl.copy();details=[];count=round(args.dt/.02)
                for sub in range(1,count+1):
                    fresh=g if sub==1 else sim.fresh()
                    measured=np.radians(fresh['angles_deg'])*sim.bridge.signs
                    applied,info=delivery.compute(origin+(target-origin)*sub/count,(target-origin)/args.dt,
                        measured,sim.d.ctrl.copy(),np.ones(20))
                    details.append(info);sim.command(applied,fresh);sim.advance(.02)
                sim.controls.append(dict(time=float(sim.d.time),target=target,delivery=details))
            for row in sim.records[-round(args.dt/.01):]:
                lost=0 if np.any(row['normal_force']>.1) else lost+1
                if row['ground'] or lost>=20:result['outcome']='support_lost';break
            if result['outcome']=='support_lost':break
            if step%10==0:print(args.name,'t=',round(sim.d.time,1),'candidate=',hypothesis['quality'],flush=True)
        result['commands']=len(sim.commands)
    finally:
        result['probe_attempts']=attempts
        write_json(root/'result.json',result);write_json(root/'hypotheses.json',hypotheses)
        if replay is not None:replay.close()
        if sim is not None:write_json(sim.folder/'result.json',result);sim.close()
    if result['outcome'] in ('response_accepted','geometry_verified_acquisition') and args.transfer:
        command=[sys.executable,'-X','utf8',str(ROOT/'validation/window_coordination_experiment.py'),'transfer',
            '--checkpoint',str(root/'candidate'),'--scene',args.scene,'--output-root',str(root),
            '--tail-mode','same','--maximum-lead','2','--supported-recovery','.35','--support-centers',
            '--radial-acquisition','--local-contact-projection','--velocity-feedback','.02',
            '--velocity-scope','all','--delivery-torque-equivalent','--velocity-filter','.02',
            '--body-damping','20','--body-speed-limit','.2','--body-reference-module','-1',
            '--seconds',str(args.transfer_seconds),'--finish-mode',args.finish_mode,
            '--dynamic-observer','--cad-support','--preserve-probe-intent',
            '--require-response-gate','--hypothesis','Frozen response takeover after automatic natural-drape candidate; no human split or pose intervention.']
        command+=['--response-state',str(response_state)]
        if args.junction_search:command.append('--junction-search')
        if args.network_calibration:command.append('--network-calibration')
        if args.feedback_progress_guard:command.append('--feedback-progress-guard')
        if args.conserve_middle:command.append('--conserve-middle')
        if args.acquire_verified_drape:command+=['--release-requires-grip','--bootstrap-front']
        completed=subprocess.run(command,check=False)
        result['transfer_exit_code']=completed.returncode
        if (root/'transfer/result.json').exists():result['transfer_result']=json.loads((root/'transfer/result.json').read_text(encoding='utf8'))
        write_json(root/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name')
    p.add_argument('--scene',choices=['horizontal','vertical_down'],default='horizontal')
    p.add_argument('--seconds',type=float,default=40.)
    p.add_argument('--phase-step',type=float,default=.15)
    p.add_argument('--initial-phase',type=float,default=0.,help='Initial nominal rolling phase, applied through the ordinary one-second EXE ramp.')
    p.add_argument('--approach-offset-mm',type=float,default=0.,help='Fixture-only initial displacement along the source pipe; negative starts farther from the junction.')
    p.add_argument('--transfer',action='store_true')
    p.add_argument('--smooth',action='store_true')
    p.add_argument('--smooth-start',action='store_true')
    p.add_argument('--body-damping',type=float,default=20.,help='Encoder-relative speed feedback during natural approach; diagnostic zero leaves joint-velocity feedback active.')
    p.add_argument('--response-cache',type=Path,help='Reuse costly fits only after the corresponding newly executed probe measurements and commands match exactly.')
    p.add_argument('--acquire-verified-drape',action='store_true',help='Geometry/constraint evidence admits local compliant forming; distributed support still gates rear release.')
    p.add_argument('--transfer-seconds',type=float,default=200.)
    p.add_argument('--finish-mode',choices=['original','continuous','supported'],default='original')
    p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--junction-search',action='store_true')
    p.add_argument('--network-calibration',action='store_true')
    p.add_argument('--feedback-progress-guard',action='store_true')
    p.add_argument('--conserve-middle',action='store_true',help='Limit rear release by actual inferred target-region membership, excluding unacquired links in the initial-front prefix.')
    p.add_argument('--resume-auto',type=Path,help='Continue an exact automatically detected boundary; retains its natural prefix for audit.')
    run(p.parse_args())
