"""Third-round complete actual attempt; frozen T01 control and independent ledger."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_GLOBAL_LOCAL_20261011/S03_ACTUAL'
import snake_control
snake_control.__path__.insert(0,str(OLD/'sources/snake_control'))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np
import mujoco
from snake_control.known90_global_local import Known90GlobalLocal,InnerCCWRoute,initial_pose
from snake_control.suspended_supply_transfer import DT,LINK
from validation.inner_ccw_compliance_experiment import Sim,write,default,shape,review

BASE=ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_FULL_ATTEMPT_20261011'
MODEL=OLD/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'
START=datetime.fromisoformat('2026-10-11T11:58:36+00:00').timestamp()
DEADLINE=START+90*60

def freeze(case):
    paths=['snake_control/known90_global_local.py','validation/known90_full_attempt.py',
           'snake_control/known90_unfold.py','snake_control/known90_progressive.py',
           'validation/inner_ccw_compliance_experiment.py','snake_control/inner_ccw_compliance.py',
           'snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py',
           'snake_control/spatial_bishop_backbone.py','snake_control/horizontal_pipe_roll.py',
           'mujoco_model/Snake_Robot_o20_full_source_inner.xml','mujoco_model/asset/c_link_H.stl','mujoco_model/asset/c_link_V.stl']
    for src in paths:
        dst=case/'sources'/src;dst.parent.mkdir(parents=True,exist_ok=True)
        origin=OLD/'sources'/src
        shutil.copy2(origin if origin.exists() else ROOT/src,dst)
    write(case/'sources.json',dict(paths=paths,hash_check=False))

def run(args):
    BASE.mkdir(exist_ok=True,parents=True)
    ledger=[]
    for path in BASE.glob('*/physics_started.json'):ledger.append(json.loads(path.read_text(encoding='utf-8')))
    full=sum(not r['local_shape_off'] for r in ledger);comparison=sum(r['local_shape_off'] for r in ledger)
    if time.time()>=DEADLINE:raise RuntimeError('Authorized 90-minute round deadline reached; saved-data review only')
    if args.local_shape_off:
        success=any(json.loads(p.read_text()).get('full_success') for p in BASE.glob('*/result.json'))
        if not success or comparison>=1:raise RuntimeError('Comparison needs a successful complete method and is limited to one')
    elif full>=3 and not args.resume:raise RuntimeError('Third-round maximum of three actual candidates reached')
    case=BASE/args.name
    if args.resume:
        if not args.complete_original or args.progressive or not (case/'result.json').exists():raise RuntimeError('Resume requires an existing original-action case')
        if (case/'continuation_started.json').exists():raise RuntimeError('Only one checkpoint continuation is allowed')
        segment=case/'segment1';segment.mkdir(exist_ok=False)
        for src in list(case.iterdir()):
            if src.is_file():shutil.copy2(src,segment/src.name)
        previous_result=json.loads((segment/'result.json').read_text(encoding='utf-8'))
        previous_commands=json.loads((segment/'commands.json').read_text(encoding='utf-8'))
        zz=np.load(segment/'trace.npz');previous_trace={k:zz[k].copy() for k in zz.files};zz.close()
    else:case.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(MODEL))
    pose=np.load(OLD/'initial_pose.npz')['qpos'].copy()
    sim=Sim(m,pose)
    if args.progressive:
        from snake_control.known90_progressive import ProgressiveFeed
        ctrl=ProgressiveFeed(m,sim.d,not args.local_shape_off,args.speed)
        route=ctrl.route;guide,_=route.window(0.)
    else:
        from snake_control.known90_unfold import Known90Unfold
        ctrl=Known90Unfold(m,sim.d,not args.local_shape_off,.005)
        route=ctrl.route;guide,_,_=ctrl.nominal(0.)
    if args.resume:
        last=previous_commands[-1]
        ctrl.shift=last['shared_shift_mm']/1000;ctrl.phase=last['roll_phase_rad'];ctrl.angle=last['source_pose_azimuth_rad'];ctrl.axial=last['source_pose_axial_mm']/1000
        ctrl.coeff=np.asarray(last['bridge_coeff_mm'])[0,1:]/1000;ctrl.target_delta=last['radius_delta_mm'][1]/1000;ctrl.target_angle=last['receiving_pose_phase_rad']
        ctrl.kp=np.asarray(last['kp_nm_per_rad']);ctrl.last=np.asarray(last['ctrl_rad']);ctrl.acquired=False;ctrl.capture_age=0.;ctrl.events=previous_result['events'];ctrl.gated=last['advancement_gate']
        if previous_result['maximum_actual_target_prefix']>=5 or previous_result['active_capture']:raise RuntimeError('Current reconstruction is limited to uncaptured original-action state')
        st=np.load(case/'segment1/final_state.npz');sim.d.qpos[:]=st['qpos'];sim.d.qvel[:]=st['qvel'];sim.d.ctrl[:]=st['ctrl'];sim.d.time=float(st['time']);st.close()
        m.actuator_gainprm[:,0]=ctrl.kp;m.actuator_biasprm[:,1]=-ctrl.kp;mujoco.mj_forward(m,sim.d)
        sim.rows=[{k:previous_trace[k][i].copy() if isinstance(previous_trace[k][i],np.ndarray) else previous_trace[k][i] for k in previous_trace} for i in range(len(previous_trace['t_s']))]
        sim.prev=previous_trace['actual_source_view_angle_rad'][-1].copy();sim.ccw=previous_trace['actual_source_ccw_rad'][-1].copy();sim.info=last.copy()
        rec=case/'continuation_sources/validation';rec.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),rec/Path(__file__).name)
    geom=json.loads((OLD/'initial_geometry.json').read_text(encoding='utf-8'))
    geom['inherited_from']='second-round S03_ACTUAL: same qpos and frozen controller/model; no new offline screen'
    path_shared,path,path_arc=ctrl.progress_path()
    np.savez(case/'registered_head_route.npz',shared_m=path_shared,points_m=path,arc_m=path_arc)
    write(case/'initial_geometry.json',geom);write(case/'route.json',route.description())
    if not args.resume:np.savez(case/'initial_pose.npz',qpos=pose,ctrl=pose[m.jnt_qposadr[m.actuator_trnid[:,0]]],points_m=guide)
    _,s0=route.window(0);vv=np.linspace(s0[-1],route.total,6001)
    np.savez(case/'full_global_route.npz',stations_m=vv,points_m=route.sample(vv))
    if not args.resume:freeze(case)
    plan=dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),round_start_utc=datetime.fromtimestamp(START,timezone.utc).isoformat(),
              first_actual_result_due_utc=datetime.fromtimestamp(START+15*60,timezone.utc).isoformat(),round_deadline_utc=datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),
              maximum_actual_seconds=600,maximum_full_methods=3,maximum_local_shape_off_comparisons=1,local_shape_off=args.local_shape_off,
              starting_state='all21 source-wrapped, no target hook; artificial pose only before mj_step',speed_m_s=args.speed,
              localization='read-only actual body positions/orientations explicitly permitted by third-round task; not claimed available on hardware',
              local_shape_inputs=['joint tracking','actuator loads','actual body pose'],diagnostic_only=['contact identities','contact forces','ground'],
              source_control='calibrated canonical source cross-section; actual azimuth/axial registration and actual bearing material-frame anchor; no source-radius adaptation',
              local_dofs='bridge inner Y bending and vertical Z bending, zero endpoint value/slope; receiving radius only after geometric capture',
              unfolding=args.unfold,planned_front_active_while_source_wrapped=bool(args.unfold),
              controller_pose_updates='source and adjacent bridge local execution only; target helix and original global curve remain fixed',
              progress=dict(path='actual head-tip distance to fixed target entry decreases3mm, execution tracking<=45mm and Y>=-10mm; changing local curve alone never qualifies',
                            wrap='target prefix wrap increases0.10rad and prefix>=3, debounced0.25s',
                            intake='new contiguous geometrically-on-target module with true target bearing, debounced0.5s',
                            axial='after actual capture target head advances5mm with maintained target prefix, debounced0.25s',
                            stall='30s after action start or latest valid new progress; local changes/shift/shaking never reset timer'),
              source_loss='Contact/geometry counts do not independently stop. No pre-ground fall classifier; observed ground or genuine stall stops.',
              limits=dict(command_deg=88,rate_deg_s=20,torque_nm=2.355,physical_joint_stop_deg=90.5),
              physics='original gravity/friction/masses/drive limits; original Kp127, bridge-only Kp95.25 on high load and tracking resistance',
              direction='tail looks along+X; screen right=-Y; positive actual atan2(Z-1,-Y) increments=CCW; posterior median>=5deg and >=70%positive for automatic full-success credit',
              stops='600s;30s no actual stage progress;ground;out-of-range command;actual joint/drive limit;nonfinite;self depth3mm;round deadline',
              progress_execution_error_filter=False,
              comparator='same initialization/route/pose localization/phase cooperation/stiffness/servo/limits/progress protections; only bridge and receiving feedback curve modifications disabled')
    write(case/'plan.json',plan);print(json.dumps(dict(initial_geometry=geom,route=route.description()),default=default),flush=True)
    if not geom['ready']:
        write(case/'result.json',dict(outcome='initial_geometry_rejected',actual_seconds=0,full_success=False,physical_run=False));return
    if not args.resume:
        sim.info=dict(global_points_mm=guide*1000,execution_points_mm=guide*1000,estimated_points_mm=guide*1000);sim.snapshot()
    if not args.resume:
        write(case/'physics_started.json',dict(name=args.name,start_utc=datetime.now(timezone.utc).isoformat(),local_shape_off=args.local_shape_off,ordinal=full+1 if not args.local_shape_off else 1))
        write(BASE/'ledger.json',dict(round_start_utc=plan['round_start_utc'],deadline_utc=plan['round_deadline_utc'],
             complete_methods=full+int(not args.local_shape_off),comparisons=comparison+int(args.local_shape_off),
             runs=ledger+[json.loads((case/'physics_started.json').read_text())]))
    else:
        record=dict(name=args.name,start_utc=datetime.now(timezone.utc).isoformat(),same_candidate=True,checkpoint_time_s=float(sim.d.time),physics_solver_warmstart_saved=False,controller_state_reconstructed=True,reason='Corrected nominal-stage entry-progress filter; no controller action change')
        write(case/'continuation_started.json',record)
        ll=json.loads((BASE/'ledger.json').read_text(encoding='utf-8'));ll.setdefault('continuations',[]).append(record);ll['physical_segments']=len(ll['runs'])+len(ll['continuations']);write(BASE/'ledger.json',ll)
    started=time.monotonic();outcome='maximum_actual_duration';commands=[];last_progress=0.;best_station=0.;best_wrap=0.;best_prefix=0;best_y=-9.
    complete_age=0.;capture_age=0.;loss_age=0.;wrap_age=0.;intake_age=0.;axial_age=0.;capture=False;capture_t=None;inner_lower=False;upward=False;error=None
    if args.unfold:
        path_station=path_arc
        plan['progress']['path']='Actual head-tip closest progress on the PRE-REGISTERED CCW front-unfold/open-upward/head route: best path arc advances3mm, pose corridor<=45mm and execution tracking<=45mm; curve changes alone never count.'
        plan['front_macro']='10 rods front: source winding removed by polar homotopy -> open lower bow -> upward approach ->5-rod target wrap;11 rods posterior source support. One shared coordinate then continues material intake.'
        if args.progressive:
            plan['front_macro']='Progressive physical-chord source->inner-upward bridge->target window; source body fed continuously, no whole-front unwind.'
            plan['progress']['path']='Actual head-tip advances3mm along fixed registered head route within45mm; no execution-error filter. Nominal shift/feedback changes never count.'
            plan['source_control']='Fixed effective source cross-section; located phase, bounded outlet axial correction, axial feed cooperation; deformed curve re-walked at exact64.5mm physical chords.'
        if not args.progressive and args.complete_original:
            plan['progress']['path']='Actual head-tip advances3mm on fixed registered route within45mm; no execution-error filter. During original front unwind also credit first planned CCW head orbit in5deg increments, capped at planned first-unwind angle plus5deg. Actual geometry motion only.'
            plan['progress']['unwind']='Unwrapped actual head orbit atan2(Z-1,-Y), positiveCCW, new5deg, limited to registered nominal first-unwind angular amount+5deg; not repeated cycles.'
            plan['progress']['approach']='Actual head-tip new best distance to fixed entry decreases3mm in every phase. No planned-stage or reference-error gate.'
        write(case/'plan.json',plan)
    else:path_station=np.linspace(0,route.total,3001);path=route.sample(path_station)
    progress_events=[];maximum_prefix=0;maximum_wrap=0.;full_success=False
    best_entry_distance=float(np.linalg.norm(sim.rows[0]['actual_points_mm'][0]/1000-route.e))
    head_angle0=float(np.arctan2(sim.rows[0]['actual_points_mm'][0,2]-1000,-sim.rows[0]['actual_points_mm'][0,1]));head_previous=head_angle0;head_unwrapped=0.;best_head_unwind=0.
    nominal_angle=np.unwrap(np.arctan2(path[:,2]-1,-path[:,1]));unwind_limit=float(max(0.,nominal_angle[path_shared<=.50].max()-nominal_angle[0])+np.radians(5)) if not args.progressive else 0.
    approach_best=best_entry_distance
    if args.resume:
        commands=previous_commands.copy();progress_events=previous_result['progress_events'].copy();last_progress=previous_result['last_actual_progress_s'];best_station=previous_result['actual_progress_station_mm']/1000
        maximum_prefix=previous_result['maximum_actual_target_prefix'];maximum_wrap=previous_result['maximum_actual_wrap_rad'];inner_lower=previous_result['inner_lower_entry_observed'];upward=previous_result['upward_wrap_observed']
        best_prefix=max([e['prefix'] for e in progress_events if e['event']=='new_actual_bearing_prefix']+[0]);best_wrap=max([e['wrap_rad'] for e in progress_events if e['event']=='actual_wrap_0p10rad']+[0.])
        head_unwrapped=np.radians(previous_result['actual_head_unwind_deg']);best_head_unwind=np.radians(previous_result['maximum_credited_unwind_deg']);head_previous=float(np.arctan2(sim.rows[-1]['actual_points_mm'][0,2]-1000,-sim.rows[-1]['actual_points_mm'][0,1]))
        for rr in sim.rows[::5]:
            distance=float(np.linalg.norm(rr['actual_points_mm'][0]/1000-route.e))
            if distance<=approach_best-.003:
                approach_best=distance;last_progress=max(last_progress,float(rr['t_s']))
        write(case/'recovered_controller_state.json',dict(shared_mm=ctrl.shift*1000,phase_rad=ctrl.phase,source_angle_rad=ctrl.angle,source_axial_mm=ctrl.axial*1000,coeff_mm=ctrl.coeff*1000,target_angle_rad=ctrl.target_angle,target_delta_mm=ctrl.target_delta*1000,last_ctrl=ctrl.last,kp=ctrl.kp,acquired=ctrl.acquired,capture_age=ctrl.capture_age,gated=ctrl.gated,last_progress_s=last_progress,entry_best_mm=approach_best*1000,original_start_time_retained=True,solver_warmstart_saved=False,actual_root_placement='saved physical checkpoint initialization before resumed first step; no control-time assignment'))
    try:
        while sim.d.time<600:
            if time.time()>=DEADLINE:outcome='round_90minute_deadline';break
            target,info=ctrl.command(sim.d,float(sim.d.time))
            if np.max(abs(target))>np.radians(88)+1e-8:outcome='proposed_command_out_of_range';break
            if info['maximum_dispatch_step_deg']>1.00001:outcome='proposed_command_rate_out_of_range';break
            sim.info=info;commands.append(dict(t_s=float(sim.d.time),ctrl_rad=target.copy(),**info));sim.command(target)
            r=sim.rows[-1];t=r['t_s'];p=r['actual_points_mm']/1000;c=r['actual_centers_mm']/1000;f=r['diagnostic_contact_force_N']
            prefix=r['actual_target_prefix'];wrap=r['actual_target_wrap_rad'];maximum_prefix=max(maximum_prefix,prefix);maximum_wrap=max(maximum_wrap,wrap)
            distances=np.linalg.norm(path-p[0],axis=1);nearest=int(np.argmin(distances));station=float(path_station[nearest])
            event=None
            execution_head_error=float(np.linalg.norm(p[0]-r['execution_points_mm'][0]/1000))
            entry_distance=float(np.linalg.norm(p[0]-route.e))
            if args.unfold and (args.progressive or args.complete_original or execution_head_error<=.045) and distances[nearest]<=.045 and station>=best_station+.003:
                best_station=station;last_progress=t;event='actual_registered_progressive_route_3mm' if args.progressive else 'actual_registered_unfold_route_3mm'
            elif not args.unfold and execution_head_error<=.045 and p[0,1]>=-.010 and entry_distance<=best_entry_distance-.003:
                best_entry_distance=entry_distance;best_station=max(best_station,station);last_progress=t;event='actual_entry_distance_reduction_3mm'
            raw_head=float(np.arctan2(p[0,2]-1,-p[0,1]));head_unwrapped+=float(np.angle(np.exp(1j*(raw_head-head_previous))));head_previous=raw_head
            if args.complete_original and info['stage']=='front_polar_unwinding' and head_unwrapped<=unwind_limit and head_unwrapped>=best_head_unwind+np.radians(5):
                best_head_unwind=head_unwrapped;last_progress=t;event='actual_first_unwind_head_ccw_5deg'
            if args.complete_original:
                if approach_best is None:approach_best=best_entry_distance
                if entry_distance<=approach_best-.003:approach_best=entry_distance;last_progress=t;event='actual_phase_entry_approach_3mm'
            wrap_age=wrap_age+DT if prefix>=3 and wrap>=best_wrap+.10 else 0
            if wrap_age>=.25:best_wrap=wrap;last_progress=t;event='actual_wrap_0p10rad';wrap_age=0
            intake_age=intake_age+DT if prefix>best_prefix and f[1,:prefix].sum()>.5 else 0
            if intake_age>=.5:best_prefix=prefix;last_progress=t;event='new_actual_bearing_prefix';intake_age=0
            actual_capture=prefix>=5 and wrap>2.3 and f[1,:prefix].sum()>.5
            capture_age=capture_age+DT if actual_capture else 0
            if capture_age>=1. and not capture:
                capture=True;capture_t=t;best_y=float(c[0,1]);last_progress=t;event='active_capture'
            axial_age=axial_age+DT if capture and prefix>=5 and c[0,1]>=best_y+.005 else 0
            if axial_age>=.25:best_y=float(c[0,1]);last_progress=t;event='maintained_target_advance_5mm';axial_age=0
            if event:progress_events.append(dict(t_s=t,event=event,station_mm=station*1000,prefix=prefix,wrap_rad=wrap))
            rt=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
            if 1.0<p[0,0]<1.10 and .15<p[0,1]<.30 and .88<p[0,2]<1.0:inner_lower=True
            if inner_lower and p[0,2]>1.02 and prefix>=2:upward=True
            source_geom=(sr>.074)&(sr<.120)&(c[:,0]<1.035)
            fs=shape(p);net=np.degrees(r['actual_source_ccw_rad'][12:]);ccw=bool(np.median(net)>=5 and np.mean(net>0)>=.70)
            end=fs['strict_single_helix'] and f[0].sum()<.1 and not r['diagnostic_source_contact_any'].any() and f[1,18:].sum()>.2 and r['kp_nm_per_rad'].min()>126.9
            complete_age=complete_age+DT if end else 0
            if complete_age>=1:
                full_success=bool(capture and inner_lower and upward and ccw)
                outcome='complete_active_inner_ccw_single_target_helix' if full_success else 'single_target_shape_but_entry_or_direction_failed';break
            if r['diagnostic_ground']:outcome='ground_contact';break
            if np.max(abs(r['q']))>np.radians(90.5):outcome='actual_joint_limit';break
            if np.max(abs(r['torque_nm']))>2.35501:outcome='actuator_limit_exceeded';break
            if r['diagnostic_self_depth_mm']>3:outcome='self_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if t-last_progress>=30:outcome='no_actual_progress_30s';break
            if len(commands)%100==0:
                msg=dict(t_s=round(t,2),wall_s=round(time.monotonic()-started,1),shift_mm=round(ctrl.shift*1000,1),stage=info['stage'],actual_prefix=prefix,actual_wrap=round(wrap,3),
                         source_support=r['source_support_count'],posterior_source_geometry=int(source_geom[-7:].sum()),head_mm=r['actual_centers_mm'][0],target_N=round(f[1].sum(),2),
                         tracking_deg=round(info['tracking_deg'],1),projection_mm=round(info['projection_rms_mm'],1),head_global_error_mm=round(info['head_global_error_mm'],1),head_execution_error_mm=round(info['head_execution_error_mm'],1),
                         actual_entry_distance_mm=round(entry_distance*1000,1),best_actual_entry_distance_mm=round(best_entry_distance*1000,1),
                         actual_progress_station_mm=round(best_station*1000,1),last_progress_s=round(last_progress,2),source_azimuth_deg=round(np.degrees(ctrl.angle),1))
                print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:
        error=traceback.format_exc();outcome='implementation_exception';print(error,flush=True)
    finally:
        np.savez_compressed(case/'trace.npz',**{k:np.asarray([r[k] for r in sim.rows]) for k in sim.rows[0]});write(case/'commands.json',commands)
        np.savez(case/'final_state.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl,time=sim.d.time)
        r=sim.rows[-1]
        result=dict(name=args.name,outcome=outcome,full_success=full_success,physical_run=True,actual_seconds=float(sim.d.time),wall_seconds=time.monotonic()-started,
             local_shape_off=args.local_shape_off,active_capture=capture,active_capture_first_s=capture_t,inner_lower_entry_observed=inner_lower,upward_wrap_observed=upward,
             maximum_actual_target_prefix=maximum_prefix,maximum_actual_wrap_rad=maximum_wrap,actual_progress_station_mm=best_station*1000,last_actual_progress_s=last_progress,
             final_shape=shape(r['actual_points_mm']/1000),source_ccw_deg=np.degrees(r['actual_source_ccw_rad']),
             final_global_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['global_points_mm'])**2,axis=1)))),
             final_execution_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['execution_points_mm'])**2,axis=1)))),
             maximum_actual_joint_deg=float(max(np.degrees(abs(x['q']).max()) for x in sim.rows)),maximum_torque_nm=float(max(abs(x['torque_nm']).max() for x in sim.rows)),
             maximum_pipe_depth_mm=float(max(x['diagnostic_pipe_depth_mm'] for x in sim.rows)),maximum_self_depth_mm=float(max(x['diagnostic_self_depth_mm'] for x in sim.rows)),
             ground=any(x['diagnostic_ground'] for x in sim.rows),physical_root_reset=False,events=ctrl.events,progress_events=progress_events,error=error,
             checkpoint_continuation_used=bool(args.resume),solver_warmstart_saved=False if args.resume else None,
             actual_head_unwind_deg=float(np.degrees(head_unwrapped)),maximum_credited_unwind_deg=float(np.degrees(best_head_unwind)),unwind_progress_cap_deg=float(np.degrees(unwind_limit)),progress_execution_error_filter=not(args.complete_original or args.progressive),
             frozen_T01_controller=None if args.progressive else str(OLD/'sources/snake_control/known90_unfold.py'),source_contact_count_hard_stop=False,source_geometry_count_hard_stop=False,source_radius_feedback=False,observations='MuJoCo actual pose permitted; contacts diagnostic-only',controller_estimated_capture=ctrl.acquired)
        write(case/'result.json',result);print(json.dumps(result,default=default),flush=True)
    if time.time()<DEADLINE:review(case)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--speed',type=float,default=.005)
    ap.add_argument('--source-x',type=float,default=.95);ap.add_argument('--source-phi',type=float,default=-45);ap.add_argument('--target-phi',type=float,default=-150)
    ap.add_argument('--local-shape-off',action='store_true');ap.add_argument('--initial-case');ap.add_argument('--review',action='store_true');ap.add_argument('--unfold',action='store_true',default=True);ap.add_argument('--progressive',action='store_true');ap.add_argument('--complete-original',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.review:review(BASE/args.name)
    else:run(args)
