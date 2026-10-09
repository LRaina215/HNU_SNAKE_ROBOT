"""Bounded all-source physical trial, with no post-step physical-state assignment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np,mujoco
from validation.full_source_inner_experiment import BASE,IMODEL,freeze
from validation.inner_hook_transfer_experiment import InnerSim
from validation.hook_start_transfer_experiment import shape_stats,render
from validation.suspended_supply_experiment import write,default
from snake_control.head_first_inner_control import HeadFirstInnerControl
from snake_control.suspended_supply_transfer import DT

def run(args):
    pre=json.loads((BASE/'HEAD_FIRST_LOWER_PREFLIGHT/summary.json').read_text())
    if not pre['ready']:raise RuntimeError('Offline geometry did not pass; no formal dynamics consumed')
    if args.name not in ['F06','F07']:raise ValueError('This authorization allows only F06 and F07')
    case=BASE/args.name;case.mkdir(exist_ok=False)
    initial=np.load(BASE/'BRACED_PREPARE_13/initial_pose.npz');sim=InnerSim(case,initial,model_path=IMODEL);r0=sim.snapshot()
    geometry=json.loads((BASE/'BRACED_PREPARE_13/initial_geometry.json').read_text());geometry['source_start_module']=0;geometry['physical_steps']=0;geometry['initial_points_mm']=r0['points_mm'];geometry['initial_state']='Same artificial all21 source start as F04/F05, before first physical step; no initial hook';write(case/'initial_geometry.json',geometry)
    np.savez(case/'initial_pose.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl)
    mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'initial_state.npz',integration_state=sim.state,sim_time=0.)
    freeze(case)
    for f in ['snake_control/head_first_inner_control.py','snake_control/head_first_inner_route.py','validation/head_first_experiment.py','validation/head_first_preflight.py']:
        dest=case/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    shutil.copy2(BASE/'HEAD_FIRST_LOWER_PREFLIGHT/summary.json',case/'offline_preflight.json')
    plan=dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),maximum_new_formal_trials=2,action='Normal source helix approach; change source pitch180 to80mm, actual backoff; change80 to300mm, actual reapproach to lower inner head exit; then head-first fixed source/bridge/target route with every rod exact64.5mm and one shared shift',repositioning_hypothesis='Different source pitches may produce different circumferential/axial rolling ratios. This is unverified until actual dynamics.',initial_state_is_artificial=True,physical_root_reset_during_run=False,original_physics='I01 robot, gravity, friction, inertia, damping1.5, torque2.355Nm, Kp127 source, ctrl88deg; actual limit90.5deg stop',max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,source_reposition_limit_seconds=220,completion='Strict actual all21 target single coil, source clear, posterior target bearing, all Kp127 for one actual second',yamada_compliance_enabled=False)
    if args.actual_exit_gate:plan.update(action='Same real approach/pitch/backoff as F06, but enter the head-first route immediately when actual lower-inner head, upward entry and16 posterior source radii qualify; actual-state exact-rod20-hinge preflight before feed; measured source spline and contact-registered joint projection during supported intake',basis='F06 actual111.2-113.1s lower inner upward exit, with18 posterior rods still near source; fixed750mm retreat target missed this usable event',actual_exit_gate=True,anchored_feed=args.anchored_feed)
    write(case/'plan.json',plan);print(json.dumps(plan),flush=True)
    ctrl=HeadFirstInnerControl(sim.m,sim.scratch,actual_exit_gate=args.actual_exit_gate,anchored_feed=args.anchored_feed);start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;acquired=False;hold=0.;complete_since=None;window_time=None;error=None;milestones={};written=set()
    try:
        while sim.d.time<args.seconds:
            t=float(sim.d.time)
            if (case/'stop_and_save.request').exists():outcome='requested_bounded_stop';break
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,t,r['contact_force_N'],r['source_contact_any']);sim.info=info;sim.commands.append(dict(t_s=t,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r)
            for name,event in ctrl.events.items():
                if name not in written:write(case/(name+'.json'),event);written.add(name)
            if ctrl.route is not None and 'continuous_curve' not in written:
                grid=np.linspace(ctrl.route.knots[-1],ctrl.route.total,3001);write(case/'continuous_curve.json',dict(stations_mm=grid*1000,points_mm=ctrl.route.sample(grid)*1000,bridge_length_mm=ctrl.route.bridge_length*1000,source_cut_mm=0,target_join_mm=ctrl.route.target_join*1000,shared_material_offset_mm=ctrl.material_offset*1000));written.add('continuous_curve')
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if r['contact_force_N'][1].sum()>.2 and 'first_target_bearing' not in milestones:milestones['first_target_bearing']=dict(t_s=r['sim_time'],shared_material_shift_mm=ctrl.material_shift*1000)
            prefix=0
            while prefix in ctrl.received:prefix+=1
            c=r['centers_mm']/1000;wrap=np.degrees(np.ptp(np.unwrap(np.arctan2(c[:8,2]-1,c[:8,0]-1.1))))
            if prefix>=8 and wrap>180 and r['contact_force_N'][1,:8].sum()>1:
                if not acquired:milestones['actual_front_acquisition']=dict(t_s=r['sim_time'],prefix=prefix,wrap_deg=wrap,points_mm=r['points_mm']);acquired=True
            if acquired:
                rr=np.linalg.norm(c[:8,[0,2]]-[1.1,1],axis=1);falling=r['contact_force_N'][1,:8].sum()<.1 and (np.median(rr)>.145 or c[:8,2].mean()<.9);fall=fall+DT if falling else 0.
                if fall>=1:outcome='acquired_front_bearing_collapsed_and_falling';break
            if ctrl.failure:outcome=ctrl.failure;break
            full=ctrl.route is not None and ctrl.shift>=ctrl.route.total-1e-9
            if full and window_time is None:window_time=float(r['sim_time'])
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2 and np.all(ctrl.kp>=126.999)
            if complete and complete_since is None:complete_since=float(r['sim_time'])
            if not complete:complete_since=None
            hold=0 if complete_since is None else float(r['sim_time'])-complete_since
            if hold>=1:outcome='complete_from_full_source';break
            if full and r['sim_time']-window_time>30:outcome='whole_reference_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),stage=ctrl.stage,pitch_mm=ctrl.pitch*1000,shared_material_shift_mm=ctrl.material_shift*1000,local_feed_shift_mm=ctrl.shift*1000,force_n=r['contact_force_N'].sum(1),head_mm=r['points_mm'][0],received=sorted(ctrl.received),tracking_deg=info['tracking_deg'],predicted_pipe_depth_mm=info['projection']['predicted_pipe_depth_mm']);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],wall_seconds=time.monotonic()-start,stage=ctrl.stage,shared_material_shift_mm=ctrl.material_shift*1000,local_feed_shift_mm=ctrl.shift*1000,initial_all_source=True,active_hook_acquisition_demonstrated=acquired,whole_transfer_complete=outcome=='complete_from_full_source',events=ctrl.events,milestones=milestones,final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),actual_current_received_target=sorted(ctrl.received),final_kp=ctrl.kp,maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),actual_completion_hold_seconds=hold,physical_root_reset_during_run=False,initial_state_is_artificial=True,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,stage=ctrl.stage,shared_material_shift_mm=ctrl.material_shift*1000,local_feed_shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase));print(json.dumps(result,default=default),flush=True)
        try:render(case,sim.rows,model_name=IMODEL.name,initial_label='Full source head-first start')
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='F06');ap.add_argument('--seconds',type=float,default=650);ap.add_argument('--max-wall',type=float,default=800);ap.add_argument('--deadline',default='2026-10-09T09:05:00+00:00');ap.add_argument('--actual-exit-gate',action='store_true');ap.add_argument('--anchored-feed',action='store_true');run(ap.parse_args())
