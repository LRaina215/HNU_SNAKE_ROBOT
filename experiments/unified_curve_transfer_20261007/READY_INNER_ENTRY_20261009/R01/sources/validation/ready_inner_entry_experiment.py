"""Bounded direct inner probing from a disclosed artificial all-source start."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np,mujoco
from scipy.optimize import brentq
from validation.full_source_inner_experiment import IMODEL,freeze
from validation.inner_hook_transfer_experiment import InnerSim
from validation.hook_start_transfer_experiment import shape_stats,render
from validation.suspended_supply_experiment import write,default
from snake_control.suspended_supply_transfer import WholeCurveMapper,nodes,LINK,DT
from snake_control.suspended_transfer_route import R,ALPHA,BETA
from snake_control.ready_inner_entry import ReadyInnerEntry

BASE=ROOT/'experiments/unified_curve_transfer_20261007/READY_INNER_ENTRY_20261009'

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(IMODEL));mapper=WholeCurveMapper(m)
    if args.initial:initial=np.load(args.initial);fit=dict(reused_artificial_all_source_pose=args.initial)
    else:
        h=brentq(lambda x:(ALPHA*x)**2+(2*R*np.sin(BETA*x/(2*R)))**2-LINK**2,LINK,LINK*1.2);u=-np.arange(22)*h
        guide=np.column_stack([.98+ALPHA*u,R*np.cos(args.phi+BETA*u/R),1+R*np.sin(args.phi+BETA*u/R)])
        pose,fit=mapper.initialize(guide);initial=dict(qpos=pose,qvel=np.zeros(m.nv),ctrl=pose[mapper.qa])
    sim=InnerSim(case,initial,model_path=IMODEL);r0=sim.snapshot();p=nodes(sim.scratch);c=(p[:-1]+p[1:])/2;sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);entry=p[0]-p[1]
    ready=bool(np.all((sr>.075)&(sr<.110)) and p[0,1]>.02 and p[0,2]<1 and entry[1]>0 and entry[2]>0 and r0['contact_force_N'][1].sum()<.1 and r0['pipe_depth_mm']<1.5 and r0['self_depth_mm']<.5 and not r0['ground'] and np.degrees(abs(r0['q']).max())<=88)
    geometry=dict(initial_state='Artificial all21 original-pipe helix, head lower inner and upward; zero velocity, no target contact; set only before first physical step',physical_steps=0,source_start_module=0,source_modules=21,front_modules=0,initial_points_mm=p*1000,source_radius_mm=sr*1000,head_entry_vector_mm=entry*1000,initial_source_target_force_n=r0['contact_force_N'].sum(1),pipe_depth_mm=r0['pipe_depth_mm'],self_depth_mm=r0['self_depth_mm'],fit=fit,ready=ready)
    write(case/'initial_geometry.json',geometry);np.savez(case/'initial_pose.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl)
    mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'initial_state.npz',integration_state=sim.state,sim_time=0.)
    freeze(case)
    for name in ['snake_control/ready_inner_entry.py','snake_control/head_first_inner_control.py','snake_control/head_first_inner_route.py','validation/ready_inner_entry_experiment.py']:
        dest=case/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    ctrl=ReadyInnerEntry(sim.m,sim.scratch,anchored_feed=args.anchored_feed);route=ctrl.route;v=np.linspace(route.knots[-1],route.total,3001)
    write(case/'complete_curve.json',dict(status='planned complete route; check each current command before actual dispatch',stations_mm=v*1000,points_mm=route.sample(v)*1000,bridge_length_mm=route.bridge_length*1000,target_join_mm=route.target_join*1000,complete_shift_mm=route.total*1000))
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),maximum_formal_trials=3,action='Head-first inner lower probe directly from artificial all-source helix; exact64.5mm chords along one fixed source/bridge/target curve, sharedshift, local roll and I01 receipt/soft supply logic',anchored_feed=args.anchored_feed,initial_state_is_artificial=True,physical_root_reset_during_run=False,yamada_compliance_enabled=False,immediate_guard='current candidate actual-root FK and projected pose pipe/self depth before dispatch; do not gate on all future fit windows',max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,completion='Strict all21 target coil, source absent, posterior target bearing and allKp127 held one actual second'))
    print(json.dumps(geometry,default=default),flush=True)
    if not ready:write(case/'result.json',dict(outcome='artificial_initial_pose_not_ready',physical_steps=0));return
    if args.prepare_only:render(case,sim.rows,model_name=IMODEL.name,initial_label='Full source ready lower inner start');write(case/'result.json',dict(outcome='artificial_all_source_inner_start_ready',physical_steps=0));return
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;acquired=False;hold=0.;complete_since=None;window_time=None;error=None;milestones={};rejected=None
    try:
        while sim.d.time<args.seconds:
            t=float(sim.d.time)
            if (case/'stop_and_save.request').exists():outcome='requested_bounded_stop';break
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];q,info=ctrl.command(sim.scratch,t,r['contact_force_N'],r['source_contact_any'])
            if ctrl.failure=='unsafe_current_joint_command_before_dispatch':outcome=ctrl.failure;rejected=dict(t_s=t,ctrl_rad=q,**info);break
            sim.info=info;sim.commands.append(dict(t_s=t,ctrl_rad=q.copy(),**info));sim.command(q);r=sim.rows[-1];stats=shape_stats(r)
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if r['contact_force_N'][1].sum()>.2 and 'first_target_bearing' not in milestones:milestones['first_target_bearing']=dict(t_s=r['sim_time'],shift_mm=ctrl.shift*1000)
            prefix=0
            while prefix in ctrl.received:prefix+=1
            c=r['centers_mm']/1000;wrap=np.degrees(np.ptp(np.unwrap(np.arctan2(c[:8,2]-1,c[:8,0]-1.1))))
            if prefix>=8 and wrap>180 and r['contact_force_N'][1,:8].sum()>1:
                if not acquired:milestones['actual_front_acquisition']=dict(t_s=r['sim_time'],prefix=prefix,wrap_deg=wrap,points_mm=r['points_mm']);acquired=True
            if acquired:
                rr=np.linalg.norm(c[:8,[0,2]]-[1.1,1],axis=1);falling=r['contact_force_N'][1,:8].sum()<.1 and (np.median(rr)>.145 or c[:8,2].mean()<.9);fall=fall+DT if falling else 0.
                if fall>=1:outcome='acquired_front_bearing_collapsed_and_falling';break
            if ctrl.failure:outcome=ctrl.failure;break
            full=ctrl.shift>=route.total-1e-9
            if full and window_time is None:window_time=float(r['sim_time'])
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2 and np.all(ctrl.kp>=126.999)
            if complete and complete_since is None:complete_since=float(r['sim_time'])
            if not complete:complete_since=None
            hold=0 if complete_since is None else float(r['sim_time'])-complete_since
            if hold>=1:outcome='complete_from_ready_inner_all_source';break
            if full and r['sim_time']-window_time>30:outcome='whole_reference_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),shift_mm=ctrl.shift*1000,received=sorted(ctrl.received),source_target_force_n=r['contact_force_N'].sum(1),head_mm=r['points_mm'][0],tracking_deg=info['tracking_deg'],private_pipe_depth_mm=info['projection']['predicted_pipe_depth_mm']);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        if rejected:write(case/'rejected_current_command.json',rejected)
        result=dict(name=args.name,outcome=outcome,physical_steps=int(round(sim.d.time/sim.m.opt.timestep)),actual_seconds=r['sim_time'],wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,complete_shift_mm=route.total*1000,initial_all_source=ready,active_hook_acquisition_demonstrated=acquired,whole_transfer_complete=outcome=='complete_from_ready_inner_all_source',milestones=milestones,final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),actual_current_received_target=sorted(ctrl.received),final_kp=ctrl.kp,maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),actual_completion_hold_seconds=hold,physical_root_reset_during_run=False,initial_state_is_artificial=True,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats);write(case/'result.json',result);print(json.dumps(result,default=default),flush=True)
        render(case,sim.rows,model_name=IMODEL.name,initial_label='Full source ready lower inner start')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='R01');ap.add_argument('--phi',type=float,default=-.6);ap.add_argument('--initial');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--anchored-feed',action='store_true');ap.add_argument('--seconds',type=float,default=650);ap.add_argument('--max-wall',type=float,default=800);ap.add_argument('--deadline',default='2026-10-09T09:45:43+00:00');run(ap.parse_args())
