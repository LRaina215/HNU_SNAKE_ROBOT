"""One continuous actual run: all source, inner probe, acquisition, transfer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback,xml.etree.ElementTree as ET
from datetime import datetime,timezone
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.suspended_supply_experiment import write,default,MODEL
from validation.inner_hook_transfer_experiment import InnerSim
from validation.hook_start_transfer_experiment import shape_stats,diagrams,render
from snake_control.suspended_supply_transfer import WholeCurveMapper,nodes,DT,LINK
from snake_control.full_source_inner_route import FullSourceInnerRoute
from snake_control.l_three_segment_transfer import ThreeSegmentTransfer
from snake_control.braced_inner_probe import BracedInnerRoute,BracedProbe

BASE=ROOT/'experiments/unified_curve_transfer_20261007/FULL_SOURCE_INNER_20261009'
IMODEL=ROOT/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'


def freeze(case):
    files=['snake_control/braced_inner_probe.py','snake_control/full_source_inner_route.py','snake_control/inner_hook_route.py','snake_control/hook_start_transfer.py','snake_control/hook_reform_route.py','snake_control/l_three_segment_transfer.py','snake_control/horizontal_pipe_roll.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py','snake_control/spatial_bishop_backbone.py','snake_control/hook_pull_review_support.py','snake_control/lower_hook_initial.py','snake_control/lower_bridge_hook_route.py','validation/full_source_inner_experiment.py','validation/inner_hook_transfer_experiment.py','validation/hook_start_transfer_experiment.py','validation/l_three_segment_experiment.py','validation/suspended_supply_experiment.py',str(IMODEL.relative_to(ROOT)),str(MODEL.relative_to(ROOT))]
    for f in files:
        dest=case/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    for mesh in ET.parse(IMODEL).findall('./asset/mesh'):
        f=mesh.get('file')
        if f:
            dest=case/'sources/mujoco_model'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'mujoco_model'/f,dest)


def curve_plot(case,route):
    grid=np.linspace(route.origin-21*LINK,21*LINK,3001);p=route.sample(grid)
    fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(111,projection='3d')
    for mask,label,color in [(grid<route.source_cut,'Source coil / +Y exit','darkorange'),((grid>=route.source_cut)&(grid<0),'Inner bridge','purple'),(grid>=0,'Target coil / -X upward entry','darkgreen')]:ax.plot(*p[mask].T,color=color,label=label)
    ax.plot([-.2,1.1],[0,0],[1,1],color='gray',lw=12,alpha=.3);ax.plot([1.1,1.1],[0,2],[1,1],color='gray',lw=12,alpha=.3)
    initial,_=route.window(0);ax.plot(*initial.T,'o-',ms=3,label='Actual artificial all-source start')
    ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)',title='One material curve, from all-source start to all-target end');ax.view_init(27,140);ax.legend();fig.tight_layout();fig.savefig(case/'complete_reference_curve.png',dpi=160);plt.close(fig)
    write(case/'complete_reference_curve.json',dict(material_coordinate_mm=grid*1000,points_mm=p*1000,origin_mm=route.origin*1000,source_cut_mm=route.source_cut*1000,complete_shift_mm=route.total*1000))


def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    tree=ET.parse(MODEL);tree.getroot().set('model','full-source-inner-probe-transfer');tree.find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto','0.15 0 1.1 2.0 0 1.1');tree.write(IMODEL,encoding='utf-8',xml_declaration=True)
    route=BracedInnerRoute() if args.braced else FullSourceInnerRoute(bridge=args.bridge)
    m=mujoco.MjModel.from_xml_path(str(IMODEL));mapper=WholeCurveMapper(m)
    if args.initial:
        initial=np.load(args.initial);fit={'reused_full_source_pose':args.initial,'physical_steps':0}
    else:
        guide=route.initial_window() if args.braced else route.window(0)[0];pose,fit=mapper.initialize(guide)
        initial=dict(qpos=pose,qvel=np.zeros(m.nv),ctrl=pose[mapper.qa])
    sim=InnerSim(case,initial,model_path=IMODEL);r0=sim.snapshot();p=r0['points_mm']/1000;c=r0['centers_mm']/1000
    if args.braced:route.bind_source(p)
    else:route.bind_initial(p)
    sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,1]));dx=np.diff(c[:,0]);da=np.diff(phi)
    ready=bool(np.all((sr>.075)&(sr<.110)) and np.all(dx<0) and np.all(da<0) and np.ptp(phi)>4*np.pi and r0['pipe_depth_mm']<1.2 and r0['self_depth_mm']<.5 and not r0['ground'] and r0['contact_force_N'][1].sum()<.1)
    geometry=dict(initial_state='artificial full-source twenty-hinge fit before first physical step, zero velocity, ctrl equals actual hinges',physical_steps=0,source_start_module=0,front_modules=0,source_modules=21,initial_points_mm=p*1000,source_radius_mm=sr*1000,source_wrap_deg=np.degrees(np.ptp(phi)),source_force_n=r0['contact_force_N'][0].sum(),target_force_n=r0['contact_force_N'][1].sum(),pipe_depth_mm=r0['pipe_depth_mm'],self_depth_mm=r0['self_depth_mm'],maximum_joint_deg=np.degrees(abs(r0['q']).max()),ready=ready,fit=fit,complete_shift_mm=route.total*1000)
    write(case/'initial_geometry.json',geometry);np.savez(case/'initial_pose.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl)
    mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'initial_state.npz',integration_state=sim.state,ctrl=sim.d.ctrl,sim_time=0.)
    freeze(case);curve_plot(case,route);diagrams(case,route)
    if args.braced:
        write(case/'braced_probe_reference.json',dict(initial_points_mm=p*1000,unfold_end_points_mm=route.probe_window('unfold',1)[0]*1000,coil_end_points_mm=route.probe_window('coil',1)[0]*1000,probe_curve_mm=route.probe*1000,source_hold_modules=list(range(12,21)),target_entry_point_mm=route.e*1000,target_entry_tangent=route.entry_tangent,probe_inside=bool(np.all((route.probe[:,0]<1.1)&(route.probe[:,1]>0))),coil_bridge_inside=bool(np.all((route.connector[:,0]<1.1)&(route.connector[:,1]>0)))))
    print(json.dumps(geometry,default=default),flush=True)
    if not ready:write(case/'result.json',dict(outcome='initial_geometry_not_ready',physical_steps=0));return
    if args.prepare_only:
        render(case,sim.rows,model_name=IMODEL.name,initial_label='Full source inner start');write(case/'result.json',dict(outcome='all_source_geometry_ready',physical_steps=0));return
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),maximum_branch_trials=3,action='same shared material shift from source helix through inner bridge into target helix; source/bridge/target chord frame then local roll; no initial target admission',speed_mm_s=args.speed*1000,rate_deg_s=args.rate,bridge_rods=args.bridge,complete_shift_mm=route.total*1000,max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,probe_limit_seconds=180,acquisition_limit_seconds=280,completion='I01 strict actual all21 coil, original source absent, posterior target bearing, all Kp127 for one actual second',physical_root_reset=False,yamada_compliance_enabled=False))
    ctrl=BracedProbe(sim.m,sim.scratch,route,unfold_seconds=args.unfold_seconds,coil_seconds=args.coil_seconds) if args.braced else ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,adaptive_supply=True,route=route,source_start=0,intake_y=.15,preserve_preload=True,handoff_on_posterior_exit=True,continuous_reform_handoff=True,event_progress=True,initial_front=0,acquisition=True,anchored_probe=args.anchored_probe)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;complete_since=None;window_time=None;hold=0.;error=None;milestones={};acquired=False
    try:
        while sim.d.time<args.seconds:
            elapsed=float(sim.d.time)
            if (case/'stop_and_save.request').exists():outcome='requested_bounded_stop';break
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,elapsed,r['contact_force_N'],r['source_contact_any']);sim.info=info;sim.commands.append(dict(t_s=elapsed,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r)
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if r['contact_force_N'][1].sum()>.2 and 'first_target_bearing' not in milestones:milestones['first_target_bearing']=dict(t_s=r['sim_time'],shift_mm=ctrl.shift*1000)
            prefix=0
            while prefix in ctrl.received:prefix+=1
            front=6 if args.braced else 8;cc=r['centers_mm']/1000;wrap=np.degrees(np.ptp(np.unwrap(np.arctan2(cc[:front,2]-1,cc[:front,0]-1.1))))
            if prefix>=front and wrap>180 and r['contact_force_N'][1,:front].sum()>1:
                if not acquired:milestones['actual_front_acquisition']=dict(t_s=r['sim_time'],shift_mm=ctrl.shift*1000,prefix=prefix,wrap_deg=np.degrees(r['actual_front_span_rad']),points_mm=r['points_mm']);acquired=True
            if isinstance(ctrl,BracedProbe) and ctrl.ready:
                route.bind_acquired(r['points_mm']/1000);old_kp=ctrl.kp.copy();old_ctrl=ctrl.last.copy()
                mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'actual_acquired_state.npz',integration_state=sim.state,ctrl=sim.d.ctrl,sim_time=sim.d.time)
                ctrl=ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,adaptive_supply=True,route=route,source_start=route.source_start,intake_y=.15,preserve_preload=True,handoff_on_posterior_exit=True,continuous_reform_handoff=True,event_progress=True,initial_front=6)
                ctrl.kp=old_kp;ctrl.last=old_ctrl;ctrl.last_progress=elapsed
                milestones['continuous_transfer_handoff']=dict(t_s=r['sim_time'],material_shift_mm=0.,material_coordinate_mm=route.origin-np.arange(22)*LINK,root_reset=False,ctrl_difference_rad=ctrl.last-old_ctrl)
            if not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and 'source_clear' not in milestones:milestones['source_clear']=dict(t_s=r['sim_time'],shift_mm=ctrl.shift*1000)
            if acquired:
                cc=r['centers_mm']/1000;rr=np.linalg.norm(cc[:8,[0,2]]-[1.1,1],axis=1);falling=r['contact_force_N'][1,:8].sum()<.1 and (np.median(rr)>.145 or cc[:8,2].mean()<.9);fall=fall+DT if falling else 0.
                if fall>=1:outcome='acquired_front_bearing_collapsed_and_falling';break
            if elapsed>180 and 'first_target_bearing' not in milestones:outcome='probe_without_target_bearing_180s';break
            if elapsed>280 and not acquired:outcome='probe_without_coherent_front_acquisition_280s';break
            if ctrl.failure:outcome=ctrl.failure;break
            full=ctrl.shift>=route.total-1e-9
            if full and window_time is None:window_time=float(r['sim_time'])
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2 and np.all(ctrl.kp>=126.999)
            if complete and complete_since is None:complete_since=float(r['sim_time'])
            if not complete:complete_since=None
            hold=0. if complete_since is None else float(r['sim_time'])-complete_since
            if hold>=1:outcome='complete_from_full_source';break
            if full and r['sim_time']-window_time>30:outcome='whole_reference_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),stage=info['stage'],shift_mm=round(ctrl.shift*1000,2),exited=sorted(ctrl.exited),received=sorted(ctrl.received),force_n=r['contact_force_N'].sum(1),head_mm=r['points_mm'][0],front_wrap_deg=np.degrees(r['actual_front_span_rad']),tracking_deg=info['tracking_deg']);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,complete_shift_mm=route.total*1000,initial_all_source=ready,active_hook_acquisition_demonstrated=acquired,milestones=milestones,final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),actual_current_received_target=sorted(ctrl.received),final_kp=ctrl.kp,maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),actual_completion_hold_seconds=hold,whole_transfer_complete=outcome=='complete_from_full_source',physical_root_reset_during_run=False,initial_state_is_artificial=True,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,engaged=sorted(getattr(ctrl,'engaged',ctrl.received)),exited=sorted(ctrl.exited),bridge_coeff=getattr(ctrl,'coeff',np.zeros((2,3)))));print(json.dumps(result,default=default),flush=True)
        diagrams(case,route,sim.rows)
        try:render(case,sim.rows,model_name=IMODEL.name,initial_label='Full source inner probe')
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='F01');ap.add_argument('--bridge',type=int,default=5);ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--rate',type=float,default=20);ap.add_argument('--seconds',type=float,default=650);ap.add_argument('--max-wall',type=float,default=800);ap.add_argument('--deadline',default='2026-10-09T08:20:27+00:00');ap.add_argument('--initial');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--anchored-probe',action='store_true');ap.add_argument('--braced',action='store_true');ap.add_argument('--unfold-seconds',type=float,default=70);ap.add_argument('--coil-seconds',type=float,default=80);run(ap.parse_args())
