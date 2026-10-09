"""Bounded actual transfer from an artificial inner-side upward hook."""
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
from validation.l_three_segment_experiment import LSim
from validation.hook_start_transfer_experiment import shape_stats,diagrams,render
from snake_control.suspended_supply_transfer import WholeCurveMapper,nodes,DT,LINK
from snake_control.inner_hook_route import InnerHookRoute
from snake_control.l_three_segment_transfer import ThreeSegmentTransfer

BASE=ROOT/'experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009'
IMODEL=ROOT/'mujoco_model/Snake_Robot_o20_inner_hook_transfer.xml'


class InnerSim(LSim):
    def __init__(self,case,initial):
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(IMODEL));self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]];self.head=self.m.body('head').id
        self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']];self.target=[self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor=self.m.geom('floor').id;self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION));self.d.qpos[:]=initial['qpos'];self.d.qvel[:]=initial['qvel'];self.d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(self.m,self.d)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21);self.exited=set();self.received=set()


def direction_plots(case,route,points):
    old=json.loads((ROOT/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009/U04/initial_geometry.json').read_text(encoding='utf-8'))
    oldp=np.asarray(old['initial_reference_points_mm'])/1000;fig,axes=plt.subplots(2,2,figsize=(14,11));theta=np.linspace(0,2*np.pi,200)
    for row,(p,front,start,title) in enumerate([(oldp,8,16,'U04 prior actual initial'),(points,route.front,route.source_start,'New actual inner-side upward hook')]):
        ax=axes[row,0];ax.fill([.4,1.1,1.1,.4],[0,0,.85,.85],color='gold',alpha=.10,label='Inner angle: X<1.1, Y>0')
        ax.plot([-.2,1.1],[0,0],color='gray',lw=20,alpha=.3,label='Finite source pipe');ax.plot([1.1,1.1],[0,2],color='slategray',lw=20,alpha=.3,label='Finite target pipe')
        ax.plot(p[:,0],p[:,1],'.-',color='steelblue',label='Actual N0 head -> N21 tail');ax.plot(p[front:start+1,0],p[front:start+1,1],'o-',color='darkorange',lw=3,label='Actual transition')
        for a,b in [(start,start-1),(front+1,front),(front,front-1),(3,2),(1,0)]:ax.annotate('',xy=p[b,:2],xytext=p[a,:2],arrowprops=dict(arrowstyle='->',color='crimson',lw=2))
        for j in [0,front,front+1,start,21]:ax.annotate('Head N0' if j==0 else 'Tail N21' if j==21 else f'N{j}',p[j,:2],xytext=(4,7),textcoords='offset points',fontsize=8)
        ax.set(title=title+' / top XY',xlabel='World X(m)',ylabel='World Y(m)',xlim=(.4,1.3),ylim=(-.16,.85),aspect='equal');ax.legend(fontsize=7,loc='upper left')
        ax=axes[row,1];ax.fill(1.1+.0565*np.cos(theta),1+.0565*np.sin(theta),color='gray',alpha=.3,label='Target cross section')
        ax.plot(p[:,0],p[:,2],'.-',color='steelblue',alpha=.6,label='Actual projected body');ax.plot(p[front:start+1,0],p[front:start+1,2],'o-',color='darkorange',lw=3)
        ax.plot(p[:front+1,0],p[:front+1,2],'o-',color='darkgreen',label='Actual target wrap')
        for a,b in [(front+1,front),(front,front-1),(front-1,front-2),(3,2),(1,0)]:ax.annotate('',xy=p[b,[0,2]],xytext=p[a,[0,2]],arrowprops=dict(arrowstyle='->',color='crimson',lw=2))
        for j in [0,front-1,front,front+1,21]:ax.annotate('Head N0' if j==0 else 'Tail N21' if j==21 else f'N{j}',p[j,[0,2]],xytext=(4,7),textcoords='offset points',fontsize=8)
        ax.axvline(1.1,color='gray',ls=':');ax.axhline(1,color='gray',ls=':');ax.annotate('',xy=(.70,1.16),xytext=(.70,1.07),arrowprops=dict(arrowstyle='->'));ax.text(.64,1.055,'World Z UP',fontsize=8)
        ax.set(title=title+' / target XZ',xlabel='World X(m); target inner side is left',ylabel='World Z(m)',xlim=(.6,1.25),ylim=(.80,1.20),aspect='equal');ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Actual initial poses in unchanged world coordinates; red arrows follow tail -> head')
    fig.tight_layout();fig.savefig(case/'inner_entry_top_side_comparison.png',dpi=160);plt.close(fig)
    grid=np.linspace(route.origin-21*LINK,route.origin+route.total,2401);p=route.sample(grid);fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(111,projection='3d')
    for mask,label,color in [(grid<route.source_cut,'Source helix / +Y outlet','darkorange'),((grid>=route.source_cut)&(grid<0),'Inner transition','purple'),(grid>=0,'Target helix / -X upward entry','darkgreen')]:ax.plot(p[mask,0],p[mask,1],p[mask,2],color=color,lw=2,label=label)
    ax.plot(points[:,0],points[:,1],points[:,2],'o--',ms=3,color='steelblue',label='Actual artificial initial')
    ax.plot([-.2,1.1],[0,0],[1,1],color='gray',lw=10,alpha=.3);ax.plot([1.1,1.1],[0,2],[1,1],color='gray',lw=10,alpha=.3)
    ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)',title='Complete reference: source +Y outlet / inner transition / target -X hook');ax.view_init(27,140);ax.legend();fig.tight_layout();fig.savefig(case/'complete_reference_curve.png',dpi=150);plt.close(fig)
    write(case/'complete_reference_curve.json',dict(material_coordinate_mm=grid*1000,points_mm=p*1000,source_cut_mm=route.source_cut*1000,target_join_mm=0,origin_mm=route.origin*1000,complete_shift_mm=route.total*1000))


def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    tree=ET.parse(MODEL);tree.getroot().set('model','inner-side-upward-hook-transfer');tree.find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto','0.15 0 1.1 2.0 0 1.1');tree.write(IMODEL,encoding='utf-8',xml_declaration=True)
    route=InnerHookRoute(front=args.front,bridge=args.bridge);initial=np.load(BASE/'PREPARE/initial_pose.npz');sim=InnerSim(case,initial);r0=sim.snapshot();p=r0['points_mm']/1000;route.bind_initial(p)
    prep=json.loads((BASE/'PREPARE/initial_geometry.json').read_text(encoding='utf-8'));entry=p[route.front-1]-p[route.front];transition=p[route.front:route.source_start+1]
    geometry=dict(initial_state='new artificial inner-side full-curve twenty-hinge fit; zero velocity; ctrl equals actual joints',physical_steps=0,active_hook_acquisition_demonstrated=False,
        initial_front8_wrap_deg=float(np.degrees(r0['actual_front_span_rad'])),initial_front9_wrap_deg=float(np.degrees(np.ptp(np.unwrap(np.arctan2(r0['centers_mm'][:9,2]-1000,r0['centers_mm'][:9,0]-1100))))),
        initial_points_mm=p*1000,source_start_module=route.source_start,front_modules=route.front,bridge_modules=route.bridge,source_modules=21-route.source_start,
        source_outlet_point_mm=p[route.source_start]*1000,target_entry_point_mm=p[route.front]*1000,target_tail_to_head_entry_vector=entry,
        actual_transition_all_inside=bool(np.all((transition[:,0]<1.1)&(transition[:,1]>0))),source_outlet_toward_positive_y=bool(p[route.source_start,1]>0),target_entry_inward_and_upward=bool(entry[0]<0 and entry[2]>0 and p[route.front,0]<1.1 and p[route.front,2]<1),
        initial_source_target_force_n=r0['contact_force_N'].sum(1),initial_pipe_depth_mm=r0['pipe_depth_mm'],maximum_joint_deg=float(np.degrees(abs(r0['q']).max())),initial_geometry_fit=prep,complete_shift_mm=route.total*1000,target_pipe_end_y_m=2.)
    geometry['ready']=bool(prep['ready'] and geometry['actual_transition_all_inside'] and geometry['source_outlet_toward_positive_y'] and geometry['target_entry_inward_and_upward'] and geometry['initial_front8_wrap_deg']>180 and r0['pipe_depth_mm']<3 and not r0['ground'])
    write(case/'initial_geometry.json',geometry);print(json.dumps(geometry,default=default),flush=True)
    np.savez(case/'initial_pose.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl);mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'initial_state.npz',integration_state=sim.state,ctrl=sim.d.ctrl,sim_time=0.)
    files=['snake_control/inner_hook_route.py','snake_control/hook_start_transfer.py','snake_control/hook_reform_route.py','snake_control/l_three_segment_transfer.py','snake_control/horizontal_pipe_roll.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py','snake_control/spatial_bishop_backbone.py','snake_control/hook_pull_review_support.py','snake_control/lower_hook_initial.py','snake_control/lower_bridge_hook_route.py','validation/inner_hook_transfer_experiment.py','validation/hook_start_transfer_experiment.py','validation/l_three_segment_experiment.py','validation/suspended_supply_experiment.py',str(IMODEL.relative_to(ROOT)),str(MODEL.relative_to(ROOT))]
    for f in files:dest=case/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    for mesh in ET.parse(IMODEL).findall('./asset/mesh'):
        f=mesh.get('file')
        if f:dest=case/'sources/mujoco_model'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'mujoco_model'/f,dest)
    direction_plots(case,route,p);diagrams(case,route)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),first_candidate_maximum_dynamics_trials=3,initial_state_is_artificial=True,
        action='source+Y outlet, five-rod inner bridge, target-X upward hook; same shared material shift, whole-curve hinge projection and roll; U04 continuous front reform handoff',
        source_start_module=route.source_start,complete_shift_mm=route.total*1000,speed_mm_s=args.speed*1000,rate_setting_deg_s=args.rate,morph_length_mm=200,max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,
        original_physics='same robot/gravity/friction/drive/damping1.5/torque2.355Nm/actualjoint90deg/ctrl88deg; prior target length2m',
        complete='whole material window; strict all21 radius/pitch/order/axis/localseam, source absent, posterior bearing, Kp127, actual elapsed1s',
        stop='ground; actualjoint>90.5deg; self>3mm; pipe>3mm sustained0.2s; nonfinite; front bearing collapsed/falling1s; no actual new event or continuous intake45s; full window without shape30s; same overall limits',yamada_compliance_enabled=False))
    if not geometry['ready']:write(case/'result.json',dict(outcome='initial_geometry_not_ready',physical_steps=0));return
    if args.prepare_only:render(case,sim.rows,model_name=IMODEL.name,initial_label='Artificial inner upward hook');write(case/'result.json',dict(outcome='geometry_ready_without_dynamics',physical_steps=0));return
    ctrl=ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,adaptive_supply=True,route=route,source_start=route.source_start,intake_y=.15,preserve_preload=True,handoff_on_posterior_exit=True,continuous_reform_handoff=True,event_progress=True)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;complete_since=None;window_time=None;hold=0.;error=None
    try:
        while sim.d.time<args.seconds:
            elapsed=float(sim.d.time)
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,elapsed,r['contact_force_N'],r['source_contact_any']);sim.info=info;sim.commands.append(dict(t_s=elapsed,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r)
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:8,[0,2]]-[1.1,1],axis=1);falling=r['contact_force_N'][1,:8].sum()<.1 and (np.median(rr)>.145 or c[:8,2].mean()<.9);fall=fall+DT if falling else 0.
            if fall>=1:outcome='front_bearing_collapsed_and_falling';break
            if ctrl.failure:outcome=ctrl.failure;break
            full=ctrl.shift>=route.total-1e-9
            if full and window_time is None:window_time=float(r['sim_time'])
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2 and np.all(ctrl.kp>=126.999)
            if complete and complete_since is None:complete_since=float(r['sim_time'])
            if not complete:complete_since=None
            hold=0. if complete_since is None else float(r['sim_time'])-complete_since
            if hold>=1:outcome='complete_from_artificial_inner_hook';break
            if full and r['sim_time']-window_time>30:outcome='whole_reference_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),shift_mm=round(ctrl.shift*1000,2),phase=round(ctrl.phase,3),exited=sorted(ctrl.exited),received=sorted(ctrl.received),force_n=r['contact_force_N'].sum(1),front_wrap_deg=np.degrees(r['actual_front_span_rad']),tracking_deg=info['tracking_deg'],tail_radius_mm=stats['center_radius_mm'][-1]);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,complete_shift_mm=route.total*1000,
            initial_front8_wrap_deg=geometry['initial_front8_wrap_deg'],final_front8_wrap_deg=float(np.degrees(r['actual_front_span_rad'])),final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),actual_current_received_target=sorted(ctrl.received),final_kp=ctrl.kp,
            maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),
            maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),actual_completion_hold_seconds=hold,whole_transfer_complete=outcome=='complete_from_artificial_inner_hook',physical_root_reset_during_run=False,initial_state_is_artificial=True,active_hook_acquisition_demonstrated=False,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,engaged=sorted(ctrl.engaged),exited=sorted(ctrl.exited),bridge_coeff=ctrl.coeff));print(json.dumps(result,default=default),flush=True)
        diagrams(case,route,sim.rows)
        try:render(case,sim.rows,model_name=IMODEL.name,initial_label='Artificial inner upward hook')
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='I01');ap.add_argument('--front',type=int,default=9);ap.add_argument('--bridge',type=int,default=5);ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--rate',type=float,default=20);ap.add_argument('--seconds',type=float,default=420);ap.add_argument('--max-wall',type=float,default=600);ap.add_argument('--deadline',default='2026-10-09T07:36:07+00:00');ap.add_argument('--prepare-only',action='store_true');run(ap.parse_args())
