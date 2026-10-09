"""Full dynamics from the lossless saved CLOSE1B state, no artificial pose."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback,xml.etree.ElementTree as ET
from datetime import datetime,timezone
import numpy as np,mujoco,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.suspended_supply_experiment import ActualSim,write,default,MODEL
from validation.l_three_segment_experiment import LSim
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.suspended_supply_transfer import DT,nodes
from snake_control.l_three_segment_transfer import ThreeSegmentTransfer
from snake_control.hook_start_transfer import MeasuredHookRoute
from snake_control.hook_reform_route import ReformHookRoute

BASE=ROOT/'experiments/unified_curve_transfer_20261007/HOOK_START_TRANSFER_20261009'
INITIAL=ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
HMODEL=ROOT/'mujoco_model/Snake_Robot_o20_hook_start_transfer.xml'
SOURCE_START=10

class HookSim(LSim):
    def __init__(self,case):
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(HMODEL));self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]];self.head=self.m.body('head').id
        self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']];self.target=[self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor=self.m.geom('floor').id;self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION));self.history=restore_saved_state(self.m,self.d,INITIAL);self.initial_time=float(self.d.time)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21);self.exited=set();self.received=set()
    def snapshot(self):
        r=super().snapshot();r['saved_absolute_sim_time']=r['sim_time'];r['sim_time']-=self.initial_time;return r

def shape_stats(r):
    c=r['centers_mm']/1000;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1]);fit=np.polyfit(phi,c[:,1],1)
    pitch=-fit[0]*2*np.pi;rms=np.sqrt(np.mean((c[:,1]-np.polyval(fit,phi))**2));local=dy+.180/(2*np.pi)*da
    strict=bool(np.all((rr>.075)&(rr<.105)&(c[:,1]>.15)) and np.max(abs(rr-np.median(rr)))<.006 and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)) and np.max(abs(local))<.008 and .162<pitch<.198 and rms<.004)
    return dict(strict_actual_shape=strict,center_radius_mm=rr*1000,pitch_mm=pitch*1000,axis_rms_mm=rms*1000,local_pitch_residual_mm=local*1000,tail_dy_mm=dy[-1]*1000,tail_dphi_rad=da[-1],tail_direct_force_n=r['contact_force_N'][1,20])

def diagrams(case,route,rows=None):
    fig=plt.figure(figsize=(16,6));reports=[]
    ids=[0,len(rows)//2,len(rows)-1] if rows is not None else None
    for i,s in enumerate([0.,route.total/2,route.total]):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        if rows is None:p,v=route.window(s);label=f'Nominal reference shift {s*1000:.1f}mm'
        else:
            r=rows[ids[i]];p=r['reference_points_mm']/1000;actual=r['points_mm']/1000;v=getattr(route,'origin',0.)+r['shift_mm']/1000-np.arange(22)*.0645;label=f'Actual time {r["sim_time"]:.2f}s / shift {r["shift_mm"]:.1f}mm';ax.plot(actual[:,0],actual[:,1],actual[:,2],'o-',ms=3,label='Actual FK')
        ax.plot([-.2,1.1],[0,0],[1,1],color='grey',lw=10,alpha=.25);ax.plot([1.1,1.1],[0,2],[1,1],color='grey',lw=10,alpha=.25);ax.plot(p[:,0],p[:,1],p[:,2],'--',lw=1.4,label='Reference')
        ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)',title=label,xlim=(.4,1.4),ylim=(-.25,1.7),zlim=(.4,1.2));ax.view_init(27,140);ax.legend(fontsize=8)
        reports.append(dict(label=label,reference_points_mm=p*1000,material_coordinate_mm=v*1000))
    fig.tight_layout();name='reference_initial_middle_final' if rows is None else 'actual_and_executed_reference';fig.savefig(case/(name+'.png'),dpi=150);plt.close(fig);write(case/(name+'.json'),reports)

def render(case,rows):
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_hook_start_transfer.xml'));d=mujoco.MjData(m);ren=mujoco.Renderer(m,height=600,width=900);cam=mujoco.MjvCamera()
    ymax=max(r['centers_mm'][:,1].max() for r in rows)/1000;cam.lookat[:]=[.98,max(.3,ymax/2),.95];cam.distance=2.15;cam.azimuth=140;cam.elevation=-26
    source_start=json.loads((case/'initial_geometry.json').read_text())['source_start_module'];source_count=21-source_start
    writer=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1260,600));ids=list(range(0,len(rows),5))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    for fi,i in enumerate(ids):
        r=rows[i];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d);ren.update_scene(d,camera=cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);panel=np.full((600,360,3),25,np.uint8)
        cv2.putText(im,f'{case.name} CLOSE1B actual continuous {r["sim_time"]:.2f}s',(10,25),0,.6,(255,255,255),1)
        lines=['Saved true hook / one L-pipe',f'Shared shift {r["shift_mm"]:.1f}mm',f'Source clear {r["actual_exit_mask"][source_start:].sum()}/{source_count}',f'Target connected {r["actual_received_mask"].sum()}/21',f'Bearing N {r["contact_force_N"].sum(1).round(1)}',f'Front wrap8 {np.degrees(r["actual_front_span_rad"]):.1f}deg',f'Actual pipe/self {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f}mm','Joint   Kp    DriveNm / original damping1.5']
        for j,line in enumerate(lines):cv2.putText(panel,line,(10,22+23*j),0,.43,(225,225,225),1)
        for j in range(20):cv2.putText(panel,f'{j+1:02d} {r["kp_nm_per_rad"][j]:5.1f} {r["torque"][j]:+4.2f}',(10+(j//10)*176,235+(j%10)*32),0,.43,(220,225,180),1)
        frame=np.hstack([im,panel]);writer.write(frame)
        if fi in {0,len(ids)//2,len(ids)-1}:cv2.imwrite(str(case/f'frame_{fi:05d}.jpg'),frame)
    writer.release();ren.close();cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for i in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,_=cap.read();checks.append(dict(frame=i,readable=bool(ok)))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['sim_time'],speed=1.,source='saved actual qpos from lossless CLOSE1B restore through every physical step',readback=checks));cap.release()

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    tree=ET.parse(MODEL);root=tree.getroot();root.set('model','lossless-CLOSE1B-full-transfer');root.find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto','0.15 0 1.1 2.0 0 1.1');tree.write(HMODEL,encoding='utf-8',xml_declaration=True)
    sim=HookSim(case);r0=sim.snapshot();initial_points=r0['points_mm']/1000;route=ReformHookRoute(initial_points,morph_length=args.morph_length) if args.reform else MeasuredHookRoute(initial_points,extension_length=args.extension);source_start=route.source_start if args.reform else SOURCE_START
    original=mujoco.MjModel.from_xml_path(str(MODEL));same=all(np.array_equal(getattr(original,k),getattr(sim.m,k)) for k in ['dof_damping','actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_friction','jnt_range']) and np.array_equal(original.body_mass[sim.head:sim.head+21],sim.m.body_mass[sim.head:sim.head+21]) and np.array_equal(original.opt.gravity,sim.m.opt.gravity)
    old=np.load(ROOT/'experiments/hook_pull_20261006/CLOSE1B/trace.npz');extension=getattr(route,'extension_length',0.);sample=route.sample(np.linspace(0,max(extension,.16125),1001));rad=np.linalg.norm(sample[:,[0,2]]-[1.1,1.],axis=1)
    geometry=dict(initial_state='lossless CLOSE1B final integration state, saved qpos/qvel/ctrl/warmstart retained',physical_steps=0,
        saved_history=sim.history,initial_front8_wrap_deg=float(np.degrees(r0['actual_front_span_rad'])),initial_front_wrap_deg=float(np.degrees(np.ptp(np.unwrap(np.arctan2(r0['centers_mm'][:9,2]-1000,r0['centers_mm'][:9,0]-1100))))),original_physics_arrays_equal=same,
        qpos_difference_from_saved_trace=float(abs(sim.d.qpos-old['qpos'][-1]).max()),qvel_difference_from_saved_trace=float(abs(sim.d.qvel-old['qvel'][-1]).max()),ctrl_difference_from_saved_trace=float(abs(sim.d.ctrl-old['ctrl'][-1]).max()),
        initial_source_target_force_n=r0['contact_force_N'].sum(1),initial_pipe_depth_mm=r0['pipe_depth_mm'],max_joint_deg=float(np.degrees(abs(r0['q']).max())),
        initial_reference_points_mm=initial_points*1000,initial_reference_point_error_mm=float(abs(route.window(0)[0]-initial_points).max()*1000),
        extension_handle_mm=route.handle*1000,extension_length_mm=extension*1000,extension_reference_radius_range_mm=[float(rad.min()*1000),float(rad.max()*1000)],
        continuous_hook_reform=args.reform,morph_length_mm=args.morph_length*1000,source_start_module=source_start,existing_target_intake_y_m=.075,target_pipe_end_y_m=2.,complete_shift_mm=route.total*1000,
        ready=bool(same and abs(r0['q']).max()<np.radians(90.5) and r0['pipe_depth_mm']<3 and not r0['ground'] and r0['contact_force_N'][0].sum()>1 and r0['contact_force_N'][1].sum()>1))
    write(case/'initial_geometry.json',geometry);print(json.dumps(geometry,default=default),flush=True)
    for src in ['snake_control/hook_start_transfer.py','snake_control/hook_reform_route.py','snake_control/l_three_segment_transfer.py','snake_control/horizontal_pipe_roll.py','snake_control/hook_pull_review_support.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py','snake_control/spatial_bishop_backbone.py','validation/hook_start_transfer_experiment.py','validation/l_three_segment_experiment.py','validation/suspended_supply_experiment.py',str(HMODEL.relative_to(ROOT))]:
        dest=case/'sources'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    for mesh in ET.parse(HMODEL).findall('./asset/mesh'):
        file=mesh.get('file')
        if file:dest=case/'sources/mujoco_model'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'mujoco_model'/file,dest)
    shutil.copy2(INITIAL,case/'initial_saved_state.npz');diagrams(case,route)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),first_candidate_maximum_dynamics_trials=3,source_state=str(INITIAL),
        action=('actual reverse hook smoothly reforms over first200mm sharedshift into8 target helix /8 underside bridge /5 retained source, with whole-head-axis alignment and common roll' if args.reform else 'existing measured source helix / actual bridge / reverse hook followed by smooth head-tangent connector and forward target helix; one shared material shift and J03/L04 common roll'),
        preload='retain complete saved ctrl; preserve its all-joint preload while smooth fading with first200mm sharedshift, no special joint angles',
        physics='original robot/gravity/friction/damping1.5/joint90deg/ctrl88deg/torque2.355Nm; target end extended toY2.0m before lossless initial restore',
        speed_mm_s=args.speed*1000,rate_setting_deg_s=args.rate,complete_shift_mm=route.total*1000,max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,
        complete='whole reference window including connector passed; all21 center radii75-105mm, within6mm median; all adjacent positive winding and negativeY feed, local pitch error<8mm; pitch162-198mm and axisRMS<4mm; source absent, posterior bearing, allKp127, actual elapsed hold1s',
        stop='ground; actualjoint>90.5deg; self>3mm; pipe>3mm sustained0.2s; nonfinite; supported front collapses/falls1s; no source exit or target intake45s after2rod action; full reference complete without shape30s; wall/deadline',yamada_compliance_enabled=False))
    if not geometry['ready']:write(case/'result.json',dict(outcome='initial_restore_or_geometry_unavailable',physical_steps=0));return
    ctrl=ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,adaptive_supply=True,route=route,source_start=source_start,intake_y=.075,preserve_preload=True)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;complete_since=None;window_time=None;error=None;hold=0.
    try:
        while sim.d.time-sim.initial_time<args.seconds:
            elapsed=float(sim.d.time-sim.initial_time)
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,elapsed,r['contact_force_N'],r['source_contact_any']);sim.info=info;sim.commands.append(dict(t_s=elapsed,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r)
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:8,[0,2]]-[1.1,1.],axis=1)
            falling=r['contact_force_N'][1,:8].sum()<.1 and (np.median(rr)>.145 or c[:8,2].mean()<.9);fall=fall+DT if falling else 0.
            if fall>=1:outcome='front_bearing_collapsed_and_falling';break
            if ctrl.failure:outcome=ctrl.failure;break
            full=ctrl.shift>=route.total-1e-9
            if full and window_time is None:window_time=float(r['sim_time'])
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2 and np.all(ctrl.kp>=126.999)
            if complete and complete_since is None:complete_since=float(r['sim_time'])
            if not complete:complete_since=None
            hold=0. if complete_since is None else float(r['sim_time'])-complete_since
            if hold>=1:outcome='complete_from_saved_hook';break
            if full and r['sim_time']-window_time>30:outcome='whole_reference_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),shift_mm=round(ctrl.shift*1000,2),phase=round(ctrl.phase,3),exited=sorted(ctrl.exited),received=sorted(ctrl.received),force_n=r['contact_force_N'].sum(1),front_wrap_deg=np.degrees(r['actual_front_span_rad']),tracking_deg=info['tracking_deg'],front_fit_rms_mm=info['projection']['front_fit_rms_mm'],tail_radius_mm=stats['center_radius_mm'][-1]);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],absolute_saved_time=float(sim.d.time),wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,complete_shift_mm=route.total*1000,
            initial_front_wrap_deg=geometry['initial_front_wrap_deg'],final_front_wrap_deg=float(np.degrees(r['actual_front_span_rad'])),actual_current_source_exited=sorted(ctrl.exited),actual_current_received_target=sorted(ctrl.received),
            final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),final_kp=ctrl.kp,
            maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),
            maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),
            actual_completion_hold_seconds=hold,whole_transfer_complete=outcome=='complete_from_saved_hook',physical_root_reset_during_run=False,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,engaged=sorted(ctrl.engaged),exited=sorted(ctrl.exited),bridge_coeff=ctrl.coeff,mapper_root=ctrl.mapper.root,mapper_tilt=ctrl.mapper.tilt,source_release_latch=sorted(ctrl.source_release_latch)))
        np.savez(case/'final_model_parameters.npz',gain=sim.m.actuator_gainprm,bias=sim.m.actuator_biasprm,damping=sim.m.dof_damping,force_range=sim.m.actuator_forcerange);print(json.dumps(result,default=default),flush=True)
        diagrams(case,route,sim.rows)
        try:render(case,sim.rows)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='H01');ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--rate',type=float,default=20.);ap.add_argument('--extension',type=float,default=.16125);ap.add_argument('--reform',action='store_true');ap.add_argument('--morph-length',type=float,default=.20);ap.add_argument('--seconds',type=float,default=420);ap.add_argument('--max-wall',type=float,default=550);ap.add_argument('--deadline',default='2026-10-09T03:47:56+00:00');run(ap.parse_args())
