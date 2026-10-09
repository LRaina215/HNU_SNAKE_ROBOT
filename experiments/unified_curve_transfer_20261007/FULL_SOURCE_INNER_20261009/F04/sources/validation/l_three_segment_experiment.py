"""Bounded full L-transfer trials, one robot and one continuously driven run."""
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
from snake_control.suspended_supply_transfer import WholeCurveMapper,LINK,DT,nodes
from snake_control.l_three_segment_transfer import ThreeSegmentRoute,ThreeSegmentTransfer,FRONT,BRIDGE,SOURCE,TOTAL,CUT

BASE=ROOT/'experiments/unified_curve_transfer_20261007/L_THREE_SEGMENT_20261009'
LMODEL=ROOT/'mujoco_model/Snake_Robot_o20_L_three_segment.xml'

def model_create():
    tree=ET.parse(MODEL);root=tree.getroot();root.set('model','real-snake20-simultaneous-L-transfer')
    root.find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto','0.15 0 1.1 1.65 0 1.1')
    tree.write(LMODEL,encoding='utf-8',xml_declaration=True)

def reference_plot(out,route):
    fig=plt.figure(figsize=(16,6));phases=[0.,TOTAL/2,TOTAL];reports=[]
    for i,shift in enumerate(phases):
        ax=fig.add_subplot(1,3,i+1,projection='3d');p,v=route.window(shift)
        theta=np.linspace(0,2*np.pi,100);ax.plot([-.2,1.1],[0,0],[1,1],color='grey',lw=12,alpha=.3)
        ax.plot([1.1,1.1],[0,1.65],[1,1],color='grey',lw=12,alpha=.3)
        ax.plot(p[:,0],p[:,1],p[:,2],'o-',ms=3,color='navy');ax.scatter(*p[-1],c='red',label='Tail')
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f'Reference shift {shift*1000:.1f}mm')
        ax.set_xlim(.45,1.25);ax.set_ylim(-.18,.75);ax.set_zlim(.7,1.18);ax.view_init(27,140)
        reports.append(dict(shift_mm=shift*1000,points_mm=p*1000,material_mm=v*1000,nominal_source_rods=int(np.count_nonzero(.5*(v[:-1]+v[1:])<CUT))))
    fig.tight_layout();fig.savefig(out/'reference_initial_middle_final.png',dpi=160);plt.close(fig);write(out/'reference_windows.json',reports)

def prepare(name):
    BASE.mkdir(parents=True,exist_ok=True);out=BASE/name;out.mkdir(exist_ok=False);model_create()
    route=ThreeSegmentRoute();guide,v=route.window(0);m=mujoco.MjModel.from_xml_path(str(LMODEL));mapper=WholeCurveMapper(m)
    pose,report=mapper.initialize(guide);p=nodes(mapper.d);c=.5*(p[:-1]+p[1:]);rt=np.linalg.norm(c[:FRONT,[0,2]]-[1.1,1.],axis=1);rs=np.linalg.norm(c[15:,[1,2]]-[0.,1.],axis=1)
    tf=np.unwrap(np.arctan2(c[:FRONT,2]-1,c[:FRONT,0]-1.1));sf=np.unwrap(np.arctan2(c[15:,2]-1,c[15:,1]))
    eps=1e-6;seams=[]
    for s in [CUT,-.09675,0.]:
        pm=route.sample([s-eps,s,s+eps]);ta=(pm[1]-pm[0])/eps;tb=(pm[2]-pm[1])/eps
        seams.append(dict(material_mm=s*1000,position_gap_mm=float(np.linalg.norm(pm[2]-pm[0])*1000),tangent_angle_deg=float(np.degrees(np.arccos(np.clip(ta@tb/np.linalg.norm(ta)/np.linalg.norm(tb),-1,1))))))
    report.update(front_rods=FRONT,bridge_rods=BRIDGE,source_rods=SOURCE,target_actual_span_rad=float(np.ptp(tf)),source_actual_span_rad=float(np.ptp(sf)),
        source_actual_center_radius_mm=rs*1000,target_actual_center_radius_mm=rt*1000,connector_handle_mm=route.handle*1000,
        seam_checks=seams,reference_speed_mm_s=5.,nominal_full_seconds=TOTAL/.005,command_rate_setting_deg_s=20.,
        pipe_axis_target_end_y_m=1.65,ready=bool(report['pipe_depth_mm']<1.2 and report['self_depth_mm']<.5 and np.ptp(tf)>np.pi
            and np.ptp(sf)>2.7 and np.count_nonzero((rt>.065)&(rt<.125))>=7 and np.count_nonzero((rs>.065)&(rs<.13))>=4 and p[:,2].min()>.15))
    np.savez(out/'initial_pose.npz',qpos=pose,qvel=np.zeros(m.nv),ctrl=pose[mapper.qa],reference_points=guide)
    write(out/'initial_geometry.json',report);reference_plot(out,route);print(json.dumps(report,default=default),flush=True)

class LSim(ActualSim):
    def __init__(self,case,initial):
        # Loading and initial artificial placement precede the first mj_step.
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(LMODEL));self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]];self.head=self.m.body('head').id
        self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']];self.target=[self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor=self.m.geom('floor').id;self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION));self.d.qpos[:]=initial['qpos'];self.d.qvel[:]=initial['qvel'];self.d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(self.m,self.d)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21);self.exited=set();self.received=set()
    def snapshot(self):
        r=super().snapshot();r.update(kp_nm_per_rad=self.m.actuator_gainprm[:,0].copy(),damping_nm_s_per_rad=self.m.dof_damping[self.va].copy(),
            passive_torque_nm=self.scratch.qfrc_passive[self.va].copy(),force_range_nm=self.m.actuator_forcerange.copy(),
            actual_received_mask=np.array([j in self.info.get('actual_received_modules',[]) for j in range(21)]),
            actual_new_retained_mask=np.array([j in self.info.get('actual_new_retained_modules',[]) for j in range(21)]),
            body_route_regions=np.asarray(self.info.get('body_route_regions',np.full(21,1))),roll_factor=self.info.get('roll_factor',0.))
        r['actual_receive_mask']=r['actual_received_mask'].copy();r['actual_exit_mask']=np.array([j in self.info.get('actual_source_exited_modules',[]) for j in range(21)])
        c=r['centers_mm']/1000;r['actual_front_span_rad']=float(np.ptp(np.unwrap(np.arctan2(c[:FRONT,2]-1,c[:FRONT,0]-1.1))))
        return r

def render(case,rows):
    m=mujoco.MjModel.from_xml_path(str(LMODEL));d=mujoco.MjData(m);ren=mujoco.Renderer(m,height=600,width=880);cam=mujoco.MjvCamera()
    cam.lookat[:]=[.98,.31,.95];cam.distance=1.65;cam.azimuth=140;cam.elevation=-26
    w=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1240,600));ids=list(range(0,len(rows),5))
    if not w.isOpened():raise RuntimeError('Video writer unavailable')
    for fi,i in enumerate(ids):
        r=rows[i];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d);ren.update_scene(d,camera=cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);panel=np.full((600,360,3),25,np.uint8)
        cv2.putText(im,f'{case.name} actual {r["sim_time"]:.2f}s / continuous1x',(12,25),0,.6,(255,255,255),1)
        lines=[f'One L-pipe / real twenty hinges',f'Shared reference {r["shift_mm"]:.1f} mm',f'Source current cleared {r["actual_exit_mask"][15:].sum()}/6',
            f'Target current connected {r["actual_received_mask"].sum()}/21',f'Bearing source/target {r["contact_force_N"][0].sum():.1f}/{r["contact_force_N"][1].sum():.1f} N',
            f'Actual pipe/self {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f} mm','Joint  Kp Nm/rad   drive Nm','Moving bridge guide, original damping1.5']
        for j,label in enumerate(lines):cv2.putText(panel,label,(10,22+23*j),0,.43,(225,225,225),1)
        for j in range(20):
            x=10+(j//10)*176;y=230+(j%10)*32;cv2.putText(panel,f'{j+1:02d} {r["kp_nm_per_rad"][j]:5.1f} {r["torque"][j]:+4.2f}',(x,y),0,.43,(220,225,180),1)
        whole=np.hstack([im,panel]);w.write(whole)
        if fi in {0,len(ids)//2,len(ids)-1}:cv2.imwrite(str(case/f'frame_{fi:05d}.jpg'),whole)
    w.release();ren.close();cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for i in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,_=cap.read();checks.append(dict(frame=i,readable=bool(ok)))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['sim_time'],speed=1.,source='saved actual qpos only',readback=checks));cap.release()

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False);initial_dir=BASE/args.initial;report=json.loads((initial_dir/'initial_geometry.json').read_text())
    if not report['ready']:raise RuntimeError('Initial dual-pipe geometry did not pass; zero dynamics started')
    with np.load(initial_dir/'initial_pose.npz') as z:initial={k:z[k].copy() for k in z.files}
    for src in ['snake_control/l_three_segment_transfer.py','snake_control/horizontal_pipe_roll.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py',
        'snake_control/spatial_bishop_backbone.py','validation/l_three_segment_experiment.py','validation/suspended_supply_experiment.py',str(LMODEL.relative_to(ROOT))]:
        dest=case/'sources'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    for name in ['initial_pose.npz','initial_geometry.json','reference_initial_middle_final.png','reference_windows.json']:shutil.copy2(initial_dir/name,case/name)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),task='one robot, one L-pipe, simultaneous source exit and target helix uptake',
        initial='manual dual-pipe8 front,7 continuous bridge,6 source; zero root assignments after first step',same_version_continuous=True,
        total_reference_shift_mm=TOTAL*1000,speed_mm_s=args.speed*1000,nominal_seconds=TOTAL/args.speed,rate_setting_deg_s=args.rate,
        roll_factor=args.roll_factor,adaptive_released_supply=args.adaptive_supply,maximum_branch_dynamics_trials=4 if args.strict_window else 3,additional_trial_explicitly_authorized=args.strict_window,max_actual_seconds=args.seconds,max_wall=args.max_wall,deadline_utc=args.deadline,
        physics='original robot, gravity, friction, damping1.5, hinge range90deg, ctrl88deg, force2.355Nm; target pipe extended before start to Y1.65m',
        control_layers='common shift; J03 head-local roll formula; full unrotated source/bridge/target guide; exact twenty-hinge numerical geometry; moving guided compliant bridge; actual connection stiffness restoration; smooth bridge posture feedback; no Yamada compliance',
        complete=('strict: shift>=838.5mm; all21 center radii75-105mm and within6mm of median; adjacent pitch residual<8mm; fitted pitch162-198mm and axisRMS<4mm; source contact/force absent; posterior bearing; Kp127; elapsed actual hold>=1s' if args.strict_window else 'all21 in a single same-pitch target helix including tail, all6 original source rods currently clear0.5s, source force/contact absent, target posterior bearing, all joints recovered full stiffness and1s actual retention; reference shift alone insufficient'),
        stop='ground, actual joint>90.5deg, nonfinite, actual self>3mm, actual pipe>3mm sustained0.2s, target support collapsed and falling1s, initial bearing unavailable3s, no actual transfer progress45s'))
    sim=LSim(case,initial);sim.snapshot();ctrl=ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,roll_factor=args.roll_factor,bridge_gain=args.bridge_gain,adaptive_supply=args.adaptive_supply)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';error=None;deep=fall=complete_age=0.;complete_since=None
    try:
        while sim.d.time<args.seconds:
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,float(sim.d.time),r['contact_force_N'],r['source_contact_any']);sim.info=info
            sim.commands.append(dict(t_s=float(sim.d.time),ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1]
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
            falling=r['contact_force_N'][1,:FRONT].sum()<.1 and (np.median(rr[:FRONT])>.145 or c[:FRONT,2].mean()<.9)
            fall=fall+DT if falling else 0.
            if fall>=1:outcome='front_bearing_collapsed_and_falling';break
            if sim.d.time>3 and len(sim.commands)<65 and (r['contact_force_N'][0].sum()<1 or r['contact_force_N'][1].sum()<1):outcome='initial_dual_bearing_not_established';break
            if ctrl.failure:outcome=ctrl.failure;break
            phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));dy=np.diff(c[:,1]);da=np.diff(phi);fit=np.polyfit(phi,c[:,1],1);pitch=-fit[0]*2*np.pi;axis_rms=np.sqrt(np.mean((c[:,1]-np.polyval(fit,phi))**2))
            complete=bool(np.all((rr>.065)&(rr<.125)&(c[:,1]>.15)) and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045))
                and .126<pitch<.234 and axis_rms<.012 and c[0,1]-c[-1,1]>.30 and len(ctrl.exited)==6 and len(ctrl.received)==21
                and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,15:].sum()>.2 and np.all(ctrl.kp>120))
            if args.strict_window:
                complete=complete and ctrl.shift>=TOTAL-1e-9 and np.all((rr>.075)&(rr<.105)) and np.max(np.abs(rr-np.median(rr)))<.006 and np.max(np.abs(dy+.180/(2*np.pi)*da))<.008 and .162<pitch<.198 and axis_rms<.004 and np.all(ctrl.kp>=126.999)
                if complete and complete_since is None:complete_since=float(sim.d.time)
                if not complete:complete_since=None
                complete_age=0. if complete_since is None else float(sim.d.time)-complete_since
            else:complete_age=complete_age+DT if complete else 0.
            if complete_age>=1:outcome='complete_whole_L_transfer';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(float(sim.d.time),2),shift_mm=round(ctrl.shift*1000,2),phase=round(ctrl.phase,3),exited=sorted(ctrl.exited),received=sorted(ctrl.received),
                    normal_n=r['contact_force_N'].sum(1).round(1).tolist(),front_span_rad=round(r['actual_front_span_rad'],3),tracking_deg=round(info['tracking_deg'],2),
                    head_axis_error_mm=round(info['head_axis_reference_error_mm'],2),fit_rms_mm=round(info['projection']['whole_fit_rms_mm'],2))
                print(json.dumps(msg),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];c=r['centers_mm']/1000;rt=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        result=dict(name=args.name,outcome=outcome,actual_seconds=float(sim.d.time),wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,roll_phase_rad=ctrl.phase,
            actual_current_source_exited=sorted(ctrl.exited),actual_current_received_target=sorted(ctrl.received),actual_target_near_modules=np.flatnonzero((rt>.065)&(rt<.125)&(c[:,1]>.15)).tolist(),
            maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),
            maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),
            maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),
            initial_front_span_rad=sim.rows[0]['actual_front_span_rad'],final_front_span_rad=r['actual_front_span_rad'],
            final_source_target_force_n=r['contact_force_N'].sum(1),final_kp=ctrl.kp,save_errors=errors,error=error,physical_root_reset_during_run=False,
            whole_L_transfer_complete=outcome=='complete_whole_L_transfer',strict_full_window_and_tail_geometry=args.strict_window,actual_completion_hold_seconds=complete_age,
            final_center_radius_mm=rt*1000,final_tail_local_pitch_residual_mm=float((dy[-1]+.180/(2*np.pi)*da[-1])*1000),yamada_compliance_enabled=False)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,received=sorted(ctrl.received),
            engaged=sorted(ctrl.engaged),exited=sorted(ctrl.exited),bridge_coeff=ctrl.coeff,mapper_root=ctrl.mapper.root,mapper_tilt=ctrl.mapper.tilt,frame_offset=ctrl.frame_offset))
        np.savez(case/'final_model_parameters.npz',gain=sim.m.actuator_gainprm,bias=sim.m.actuator_biasprm,damping=sim.m.dof_damping,force_range=sim.m.actuator_forcerange)
        print(json.dumps(result,default=default),flush=True)
        try:render(case,sim.rows)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--prepare-name',default='PREPARE_1');ap.add_argument('--initial',default='PREPARE_1');ap.add_argument('--name',default='L01')
    ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--rate',type=float,default=20.);ap.add_argument('--roll-factor',type=float,default=None);ap.add_argument('--bridge-gain',type=float,default=.25);ap.add_argument('--adaptive-supply',action='store_true');ap.add_argument('--strict-window',action='store_true')
    ap.add_argument('--seconds',type=float,default=360);ap.add_argument('--max-wall',type=float,default=420);ap.add_argument('--deadline',default='2026-10-09T02:53:55+00:00');args=ap.parse_args()
    if args.prepare_only:prepare(args.prepare_name)
    else:run(args)
