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
from snake_control.lower_hook_initial import construct_lower_entry
from snake_control.lower_bridge_hook_route import LowerBridgeReformHookRoute

BASE=ROOT/'experiments/unified_curve_transfer_20261007/HOOK_START_TRANSFER_20261009'
INITIAL=ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
HMODEL=ROOT/'mujoco_model/Snake_Robot_o20_hook_start_transfer.xml'
SOURCE_START=10

class HookSim(LSim):
    def __init__(self,case,lower_entry=False):
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(HMODEL));self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]];self.head=self.m.body('head').id
        self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']];self.target=[self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor=self.m.geom('floor').id;self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION));self.history=restore_saved_state(self.m,self.d,INITIAL);self.initial_time=float(self.d.time)
        self.lower_report=construct_lower_entry(self.m,self.d) if lower_entry else None
        self.initial_time=float(self.d.time)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21);self.exited=set();self.received=set()
    def snapshot(self):
        r=super().snapshot();r['saved_absolute_sim_time']=r['sim_time'];r['sim_time']-=self.initial_time;return r

def shape_stats(r,winding_sign=1):
    c=r['centers_mm']/1000;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1);phi=winding_sign*np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1]);fit=np.polyfit(phi,c[:,1],1)
    pitch=-fit[0]*2*np.pi;rms=np.sqrt(np.mean((c[:,1]-np.polyval(fit,phi))**2));local=dy+.180/(2*np.pi)*da
    strict=bool(np.all((rr>.075)&(rr<.105)&(c[:,1]>.15)) and np.max(abs(rr-np.median(rr)))<.006 and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)) and np.max(abs(local))<.008 and .162<pitch<.198 and rms<.004)
    return dict(strict_actual_shape=strict,center_radius_mm=rr*1000,pitch_mm=pitch*1000,axis_rms_mm=rms*1000,local_pitch_residual_mm=local*1000,tail_dy_mm=dy[-1]*1000,tail_dphi_rad=da[-1]*winding_sign,oriented_tail_dphi_rad=da[-1],actual_winding_sign=winding_sign,tail_direct_force_n=r['contact_force_N'][1,20])

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

def lower_diagrams(case,route,report):
    """Fixed world coordinates: direction evidence, not a flipped camera."""
    fig,axes=plt.subplots(1,2,figsize=(14,7));theta=np.linspace(0,2*np.pi,200)
    for ax,key,title in zip(axes,['old_upper_points_mm','new_lower_points_mm'],['H03 parent: upper bridge entry','New artificial initial: lower bridge entry']):
        p=np.asarray(report[key]);ax.fill(1100+56.5*np.cos(theta),1000+56.5*np.sin(theta),color='gray',alpha=.35,label='Target pipe cross section')
        ax.plot(p[:,0],p[:,2],'.-',color='steelblue',alpha=.65,label='N0 head to N21 tail projection')
        ax.plot(p[8:11,0],p[8:11,2],'o-',color='darkorange',lw=3,label='Actual bridge N10 -> N9 -> N8')
        ax.plot(p[:9,0],p[:9,2],'o-',color='darkgreen',lw=2,label='Front wrapped rods M0..M7')
        for j in [0,1,3,5,7,8,9,10,16,21]:ax.annotate('Head N0' if j==0 else 'Tail N21' if j==21 else f'N{j}',p[j,[0,2]],xytext=(5,6),textcoords='offset points',fontsize=8)
        for a,b in [(10,9),(9,8),(7,6),(5,4),(3,2),(1,0)]:ax.annotate('',xy=p[b,[0,2]],xytext=p[a,[0,2]],arrowprops=dict(arrowstyle='->',color='crimson',lw=2))
        ax.axhline(1000,color='gray',ls=':');ax.annotate('',xy=(740,1190),xytext=(740,1110),arrowprops=dict(arrowstyle='->',color='black'));ax.text(700,1095,'World +Z UP',fontsize=9)
        ax.set(title=title,xlabel='World X (mm)',ylabel='World Z (mm)',xlim=(680,1240),ylim=(780,1220),aspect='equal');ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Same target axis (X=1100,Z=1000mm); red arrows follow tail -> head; gravity stays -Z')
    fig.tight_layout();fig.savefig(case/'initial_cross_section_vs_H03.png',dpi=160);plt.close(fig)
    grid=np.linspace(route.origin-21*.0645,route.origin+route.total,2401);p=route.sample(grid)
    fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(111,projection='3d')
    for mask,label,color in [(grid<route.source_cut,'Retained source','darkorange'),((grid>=route.source_cut)&(grid<0),'Continuous nominal bridge','purple'),(grid>=0,'Opposite-winding target helix','darkgreen')]:
        ax.plot(p[mask,0],p[mask,1],p[mask,2],color=color,lw=2,label=label)
    initial=np.asarray(report['new_lower_points_mm'])/1000;ax.plot(initial[:,0],initial[:,1],initial[:,2],'o--',ms=3,color='steelblue',label='Actual artificial initial / measured seed')
    ax.plot([-.2,1.1],[0,0],[1,1],color='gray',lw=10,alpha=.3);ax.plot([1.1,1.1],[0,2],[1,1],color='gray',lw=10,alpha=.3)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z UP (m)',title='Complete nominal source / bridge / target route; seed morph over first 200mm')
    ax.view_init(27,140);ax.legend();fig.tight_layout();fig.savefig(case/'complete_reference_curve.png',dpi=150);plt.close(fig)
    write(case/'complete_reference_curve.json',dict(material_coordinate_mm=grid*1000,points_mm=p*1000,winding_sign=route.winding_sign,source_cut_mm=route.source_cut*1000,target_join_mm=0,measured_seed_morph_length_mm=route.morph_length*1000,initial_reference_is_measured_seed=True))

def render(case,rows,lower_entry=False):
    model_name='Snake_Robot_o20_lower_hook_transfer.xml' if lower_entry else 'Snake_Robot_o20_hook_start_transfer.xml'
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model'/model_name));d=mujoco.MjData(m);ren=mujoco.Renderer(m,height=600,width=900);cam=mujoco.MjvCamera()
    ymax=max(r['centers_mm'][:,1].max() for r in rows)/1000;cam.lookat[:]=[.98,max(.3,ymax/2),.95];cam.distance=2.15;cam.azimuth=140;cam.elevation=-26
    source_start=json.loads((case/'initial_geometry.json').read_text())['source_start_module'];source_count=21-source_start
    writer=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1260,600));ids=list(range(0,len(rows),5))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    for fi,i in enumerate(ids):
        r=rows[i];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d);ren.update_scene(d,camera=cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);panel=np.full((600,360,3),25,np.uint8)
        initial_label='Artificial lower hook' if lower_entry else 'CLOSE1B'
        cv2.putText(im,f'{case.name} {initial_label} actual continuous {r["sim_time"]:.2f}s',(10,25),0,.6,(255,255,255),1)
        lines=['Saved true hook / one L-pipe',f'Shared shift {r["shift_mm"]:.1f}mm',f'Source clear {r["actual_exit_mask"][source_start:].sum()}/{source_count}',f'Target connected {r["actual_received_mask"].sum()}/21',f'Bearing N {r["contact_force_N"].sum(1).round(1)}',f'Front wrap8 {np.degrees(r["actual_front_span_rad"]):.1f}deg',f'Actual pipe/self {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f}mm','Joint   Kp    DriveNm / original damping1.5']
        if lower_entry:lines[0]='Lower entry / gravity -9.81 unchanged'
        for j,line in enumerate(lines):cv2.putText(panel,line,(10,22+23*j),0,.43,(225,225,225),1)
        for j in range(20):cv2.putText(panel,f'{j+1:02d} {r["kp_nm_per_rad"][j]:5.1f} {r["torque"][j]:+4.2f}',(10+(j//10)*176,235+(j%10)*32),0,.43,(220,225,180),1)
        frame=np.hstack([im,panel]);writer.write(frame)
        if fi in {0,len(ids)//2,len(ids)-1}:cv2.imwrite(str(case/f'frame_{fi:05d}.jpg'),frame)
    writer.release();ren.close();cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for i in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,_=cap.read();checks.append(dict(frame=i,readable=bool(ok)))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['sim_time'],speed=1.,source='saved actual qpos; artificial lower placement then continuous original-gravity dynamics' if lower_entry else 'saved actual qpos from lossless CLOSE1B restore through every physical step',readback=checks));cap.release()

def run(args):
    global BASE,HMODEL
    lower_entry=getattr(args,'lower_entry',False);winding_sign=-1 if lower_entry else 1
    if lower_entry:
        BASE=ROOT/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009'
        HMODEL=ROOT/'mujoco_model/Snake_Robot_o20_lower_hook_transfer.xml'
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    tree=ET.parse(MODEL);root=tree.getroot();root.set('model','lossless-CLOSE1B-full-transfer');root.find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto','0.15 0 1.1 2.0 0 1.1');tree.write(HMODEL,encoding='utf-8',xml_declaration=True)
    sim=HookSim(case,lower_entry=lower_entry);r0=sim.snapshot();initial_points=r0['points_mm']/1000;route=ReformHookRoute(initial_points,morph_length=args.morph_length,winding_sign=winding_sign) if args.reform else MeasuredHookRoute(initial_points,extension_length=args.extension,winding_sign=winding_sign);source_start=route.source_start if args.reform else SOURCE_START
    if getattr(args,'lower_bridge',False):
        if not lower_entry or not args.reform:raise ValueError('lower bridge requires lower-entry reform')
        route=LowerBridgeReformHookRoute(initial_points,morph_length=args.morph_length);source_start=route.source_start
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
    if lower_entry:
        geometry.update(initial_state='artificial lower-entry mirror of real CLOSE1B geometry, zero velocity, mirrored all-joint preload',lower_entry_construction=sim.lower_report,actual_winding_sign=winding_sign)
        geometry['ready']=geometry['ready'] and sim.lower_report['maximum_node_mirror_error_mm']<1e-6 and sim.lower_report['bridge_entry_z_relative_to_axis_mm']<0 and geometry['initial_front8_wrap_deg']>180
        mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez(case/'initial_lower_state.npz',integration_state=sim.state,ctrl=sim.d.ctrl.copy(),sim_time=sim.d.time)
        lower_diagrams(case,route,sim.lower_report)
    write(case/'initial_geometry.json',geometry);print(json.dumps(geometry,default=default),flush=True)
    for src in ['snake_control/hook_start_transfer.py','snake_control/hook_reform_route.py','snake_control/l_three_segment_transfer.py','snake_control/horizontal_pipe_roll.py','snake_control/hook_pull_review_support.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py','snake_control/spatial_bishop_backbone.py','validation/hook_start_transfer_experiment.py','validation/l_three_segment_experiment.py','validation/suspended_supply_experiment.py',str(HMODEL.relative_to(ROOT))]:
        dest=case/'sources'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    for mesh in ET.parse(HMODEL).findall('./asset/mesh'):
        file=mesh.get('file')
        if file:dest=case/'sources/mujoco_model'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'mujoco_model'/file,dest)
    shutil.copy2(INITIAL,case/'initial_saved_state.npz');diagrams(case,route)
    if lower_entry:
        for name in ['lower_hook_initial.py','lower_bridge_hook_route.py']:
            dest=case/'sources/snake_control'/name;shutil.copy2(ROOT/'snake_control'/name,dest)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),first_candidate_maximum_dynamics_trials=3,source_state=str(INITIAL),
        action=('actual reverse hook smoothly reforms over first200mm sharedshift into8 target helix /8 underside bridge /5 retained source, with whole-head-axis alignment and common roll' if args.reform else 'existing measured source helix / actual bridge / reverse hook followed by smooth head-tangent connector and forward target helix; one shared material shift and J03/L04 common roll'),
        preload='retain complete saved ctrl; preserve its all-joint preload while smooth fading with first200mm sharedshift, no special joint angles',
        physics='original robot/gravity/friction/damping1.5/joint90deg/ctrl88deg/torque2.355Nm; target end extended toY2.0m before lossless initial restore',
        speed_mm_s=args.speed*1000,rate_setting_deg_s=args.rate,complete_shift_mm=route.total*1000,max_actual_seconds=args.seconds,max_wall_seconds=args.max_wall,deadline_utc=args.deadline,
        posterior_exit_roll_handoff=args.posterior_handoff,still_bearing_middle='keep original stiffness and measured registered guide during full-roll posterior supply',
        complete='whole reference window including connector passed; all21 center radii75-105mm, within6mm median; all adjacent positive winding and negativeY feed, local pitch error<8mm; pitch162-198mm and axisRMS<4mm; source absent, posterior bearing, allKp127, actual elapsed hold1s',
        stop='ground; actualjoint>90.5deg; self>3mm; pipe>3mm sustained0.2s; nonfinite; supported front collapses/falls1s; no source exit or target intake45s after2rod action; full reference complete without shape30s; wall/deadline',yamada_compliance_enabled=False))
    if lower_entry:
        plan=json.loads((case/'plan.json').read_text(encoding='utf-8'))
        plan.update(source_state='initial_lower_state.npz: artificial construction before first step; CLOSE1B is geometric parent only',actual_winding_sign=-1,active_upward_hook_acquisition_demonstrated=False,action='lower bridge entry; mirrored hook reformation and opposite target winding, original gravity dynamics; retain H03 posterior handoff and actual middle source support',complete=plan['complete'].replace('positive winding','negative raw winding / positive oriented winding'))
        if getattr(args,'lower_bridge',False):plan.update(action='same artificial lower hook morphs to a lower source outlet and lower target entry; nominal source turns half a revolution around source X-axis through joint control only; target head phase unchanged, opposite winding, H03 posterior handoff',reference_route='LowerBridgeReformHookRoute',complete_shift_mm=route.total*1000,reference_origin_mm=route.origin*1000)
        write(case/'plan.json',plan)
    if not geometry['ready']:write(case/'result.json',dict(outcome='initial_restore_or_geometry_unavailable',physical_steps=0));return
    if getattr(args,'prepare_only',False):
        write(case/'result.json',dict(outcome='geometry_ready_without_dynamics',physical_steps=0,ready=geometry['ready']))
        render(case,sim.rows,lower_entry=lower_entry);return
    continuous_handoff=getattr(args,'continuous_reform_handoff',False)
    ctrl=ThreeSegmentTransfer(sim.m,sim.scratch,speed=args.speed,rate_deg_s=args.rate,adaptive_supply=True,route=route,source_start=source_start,intake_y=.075,preserve_preload=True,handoff_on_posterior_exit=args.posterior_handoff,handoff_exit_count=getattr(args,'handoff_exit_count',None),continuous_reform_handoff=continuous_handoff,event_progress=continuous_handoff)
    if continuous_handoff:
        plan=json.loads((case/'plan.json').read_text(encoding='utf-8'))
        plan.update(additional_trial_explicitly_authorized=True,authorization='supervision after review of first three; one isolated U04, no parameter scan',continuous_reform_handoff=True,
            roll_supply_handoff='posterior clearance adjusts registered measured supply gradually; front keeps the complete original morph guide; only after200mm completed morph, blend guide and full roll over1.5 actual seconds',
            actual_progress_rule='first distinct received/cleared module event, or actual head advances10mm while first8 are continuously received and bear>1N; same45s/25s stalls and overall limits',
            complete='unchanged strict21-body helix, complete window, source absent and posterior bearing, Kp127, actual elapsed1s')
        write(case/'plan.json',plan)
    if getattr(args,'handoff_exit_count',None) is not None:
        plan=json.loads((case/'plan.json').read_text(encoding='utf-8'));plan['handoff_exit_count']=args.handoff_exit_count;plan['roll_supply_handoff']='start full target roll when this many original posterior modules actually clear; keep every still carrying source module at original Kp and registered measured guide';write(case/'plan.json',plan)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;complete_since=None;window_time=None;error=None;hold=0.
    try:
        while sim.d.time-sim.initial_time<args.seconds:
            elapsed=float(sim.d.time-sim.initial_time)
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,elapsed,r['contact_force_N'],r['source_contact_any']);sim.info=info;sim.commands.append(dict(t_s=elapsed,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r,winding_sign)
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
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r,winding_sign)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],absolute_saved_time=float(sim.d.time),wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,complete_shift_mm=route.total*1000,
            initial_front_wrap_deg=geometry['initial_front_wrap_deg'],final_front_wrap_deg=float(np.degrees(r['actual_front_span_rad'])),actual_current_source_exited=sorted(ctrl.exited),actual_current_received_target=sorted(ctrl.received),
            final_source_contacts=np.flatnonzero(r['source_contact_any']),final_source_target_force_n=r['contact_force_N'].sum(1),final_kp=ctrl.kp,
            maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),
            maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),ground=any(x['ground'] for x in sim.rows),
            actual_completion_hold_seconds=hold,whole_transfer_complete=outcome=='complete_from_saved_hook',physical_root_reset_during_run=False,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,engaged=sorted(ctrl.engaged),exited=sorted(ctrl.exited),bridge_coeff=ctrl.coeff,mapper_root=ctrl.mapper.root,mapper_tilt=ctrl.mapper.tilt,source_release_latch=sorted(ctrl.source_release_latch)))
        if lower_entry:
            result['initial_state_is_artificial']=True;result['active_upward_hook_acquisition_demonstrated']=False
            result['outcome']=result['outcome'].replace('complete_from_saved_hook','complete_from_artificial_lower_hook')
            write(case/'result.json',result)
        np.savez(case/'final_model_parameters.npz',gain=sim.m.actuator_gainprm,bias=sim.m.actuator_biasprm,damping=sim.m.dof_damping,force_range=sim.m.actuator_forcerange);print(json.dumps(result,default=default),flush=True)
        diagrams(case,route,sim.rows)
        try:render(case,sim.rows,lower_entry=lower_entry)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='H01');ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--rate',type=float,default=20.);ap.add_argument('--extension',type=float,default=.16125);ap.add_argument('--reform',action='store_true');ap.add_argument('--posterior-handoff',action='store_true');ap.add_argument('--handoff-exit-count',type=int);ap.add_argument('--lower-entry',action='store_true');ap.add_argument('--lower-bridge',action='store_true');ap.add_argument('--continuous-reform-handoff',action='store_true');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--morph-length',type=float,default=.20);ap.add_argument('--seconds',type=float,default=420);ap.add_argument('--max-wall',type=float,default=550);ap.add_argument('--deadline',default='2026-10-09T03:47:56+00:00');run(ap.parse_args())
