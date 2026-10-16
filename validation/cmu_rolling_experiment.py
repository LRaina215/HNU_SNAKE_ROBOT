"""Prepare and run at most two complete CMU rolling L-junction candidates."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
import shutil
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from snake_control.cmu_rolling_transfer import (CMURollingTransfer, A, WS, SOURCE_PHASE,
     FREQUENCY, MODEL_NAME, initial_pose, smooth, blend_rotation, front_status)
from snake_control.cmu_window_gait import placed_pose, wave
from snake_control.suspended_supply_transfer import LINK, DT, nodes
from validation.suspended_supply_experiment import ActualSim, write, default

BASE = ROOT/'experiments/unified_curve_transfer_20261007/CMU_ROLLING_REPRO_20261011'
MODEL = ROOT/'mujoco_model'/MODEL_NAME
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False})


class RollingSim(ActualSim):
    def __init__(self, case, initial):
        self.case = case
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d, self.scratch = mujoco.MjData(self.m), mujoco.MjData(self.m)
        self.qa = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.va = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.head = self.m.body('head').id
        self.source = [self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']]
        self.target = [self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor = self.m.geom('floor').id
        self.robot = [next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.head+j) for j in range(21)]
        self.state = np.empty(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.d.qpos[:] = initial['qpos']
        self.d.qvel[:] = initial['qvel']
        self.d.ctrl[:] = initial['ctrl']
        mujoco.mj_forward(self.m, self.d)
        self.rows, self.commands, self.info = [], [], {}
        self.exit_age, self.receive_age = np.zeros(6), np.zeros(21)
        self.exited, self.received = set(), set()

    def snapshot(self):
        row = super().snapshot()
        pipe_contacts = np.zeros(21, dtype=bool)
        target_contacts = np.zeros(21, dtype=bool)
        for con in self.scratch.contact:
            if con.dist>0:continue
            for pipe, robot in [(con.geom1,con.geom2),(con.geom2,con.geom1)]:
                body=int(self.m.geom_bodyid[robot])
                if self.head<=body<self.head+21 and (self.m.geom(pipe).name or '').startswith('pipe_'):
                    pipe_contacts[body-self.head]=True
                    if pipe in self.target:target_contacts[body-self.head]=True
        prefix, wrap, grip, near = front_status(self.scratch, row['contact_force_N'])
        row.update(actual_target_prefix=prefix, actual_front_span_rad=wrap,
                   actual_receive_mask=near, kp_nm_per_rad=self.m.actuator_gainprm[:,0].copy(),
                   window_transition_modules=self.info.get('window_transition_modules', 6.),
                   roll_phase_rad=self.info.get('roll_phase_rad', SOURCE_PHASE),
                   loosening_factor=self.info.get('loosening_factor', 1.),
                   controller_clock_s=self.info.get('controller_clock_s', 0.),
                   projection_position_error_mm=self.info.get('projection', {}).get('position_error_mm', 0.),
                   projection_orientation_error_deg=self.info.get('projection', {}).get('orientation_error_deg', 0.))
        row.update(all_pipe_contact_any=pipe_contacts,all_target_contact_any=target_contacts,raw_pipe_contact_logged=True)
        return row


def shape_stats(row):
    c = row['centers_mm']/1000
    rr = np.linalg.norm(c[:,[0,2]]-[1.1,1.], axis=1)
    phi = np.unwrap(np.arctan2(c[:,2]-1.,c[:,0]-1.1))
    dy, da = np.diff(c[:,1]), np.diff(phi)
    fit = np.polyfit(phi, c[:,1], 1)
    pitch = -fit[0]*2*np.pi
    rms = np.sqrt(np.mean((c[:,1]-np.polyval(fit,phi))**2))
    local = dy+.18/(2*np.pi)*da
    strict = bool(np.all((rr>.075)&(rr<.105)&(c[:,1]>.15)) and abs(rr-np.median(rr)).max()<.006
           and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)) and abs(local).max()<.008
           and .162<pitch<.198 and rms<.004)
    return dict(strict_actual_shape=strict, center_radius_mm=rr*1000, pitch_mm=pitch*1000,
                axis_rms_mm=rms*1000, local_pitch_residual_mm=local*1000)


def setup_axes(ax):
    ax.plot([.45,1.1],[0,0],[1,1],color='gray',lw=13,alpha=.25)
    ax.plot([1.1,1.1],[0,.75],[1,1],color='gray',lw=13,alpha=.25)
    ax.set(xlabel='X / m',ylabel='Y / m',zlabel='Z UP / m',xlim=(.45,1.3),ylim=(-.20,.8),zlim=(.60,1.20))
    ax.view_init(27,140)


def reference(case, model, pose, variant=1):
    d = mujoco.MjData(model)
    qa = model.jnt_qposadr[model.actuator_trnid[:,0]]
    d.qpos[:]=pose
    d.ctrl[:]=pose[qa]
    mujoco.mj_fwdPosition(model,d)
    ctrl = CMURollingTransfer(model,d,variant)
    records = [dict(stage='初始全身原管', t_s=0., points_mm=nodes(d)*1000, physical_steps=0)]
    for t, label in [(20.,'前部降低波幅并向下接近'),(30.,'下部接近结束'),(55.,'前部恢复目标包绕')]:
        phase = SOURCE_PHASE-2*np.pi*FREQUENCY*t
        # The carrying-frame motion is a reference assumption from the prior
        # first-cycle source displacement, not a dynamics prediction.
        source_pose,_,_=placed_pose(model,A,WS,phase,12,.83+.0023642*t)
        d.qpos[:]=source_pose
        mujoco.mj_fwdPosition(model,d)
        u=float(smooth(t/30.));co=float(smooth((t-30.)/25.))
        loosen=1-.6*float(smooth(t/15.))+.6*co
        head_phase=phase+np.pi/2*u
        raw=wave(A,WS,phase);raw[:6]=wave(A*loosen,WS,head_phase)[:6]
        if variant==2:raw[6:12]=wave(A*loosen,WS,phase)[6:12]
        goal_p,goal_r,ideal=ctrl.target.pose(head_phase,6.,.088+.09*(1-loosen))
        if u<1:
            ctrl.connector.register(d,wave(A,WS,phase),18)
            sp=ctrl.connector.d.xpos[ctrl.head+6].copy();sr=ctrl.connector.d.xmat[ctrl.head+6].reshape(3,3)
            radial0=sp[[1,2]]-[0.,1.];phi0=np.arctan2(radial0[1],radial0[0]);phi1=-np.pi/4
            while phi1<phi0:phi1+=2*np.pi
            phi=(1-u)*phi0+u*phi1;radial=(1-u)*np.linalg.norm(radial0)+u*.17
            around=np.array([(1-u)*sp[0]+u*.99,radial*np.cos(phi),1+radial*np.sin(phi)])
            late=float(smooth((u-.65)/.35));goal_p=(1-late)*around+late*goal_p
            goal_r=blend_rotation(sr,goal_r,u)
        seed=source_pose[qa].copy()
        if variant==2:seed[6:12]=raw[6:12]
        q,report=ctrl.connector.solve(d,raw,seed,np.arange(6,12),18,6,goal_p,goal_r,unlimited=True,orientation_weight=co)
        records.append(dict(stage=label,t_s=t,points_mm=report['points_mm'],q_rad=q,goal_mm=goal_p*1000,
                            loosening=loosen,projection=report,physical_steps=0,
                            carrying_frame_motion='assumed +2.3642mm/s from prior C01 first cycle; not predicted dynamics'))
    _,_,end=ctrl.target.pose(SOURCE_PHASE-2*np.pi*FREQUENCY*240+np.pi/2,20.)
    records.append(dict(stage='期望整身目标螺旋',t_s=240.,points_mm=end*1000,physical_steps=0,
                        reference_only=True, actual_transfer_proven=False))
    write(case/'candidate_reference.json',records)
    fig=plt.figure(figsize=(17,10))
    for i,r in enumerate(records):
        ax=fig.add_subplot(2,3,i+1,projection='3d');setup_axes(ax)
        p=np.asarray(r['points_mm'])/1000
        ax.plot(*p.T,'o-',ms=3,color='navy');ax.plot(*p[:8].T,'o-',ms=3,color='darkgreen')
        ax.plot(*p[7:13].T,'o-',ms=3,color='darkorange');ax.scatter(*p[0],c='crimson',s=35)
        ax.set_title(r['stage'],fontsize=11)
    ax=fig.add_subplot(2,3,6);ax.axis('off')
    text=f'同一候选 {case.name}\n\n全身正手性源管正弦波起点\n前部7杆＋转接5杆＋源管9杆\n中间6关节完成转接\n\n持续累计关节波；波幅先减后恢复\n前部真实包绕后才向尾部切窗\n展开过程中后部继续滚动\n\n本图为私有FK参考，非运动结果\n轴向配准假设不证明实际送身\n完整终点形态仅表示目标'
    if variant==2:text+='\nR02：先松开包含转接段的完整前部'
    ax.text(.03,.95,text,va='top',fontsize=12)
    fig.suptitle(f'CMU持续滚动跨管 {case.name}：初态、展开、恢复和整身目标',fontsize=17)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(case/'candidate_reference.png',dpi=145);plt.close(fig)


def freeze(case):
    files=['snake_control/cmu_rolling_transfer.py','snake_control/cmu_window_gait.py',
           'snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py',
           'snake_control/spatial_bishop_backbone.py','validation/cmu_rolling_experiment.py',
           'validation/suspended_supply_experiment.py',str(MODEL.relative_to(ROOT))]
    for f in files:
        dest=case/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    for mesh in ET.parse(MODEL).findall('./asset/mesh'):
        if mesh.get('file'):
            f=mesh.get('file');dest=case/'sources/mujoco_model'/f;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'mujoco_model'/f,dest)


def prepare(case, variant):
    case.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(MODEL))
    q=np.load(BASE/'R01/initial_pose.npz')['qpos'] if variant==2 else initial_pose(m)
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    initial=dict(qpos=q,qvel=np.zeros(m.nv),ctrl=q[qa])
    sim=RollingSim(case,initial);r=sim.snapshot();c=r['centers_mm']/1000
    sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
    target_contacts=0
    for con in sim.scratch.contact:
        if con.dist<=0 and ({con.geom1,con.geom2}&set(sim.target)):
            target_contacts+=1
    report=dict(initial_all_source=bool(np.all((sr>.075)&(sr<.11)&(c[:,0]>-.2)&(c[:,0]<1.1565))),
                initial_target_contacts=target_contacts,pipe_depth_mm=r['pipe_depth_mm'],self_depth_mm=r['self_depth_mm'],
                maximum_joint_deg=float(np.degrees(abs(q[qa])).max()),source_center_radii_mm=sr*1000,
                actual_points_mm=r['points_mm'],front_rods=7,connector_rods=5,source_rods=9,
                active_front_length_mm=12*LINK*1000,initial_placement='native positive-handed rolling sine; selected M12 on source +Y side; seam X830mm',
                previous_start_changed=True,source_and_target_geometry_same_chirality=True,
                physics_steps=0,reference_fit_is_not_admission_to_full_success=True)
    report['ready']=bool(report['initial_all_source'] and not target_contacts and report['pipe_depth_mm']<1.2
                    and report['self_depth_mm']<.5 and report['maximum_joint_deg']<=88.01 and not r['ground'])
    np.savez(case/'initial_pose.npz',**initial);write(case/'initial_geometry.json',report)
    write(case/'plan.json',dict(candidate=case.name,variant=variant,maximum_complete_candidates=2,
          controller='absolute head/tail CMU sine waves plus six connector hinge pose fit',
          unfold_controller_seconds=30,restore_controller_seconds=25,continued_roll_without_capture_seconds=50,
          window_rate_modules_s=np.pi*.058*FREQUENCY/LINK,
          wave_frequency_cycles_s=FREQUENCY,head_tail_offset_rad=-np.pi/2,
          phase_spatial_propagation_not_used_as_material_window_speed=True,
          no_extra_release_wave=True,original_physics_preserved=True,root_reset_during_run=False,
          completion='actual all21 single target helix + original source clear + posterior target bearing + hold1s',
          stop='ground; total pipe loss; numerical/joint failure; self>3mm; pipe>3mm for0.2s; completed unfold/restore/one cycle with no front capture'))
    reference(case,m,q,variant);freeze(case)
    print(json.dumps(report,default=default),flush=True)


def video(case, rows):
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model'/MODEL_NAME));d=mujoco.MjData(m)
    ren=mujoco.Renderer(m,height=600,width=900);cam=mujoco.MjvCamera()
    cam.azimuth=140;cam.elevation=-26;cam.distance=1.65;cam.lookat[:]=[.95,.20,.95]
    wr=cv2.VideoWriter(str(case/'actual_continuous_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1240,600))
    if not wr.isOpened():raise RuntimeError('Video writer failed')
    selected=list(range(0,len(rows),5))
    for fi,j in enumerate(selected):
        r=rows[j];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d)
        zmin=r['points_mm'][:,2].min()/1000;drop=np.clip((.8-zmin)/.8,0,1)
        cam.lookat[2]=.95-.4*drop;cam.distance=1.65+.8*drop
        ren.update_scene(d,camera=cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
        panel=np.full((600,340,3),25,np.uint8)
        cv2.putText(im,f'{case.name} ACTUAL continuous 1x {r["sim_time"]:.2f}s',(10,25),0,.65,(255,255,255),1)
        labels=['CMU waves + six-hinge connector',r['stage'],f'Clock {r["controller_clock_s"]:.1f}s / L {r["loosening_factor"]:.3f}',
                f'Actual target prefix {r["actual_target_prefix"]}/21',f'Window {r["window_transition_modules"]:.2f}',
                f'Joint phase {r["roll_phase_rad"]:.2f}rad',f'Source/target N {r["contact_force_N"].sum(1).round(1)}',
                f'Pipe/self mm {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f}',f'Actual tracking {r["tracking_deg"]:.1f}deg',
                f'Connector error {r["projection_position_error_mm"]:.1f}mm']
        for k,line in enumerate(labels):cv2.putText(panel,line,(8,24+24*k),0,.40,(230,230,230),1)
        for k in range(20):
            cv2.putText(panel,f'J{k+1:02d} {np.degrees(r["q"][k]):+5.1f} {r["torque"][k]:+4.2f}',
                        (8+(k//10)*163,282+(k%10)*28),0,.38,(225,225,170),1)
        frame=np.hstack([im,panel]);wr.write(frame)
        if fi in {0,len(selected)//2,len(selected)-1}:cv2.imwrite(str(case/f'actual_frame_{fi:05d}.jpg'),frame)
    wr.release();ren.close()
    cap=cv2.VideoCapture(str(case/'actual_continuous_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for j in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,_=cap.read();checks.append(dict(frame=j,readable=bool(ok)))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['sim_time'],
                               speed=1.,readback=checks,source='saved qpos from this one uninterrupted joint-driven dynamics run'))
    cap.release()


def run(case, variant, max_seconds, resume=False):
    if (case/'result.json').exists() and not resume:raise RuntimeError('Candidate already ended; not rerunning')
    ready=json.loads((case/'initial_geometry.json').read_text(encoding='utf-8'))
    if not ready['ready']:raise RuntimeError('Artificial all-source initial geometry not ready')
    initial=np.load(case/'initial_pose.npz')
    sim=RollingSim(case,initial)
    prior_wall=0.
    if resume:
        prior=json.loads((case/'result.json').read_text(encoding='utf-8'))
        write(case/'provisional_stop_diagnostic.json',prior)
        saved=np.load(case/'final_state.npz')
        mujoco.mj_setState(sim.m,sim.d,saved['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(sim.m,sim.d)
        archive=np.load(case/'trace.npz');old={k:archive[k] for k in archive.files};sim.rows=[]
        for i in range(len(old['sim_time'])):
            row={k:old[k][i].copy() if isinstance(old[k][i],np.ndarray) else old[k][i].item() for k in old}
            row.update(all_pipe_contact_any=np.zeros(21,dtype=bool),all_target_contact_any=np.zeros(21,dtype=bool),raw_pipe_contact_logged=False)
            sim.rows.append(row)
        sim.commands=json.loads((case/'commands.json').read_text(encoding='utf-8'))
        sim.info=sim.commands[-1]
        sim.snapshot();last=sim.rows.pop();sim.rows[-1]=last
        ctrl=CMURollingTransfer(sim.m,sim.scratch,variant)
        state=json.loads((case/'controller_state.json').read_text(encoding='utf-8'))
        ctrl.last=np.array(state['last_ctrl']);ctrl.clock=state['clock'];ctrl.phase=state['phase'];ctrl.boundary=state['boundary']
        ctrl.acquired=prior['active_front_capture'];ctrl.best_prefix=prior['maximum_target_prefix'];ctrl.events=prior['events']
        ctrl.last_gain_time=max([e['t_s'] for e in ctrl.events]+[float(sim.d.time)])
        prior_wall=prior['wall_seconds']
        write(case/'run_segments.json',dict(reason='correct side-bearing-zero falsely classified target-miter contact as total pipe loss',
             first_segment_s=[0,float(sim.d.time)],second_segment_starts_s=float(sim.d.time),
             integration_state_restored=True,maximum_qpos_change_at_resume=float(abs(sim.d.qpos-old['qpos'][-1]).max()),
             previous_times_not_replayed=True,new_candidate=False))
        freeze(case)
    else:
        sim.snapshot();ctrl=CMURollingTransfer(sim.m,sim.scratch,variant)
    ledger_path=BASE/'ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if not resume:
        if sum(x['actual_started'] for x in ledger)>=2:raise RuntimeError('Two-candidate limit reached')
        ledger.append(dict(candidate=case.name,actual_started=True));write(ledger_path,ledger)
    start=time.monotonic();outcome='maximum_actual_duration';error=None;deep=unsupported=0.;complete_age=0.
    try:
        while sim.d.time<max_seconds:
            t=float(sim.d.time);r=sim.rows[-1]
            if (case/'stop_and_save.request').exists():outcome='requested_stop';break
            q,info=ctrl.command(sim.scratch,t,r['contact_force_N'],r['source_contact_any'])
            sim.info=info;sim.commands.append(dict(t_s=t,ctrl_rad=q.copy(),**info));sim.command(q);r=sim.rows[-1]
            if r['ground']:outcome='ground_contact';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if np.degrees(abs(r['q'])).max()>90.5:outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            unsupported=unsupported+DT if not r['all_pipe_contact_any'].any() else 0.
            if unsupported>=1:outcome='total_pipe_loss';break
            stats=shape_stats(r)
            complete=bool(stats['strict_actual_shape'] and not r['source_contact_any'].any()
                      and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2)
            complete_age=complete_age+DT if complete else 0.
            if complete_age>=1:outcome='complete_whole_target_helix';break
            if ctrl.failure:outcome=ctrl.failure;break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),clock_s=round(ctrl.clock,2),stage=info['stage'],prefix=r['actual_target_prefix'],
                         window=round(ctrl.boundary,2),bearing_n=r['contact_force_N'].sum(1),track_deg=round(info['tracking_deg'],2),
                         connector_mm=round(info['projection']['position_error_mm'],2),pipe_mm=r['pipe_depth_mm'])
                write(case/'progress.json',msg);print(json.dumps(msg,default=default),flush=True)
    except Exception:
        outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        save_errors=sim.save();r=sim.rows[-1];stats=shape_stats(r)
        result=dict(candidate=case.name,outcome=outcome,actual_seconds=r['sim_time'],wall_seconds=prior_wall+time.monotonic()-start,
                    controller_clock_s=ctrl.clock,active_front_capture=ctrl.acquired,maximum_target_prefix=ctrl.best_prefix,
                    final_target_prefix=r['actual_target_prefix'],final_window=ctrl.boundary,events=ctrl.events,
                    whole_transfer_complete=outcome=='complete_whole_target_helix',actual_completion_hold_seconds=complete_age,
                    final_source_target_force_n=r['contact_force_N'].sum(1),final_source_contact_rods=np.flatnonzero(r['source_contact_any']),
                    maximum_joint_deg=max(float(np.degrees(abs(x['q'])).max()) for x in sim.rows),
                    maximum_drive_torque_nm=max(float(abs(x['torque']).max()) for x in sim.rows),
                    maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),
                    maximum_command_step_deg=max(x['maximum_dispatch_step_deg'] for x in sim.commands),
                    physical_root_reset_during_run=False,original_physics_preserved=True,no_extra_release_wave=True,
                    save_errors=save_errors,error=error,**stats)
        result['lossless_resume_used']=resume
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,phase=ctrl.phase,clock=ctrl.clock,boundary=ctrl.boundary))
        print(json.dumps(result,default=default),flush=True)
        try:video(case,sim.rows)
        except Exception:write(case/'video_error.json',dict(error=traceback.format_exc()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['R01','R02'],required=True)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--seconds',type=float,default=600)
    ap.add_argument('--resume-last',action='store_true')
    args=ap.parse_args();case=BASE/args.name;variant=1 if args.name=='R01' else 2
    if args.prepare_only:prepare(case,variant)
    else:run(case,variant,args.seconds,args.resume_last)
