"""Two bounded actual experiments of a growing helix on one horizontal pipe."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback,xml.etree.ElementTree as ET
from datetime import datetime,timezone
import numpy as np,mujoco
from validation.suspended_supply_experiment import ActualSim,write,default,MODEL
import validation.passive_supply_experiment as replay
from snake_control.suspended_supply_transfer import WholeCurveMapper,LINK,DT,nodes
from snake_control.horizontal_pipe_roll import HorizontalPipeRoll,backbone

BASE=ROOT/'experiments/unified_curve_transfer_20261007/HORIZONTAL_PIPE_20261009'
SINGLE=ROOT/'mujoco_model/Snake_Robot_o20_single_horizontal_pipe.xml'

def prepare():
    BASE.mkdir(parents=True,exist_ok=True);out=BASE/'PREPARE';out.mkdir(exist_ok=False)
    tree=ET.parse(MODEL);root=tree.getroot();pipe=root.find("./worldbody/body[@name='pipe_network']")
    for geom in list(pipe):
        if geom.get('name')!='pipe_straight_04':pipe.remove(geom)
    feature=root.find("./worldbody/site[@name='feature_00']")
    if feature is not None:root.find('worldbody').remove(feature)
    root.set('model','original-robot-single-horizontal-pipe');tree.write(SINGLE,encoding='utf-8',xml_declaration=True)
    m=mujoco.MjModel.from_xml_path(str(SINGLE));mapper=WholeCurveMapper(m);guide,_=backbone(0.)
    qpos,report=mapper.initialize(guide);d=mapper.d;c=.5*(nodes(d)[:-1]+nodes(d)[1:]);rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
    phi=np.unwrap(np.arctan2(c[:9,2]-1,c[:9,0]-1.1));ground=any(con.geom1==m.geom('floor').id or con.geom2==m.geom('floor').id for con in d.contact)
    report.update(initial_front_span_rad=float(np.ptp(phi)),initial_tail_height_mm=float(nodes(d)[-1,2]*1000),
        rear_joint_max_deg=float(np.degrees(abs(qpos[mapper.qa][11:]).max())),ground=ground,source_pipe_removed=True,L_elbow_removed=True,
        ready=bool(np.ptp(phi)>np.pi and np.count_nonzero((rr[:9]>.065)&(rr[:9]<.125))>=7 and report['pipe_depth_mm']<1.2 and report['self_depth_mm']<.5 and not ground))
    np.savez(out/'initial_pose.npz',qpos=qpos,qvel=np.zeros(m.nv),ctrl=qpos[mapper.qa],reference_points=guide)
    write(out/'initial_geometry.json',report);shutil.copy2(SINGLE,out/SINGLE.name);print(json.dumps(report),flush=True)

class SingleSim(replay.PassiveSim):
    def __init__(self,case,initial):
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(SINGLE));self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]];self.head=self.m.body('head').id
        self.source=[];self.target=[self.m.geom('pipe_straight_04').id];self.floor=self.m.geom('floor').id
        self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION))
        self.d.qpos[:]=initial['qpos'];self.d.qvel[:]=initial['qvel'];self.d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(self.m,self.d)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21);self.exited=set();self.received=set();self.origin_time=0.

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False);report=json.loads((BASE/'PREPARE/initial_geometry.json').read_text())
    if not report['ready']:raise RuntimeError('Artificial initial chain has not met geometric conditions; no dynamics started')
    with np.load(BASE/'PREPARE/initial_pose.npz') as z:initial={k:z[k].copy() for k in z.files}
    for src in ['snake_control/horizontal_pipe_roll.py','snake_control/suspended_transfer_route.py','snake_control/spatial_bishop_backbone.py',
        'snake_control/suspended_supply_transfer.py','validation/horizontal_pipe_experiment.py','validation/passive_supply_experiment.py',
        'validation/suspended_supply_experiment.py','mujoco_model/Snake_Robot_o20_single_horizontal_pipe.xml']:
        dest=case/'sources'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    shutil.copy2(SINGLE,case/'sources/mujoco_model'/SINGLE.name)
    shutil.copy2(BASE/'PREPARE/initial_pose.npz',case/'initial_pose.npz');shutil.copy2(BASE/'PREPARE/initial_geometry.json',case/'initial_geometry.json')
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),task='independent horizontal single-pipe growing helix; not cross-pipe success',
        initial='manual real twenty-hinge front9 helix with posterior12 suspended, first placement only',initial_front_span_rad=report['initial_front_span_rad'],
        shared_total_shift_mm=774,nominal_action_seconds=309.6,phase_sign=args.phase_sign,front_only=args.front_only,true_twenty_hinge_projection=args.true_hinge,
        phase_rate='shared shift /(29mm/BETA), material roll in every active curvature component; J02 private head tilt allows discrete-rod chord waviness, common roll gauge stays within0.07rad',
        max_seconds=args.seconds,max_wall=args.max_wall,execution_deadline_utc=args.deadline,maximum_formal_dynamics_attempts=3,
        reception='J03: current near-pipe adjacent geometry, circumferential increase0.30..1.10rad, axial decrease2..45mm and pitch residual<25mm held1s; no cumulative signed turn from hanging origin' if args.connected_intake else 'signed time winding from initial hanging pose',
        pipe_axis_end_y_m=args.pipe_end,
        complete='J03: shared shift>=774mm, all21 modules near pipe, every neighbouring pair including tail follows same chirality and decreasing axis coordinate, fitted pitch126..234mm, axis RMS<12mm, all12 new modules retained and joint20 Kp>120; actual posterior bearing and1s retention',
        physics='original gravity, contact friction, robot geometry, mass, damping, kp127 and2.355Nm caps; declaration if posterior made passive',
        stop='ground, actual joint>90.5deg, self>3mm, actual pipe>3mm sustained0.2s, front detached/falling1s, registered no intake progress',
        no_physical_root_assignment_after_first_step=True))
    sim=SingleSim(case,initial);sim.snapshot();ctrl=HorizontalPipeRoll(sim.m,sim.scratch,phase_sign=args.phase_sign,front_passive=args.front_only,true_hinge=args.true_hinge,connected_intake=args.connected_intake)
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';error=None;deep=fall=complete_age=0.
    try:
        while sim.d.time<args.seconds:
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1];target,info=ctrl.command(sim.scratch,float(sim.d.time),r['contact_force_N']);sim.info=info
            sim.commands.append(dict(t_s=float(sim.d.time),ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1]
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
            falling=r['contact_force_N'][1,:9].sum()<.1 and (np.median(rr[:9])>.145 or c[:9,2].mean()<.9)
            fall=fall+DT if falling else 0.
            if fall>=1:outcome='front_bearing_collapsed';break
            if ctrl.failure:outcome=ctrl.failure;break
            phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));dy=np.diff(c[:,1]);da=np.diff(phi);use=abs(dy)>.002
            coherent=np.count_nonzero(use)>=16 and np.mean(dy[use]*da[use]<0)>.85 and c[0,1]-c[-1,1]>.30
            complete=bool(np.all((rr>.065)&(rr<.125)&(c[:,1]>.15)) and coherent and r['contact_force_N'][1,15:].sum()>.2 and len(ctrl.received)==12)
            if args.connected_intake:
                fit=np.polyfit(phi,c[:,1],1);pitch=-fit[0]*2*np.pi;axis_rms=np.sqrt(np.mean((c[:,1]-np.polyval(fit,phi))**2))
                complete=bool(complete and ctrl.shift>=12*LINK and np.all((da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045))
                    and .126<pitch<.234 and axis_rms<.012 and ctrl.kp[-1]>120.)
            complete_age=complete_age+DT if complete else 0.
            if complete_age>=1:outcome='complete_single_pipe_helix';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(float(sim.d.time),2),shift_mm=round(ctrl.shift*1000,2),phase_rad=round(ctrl.phase,3),new_retained=sorted(ctrl.received),
                    front_span_rad=round(r['actual_front_span_rad'],3),target_normal_n=round(float(r['contact_force_N'][1].sum()),2),tracking_deg=round(info['tracking_deg'],2))
                print(json.dumps(msg),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1]
        np.savez(case/'final_model_parameters.npz',actuator_gainprm=sim.m.actuator_gainprm,actuator_biasprm=sim.m.actuator_biasprm,
            dof_damping=sim.m.dof_damping,forcerange=sim.m.actuator_forcerange)
        result=dict(name=args.name,outcome=outcome,actual_seconds=float(sim.d.time),wall_seconds=time.monotonic()-start,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,
            new_retained_modules=sorted(ctrl.received),historical_retained_max=ctrl.best_received,front_span_initial_rad=sim.rows[0]['actual_front_span_rad'],front_span_final_rad=r['actual_front_span_rad'],
            maximum_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),
            maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),
            maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),final_new_local_bearing_n=r['contact_force_N'][1,9:],ground=any(x['ground'] for x in sim.rows),
            save_errors=errors,error=error,physical_root_reset_during_run=False,cross_pipe_success=False)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,shift_mm=ctrl.shift*1000,phase_rad=ctrl.phase,
            received=sorted(ctrl.received),engaged=sorted(ctrl.engaged),received_age=ctrl.received_age,phi0=ctrl.phi0,phi=ctrl.phi,previous_raw=ctrl.previous_raw,
            private_mapper_root=ctrl.mapper.root if ctrl.mapper else None,private_mapper_tilt=ctrl.mapper.tilt if ctrl.mapper else None))
        print(json.dumps(result,default=default),flush=True);replay.MODEL=SINGLE
        try:replay.render(case,sim.rows)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--name',default='J01');ap.add_argument('--phase-sign',type=float,default=1)
    ap.add_argument('--front-only',action='store_true');ap.add_argument('--true-hinge',action='store_true');ap.add_argument('--connected-intake',action='store_true')
    ap.add_argument('--pipe-end',type=float,default=1.2);ap.add_argument('--seconds',type=float,default=450);ap.add_argument('--max-wall',type=float,default=600)
    ap.add_argument('--deadline',default='2026-10-09T01:57:00+00:00');args=ap.parse_args()
    if args.prepare_only:prepare()
    else:
        if args.pipe_end!=1.2:
            tree=ET.parse(SINGLE);tree.getroot().find("./worldbody/body[@name='pipe_network']/geom[@name='pipe_straight_04']").set('fromto',f'0.15 0 1.1 {args.pipe_end} 0 1.1')
            SINGLE=ROOT/'mujoco_model/Snake_Robot_o20_single_horizontal_pipe_long.xml';tree.write(SINGLE,encoding='utf-8',xml_declaration=True)
        run(args)
