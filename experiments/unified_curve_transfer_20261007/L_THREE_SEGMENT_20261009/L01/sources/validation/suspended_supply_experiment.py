"""Registered, bounded actual runs of the suspended-supply macro gait."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import time
import shutil
import traceback
from datetime import datetime,timezone
import cv2
import numpy as np
import mujoco
from snake_control.suspended_supply_transfer import WholeCurveMapper,SuspendedSupplyTransfer,desired_curve,nodes,LINK,DT

BASE=ROOT/'experiments/unified_curve_transfer_20261007/SUSPENDED_SUPPLY_20261009'
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'

def default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n',encoding='utf-8')

def geometry_prepare(name):
    out=BASE/name;out.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_xml_path(str(MODEL));mapper=WholeCurveMapper(model)
    guide,_=desired_curve('peel',0.)
    pose,report=mapper.initialize(guide)
    np.savez(out/'initial_pose.npz',qpos=pose,qvel=np.zeros(model.nv),ctrl=pose[mapper.qa],reference_points=guide)
    write(out/'initial_geometry.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return out,report

class ActualSim:
    def __init__(self,case,initial):
        self.case=case;self.m=mujoco.MjModel.from_xml_path(str(MODEL))
        self.d=mujoco.MjData(self.m);self.scratch=mujoco.MjData(self.m)
        self.qa=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.va=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]]
        self.head=self.m.body('head').id
        self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']]
        self.target=[self.m.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']]
        self.floor=self.m.geom('floor').id
        self.robot=[next(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.head+j) for j in range(21)]
        self.state=np.empty(mujoco.mj_stateSize(self.m,mujoco.mjtState.mjSTATE_INTEGRATION))
        self.d.qpos[:]=initial['qpos'];self.d.qvel[:]=initial['qvel'];self.d.ctrl[:]=initial['ctrl']
        mujoco.mj_forward(self.m,self.d)
        self.rows=[];self.commands=[];self.info={};self.exit_age=np.zeros(6);self.receive_age=np.zeros(21)
        self.exited=set();self.received=set(range(9));self.front_failure_age=0.

    def snapshot(self):
        mujoco.mj_getState(self.m,self.d,self.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_setState(self.m,self.scratch,self.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m,self.scratch);d=self.scratch
        p=nodes(d);c=.5*(p[:-1]+p[1:])
        forces=np.zeros((2,21));pd=sd=0.;ground=False;source_any=np.zeros(21,dtype=bool)
        for i,con in enumerate(d.contact):
            b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]])
            one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
            ground|=bool((con.geom1==self.floor and two) or (con.geom2==self.floor and one))
            if one!=two and any((self.m.geom(g).name or '').startswith('pipe_') for g in [con.geom1,con.geom2]):pd=max(pd,-con.dist)
            if one and two and abs(b1-b2)>1:sd=max(sd,-con.dist)
            for pipe,body in [(con.geom1,b2),(con.geom2,b1)]:
                if not self.head<=body<self.head+21:continue
                side=0 if pipe in self.source else 1 if pipe in self.target else None
                if side is None:continue
                if side==0 and con.dist<=0.:source_any[body-self.head]=True
                # Score cylindrical side bearing only, independently of reference regions.
                if abs(con.frame[side])>.35 or (side==1 and con.pos[1]<.15):continue
                force=np.zeros(6);mujoco.mj_contactForce(self.m,d,i,force)
                forces[side,body-self.head]+=max(0.,force[0])
        # The API distance query to the miter gave large negative values even
        # for rods far outside it in H01. Do not use it for exit credit.
        source_radius=np.linalg.norm(c[:,[1,2]]-[0.,1.],axis=1)
        gaps=source_radius-.0565-.029
        rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        rod=(p[:-1]-p[1:])/LINK
        coiled=(c[:,1]>.205)&(rr>.065)&(rr<.12)&(rod[:,1]>.10)
        local_bearing=np.array([forces[1,max(0,j-1):min(21,j+2)].sum() for j in range(21)])>.2
        self.receive_age=np.where(coiled&local_bearing,self.receive_age+.01,0.)
        self.exit_age=np.where((source_radius[15:]>.130)&(~source_any[15:])&(forces[0,15:]<.1),self.exit_age+.01,0.)
        # Exit and intake are observed and retained, never inferred from q/r/S.
        self.exited={j+15 for j in range(6) if self.exit_age[j]>=.5}
        for j in range(9,21):
            if self.receive_age[j]>=.5:self.received.add(j)
        phi=np.unwrap(np.arctan2(c[:9,2]-1.,c[:9,0]-1.1))
        current_received=np.array([j in self.received and self.receive_age[j]>.2 for j in range(21)])
        self.rows.append(dict(sim_time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),
            q=d.qpos[self.qa].copy(),torque=d.qfrc_actuator[self.va].copy(),points_mm=p*1000,centers_mm=c*1000,
            contact_force_N=forces,source_contact_any=source_any,source_gap_mm=gaps*1000,ground=ground,pipe_depth_mm=float(pd*1000),self_depth_mm=float(sd*1000),
            actual_front_span_rad=float(np.ptp(phi)),actual_exit_mask=np.array([j in self.exited for j in range(21)]),
            actual_receive_mask=current_received,stage=self.info.get('stage','initial'),shape_parameter=self.info.get('shape_parameter',0.),
            shift_mm=self.info.get('shift_mm',0.),radius_delta_mm=self.info.get('curve_radius_delta_mm',0.),
            tracking_deg=self.info.get('tracking_deg',0.),reference_points_mm=np.array(self.info.get('reference_points_mm',p*1000))))
        return self.rows[-1]

    def command(self,target):
        self.d.ctrl[:]=target
        for j in range(50):
            mujoco.mj_step(self.m,self.d)
            if j%10==9:self.snapshot()

    def save(self):
        errors=[]
        for name,operation in [('trace.npz',lambda:np.savez_compressed(self.case/'trace.npz',**{k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})),
            ('commands.json',lambda:write(self.case/'commands.json',self.commands)),
            ('final_state.npz',lambda:np.savez(self.case/'final_state.npz',integration_state=self.state,ctrl=self.d.ctrl,sim_time=self.d.time))]:
            try:operation()
            except Exception as exc:errors.append(dict(artifact=name,error=repr(exc)))
        return errors

def video(case,rows):
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=600,width=880);camera=mujoco.MjvCamera()
    camera.lookat[:]=[1.,.21,.80];camera.distance=1.85;camera.azimuth=145;camera.elevation=-22
    writer=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1240,600))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    indices=list(range(0,len(rows),5))
    for frame,i in enumerate(indices):
        r=rows[i];data.qpos[:]=r['qpos'];mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera)
        image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        cv2.putText(image,f'{case.name} {r["sim_time"]:.2f}s {r["stage"]}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(image,'ACTUAL joint-driven dynamics / continuous 1x',(12,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        panel=np.full((600,360,3),25,np.uint8)
        labels=[f'Shared shift {r["shift_mm"]:.1f} mm',f'Shape phase {r["shape_parameter"]:.3f}',
            f'Front wrap {r["actual_front_span_rad"]:.3f} rad',f'Track {r["tracking_deg"]:.1f} deg',
            f'Source exits {np.count_nonzero(r["actual_exit_mask"])} / 6',
            f'New retained {np.count_nonzero(r["actual_receive_mask"][9:])} / 12',
            f'Actual pipe depth {r["pipe_depth_mm"]:.2f} mm','Circles: actual source / target bearing']
        for line,label in enumerate(labels):cv2.putText(panel,label,(12,24+24*line),cv2.FONT_HERSHEY_SIMPLEX,.44,(225,225,225),1)
        for j in range(21):
            x,y=12+(j//11)*170,255+(j%11)*28
            cv2.putText(panel,f'M{j:02d}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
            for side in range(2):cv2.circle(panel,(x+65+side*35,y-5),6,(80,210,90) if r['contact_force_N'][side,j]>.1 else (65,65,65),-1)
        whole=np.hstack([image,panel]);writer.write(whole)
        if frame in {0,len(indices)//2,len(indices)-1}:cv2.imwrite(str(case/f'frame_{frame:05d}.jpg'),whole)
    writer.release();renderer.close()
    cap=cv2.VideoCapture(str(case/'replay_1x.mp4'))
    result=dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),actual_seconds=rows[-1]['sim_time'],speed=1.,source='actual saved qpos only')
    good=[]
    for i in [0,int(result['frames'])//2,int(result['frames'])-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,_=cap.read();good.append(dict(frame=i,readable=bool(ok)))
    result['readback']=good;cap.release();write(case/'video.json',result)

def run(name,initial_dir,max_seconds,max_wall,deadline):
    case=BASE/name;case.mkdir(parents=True,exist_ok=False)
    report=json.loads((initial_dir/'initial_geometry.json').read_text(encoding='utf-8'))
    if not report['ready']:raise RuntimeError('Initial real-chain geometry did not pass; no physics started')
    with np.load(initial_dir/'initial_pose.npz') as z:initial={k:z[k].copy() for k in z.files}
    for src in ['snake_control/suspended_transfer_route.py','snake_control/suspended_supply_transfer.py','snake_control/spatial_bishop_backbone.py','validation/suspended_supply_experiment.py','mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml']:
        target=case/'sources'/src;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,target)
    plan=dict(name=name,registered_utc=datetime.now(timezone.utc).isoformat(),initial='artificial double-pipe 9 target / 6 bridge / 6 source, real twenty-hinge constrained fit',
        nominal_stages_seconds=dict(peel=50.,unfold=55.,haul=309.6),expected_trajectory_seconds=414.6,maximum_actual_seconds=max_seconds,
        maximum_wall_seconds=max_wall,execution_deadline_utc=deadline,maximum_physics_attempts_in_branch=3,
        no_progress=dict(initial_bearing_not_established_s=8.,continuous_curve_chain_unavailable_s=8.,source_exit_s=75.,haul_s=30.,
            end_of_phase='stop when source peel/unfold or two-rod haul target completes without corresponding actual exit/intake'),
        failure_stops='ground; actual >90.5 deg; nonfinite; self depth >3 mm; target front detached and falling 1 s',
        limits=dict(command_deg=88.,rate_deg_s=10.,torque_nm=2.355),physics='original gravity/friction/actuators unchanged',
        macro_release='current cleared prefix becomes measured continuous free bow; retain only still-carried source suffix; capture its unit-tangent bow at complete release, morph to fixed hanging guide',
        predicted_soft_contact='actual observed depth +0.8 mm, minimum1.2 mm and maximum2.5 mm; original collision shells, actual ground/self/joint failure stops retained',
        exit_condition='current source radius >130mm and no source geom contact continuously0.5s; historical clearance alone never changes phase',
        physical_root_assignment='only artificial placement before first mj_step; joint ctrl only thereafter')
    write(case/'plan.json',plan);shutil.copy2(initial_dir/'initial_geometry.json',case/'initial_geometry.json')
    shutil.copy2(initial_dir/'initial_pose.npz',case/'initial_pose.npz')
    sim=ActualSim(case,initial);ctrl=SuspendedSupplyTransfer(sim.m,initial['qpos'])
    sim.snapshot();started=time.monotonic();outcome='maximum_actual_duration';complete_age=0.;failure_age=0.;error=None
    deadline_epoch=datetime.fromisoformat(deadline).timestamp()
    try:
        while sim.d.time<max_seconds:
            if time.monotonic()-started>max_wall or time.time()>deadline_epoch:outcome='execution_wall_budget';break
            row=sim.rows[-1];centers=row['centers_mm']/1000
            retained={j for j in range(9,21) if row['actual_receive_mask'][j]}
            target,info=ctrl.command(sim.scratch,float(sim.d.time),row['contact_force_N'],centers,sim.exited,retained|set(range(9)))
            sim.info=info;sim.commands.append(dict(t_s=float(sim.d.time),ctrl_rad=target.copy(),**info))
            sim.command(target);r=sim.rows[-1]
            if r['ground']:outcome='ground_contact';break
            if np.max(abs(r['q']))>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3.:outcome='actual_self_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            cc=r['centers_mm']/1000;rr=np.linalg.norm(cc[:,[0,2]]-[1.1,1.],axis=1)
            falling=bool(r['contact_force_N'][1,:9].sum()<.1 and (np.median(rr[:9])>.145 or cc[:9,2].mean()<.90))
            failure_age=failure_age+DT if falling else 0.
            if failure_age>=1.:outcome='target_bearing_collapsed_and_falling_1s';break
            if ctrl.failure:outcome=ctrl.failure;break
            phi=np.unwrap(np.arctan2(cc[:,2]-1,cc[:,0]-1.1))
            dy=np.diff(cc[:,1]);da=np.diff(phi);use=np.abs(dy)>.002
            coherent=bool(np.count_nonzero(use)>=16 and np.mean((dy[use]*da[use])<0)>.85 and cc[0,1]-cc[-1,1]>.30)
            complete=bool(np.all((cc[:,1]>.17)&(rr<.125)&(rr>.06)) and coherent and r['contact_force_N'][0].sum()<.1
                and r['contact_force_N'][1,15:].sum()>.2 and all(j in retained for j in range(9,21)))
            complete_age=complete_age+DT if complete else 0.
            if complete_age>=1.:outcome='complete_single_target_helix';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(float(sim.d.time),2),stage=ctrl.stage,phase=round(ctrl.parameter,3),shift_mm=round(ctrl.shift*1000,2),
                    actual_exited=sorted(sim.exited),actual_retained=sorted(retained),target_normal_n=round(float(r['contact_force_N'][1].sum()),2),
                    tracking_deg=round(info['tracking_deg'],2),fit_rms_mm=round(info['mapper']['curve_fit_rms_mm'],2),candidate=info['candidate_accepted'])
                print(json.dumps(msg),flush=True);write(case/'progress.json',msg)
    except Exception:
        outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();write(case/'controller_state.json',dict(stage=ctrl.stage,parameter=ctrl.parameter,shift_mm=ctrl.shift*1000,
            radius_delta_mm=ctrl.radius_delta*1000,last_ctrl=ctrl.last,orientation=ctrl.orientation,events=ctrl.events,
            initial_curve_residual_mm=ctrl.initial_curve_residual*1000,handoff_tangents=ctrl.handoff_tangents))
        r=sim.rows[-1]
        result=dict(name=name,outcome=outcome,actual_seconds=float(sim.d.time),wall_seconds=time.monotonic()-started,
            final_stage=ctrl.stage,final_shape_parameter=ctrl.parameter,reference_shift_mm=ctrl.shift*1000,
            actual_exited_source_modules=sorted(sim.exited),actually_retained_new_modules=np.flatnonzero(r['actual_receive_mask'][9:]).tolist(),
            actual_source_exit_length_mm=len(sim.exited)*LINK*1000,
            actual_retained_new_length_mm=int(np.count_nonzero(r['actual_receive_mask'][9:]))*LINK*1000,
            front_wrap_initial_rad=sim.rows[0]['actual_front_span_rad'],front_wrap_final_rad=r['actual_front_span_rad'],
            maximum_actual_joint_deg=float(np.degrees(max(np.abs(x['q']).max() for x in sim.rows))),
            maximum_torque_nm=float(max(np.abs(x['torque']).max() for x in sim.rows)),
            maximum_pipe_depth_mm=float(max(x['pipe_depth_mm'] for x in sim.rows)),
            maximum_self_depth_mm=float(max(x['self_depth_mm'] for x in sim.rows)),
            ground=any(x['ground'] for x in sim.rows),maximum_command_step_deg=float(max(x['maximum_dispatch_step_deg'] for x in sim.commands)) if sim.commands else 0.,
            physical_root_reset_during_run=False,events=ctrl.events,save_errors=errors,error=error)
        # IDs in the final result use physical module numbers, not slice offsets.
        result['actually_retained_new_modules']=[j+9 for j in result['actually_retained_new_modules']]
        write(case/'result.json',result);print(json.dumps(result,default=default),flush=True)
        try:video(case,sim.rows)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--name',default='H01')
    ap.add_argument('--prepare-name',default='PREPARE_1')
    ap.add_argument('--initial',default=str(BASE/'PREPARE_1'));ap.add_argument('--seconds',type=float,default=600.)
    ap.add_argument('--max-wall',type=float,default=900.);ap.add_argument('--deadline',default='2026-10-09T01:28:46+00:00')
    args=ap.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:geometry_prepare(args.prepare_name)
    else:run(args.name,Path(args.initial),args.seconds,args.max_wall,args.deadline)

if __name__=='__main__':main()
