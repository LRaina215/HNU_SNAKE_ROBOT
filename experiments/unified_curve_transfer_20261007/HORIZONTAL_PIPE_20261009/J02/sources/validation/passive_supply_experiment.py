"""Real continuation: active front winding with passive hanging posterior."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np
import mujoco,cv2
from validation.suspended_supply_experiment import ActualSim,write,default,BASE,MODEL
from snake_control.passive_supply_roll import PassiveSupplyRoll
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.suspended_supply_transfer import LINK,DT

class PassiveSim(ActualSim):
    def snapshot(self):
        r=super().snapshot()
        r.update(continuation_t_s=float(self.d.time-self.origin_time),kp_nm_per_rad=self.m.actuator_gainprm[:,0].copy(),
            damping_nm_s_per_rad=self.m.dof_damping[self.va].copy(),passive_torque_nm=self.scratch.qfrc_passive[self.va].copy(),
            force_range_nm=self.m.actuator_forcerange.copy(),active_joint_mask=np.asarray(self.info.get('active_joint_mask',np.ones(20,bool))),
            region_codes=np.asarray(self.info.get('region_codes',np.zeros(20,int))),
            actual_new_winding_rad=np.asarray(self.info.get('actual_new_winding_rad',np.zeros(21))),
            actual_new_retained_mask=np.array([j in self.info.get('actual_new_retained_modules',[]) for j in range(21)]))
        r['actual_receive_mask']=r['actual_new_retained_mask'].copy();r['actual_receive_mask'][:9]=True
        return r

def render(case,rows):
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    ren=mujoco.Renderer(m,height=600,width=880);cam=mujoco.MjvCamera()
    cam.lookat[:]=[1.,.25,.78];cam.distance=1.85;cam.azimuth=145;cam.elevation=-22
    w=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1240,600))
    if not w.isOpened():raise RuntimeError('Video writer unavailable')
    ids=list(range(0,len(rows),5))
    for fi,i in enumerate(ids):
        r=rows[i];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d);ren.update_scene(d,camera=cam)
        im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);panel=np.full((600,360,3),25,np.uint8)
        cv2.putText(im,f'{case.name} continuation {r["continuation_t_s"]:.2f}s',(12,25),0,.65,(255,255,255),1)
        cv2.putText(im,'ACTUAL saved dynamics / continuous 1x',(12,50),0,.5,(255,255,255),1)
        labels=[f'Shared shift {r["shift_mm"]:.1f} mm',f'Front span {r["actual_front_span_rad"]:.2f} rad',
            f'New retained {int(r["actual_new_retained_mask"][9:].sum())}/12',f'Target bearing {r["contact_force_N"][1].sum():.1f} N',
            f'Pipe/self {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f} mm',f'Tracking {r["tracking_deg"]:.1f} deg',
            'Joint   Kp Nm/rad   drive Nm','F=front E=entry P=passive; damping 1.5']
        for j,label in enumerate(labels):cv2.putText(panel,label,(10,23+23*j),0,.44,(225,225,225),1)
        for j in range(20):
            x=10+(j//10)*176;y=230+(j%10)*32
            role='FEP'[int(r['region_codes'][j])]
            cv2.putText(panel,f'{j+1:02d}{role} {r["kp_nm_per_rad"][j]:5.1f} {r["torque"][j]:+4.2f}',(x,y),0,.43,(100,230,120) if role=='P' else (230,220,180),1)
        whole=np.hstack([im,panel]);w.write(whole)
        if fi in {0,len(ids)//2,len(ids)-1}:cv2.imwrite(str(case/f'frame_{fi:05d}.jpg'),whole)
    w.release();ren.close();cap=cv2.VideoCapture(str(case/'replay_1x.mp4'))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for i in [0,n//2,n-1]:cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,i);ok,_=cap.read();checks.append(dict(frame=i,readable=bool(ok)))
    meta=dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['continuation_t_s'],speed=1.,source='actual saved qpos',readback=checks)
    cap.release();write(case/'video.json',meta)

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False)
    saved=BASE/'H03/final_state.npz'
    with np.load(BASE/'H03/trace.npz') as z:initial=dict(qpos=z['qpos'][-1],qvel=z['qvel'][-1],ctrl=z['ctrl'][-1])
    sim=PassiveSim(case,initial);restore_saved_state(sim.m,sim.d,saved);sim.origin_time=float(sim.d.time)
    for src in ['snake_control/passive_supply_roll.py','snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py',
        'snake_control/spatial_bishop_backbone.py','snake_control/hook_pull_review_support.py','validation/passive_supply_experiment.py',
        'validation/suspended_supply_experiment.py','mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml']:
        dest=case/'sources'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    shutil.copy2(saved,case/'initial_full_state.npz');sim.snapshot();ctrl=PassiveSupplyRoll(sim.m,sim.scratch,entry_gain=args.entry_gain,speed=args.speed,preserve_front=args.preserve_front)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),initial='H03 exact final INTEGRATION state, ctrl and warmstart restored before first step',
        action='front active coil, hanging rear zero position stiffness with original damping, proximal entry gradually active; no rear straightening objective',
        full_nominal_shift_mm=12*LINK*1000,nominal_seconds=12*LINK/args.speed,max_seconds=args.seconds,max_wall=args.max_wall,
        execution_deadline_utc=args.deadline,entry_gain=args.entry_gain,speed_mm_s=args.speed*1000,preserve_measured_front_screw_shape=args.preserve_front,
        limits=dict(target_deg=88,rate_deg_s=10,torque_nm=2.355),small_predicted_contact_freeze=False,
        actual_stops='ground, joint >90.5deg, self penetration >3mm, pipe >3mm for0.2s, target bearing collapsed and falling1s; no actual intake progress',
        physics='original gravity/friction/damping/force caps; only Kp changed by declared region ramp',root_assignments_during_run=0))
    started=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';error=None
    deep_age=fall_age=complete_age=0.
    try:
        while sim.d.time-sim.origin_time<args.seconds:
            if time.monotonic()-started>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            t=float(sim.d.time-sim.origin_time);r=sim.rows[-1]
            target,info=ctrl.command(sim.scratch,t,r['contact_force_N']);sim.info=info
            sim.commands.append(dict(t_s=t,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1]
            if r['ground']:outcome='ground_contact';break
            if np.max(abs(r['q']))>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3.:outcome='actual_self_interference';break
            deep_age=deep_age+DT if r['pipe_depth_mm']>3 else 0.
            if deep_age>=.2:outcome='actual_pipe_interference_0p2s';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
            falling=bool(r['contact_force_N'][1,:9].sum()<.1 and (np.median(rr[:9])>.145 or c[:9,2].mean()<.90))
            fall_age=fall_age+DT if falling else 0.
            if fall_age>=1:outcome='target_bearing_collapsed_and_falling';break
            if ctrl.failure:outcome=ctrl.failure;break
            phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));dy=np.diff(c[:,1]);da=np.diff(phi);use=abs(dy)>.002
            coherent=np.count_nonzero(use)>=16 and np.mean(dy[use]*da[use]<0)>.85 and c[0,1]-c[-1,1]>.30
            complete=bool(np.all((c[:,1]>.17)&(rr<.125)&(rr>.06)) and coherent and len(ctrl.received)==12 and r['contact_force_N'][1,15:].sum()>.2)
            complete_age=complete_age+DT if complete else 0.
            if complete_age>=1:outcome='complete_single_target_helix';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(t,2),shift_mm=round(ctrl.shift*1000,2),new_retained=sorted(ctrl.received),
                    target_normal_n=round(float(r['contact_force_N'][1].sum()),2),track_deg=round(info['tracking_deg'],2),
                    kp=ctrl.kp.round(1).tolist(),max_actual_winding_rad=round(float(info['actual_new_winding_rad'][9:].max()),3))
                print(json.dumps(msg),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1]
        np.savez(case/'final_model_parameters.npz',actuator_gainprm=sim.m.actuator_gainprm,actuator_biasprm=sim.m.actuator_biasprm,dof_damping=sim.m.dof_damping,forcerange=sim.m.actuator_forcerange)
        write(case/'controller_state.json',dict(shift_mm=ctrl.shift*1000,last_ctrl=ctrl.last,kp=ctrl.kp,received=sorted(ctrl.received),phi0=ctrl.phi0,phi=ctrl.phi))
        result=dict(name=args.name,outcome=outcome,actual_continuation_seconds=float(sim.d.time-sim.origin_time),original_sim_time=float(sim.d.time),
            wall_seconds=time.monotonic()-started,shift_mm=ctrl.shift*1000,new_retained_modules=sorted(ctrl.received),
            maximum_new_winding_rad=float(ctrl.phi0[9:].max()) if False else float((ctrl.phi0-ctrl.phi)[9:].max()),
            final_new_module_local_bearing_n=r['contact_force_N'][1,9:],front_span_initial_rad=sim.rows[0]['actual_front_span_rad'],front_span_final_rad=r['actual_front_span_rad'],
            maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),
            maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),
            maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),ground=any(x['ground'] for x in sim.rows),
            final_kp_nm_per_rad=ctrl.kp,save_errors=errors,error=error,physical_root_reset_during_run=False)
        write(case/'result.json',result);print(json.dumps(result,default=default),flush=True)
        try:render(case,sim.rows)
        except Exception as exc:write(case/'video_error.json',dict(error=repr(exc)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='P01');ap.add_argument('--entry-gain',type=float,default=.18)
    ap.add_argument('--speed',type=float,default=.0025);ap.add_argument('--seconds',type=float,default=600);ap.add_argument('--max-wall',type=float,default=480)
    ap.add_argument('--preserve-front',action='store_true');ap.add_argument('--deadline',default='2026-10-09T01:32:32+00:00');run(ap.parse_args())
