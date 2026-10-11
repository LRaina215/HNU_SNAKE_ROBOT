"""Bounded actual tests of the 2026-10-11 independently registered baseline."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np
import mujoco
import cv2
from snake_control.inner_ccw_compliance import InnerCCWRoute,InnerCCWCompliance,initial_pose
from snake_control.suspended_supply_transfer import nodes,DT,LINK

BASE=ROOT/'experiments/unified_curve_transfer_20261007/INNER_CCW_COMPLIANCE_20261011'
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'

def default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n',encoding='utf-8')

def shape(points):
    c=(points[:-1]+points[1:])/2;rr=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1)
    phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1));da=np.diff(phi);dy=np.diff(c[:,1])
    coef=np.polyfit(phi,c[:,1],1);pitch=-coef[0]*2*np.pi;resid=np.sqrt(np.mean((c[:,1]-np.polyval(coef,phi))**2))
    good=bool(np.all((rr>.075)&(rr<.105)&(c[:,1]>.15)) and np.ptp(rr)<.006 and np.all((da>.30)&(da<1.10)&(dy>-.045)&(dy<-.002))
              and np.max(abs(dy+.180/(2*np.pi)*da))<.008 and .162<pitch<.198 and resid<.004)
    return dict(strict_single_helix=good,pitch_mm=pitch*1000,radial_spread_mm=np.ptp(rr)*1000,axial_fit_rms_mm=resid*1000,
                target_radii_mm=rr*1000,spatial_step_rad=da,axial_step_mm=dy*1000)

class Sim:
    def __init__(self,model,pose):
        self.m=model;self.d=mujoco.MjData(model);self.d.qpos[:]=pose;self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.va=model.jnt_dofadr[model.actuator_trnid[:,0]];self.d.ctrl[:]=pose[self.qa];mujoco.mj_forward(model,self.d)
        self.head=model.body('head').id;self.source=[model.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']]
        self.target=[model.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']];self.floor=model.geom('floor').id
        self.rows=[];self.info={};self.theta=np.zeros(21);self.prev=None;self.ccw=np.zeros(21)

    def snapshot(self):
        d=self.d;m=self.m;p=nodes(d);c=(p[:-1]+p[1:])/2;f=np.zeros((2,21));source_any=np.zeros(21,bool);ground=False;pd=sd=0.
        for i,con in enumerate(d.contact):
            b1,b2=map(int,m.geom_bodyid[[con.geom1,con.geom2]]);one=self.head<=b1<self.head+21;two=self.head<=b2<self.head+21
            ground|=(one and con.geom2==self.floor) or (two and con.geom1==self.floor)
            if one and two and abs(b1-b2)>1:sd=max(sd,-con.dist)
            if one!=two:
                g=con.geom2 if one else con.geom1;body=b1 if one else b2
                side=0 if g in self.source else 1 if g in self.target else None
                if side is not None:
                    pd=max(pd,-con.dist)
                    if side==0:source_any[body-self.head]=True
                    ff=np.zeros(6);mujoco.mj_contactForce(m,d,i,ff);f[side,body-self.head]+=max(0,ff[0])
        sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);rt=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1)
        near=(rt>.065)&(rt<.125)&(c[:,1]>.15)
        prefix=0
        while prefix<21 and near[prefix]:prefix+=1
        span=float(np.ptp(np.unwrap(np.arctan2(c[:prefix,2]-1,c[:prefix,0]-1.1)))) if prefix>=2 else 0.
        raw=np.arctan2(c[:,2]-1,-c[:,1]) # screen right=-world Y, screen up=Z; look tail->head +X
        delta=np.zeros(21) if self.prev is None else np.arctan2(np.sin(raw-self.prev),np.cos(raw-self.prev))
        carried=source_any & (sr>.06)&(sr<.13)&(c[:,0]<1.03)
        self.ccw+=np.where(carried,delta,0);self.prev=raw.copy()
        self.rows.append(dict(t_s=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),q=d.qpos[self.qa].copy(),ctrl=d.ctrl.copy(),
            torque_nm=d.qfrc_actuator[self.va].copy(),actual_points_mm=p*1000,actual_centers_mm=c*1000,
            diagnostic_contact_force_N=f,diagnostic_source_contact_any=source_any,diagnostic_ground=bool(ground),
            diagnostic_pipe_depth_mm=pd*1000,diagnostic_self_depth_mm=sd*1000,actual_target_prefix=prefix,actual_target_wrap_rad=span,
            actual_source_ccw_rad=self.ccw.copy(),actual_source_view_angle_rad=raw,
            source_support_count=int(np.count_nonzero(source_any)),
            global_points_mm=self.info.get('global_points_mm',p*1000),execution_points_mm=self.info.get('execution_points_mm',p*1000),
            estimated_points_mm=self.info.get('estimated_points_mm',p*1000),shared_shift_mm=self.info.get('shared_shift_mm',0.),
            radius_delta_mm=self.info.get('radius_delta_mm',np.zeros(2)),bridge_coeff_mm=self.info.get('bridge_coeff_mm',np.zeros((2,3))),
            controller_stage=self.info.get('stage','initial'),tracking_deg=self.info.get('tracking_deg',0.),
            controller_estimated_capture=self.info.get('estimated_capture',False),advancement_gate=self.info.get('advancement_gate',False),
            kp_nm_per_rad=m.actuator_gainprm[:,0].copy()))
        return self.rows[-1]

    def command(self,target):
        self.d.ctrl[:]=target
        for k in range(50):
            mujoco.mj_step(self.m,self.d)
            if k%10==9:self.snapshot()

def freeze(case):
    paths=['snake_control/inner_ccw_compliance.py','validation/inner_ccw_compliance_experiment.py',
           'snake_control/suspended_supply_transfer.py','snake_control/suspended_transfer_route.py',
           'snake_control/spatial_bishop_backbone.py','snake_control/horizontal_pipe_roll.py',
           'mujoco_model/Snake_Robot_o20_full_source_inner.xml','mujoco_model/asset/c_link_H.stl','mujoco_model/asset/c_link_V.stl']
    for src in paths:
        dst=case/'sources'/src;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dst)
    write(case/'sources.json',dict(paths=paths,hash_check_performed=False))

def run(args):
    BASE.mkdir(parents=True,exist_ok=True);case=BASE/args.name;case.mkdir(exist_ok=False)
    route=InnerCCWRoute(source_x=args.source_x,source_phi=np.radians(args.source_phi),target_phi=np.radians(args.target_phi),source_hand=args.source_hand)
    m=mujoco.MjModel.from_xml_path(str(MODEL));pose,guide,geom=initial_pose(m,route)
    write(case/'initial_geometry.json',geom);write(case/'route.json',route.description())
    np.savez(case/'initial_pose.npz',qpos=pose,ctrl=pose[m.jnt_qposadr[m.actuator_trnid[:,0]]],points_m=guide)
    vv=np.linspace(-1.7,route.total,6001);np.savez(case/'full_global_route.npz',stations_m=vv,points_m=route.sample(vv))
    freeze(case)
    plan=dict(registered_utc=datetime.now(timezone.utc).isoformat(),candidate=args.name,
              maximum_full_method_runs=3,maximum_actual_seconds=args.seconds,feedback_enabled=not args.no_feedback,
              model=str(MODEL),control_inputs=['joint angles','actuator torques','known pipes','support-registered kinematic estimate'],
              diagnostic_only=['exact world qpos/body positions','exact contact identities/forces','ground','measured circumferential direction'],
              estimator_limitation='Pipe support is assumed in FK registration; world drift and loss of support are not directly observed by controller.',
              physical_root_assignment='artificial placement only before first mj_step',roll_sign=args.roll_sign,
              roll_factor=args.roll_factor,speed_m_s=args.speed,rolling_support=args.rolling_support,anchor_mapper=args.anchor_mapper,
              physics_changes='none',command_limit_deg=88,rate_deg_s=20,
              stops='600s max;30s no actual progress after local adjustment;ground;clear loss;physical joint/drive limits;nonfinite;self interference',
              success='active lower-inner upward acquisition AND source CCW AND continuous all-body single target helix with tail contact and no source contact')
    write(case/'plan.json',plan)
    print(json.dumps(dict(geometry=geom,route=route.description()),default=default),flush=True)
    if not geom['ready']:
        write(case/'result.json',dict(outcome='initial_geometry_rejected',actual_seconds=0,full_method_runs_consumed=0));return
    sim=Sim(m,pose);ctrl=InnerCCWCompliance(m,sim.d,route,not args.no_feedback,args.speed,args.roll_sign,args.roll_factor,args.rolling_support,args.anchor_mapper)
    sim.info=dict(global_points_mm=guide*1000,execution_points_mm=guide*1000,estimated_points_mm=guide*1000);sim.snapshot()
    started=time.monotonic();outcome='maximum_actual_duration';last_progress=0.;best_distance=9.;best_wrap=0.;best_prefix=0;best_clear=0
    capture=False;capture_t=None;complete_age=0.;lost_age=0.;error=None;commands=[];adjusted_t=None
    try:
        while sim.d.time<args.seconds:
            target,info=ctrl.command(sim.d.qpos[sim.qa].copy(),sim.d.qfrc_actuator[sim.va].copy(),float(sim.d.time))
            sim.info=info;commands.append(dict(t_s=float(sim.d.time),ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];t=r['t_s']
            c=r['actual_centers_mm']/1000;f=r['diagnostic_contact_force_N'];prefix=r['actual_target_prefix'];wrap=r['actual_target_wrap_rad']
            distance=float(np.linalg.norm(c[0]-route.e));clear=int(np.count_nonzero(~r['diagnostic_source_contact_any']))
            if distance<best_distance-.002:best_distance=distance;last_progress=t
            if wrap>best_wrap+.10:best_wrap=wrap;last_progress=t
            if prefix>best_prefix:best_prefix=prefix;last_progress=t
            if clear>best_clear and prefix>=3:best_clear=clear;last_progress=t
            if adjusted_t is None and (np.max(abs(r['bridge_coeff_mm']))>.1 or np.max(abs(r['radius_delta_mm']))>.1):adjusted_t=t
            actual_capture=bool(prefix>=5 and wrap>2.3 and f[1,:prefix].sum()>.5)
            if actual_capture and not capture:capture=True;capture_t=t
            finalshape=shape(r['actual_points_mm']/1000)
            complete=finalshape['strict_single_helix'] and f[0].sum()<.1 and not r['diagnostic_source_contact_any'].any() and f[1,18:].sum()>.2 and np.min(r['kp_nm_per_rad'])>126.9
            complete_age=complete_age+DT if complete else 0.
            if complete_age>=1:outcome='complete_single_target_helix';break
            if r['diagnostic_ground']:outcome='ground_contact';break
            if np.max(abs(r['q']))>np.radians(90.5):outcome='actual_joint_limit';break
            if np.max(abs(r['torque_nm']))>2.35501:outcome='actuator_limit_exceeded';break
            if r['diagnostic_self_depth_mm']>3:outcome='actual_self_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            lost=(f.sum()<.1 and c[:,2].mean()<.75);lost_age=lost_age+DT if lost else 0
            if lost_age>=.5:outcome='complete_support_loss';break
            if adjusted_t is not None and t-last_progress>=30:outcome='no_actual_progress_30s_after_local_adjustment';break
            if len(commands)%100==0:
                msg=dict(t_s=round(t,2),wall_s=round(time.monotonic()-started,1),shift_mm=round(ctrl.shift*1000,1),actual_prefix=prefix,actual_wrap=round(wrap,3),
                         source_support=r['source_support_count'],target_N=round(f[1].sum(),2),head_mm=r['actual_centers_mm'][0],tracking_deg=round(info['tracking_deg'],1),
                         source_ccw_deg=np.degrees(r['actual_source_ccw_rad']),estimate_capture=ctrl.acquired,last_actual_progress_s=round(last_progress,2))
                print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:
        outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        np.savez_compressed(case/'trace.npz',**{k:np.asarray([r[k] for r in sim.rows]) for k in sim.rows[0]})
        write(case/'commands.json',commands)
        np.savez(case/'final_state.npz',qpos=sim.d.qpos,qvel=sim.d.qvel,ctrl=sim.d.ctrl,time=sim.d.time)
        r=sim.rows[-1];esterr=np.linalg.norm((r['actual_points_mm']-r['estimated_points_mm']),axis=1)
        result=dict(outcome=outcome,actual_seconds=float(sim.d.time),wall_seconds=time.monotonic()-started,full_method_runs_consumed=1,
                    active_capture_observed=capture,active_capture_first_s=capture_t,actual_maximum_target_prefix=best_prefix,actual_maximum_target_wrap_rad=best_wrap,
                    final_global_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['global_points_mm'])**2,axis=1)))),
                    final_execution_rms_mm=float(np.sqrt(np.mean(np.sum((r['actual_points_mm']-r['execution_points_mm'])**2,axis=1)))),
                    final_estimator_rms_mm=float(np.sqrt(np.mean(esterr**2))),source_ccw_deg=np.degrees(r['actual_source_ccw_rad']),
                    direction_definition='tail looks toward head along +X; screen right=-Y; positive atan2(Z-1,-Y) change means CCW',
                    controller_estimated_capture=ctrl.acquired,controller_events=ctrl.events,final_shape=shape(r['actual_points_mm']/1000),
                    maximum_actual_joint_deg=float(max(np.degrees(abs(x['q']).max()) for x in sim.rows)),
                    maximum_torque_nm=float(max(abs(x['torque_nm']).max() for x in sim.rows)),
                    maximum_pipe_depth_mm=float(max(x['diagnostic_pipe_depth_mm'] for x in sim.rows)),
                    maximum_self_depth_mm=float(max(x['diagnostic_self_depth_mm'] for x in sim.rows)),
                    ground=any(x['diagnostic_ground'] for x in sim.rows),physical_root_reset=False,error=error)
        write(case/'result.json',result);print(json.dumps(result,default=default),flush=True)
    review(case)

def review(case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(case/'trace.npz') as z:rows={k:z[k].copy() for k in z.files}
    t=rows['t_s'];snap=np.unique([0,int(np.argmax(rows['actual_target_wrap_rad'])),len(t)//2,len(t)-1])
    fig,axes=plt.subplots(2,len(snap),figsize=(4*len(snap),7),squeeze=False)
    with np.load(case/'full_global_route.npz') as z:full=z['points_m']*1000
    for col,i in enumerate(snap):
        for row,(a,b,title) in enumerate([(0,1,'Top: X,Y'),(1,2,'Side: Y,Z')]):
            ax=axes[row,col];ax.plot(full[:,a],full[:,b],color='.8',lw=1)
            for key,label,color in [('global_points_mm','global','royalblue'),('execution_points_mm','local execution','darkorange'),('actual_points_mm','actual','black')]:
                p=rows[key][i];ax.plot(p[:,a],p[:,b],'-o',ms=2,lw=1.4,label=label,color=color)
            p=rows['actual_points_mm'][i];ax.annotate('HEAD',p[0,[a,b]]);ax.annotate('TAIL',p[-1,[a,b]])
            ax.set_title(f'{title}  t={t[i]:.2f}s');ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Tail -> head viewer: +X, screen right=-Y / source CCW evaluated from actual orbit\nEntry: L inner (+Y, X<1100), lower to upper target wrap')
    fig.tight_layout();fig.savefig(case/'curve_comparison.png',dpi=145);plt.close(fig)
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m)
    renderer=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.lookat[:]=[.9,.18,.94];cam.distance=1.65
    writer=cv2.VideoWriter(str(case/'actual_continuous_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,480))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    for frame,i in enumerate(range(0,len(t),5)):
        d.qpos[:]=rows['qpos'][i];mujoco.mj_forward(m,d)
        images=[]
        for az,el in [(90,-89),(180,-20)]:
            cam.azimuth=az;cam.elevation=el;renderer.update_scene(d,camera=cam);images.append(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        out=np.hstack(images)
        for line,label in enumerate([f'{case.name} / ACTUAL continuous 1x / t={t[i]:.2f}s',
                                    'Tail -> head viewer: +X / screen right=-Y / positive orbit angle=CCW',
                                    f'Actual target prefix {rows["actual_target_prefix"][i]}/21 / wrap {rows["actual_target_wrap_rad"][i]:.2f} rad']):
            cv2.putText(out,label,(10,22+22*line),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
        for v in range(2):
            p=rows['actual_points_mm'][i]/1000
            # labels use saved body endpoint projections via the live renderer camera.
            # Head/tail are highlighted by native geom colour in this replay only.
            m.geom_rgba[m.body('head').geomadr,:3]=[.9,.15,.15]
            m.geom_rgba[m.body('tail').geomadr,:3]=[.15,.8,.2]
        writer.write(out)
        if frame in (0,len(t)//10,(len(t)-1)//5):cv2.imwrite(str(case/f'actual_frame_{frame:05d}.jpg'),out)
    writer.release();renderer.close()
    cap=cv2.VideoCapture(str(case/'actual_continuous_1x.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for frame in [0,count//2,count-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,_=cap.read();checks.append(dict(frame=frame,readable=bool(ok)))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=count,actual_seconds=float(t[-1]),readback=checks,source='same run saved actual qpos',head='red',tail='green'));cap.release()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--seconds',type=float,default=600)
    ap.add_argument('--speed',type=float,default=.005);ap.add_argument('--source-x',type=float,default=.95)
    ap.add_argument('--source-phi',type=float,default=-45);ap.add_argument('--target-phi',type=float,default=-150)
    ap.add_argument('--source-hand',type=float,default=1);ap.add_argument('--roll-sign',type=float,default=1)
    ap.add_argument('--roll-factor',type=float,default=None);ap.add_argument('--no-feedback',action='store_true');ap.add_argument('--review',action='store_true')
    ap.add_argument('--rolling-support',action='store_true')
    ap.add_argument('--anchor-mapper',action='store_true')
    args=ap.parse_args()
    if args.review:review(BASE/args.name)
    else:run(args)
