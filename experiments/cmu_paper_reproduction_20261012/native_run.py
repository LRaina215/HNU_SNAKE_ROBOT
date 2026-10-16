"""Continuous CMU baseline using the repository's exact original L XML."""
from pathlib import Path
import argparse,json,sys,time,shutil,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np,cv2
from scipy.spatial.transform import Rotation
from snake_control.cmu_window_gait import cylinder
from snake_control.suspended_supply_transfer import nodes
from native_gait import *
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'

def snapshot(m,d):
    mujoco.mj_forward(m,d);p=nodes(d);c=(p[:-1]+p[1:])/2;h=m.body('head').id
    forces=np.zeros((2,21));depth=np.zeros(4);pairs=[]
    for i,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];bs=[int(m.geom_bodyid[g]) for g in gs]
        robot=[h<=b<h+21 for b in bs];names=[m.geom(g).name or '' for g in gs]
        kind=2 if all(robot) else 3 if 'floor' in names and any(robot) else None
        for j,name in enumerate(names):
            if name.startswith('pipe_') and robot[1-j]:
                kind=0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);forces[kind,bs[1-j]-h]+=max(0,f[0])
        if kind is not None:
            depth[kind]=max(depth[kind],-float(ct.dist))
            if ct.dist<0:pairs.append(dict(kind=kind,geoms=names,depth=-float(ct.dist)))
    return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),centers=c,points=p,forces=forces,depth=depth,torque=d.actuator_force.copy()),pairs

def placement(m,A,ws,phase,nb,seam,polar):
    d=mujoco.MjData(m);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=wave(A,ws,phase);mujoco.mj_forward(m,d)
    p=nodes(d);s=cylinder(p);R=Rotation.align_vectors([[1,0,0]],[s['axis']])[0].as_matrix()
    pp=(p-s['origin'])@R.T;cc=(pp[:-1]+pp[1:])/2
    angle=polar-np.arctan2(cc[nb,2],cc[nb,1]);R=Rotation.from_rotvec([angle,0,0]).as_matrix()@R
    pp=(p-s['origin'])@R.T
    trans=np.array([seam-pp[nb,0],0,1])-R@s['origin'];d.qpos[:3]=trans;d.qpos[3:7]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];d.ctrl[:]=d.qpos[qa]
    mujoco.mj_forward(m,d);return d,s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True)
    for k,v in dict(A=1.2351672327411283,ws=.19063617553842116,phase=-.9,f=.02,seam=.84,polar=0.,low=.4,slope=5.,rate_scale=1.,settle=.5,duration=420.,spiral_time=0.,pulse_rate=.1,slew=.34906585).items():ap.add_argument('--'+k,type=float,default=v)
    ap.add_argument('--nb',type=int,default=6);ap.add_argument('--only_prepare',action='store_true');ap.add_argument('--roll_only',action='store_true')
    a=ap.parse_args();case=BASE/a.case;case.mkdir(parents=True,exist_ok=False)
    for src in ['native_run.py','native_gait.py']:shutil.copy2(BASE/src,case/src)
    shutil.copy2(MODEL,case/'model.xml');tree=ET.parse(MODEL)
    for mesh in tree.findall('./asset/mesh'):
        if mesh.get('file'):
            src=MODEL.parent/mesh.get('file');dest=case/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d,s=placement(m,a.A,a.ws,a.phase,a.nb,a.seam,a.polar);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    init,pairs=snapshot(m,d);h=m.body('head').id;axis=d.xmat[h+a.nb].reshape(3,3)[:,2]
    # Native link length axis is -local Z for odd/even alternating frames;
    # selected direction is therefore taken from actual adjacent nodes.
    tangent=(init['points'][a.nb]-init['points'][a.nb+1])/LM
    plan=dict(parameters=vars(a),original_model=str(MODEL),model_unchanged=True,robot_mass=float(m.body_mass[h:].sum()),force_range=m.actuator_forcerange.tolist(),gain=m.actuator_gainprm[:,:3].tolist(),initial_cylinder={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()},selected_tail_to_head=tangent.tolist(),initial_depth_m=init['depth'].tolist(),initial_contacts=pairs,physical_root_assigned_after_initial=False,formula10='nt_dot=pi*DM*f/LM from prose, correction to inconsistent print',expected='rolling source; bend head loosens and wraps original +Y target; nominal tailward window; full-body target rolling',video_speed=2)
    (case/'plan.json').write_text(json.dumps(plan,indent=2));np.savez(case/'initial.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl);print('INITIAL',json.dumps({k:plan[k] for k in ['parameters','initial_depth_m','selected_tail_to_head','initial_cylinder']}),flush=True)
    ren=mujoco.Renderer(m,width=900,height=600);cam=mujoco.MjvCamera();cam.lookat[:]=[.95,.22,.9];cam.distance=1.8;cam.azimuth=140;cam.elevation=-26
    ren.update_scene(d,cam);cv2.imwrite(str(case/'initial.png'),cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR))
    if a.only_prepare:ren.close();return
    if max(init['depth'][:3])>.002:
        (case/'result.json').write_text(json.dumps(dict(stop='initial_penetration',physical_steps=0,depth=init['depth'].tolist()),indent=2));ren.close();print('initial penetration, no dynamics',flush=True);return
    wr=cv2.VideoWriter(str(case/'actual_continuous.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(900,600))
    rows=[];last=d.ctrl.copy();next_sample=0.;next_frame=0.;deep_age=0.;stop='duration';wall=time.time();info={};final_phase=a.phase
    while d.time<a.duration:
        t=float(d.time);tau=max(0,t-a.settle-a.spiral_time)
        if t<a.settle:q=wave(a.A,a.ws,a.phase);info=dict(nt=0.,L=1.,phase=a.phase,delta=0.,stage='settle_source')
        elif t<a.settle+a.spiral_time:
            st=t-a.settle;final_phase=a.phase-2*np.pi*a.f*st;ns=(N-1-a.pulse_rate*st)%N
            q=spiral(a.A,a.ws,final_phase,ns);info=dict(nt=0.,L=1.,phase=final_phase,delta=0.,stage='spiraling')
        elif a.roll_only:
            q=wave(a.A,a.ws,a.phase-2*np.pi*a.f*(t-a.settle));info=dict(nt=0.,L=1.,phase=a.phase-2*np.pi*a.f*(t-a.settle),delta=0.,stage='source_roll_only')
        else:
            q,info=window(a.A,a.ws,final_phase,tau,a.f,a.nb,a.low,a.slope,a.rate_scale);info['stage']='window' if info['nt']<N else 'continue_target_rolling'
        q=np.clip(q,last-a.slew*m.opt.timestep,last+a.slew*m.opt.timestep);q=np.clip(q,-np.radians(88),np.radians(88));d.ctrl[:]=q;last=q.copy();mujoco.mj_step(m,d)
        if t>=next_sample:
            row,_=snapshot(m,d);row.update(t=t,tracking=float(np.max(abs(q-d.qpos[qa]))),**info);rows.append(row);next_sample+=.1
            deep_age=deep_age+.1 if max(row['depth'][:2])>.003 else 0.
            if int(round(t*10))%100==0:
                print('t',round(t,1),'nt',round(info['nt'],2),'forces',row['forces'].sum(1).round(1),'head',row['points'][0].round(3),'depthmm',(row['depth']*1000).round(2),'trackdeg',round(np.rad2deg(row['tracking']),2),flush=True)
            if deep_age>=.3:stop='sustained_pipe_penetration';break
            if row['depth'][2]>.003:stop='self_penetration';break
            if row['depth'][3]>0:stop='floor_contact';break
            if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
            if (case/'stop_and_save.request').exists():stop='requested_stop';break
        if t>=next_frame:
            zmin=nodes(d)[:,2].min();cam.lookat[2]=.9-.35*np.clip((.8-zmin)/.8,0,1);cam.distance=1.8+.5*np.clip((.8-zmin)/.8,0,1)
            ren.update_scene(d,cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,f'ACTUAL native model {t:.1f}s nt={info["nt"]:.2f} L={info["L"]:.2f}',(15,28),0,.6,(255,255,255),1);wr.write(im);next_frame+=.1
    wr.release();ren.close();np.savez_compressed(case/'trace.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]});np.savez(case/'final_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
    c=rows[-1]['centers'];rr=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);s=cylinder(rows[-1]['points'])
    summary=dict(stop=stop,sim_time=d.time,wall_seconds=time.time()-wall,success=False,final_window_nt=info['nt'],target_radius_m=rr.tolist(),final_cylinder={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()},final_force=rows[-1]['forces'].sum(1).tolist(),maximum_depth_mm=(np.max([r['depth'] for r in rows],axis=0)*1000).tolist(),max_tracking_deg=float(np.rad2deg(max(r['tracking'] for r in rows))),max_torque=float(max(np.max(abs(r['torque'])) for r in rows)))
    (case/'result.json').write_text(json.dumps(summary,indent=2));print('RESULT',summary,flush=True)

if __name__=='__main__':main()
