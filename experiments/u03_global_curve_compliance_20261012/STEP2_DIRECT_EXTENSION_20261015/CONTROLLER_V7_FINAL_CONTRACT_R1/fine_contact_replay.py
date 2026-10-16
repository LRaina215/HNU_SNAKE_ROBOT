"""Kinematic contact replay at the registered command clock, without integration.

Original-scene forward contacts are exposed only through read_sensor.  This
diagnostic follows commanded C ideally; it cannot establish physical tracking,
load sharing, estimator reliability, or capture.  Never consumed by runtime.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import argparse,json
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner
from contact_sensor import read_sensor,known_source

def run(args):
    for name in ['mj_step','mj_step1','mj_step2']:
        setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'))
    if args.released_hinges is not None:
        reg['direct_extension']['released_hinges']=args.released_hinges;reg['kinematic_solver']['release_hinges']=args.released_hinges
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial)
    mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),d.qpos.copy(),d.ctrl.copy(),reg['kinematic_solver'])
    dt=.025;duration=reg['direct_extension']['release_duration_s'];wrap_duration=reg['direct_extension']['wrap_duration_s'];rows=[];poses=[];guides=[];failure='';first=None;wrap_t=None;alpha=0.;beta=0.
    resume_time=0.
    if args.resume:
        prefix=np.load(args.resume);index=int(np.flatnonzero(prefix['beta']>0.)[0]);resume_time=float(prefix['t'][index]);alpha=float(prefix['alpha'][index]);wrap_t=resume_time;first=resume_time
        pose_index=max(0,index-1) if args.resume_before_wrap else index
        d.qpos[:]=prefix['qpos'][pose_index];d.ctrl[:]=d.qpos[plan.qa];d.qvel[:]=0.;mujoco.mj_forward(m,d);plan.q=d.qpos[plan.qa].copy();plan.start_wrap(alpha)
    count=int(round((duration+(wrap_duration if args.early_wrap else 0.))/dt))+1
    for step in range(count):
        t=resume_time+step*dt
        if wrap_t is None:alpha=min(1.,t/duration)
        else:beta=min(1.,(t-wrap_t)/wrap_duration)
        sensor=read_sensor(m,d)
        touched=any(c['material']<8 and not known_source(c,reg['source_known']) for c in sensor['contacts'])
        if touched and first is None:first=t
        if touched and args.early_wrap and wrap_t is None:wrap_t=t;plan.start_wrap(alpha);beta=1e-12
        n=reg['direct_extension']['wrap_hinges'] if wrap_t is not None else reg['direct_extension']['released_hinges']
        plan.observe(d.qpos.copy(),n);plan.observe_contacts(sensor['contacts'])
        previous=plan.q.copy();oldC=plan.fk(previous)
        if step==0:q=previous;G=plan.guide(alpha,max(beta,1e-12))[0] if wrap_t is not None else plan.given.copy();metrics=dict(accepted=True,head_error_m=0.)
        else:q,G,metrics=plan.solve(alpha,beta,position_cap=reg['kinematic_solver']['position_step_m'],joint_step_rad=np.radians(20)*dt)
        # Same native command slew as runtime, applied to this ideal geometric C.
        change=q-previous;factor=min(1.,np.radians(20)*dt/max(abs(change).max(),1e-12));plan.q=previous+factor*change;q=plan.q.copy()
        C=plan.fk(q);d.qpos[:]=plan.d.qpos;d.ctrl[:]=q;d.qvel[:]=0.;mujoco.mj_forward(m,d)
        depths=np.zeros(4);h=m.body('head').id
        for ct in d.contact:
            gs=[ct.geom1,ct.geom2];robot=[h<=m.geom_bodyid[g]<h+21 for g in gs]
            if all(robot):depths[2]=max(depths[2],-ct.dist)
            elif sum(robot)==1:
                other=m.geom(gs[0] if not robot[0] else gs[1]).name
                if other=='floor':depths[3]=max(depths[3],-ct.dist)
                elif other and other.startswith('pipe_'):
                    j=0 if other in ['pipe_straight_01','pipe_sharp_in_02'] else 1;depths[j]=max(depths[j],-ct.dist)
        metrics.update(t=t,alpha=alpha,beta=beta,original_depths_mm=(depths*1000).tolist(),
            C_step_mm=float(np.linalg.norm(C-oldC,axis=1).max()*1000),joint_step_deg=float(abs(q-previous).max()*180/np.pi),
            first_unlabeled_front_contact_t=first,contact_plane_count=len(plan.contact_planes),
            G_step_mm=0. if not guides else float(np.linalg.norm(G-guides[-1],axis=1).max()*1000),
            sensed_front_materials=sorted(set(c['material'] for c in sensor['contacts'] if c['material']<8 and not known_source(c,reg['source_known']))),
            source_node_motion_mm=float(np.linalg.norm(C[n:]-plan.given[n:],axis=1).max()*1000))
        rows.append(metrics);poses.append(d.qpos.copy());guides.append(G)
        if step%10==0:print(json.dumps(metrics),flush=True)
        if max(depths[:3])>.003 or depths[3]>0:failure='original_collision_limit';break
        if (out/'stop_and_save.request').exists():failure='requested_kinematic_stop';break
        if wrap_t is not None and beta>=1.:break
        if wrap_t is None and alpha>=1. and args.early_wrap:failure='no_front_contact';break
    np.savez_compressed(out/'KINEMATIC_CONTACT_REPLAY.npz',t=[x['t'] for x in rows],alpha=[x['alpha'] for x in rows],beta=[x['beta'] for x in rows],qpos=poses,G=guides)
    report=dict(simulation_steps=0,ideal_C_following=True,initial_file=str(Path(args.initial).resolve()),
        runtime_consumes_this_path=False,condition='DIRECTION',command_dt_s=dt,failure=failure,
        completed_release=rows[-1]['alpha']>=1.,first_unlabeled_front_contact_t=first,
        early_contact_wrap_enabled=args.early_wrap,completed_kinematic_wrap=beta>=1.,estimator_gates_used=False,
        reused_kinematic_prefix=args.resume,reused_pre_wrap_pose=args.resume_before_wrap,kinematic_continuation_start_s=resume_time,
        maximum_original_depths_mm=np.max([x['original_depths_mm'] for x in rows],axis=0).tolist(),
        maximum_C_step_mm=max(x['C_step_mm'] for x in rows),rows=rows,
        scope='Kinematic measured-contact constraint diagnostic only. No physical load, tracking, fit, capture or dynamic success.')
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--out',required=True);ap.add_argument('--released-hinges',type=int);ap.add_argument('--early-wrap',action='store_true');ap.add_argument('--resume');ap.add_argument('--resume-before-wrap',action='store_true');run(ap.parse_args())
