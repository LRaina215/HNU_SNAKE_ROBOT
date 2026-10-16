from pathlib import Path
import argparse,json,time,shutil,sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco,cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from paper_gait import *
BASE=Path(__file__).resolve().parent

def make_model(mass=.16,friction=1.,kp=80.,kd=1.,dt=.002):
    root=ET.Element('mujoco',model='CMU16 approximate SEA geometry horizontal T')
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep=str(dt),iterations='80',solver='Newton',cone='elliptic',integrator='implicitfast')
    vis=ET.SubElement(root,'visual');ET.SubElement(vis,'global',offwidth='900',offheight='600');asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='grid',type='2d',builtin='checker',rgb1='.8 .8 .8',rgb2='.6 .65 .7',width='128',height='128')
    ET.SubElement(asset,'material',name='ground',texture='grid',texrepeat='6 6')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'joint',damping='.02',armature='.0001',limited='true',range='-1.57079632679 1.57079632679')
    ET.SubElement(default,'geom',friction=f'{friction} .005 .0001',condim='4',solref='.01 1',solimp='.95 .99 .001')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 -1 3',dir='0 0 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .01',material='ground')
    ET.SubElement(world,'geom',name='source',type='cylinder',fromto='-1.5 0 1 0 0 1',size=str(DP/2),rgba='.35 .4 .48 1')
    ET.SubElement(world,'geom',name='target',type='cylinder',fromto='0 -1.5 1 0 1.5 1',size=str(DP/2),rgba='.25 .5 .75 1')
    body=ET.SubElement(world,'body',name='M0');ET.SubElement(body,'freejoint',name='root')
    for n in range(N+1):
        if n:
            body=ET.SubElement(body,'body',name=f'M{n}',pos=f'{LM} 0 0')
            ET.SubElement(body,'joint',name=f'J{n-1}',axis='0 0 1' if (n-1)%2==0 else '0 1 0')
        ET.SubElement(body,'geom',name=f'robot{n}',type='capsule',fromto=f'{DM/2} 0 0 {LM-DM/2} 0 0',size=str(DM/2),mass=str(mass),rgba='.82 .12 .10 1' if n%2==0 else '.3 .3 .35 1')
        ET.SubElement(body,'site',name=f'C{n}',pos=f'{LM/2} 0 0',size='.002')
    act=ET.SubElement(root,'actuator')
    for n in range(N):
        ET.SubElement(act,'position',name=f'A{n}',joint=f'J{n}',kp=str(kp),kv=str(kd),forcerange='-4 4',ctrllimited='true',ctrlrange='-1.57079632679 1.57079632679')
    return ET.tostring(root,encoding='unicode')

def centers(m,d):return d.site_xpos[[m.site(f'C{n}').id for n in range(N+1)]].copy()

def cylinder(c):
    origin=c.mean(0);_,_,v=np.linalg.svd(c-origin);a=v[0]
    if a@(c[0]-c[-1])<0:a=-a
    u=np.eye(3)[np.argmin(abs(a))];u-=u@a*a;u/=np.linalg.norm(u);v=np.cross(a,u)
    def unpack(x):
        aa=a+x[0]*u+x[1]*v;aa/=np.linalg.norm(aa)
        return aa,origin+x[2]*u+x[3]*v
    def fun(x):
        aa,o=unpack(x);r=c-o
        return np.sqrt(np.maximum(0,(r*r).sum(1)-(r@aa)**2))-x[4]
    res=least_squares(fun,[0,0,0,0,.070],max_nfev=100)
    aa,o=unpack(res.x);uu=u-u@aa*aa;uu/=np.linalg.norm(uu);vv=np.cross(aa,uu);rr=c-o
    phi=np.unwrap(np.arctan2(rr@vv,rr@uu));p=np.polyfit(phi,rr@aa,1)
    return aa,o,float(abs(res.x[4])),float(abs(p[0])*2*np.pi),float(np.sqrt(np.mean(res.fun**2)))

def initial(m,A,ws,phase,nb,offset,polar):
    d=mujoco.MjData(m);d.qpos[3]=1;d.qpos[7:]=helix(A,ws,phase);mujoco.mj_forward(m,d)
    c=centers(m,d);a,o,r,p,rms=cylinder(c)
    R=Rotation.align_vectors([[1,0,0]],[a])[0].as_matrix()
    cc=(c-o)@R.T
    angle=polar-np.arctan2(cc[nb,2],cc[nb,1])
    R=Rotation.from_rotvec([angle,0,0]).as_matrix()@R
    cc=(c-o)@R.T
    trans=np.array([offset-cc[nb,0],0,1])-R@o
    d.qpos[:3]=trans;d.qpos[3:7]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    d.ctrl[:]=d.qpos[7:];mujoco.mj_forward(m,d)
    return d,dict(radius=r,pitch=p,radial_rms=rms,estimated_diameter=estimate_diameter(centers(m,d)))

def contact(m,d):
    forces=np.zeros((2,N+1));min_dist=0.
    for ci,con in enumerate(d.contact):
        min_dist=min(min_dist,float(con.dist));names=[m.geom(g).name for g in [con.geom1,con.geom2]]
        for i,name in enumerate(names):
            if name in ['source','target'] and names[1-i].startswith('robot'):
                n=int(names[1-i][5:]);force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                forces[['source','target'].index(name),n]+=np.linalg.norm(force[:3])
    return forces,min_dist

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True)
    for key,value in dict(A=1.2,ws=.22,f=.09,phase=0.,offset=-.045,polar=0.,mass=.16,friction=1.,kp=80.,kd=1.,low=.4,slope=5.,rate_scale=1.,settle=2.,duration=100.,spiral_time=0.,pulse_rate=.2).items():ap.add_argument('--'+key,type=float,default=value)
    ap.add_argument('--nb',type=int,default=4);ap.add_argument('--render',action='store_true');ap.add_argument('--only_prepare',action='store_true')
    args=ap.parse_args();case=BASE/args.case;case.mkdir(parents=True,exist_ok=False)
    for src in ['run.py','paper_gait.py']:shutil.copy2(BASE/src,case/src)
    xml=make_model(args.mass,args.friction,args.kp,args.kd);(case/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d,stats=initial(m,args.A,args.ws,args.phase,args.nb,args.offset,args.polar)
    plan=dict(parameters=vars(args),initial_stats=stats,nominal_rate=window_rate(args.f)*args.rate_scale,expected='source rolling; head phase +nu*pi/2 and loose amplitude; nominal window travels head to tail; target +Y or -Y depends on selected polar orientation',physical_root_assigned_after_initial=False)
    (case/'plan.json').write_text(json.dumps(plan,indent=2));print('INITIAL',plan,flush=True)
    np.savez(case/'initial.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl)
    ren=mujoco.Renderer(m,height=600,width=900);cam=mujoco.MjvCamera();cam.lookat[:]=[-.12,0,1];cam.distance=1.6;cam.azimuth=125;cam.elevation=-32
    ren.update_scene(d,cam);cv2.imwrite(str(case/'initial.png'),cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR))
    if args.only_prepare:ren.close();return
    writer=cv2.VideoWriter(str(case/'actual_continuous.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(900,600))
    rows=[];commands=[];next_sample=0.;next_frame=0.;wall=time.time();stop='duration';phase=args.phase;last=d.ctrl.copy()
    delta=-np.sign(args.A*args.ws)*np.pi/2;freq=-2*np.pi*args.f
    while d.time<args.duration:
        t=float(d.time);tau=max(0,t-args.settle-args.spiral_time)
        if t<args.settle:q=helix(args.A,args.ws,args.phase);nt=0.;L=1.;stage='settle'
        elif t<args.settle+args.spiral_time:
            st=t-args.settle;phase=args.phase+freq*st;ns=(N-1-args.pulse_rate*st)%N
            q=spiral(args.A,args.ws,phase,ns);nt=0.;L=1.;stage='spiraling'
        else:
            phase=args.phase+freq*(t-args.settle)-delta
            nt=window_rate(args.f)*args.rate_scale*tau
            q,L=window(args.A,args.ws,phase,nt,args.nb,args.low,args.slope);stage='window' if nt<N else 'continue_target_rolling'
        # Finite servo command slew, no Cartesian fit and no pose/state reset.
        q=np.clip(q,last-3*m.opt.timestep,last+3*m.opt.timestep);q=np.clip(q,-np.pi/2,np.pi/2);d.ctrl[:]=q;last=q.copy()
        mujoco.mj_step(m,d)
        if t>=next_sample:
            mujoco.mj_forward(m,d);forces,mind=contact(m,d);c=centers(m,d)
            row=dict(t=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),centers=c,forces=forces,phase=phase,nt=nt,L=L,ctrl=q.copy(),torque=d.actuator_force.copy(),tracking=np.max(abs(q-d.qpos[7:])),stage=stage,min_contact_dist=mind)
            rows.append(row);next_sample+=.1
            if int(t*10)%100==0:print(f't={t:.1f} nt={nt:.2f} contacts={np.sum(forces>0,axis=1)} head={c[0].round(3)} zmin={c[:,2].min():.3f} tracking={row["tracking"]:.3f}',flush=True)
        if t>=next_frame:
            ren.update_scene(d,cam);frame=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
            cv2.putText(frame,f'ACTUAL dynamics t={t:.1f}s  nt={nt:.2f}  L={L:.2f}  {stage}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
            writer.write(frame);next_frame+=.1
        if not np.all(np.isfinite(d.qpos)):stop='nonfinite';break
        if centers(m,d)[:,2].min()<.3:stop='fall_below_pipe';break
    writer.release();ren.close()
    np.savez_compressed(case/'trace.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    c=rows[-1]['centers'];rr=np.linalg.norm(c[:,[0,2]]-[0,1],axis=1)
    summary=dict(stop=stop,sim_time=float(d.time),wall_seconds=time.time()-wall,all_target_band=bool(np.all((rr>.052)&(rr<.11))),target_radius=rr.tolist(),source_contact_modules=np.flatnonzero(rows[-1]['forces'][0]>.01).tolist(),target_contact_modules=np.flatnonzero(rows[-1]['forces'][1]>.01).tolist(),max_tracking_deg=float(np.rad2deg(max(r['tracking'] for r in rows))),max_torque=float(max(np.max(abs(r['torque'])) for r in rows)),success=False)
    (case/'result.json').write_text(json.dumps(summary,indent=2));print('RESULT',summary,flush=True)

if __name__=='__main__':main()
