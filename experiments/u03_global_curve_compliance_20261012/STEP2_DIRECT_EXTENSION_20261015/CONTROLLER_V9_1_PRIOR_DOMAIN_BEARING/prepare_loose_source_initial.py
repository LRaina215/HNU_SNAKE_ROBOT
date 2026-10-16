"""Prescribed all-source relaxed front coil; no released bridge or dynamics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from physics import *
from geometry import Geometry,normalize
from rough_planner import RoughPlanner

def run(args):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));source=BASE.parent/'INITIAL_V5_POSITIVE_PHASE_45/CANDIDATE_INITIAL.npz';m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(source);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),d.qpos,d.ctrl,reg['kinematic_solver']);n=args.released_hinges;plan.observe(d.qpos.copy(),n);initial=plan.initial_q.copy();given=plan.given.copy();oldrho=np.linalg.norm(given[:,1:]-[0,1],axis=1);rho=oldrho.copy();G=given.copy();q=initial.copy();stages=[]
    for extension in np.linspace(.004,.026,7):
        taper=np.clip((n-np.arange(22))/args.taper_length,0,1);taper=taper*taper*(3-2*taper);rho=oldrho+extension*taper;phase=np.zeros(22);phase[n]=np.arctan2(given[n,2]-1,given[n,1])
        for i in range(n-1,-1,-1):
            dx=given[i,0]-given[i+1,0];co=(rho[i]**2+rho[i+1]**2+dx*dx-.0645**2)/(2*rho[i]*rho[i+1]);phase[i]=phase[i+1]+np.arccos(np.clip(co,-1,1));G[i]=[given[i,0],rho[i]*np.cos(phase[i]),1+rho[i]*np.sin(phase[i])]
        def residual(v):
            command=initial.copy();command[:n]=v;p=plan.fk(command);centers=.5*(p[:-1]+p[1:]);cr=np.linalg.norm(centers[:n,1:]-[0,1],axis=1);r=[((p[:n]-G[:n])/.003).ravel(),np.maximum(cr-.122,0)/.001,np.minimum(cr-.070,0)/.001,.002*(v-initial[:n])]
            for i in range(n):
                for other in plan.source:r.append(np.array([min(0.,plan.distance(plan.robot[i],other)[0]+.0008)/.0005]))
                for j in range(i+2,21):r.append(np.array([min(0.,plan.distance(plan.robot[i],plan.robot[j])[0]-.002)/.0005]))
            return np.concatenate(r)
        fit=least_squares(residual,q[:n],bounds=(-np.radians(88),np.radians(88)),diff_step=1e-5,max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-7);q[:n]=fit.x;p=plan.fk(q);stages.append(dict(extension_m=float(extension),max_point_error_mm=float(np.linalg.norm(p[:n]-G[:n],axis=1).max()*1000),solver_success=bool(fit.success)))
    plan.fk(q);pose=plan.d.qpos.copy();points=nodes(plan.d);headphase=np.arctan2(points[0,2]-1,points[0,1]);turn=Rotation.from_rotvec((np.radians(-45)-headphase)*np.array([1.,0.,0.]));origin=np.array([0.,0.,1.]);pose[:3]=origin+turn.apply(pose[:3]-origin);pose[3:7]=(turn*Rotation.from_quat(pose[3:7][[1,2,3,0]])).as_quat()[[3,0,1,2]]
    # Place the whole prescribed source setup once, before physical t0.
    pvt=mujoco.MjModel.from_xml_path(str(MODEL));pvt.opt.enableflags|=64;dp=mujoco.MjData(pvt);dp.qpos[:]=pose;mujoco.mj_kinematics(pvt,dp);points=nodes(dp);centers=.5*(points[:-1]+points[1:]);headX=points[0,0];upper=headX+1.095-float((centers[:,0]+centers[:,1]).max());targets=[pvt.geom(x).id for x in ['pipe_sharp_out_03','pipe_straight_04']];physical_robot_geoms=[int(pvt.body_geomadr[pvt.body('head').id+i]) for i in range(21)]
    def assign(x):dp.qpos[:]=pose;dp.qpos[0]+=x-headX;mujoco.mj_kinematics(pvt,dp)
    lo=.85;hi=upper
    for _ in range(24):
        mid=.5*(lo+hi);assign(mid);gap=min(min(.03,float(mujoco.mj_geomDistance(pvt,dp,a,b,.03,None))) for a in physical_robot_geoms for b in targets)
        if gap>=.002:lo=mid
        else:hi=mid
    assign(lo);d.qpos[:]=dp.qpos;d.qvel[:]=0.;d.ctrl[:]=d.qpos[plan.qa]+(z['ctrl']-z['qpos'][plan.qa]);d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d);p=nodes(d).copy();c=.5*(p[:-1]+p[1:]);depth=np.zeros(6);normal=np.zeros(4)
    for k,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];robot=[plan.head<=m.geom_bodyid[g]<plan.head+21 for g in gs]
        if all(robot):depth[4]=max(depth[4],-ct.dist);continue
        if sum(robot)!=1:continue
        name=m.geom(gs[0] if not robot[0] else gs[1]).name;names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
        if name in names:
            j=names.index(name);j=5 if j==4 else j;depth[j]=max(depth[j],-ct.dist)
            if j<4:f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);normal[j]+=max(0.,f[0])
    out=Path(args.out).resolve();out.mkdir(exist_ok=False);state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=p)
    np.savez_compressed(out/'INITIAL_TRACE.npz',alpha=np.array([0.]),qpos=d.qpos[None,:])
    r=np.linalg.norm(c[:,1:]-[0,1],axis=1);report=dict(role='Unreviewed prescribed relaxed source coil, no dynamics or preextended receiving bridge',simulation_steps=0,stages=stages,head_m=p[0].tolist(),head_T=((p[0]-p[1])/.0645).tolist(),center_radii_m=r.tolist(),all21_source_envelope=bool(np.all((r>.065)&(r<.13)&(c[:,0]+c[:,1]<1.1))),online_source_chart_all21=bool(np.all(abs(r-(.0565+.036))<.031)),original_depths_mm=(depth*1000).tolist(),pipe_normals_N=normal.tolist(),initial_preload_not_balanced=True,source_retained_M13to20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[13:,2]-1,c[13:,1])))*180/np.pi),joint_peak_deg=float(abs(d.qpos[plan.qa]).max()*180/np.pi),not_selected=True)
    (out/'CANDIDATE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--released-hinges',type=int,default=13);ap.add_argument('--taper-length',type=float,default=9.);ap.add_argument('--out',required=True);run(ap.parse_args())
