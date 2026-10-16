"""Prescribed source cap loop/neck/rear coil candidate, not an action state."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json
from math import pi
from scipy.optimize import least_squares,brentq
from scipy.spatial.transform import Rotation
from physics import *
from rough_planner import RoughPlanner
from geometry import Geometry,normalize

def run():
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d);oldpoints=nodes(d);n=13
    oldphi=np.arctan2(oldpoints[n,2]-1,oldpoints[n,1]);phase=np.radians(-110)-oldphi;around=Rotation.from_rotvec(phase*np.array([1.,0.,0.]));origin=np.array([0.,0.,1.]);q0=d.qpos.copy();q0[:3]=origin+around.apply(q0[:3]-origin);q0[0]+=.93-oldpoints[n,0];q0[3:7]=(around*Rotation.from_quat(q0[3:7][[1,2,3,0]])).as_quat()[[3,0,1,2]]
    plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),q0,d.ctrl,reg['kinematic_solver']);plan.observe(q0,n);initial=plan.initial_q.copy();headphi=np.radians(-150);radius=.095;pitch=.075;plane=1.05
    def loop(phi):return np.array([plane-radius*np.cos(phi)-pitch*(phi-headphi)/(2*pi),radius*np.cos(phi),1+radius*np.sin(phi)])
    G=plan.given.copy();phis=[headphi];G[0]=loop(headphi)
    for i in range(1,11):
        before=G[i-1].copy();last=phis[-1];lo=last+.05;hi=last+.95;phi=brentq(lambda phi:np.linalg.norm(loop(phi)-before)-.0645,lo,hi);phis.append(phi);G[i]=loop(phi)
    # Three source-side neck rods connect the cap loop to the unchanged rear.
    A=G[10];B=G[13];middle=.5*(A+B);rad=normalize(np.r_[0.,middle[1],middle[2]-1]);G[11]=A+(B-A)/3+.035*rad;G[12]=A+2*(B-A)/3+.035*rad
    def neck(v):
        p=v.reshape(2,3);allp=np.vstack([A,p,B]);centers=.5*(allp[:-1]+allp[1:]);rho=np.linalg.norm(centers[:,1:]-[0,1],axis=1)
        return np.r_[(np.linalg.norm(np.diff(allp,axis=0),axis=1)-.0645)/.001,(rho-.087)/.005,.01*(p-np.array([G[11],G[12]])).ravel()]
    fit=least_squares(neck,G[11:13].ravel(),max_nfev=200);G[11:13]=fit.x.reshape(2,3)
    plan.fk(initial);child=plan.d.xmat[plan.head+n].reshape(3,3).copy();seed=initial.copy()
    for i in range(n-1,-1,-1):
        desired=child.T@normalize(G[i]-G[i+1]);axis=plan.m.jnt_axis[plan.m.actuator_trnid[i,0]];zero=np.array([0.,0.,1.]);u=zero-axis*(axis@zero);v=desired-axis*(axis@desired);seed[i]=np.clip(-np.arctan2(np.cross(axis,u)@v,u@v),-np.radians(88),np.radians(88));base=Rotation.from_quat(plan.m.body_quat[plan.head+i+1][[1,2,3,0]]).as_matrix();child=child@Rotation.from_rotvec(-seed[i]*axis).as_matrix()@base.T
    cache={};pairs=[(plan.robot[i],g,'source') for i in range(n) for g in plan.source]+[(plan.robot[i],plan.robot[j],'self') for i in range(n) for j in range(i+2,21)]
    def evaluate(front):
        q=initial.copy();q[:n]=front;p=plan.fk(q);centers=.5*(p[:-1]+p[1:]);res=[((p[:n]-G[:n])/.004).ravel(),.002*(front-seed[:n])]
        gaps=[]
        for a,b,kind in pairs:
            gap,_=plan.distance(a,b);gaps.append(gap);limit=-.0008 if kind=='source' else .002;res.append(np.array([min(0.,gap-limit)/.0005]))
        rho=np.linalg.norm(centers[:n,1:]-[0,1],axis=1);res.extend([np.minimum(rho-.075,0)/.001,np.maximum(rho-.115,0)/.001,np.maximum(centers[:n,0]+centers[:n,1]-1.055,0)/.001])
        for i in range(n):
            closest=min(plan.distance(plan.robot[i],g)[0] for g in plan.source);res.append(np.array([max(0.,closest-.0015)/.002]))
        cache.update(q=q.copy(),points=p.copy(),gaps=gaps);return np.concatenate(res)
    candidates=[]
    for start in [seed[:n],initial[:n]]:
        fit=least_squares(evaluate,start,bounds=(-np.radians(88),np.radians(88)),diff_step=1e-5,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-7);r=evaluate(fit.x);candidates.append((float(r@r),fit.x.copy(),bool(fit.success)))
    _,front,success=min(candidates,key=lambda x:x[0]);evaluate(front);plan.fk(cache['q']);d.qpos[:]=plan.d.qpos;d.qvel[:]=0.;d.ctrl[:]=d.qpos[plan.qa];mujoco.mj_forward(m,d);p=nodes(d);c=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(c[:,1:]-[0,1],axis=1);depth=np.zeros(6);normal=np.zeros(4)
    for index,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];robot=[plan.head<=m.geom_bodyid[g]<plan.head+21 for g in gs]
        if all(robot):depth[4]=max(depth[4],-ct.dist);continue
        if sum(robot)!=1:continue
        j=int(robot[1]);name=m.geom(gs[1-j]).name;names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
        if name not in names:continue
        kind=names.index(name);kind=5 if kind==4 else kind;depth[kind]=max(depth[kind],-ct.dist)
        if kind<4:f=np.zeros(6);mujoco.mj_contactForce(m,d,index,f);normal[kind]+=max(0.,f[0])
    out=BASE.parent/'INITIAL_V4_CAP_LOOP_LEFT';out.mkdir(exist_ok=True);state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=p,G=G)
    report=dict(role='Unreviewed prescribed cap-loop source candidate, no dynamics',simulation_steps=0,head_m=p[0].tolist(),head_T=((p[0]-p[1])/.0645).tolist(),head_G_error_mm=float(np.linalg.norm(p[0]-G[0])*1000),G_material_length_error_mm=float(abs(np.linalg.norm(np.diff(G,axis=0),axis=1)-.0645).max()*1000),
        all21_source_envelope=bool(np.all((rho>.065)&(rho<.13)&(c[:,0]+c[:,1]<1.1))),max_center_X_plus_Y=float((c[:,0]+c[:,1]).max()),original_depths_mm=(depth*1000).tolist(),pipe_normals_N=normal.tolist(),joint_peak_deg=float(abs(d.qpos[plan.qa]).max()*180/pi),source_retained_M13to20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[n:,2]-1,c[n:,1])))*180/pi),source_front_M0to12_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[:n,2]-1,c[:n,1])))*180/pi),solver_success=success,head_goal_m=G[0].tolist(),not_selected=True)
    (out/'CANDIDATE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
