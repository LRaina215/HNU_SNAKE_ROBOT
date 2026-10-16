"""Mechanical endpoint IK, source-native wrap exactly fixed; no path/dynamics claim."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner

def run(folder):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    P=Path(folder).resolve();reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(P/'CANDIDATE_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    geometry=Geometry([1.1,.2,1],[0,1,0],.0565);n=reg['kinematic_solver']['wrap_hinges'];plan=RoughPlanner(geometry,d.qpos,d.ctrl,reg['kinematic_solver']);plan.observe(d.qpos.copy(),n);cal=np.load(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.npz');fixed=cal['joint_curvature_integrals'][:5].copy();fixed[::2]*=-1;cp=cal['points'];e,b=geometry.frame();phi=np.radians(-165-reg['kinematic_solver']['maximum_wrap_deg']);delta=phi-np.arctan2(cp[0,2]-1,cp[0,1]);Rmap=np.column_stack([geometry.axis,np.cos(delta)*e+np.sin(delta)*b,-np.sin(delta)*e+np.cos(delta)*b]);head_ax=.18266863819422827*reg['kinematic_solver']['maximum_wrap_deg']/360+.02
    origin=np.array([cp[0,0],0,1]);wantedP=geometry.entry+head_ax*geometry.axis+Rmap@(cal['body_xpos'][5]-origin);wantedR=Rmap@cal['body_xmat'][5]@np.diag([1.,-1.,1.])
    assert np.linalg.det(wantedR)>.999999
    pairs=[(plan.robot[i],g,'source') for i in range(n) for g in plan.source]+[(plan.robot[i],plan.target,'target') for i in range(n)]+[(plan.robot[i],plan.robot[j],'self') for i in range(n) for j in range(i+2,21)]
    base=plan.initial_q.copy();base[:5]=fixed;cache={}
    pose_only=False
    def evaluate(v):
        q=base.copy();q[5:n]=v;p=plan.fk(q);R=plan.d.xmat[plan.head+5].reshape(3,3);res=[(plan.d.xpos[plan.head+5]-wantedP)/.001,Rotation.from_matrix(wantedR.T@R).as_rotvec()/.02,.0001*v];gaps=[]
        for a,b,kind in pairs:
            gap,_=plan.distance(a,b);limit=.001 if kind=='self' else -.0008 if kind=='target' else min(-.0008,plan.initial_gaps.get((int(plan.m.geom_bodyid[a])-plan.head,b),-.0008)-.00005);gaps.append(gap)
            if not pose_only:res.append(np.array([min(0.,gap-limit)/.0003]))
        cache.update(q=q,p=p.copy(),gaps=np.array(gaps),position_error=float(np.linalg.norm(plan.d.xpos[plan.head+5]-wantedP)),orientation_error=float(np.linalg.norm(Rotation.from_matrix(wantedR.T@R).as_rotvec())))
        return np.concatenate(res)
    rng=np.random.default_rng(20261015);saved=[];start=plan.initial_q[5:n].copy();path=P/'CONTACT_REPLAY_DIAGNOSTIC_TRUE_AXIS_T_WRAP.npz'
    if path.exists():start=np.load(path)['qpos'][-1,plan.qa[5:n]].copy()
    for index,guess in enumerate([start,np.zeros(n-5),rng.uniform(-1.,1.,n-5),rng.uniform(-1.2,1.2,n-5)]):
        pose_only=True;pose=least_squares(evaluate,guess,bounds=(-np.radians(88),np.radians(88)),diff_step=1e-5,max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-7)
        pose_only=False;fit=least_squares(evaluate,pose.x,bounds=(-np.radians(88),np.radians(88)),diff_step=1e-5,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-7);res=evaluate(fit.x);saved.append((float(res@res),fit.x.copy()));print(json.dumps(dict(seed=index,cost=float(res@res),position_error_mm=cache['position_error']*1000,orientation_error_deg=cache['orientation_error']*180/np.pi,minimum_gap_mm=float(cache['gaps'].min()*1000))),flush=True)
    _,best=min(saved,key=lambda v:v[0]);evaluate(best);plan.fk(cache['q']);d.qpos[:]=plan.d.qpos;mujoco.mj_fwdPosition(m,d);p=nodes(d);c=.5*(p[:-1]+p[1:]);theta=np.unwrap(np.arctan2(c[:6,2]-1,c[:6,0]-1.1));depth=np.zeros(4)
    for ct in d.contact:
        gs=[ct.geom1,ct.geom2];robot=[plan.head<=m.geom_bodyid[g]<plan.head+21 for g in gs]
        if all(robot):depth[2]=max(depth[2],-ct.dist)
        elif sum(robot)==1:
            name=m.geom(gs[0] if not robot[0] else gs[1]).name
            if name=='floor':depth[3]=max(depth[3],-ct.dist)
            elif name and name.startswith('pipe_'):depth[0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1]=max(depth[0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1],-ct.dist)
    np.savez_compressed(P/'NATIVE_CAPTURE_ENDPOINT.npz',qpos=d.qpos.copy(),points=p,joint_integrals=cache['q'])
    out=dict(role='Accurate-axis mechanical endpoint counterfactual only, not a path, sensor fit or dynamics',simulation_steps=0,position_error_mm=cache['position_error']*1000,orientation_error_deg=cache['orientation_error']*180/np.pi,front6_span_deg=float(np.ptp(theta)*180/np.pi),front6_axial_Y_m=c[:6,1].tolist(),original_depths_mm=(depth*1000).tolist(),joint_peak_deg=float(abs(cache['q']).max()*180/np.pi),source_M14to20_fixed=True,head_m=p[0].tolist())
    (P/'NATIVE_CAPTURE_ENDPOINT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');run(ap.parse_args().folder)
