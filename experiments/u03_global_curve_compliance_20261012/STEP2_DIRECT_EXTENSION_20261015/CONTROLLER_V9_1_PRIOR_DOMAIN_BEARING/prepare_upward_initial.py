"""New prescribed all-source candidate; no saved action state or integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner,skew

def run():
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    q0=d.qpos.copy();around=Rotation.from_rotvec(np.radians(-96.)*np.array([1.,0.,0.]));origin=np.array([0.,0.,1.]);q0[:3]=origin+around.apply(q0[:3]-origin);q0[3:7]=(around*Rotation.from_quat(q0[3:7][[1,2,3,0]])).as_quat()[[3,0,1,2]]
    plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),q0,d.ctrl.copy(),reg['kinematic_solver']);n=8;cache={};goalX=1.13;initial=plan.initial_q.copy()
    def evaluate(q):
        if 'q' in cache and np.array_equal(q,cache['q']):return cache['r'],cache['J']
        command=initial.copy();command[:n]=q;p=plan.fk(command);jpa=np.zeros((3,plan.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(plan.m,plan.d,jpa,jra,plan.snapshot['anchorP'],plan.head+8);jpa=jpa[:,plan.va[:n]];jra=jra[:,plan.va[:n]]
        jp,jr=plan.point_J(p[0],plan.head,jpa,jra,n);r=[np.array([(p[0,0]-goalX)/.001])];J=[(jp[0]/.001)[None,:]];gaps=[]
        for i in range(n):
            a=plan.robot[i];values=[plan.distance(a,b) for b in plan.source];k=int(np.argmin([v[0] for v in values]));gap,near=values[k];gaps.append(gap)
            grad=np.zeros(n)
            if abs(gap)>1e-9 and gap<.0249:grad=((near[:3]-near[3:])/gap)@plan.point_J(near[:3],plan.head+i,jpa,jra,n)[0]
            r.append(np.array([(gap+.0005)/.001]));J.append((grad/.001)[None,:])
            center=.5*(p[i]+p[i+1]);Jc=.5*(plan.point_J(p[i],plan.head+i,jpa,jra,n)[0]+plan.point_J(p[i+1],plan.head+i,jpa,jra,n)[0]);desiredX=goalX-(i+.5)*(goalX-plan.given[8,0])/8
            r.append(np.array([(center[0]-desiredX)/.03]));J.append((Jc[0]/.03)[None,:])
        r.append(.01*(q-initial[:n]));J.append(.01*np.eye(n));cache.update(q=q.copy(),r=np.concatenate(r),J=np.vstack(J),command=command,points=p.copy(),gaps=np.array(gaps))
        return cache['r'],cache['J']
    fit=least_squares(lambda q:evaluate(q)[0],initial[:n],jac=lambda q:evaluate(q)[1],bounds=(-np.radians(88),np.radians(88)),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-9)
    evaluate(fit.x);plan.fk(cache['command']);d.qpos[:]=plan.d.qpos;d.qvel[:]=0.;d.ctrl[:]=d.qpos[plan.qa];mujoco.mj_forward(m,d);p=nodes(d).copy();c=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(c[:,1:]-[0,1],axis=1);depth=np.zeros(6);normal=np.zeros(4)
    for k,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];robot=[plan.head<=m.geom_bodyid[g]<plan.head+21 for g in gs]
        if all(robot):depth[4]=max(depth[4],-ct.dist);continue
        if sum(robot)!=1:continue
        j=int(robot[1]);name=m.geom(gs[1-j]).name;names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
        if name not in names:continue
        kind=names.index(name);kind=5 if kind==4 else kind;depth[kind]=max(depth[kind],-ct.dist)
        if kind<4:f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);normal[kind]+=max(0.,f[0])
    state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    out=BASE.parent/'INITIAL_V2_UPWARD_EXIT';out.mkdir(exist_ok=True);np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=p)
    report=dict(role='Unreviewed prescribed all-source candidate only, never a dynamic state',simulation_steps=0,new_dynamics_starts=0,source_axis_rotation_deg=-96.,goal_head_tip_X_m=goalX,
        actual_head_tip_m=p[0].tolist(),head_T=((p[0]-p[1])/.0645).tolist(),all21_source_envelope=bool(np.all((rho>=.065)&(rho<=.13)&(c[:,0]+c[:,1]<=1.1)&(c[:,0]>-.2))),
        maximum_center_X_plus_Y_m=float(np.max(c[:,0]+c[:,1])),source_center_radius_m=rho.tolist(),original_depths_mm=(depth*1000).tolist(),original_pipe_normals_N=normal.tolist(),
        actual_joint_peak_deg=float(abs(d.qpos[plan.qa]).max()*180/np.pi),source_retained_M8to20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[8:,2]-1,c[8:,1])))*180/np.pi),
        initial_preload_not_balanced=True,solver_success=bool(fit.success),head_X_error_mm=float(abs(p[0,0]-goalX)*1000),current_GIVEN_INITIAL_unchanged=True)
    (out/'CANDIDATE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
