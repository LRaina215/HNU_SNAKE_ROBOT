"""Original-physics static preload, preserving given head and M8 boundary."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,numpy as np,mujoco
from scipy.optimize import least_squares
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m)
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
z=np.load(P/'SUPPORT_GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);q0=d.qpos.copy();qa=m.jnt_qposadr[m.actuator_trnid[:,0]];source=qa[8:]
lp=json.loads((P/'SOURCE_WRENCH_SUPPORT_GIVEN.json').read_text());torque=np.array(lp['records'][0]['motor_torques_Nm']);x0=np.r_[np.zeros(12),torque];stage='force'
def assign(x):
 d.qpos[:]=q0;d.qpos[source]+=x[:12];d.ctrl[:]=d.qpos[qa]+x[12:]/m.actuator_gainprm[:,0];d.qvel[:]=0.;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d)
def residual(x):
 assign(x)
 if stage=='acceleration':return np.r_[d.qacc*np.r_[np.full(6,5.),np.ones(20)],.0001*x[:12]]
 mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM);return np.r_[(mass@d.qacc)/np.r_[np.ones(3),np.full(3,.1),np.full(20,.05)],.0001*x[:12]]
assign(x0);before=dict(qacc_max=float(abs(d.qacc).max()),qacc_norm=float(np.linalg.norm(d.qacc)),root_qacc=d.qacc[:6].tolist())
lower=np.r_[np.full(12,-np.radians(1.5)),np.full(20,-2.30)];upper=-lower;scales=np.r_[np.full(12,.01),np.full(20,.5)]
steps=np.r_[np.full(12,1e-5),np.full(20,.001)]
def derivative(x):
 columns=[]
 for j,h in enumerate(steps):
  hi=x.copy();lo=x.copy();hi[j]=min(upper[j],x[j]+h);lo[j]=max(lower[j],x[j]-h);columns.append((residual(hi)-residual(lo))/(hi[j]-lo[j]))
 return np.column_stack(columns)
fit=least_squares(residual,x0,jac=derivative,bounds=(lower,upper),x_scale=scales,max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-8)
stage='acceleration';steps*=.1
fit=least_squares(residual,fit.x,jac=derivative,bounds=(lower,upper),x_scale=scales,max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-8);assign(fit.x)
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(P/'STATIC_READY_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy())
report=dict(role='Prescribed source preload, no dynamic formation',simulation_steps=0,original_physical_enableflags=int(m.opt.enableflags),before=before,
 qacc_max=float(abs(d.qacc).max()),qacc_norm=float(np.linalg.norm(d.qacc)),root_qacc=d.qacc[:6].tolist(),
 source_hinge_adjustment_deg=np.degrees(fit.x[:12]).tolist(),torques_Nm=d.actuator_force.tolist(),torque_max_Nm=float(abs(d.actuator_force).max()),
 head_and_fore_qpos_unchanged=bool(np.array_equal(d.qpos[:7],q0[:7]) and np.array_equal(d.qpos[qa[:8]],q0[qa[:8]])),
 optimizer_success=bool(fit.success),optimizer_message=fit.message,optimizer_optimality=float(fit.optimality),accepted_initial=False,
 note='Forward residual is reported honestly; logged source hold and direct-extension dynamics remain required')
(P/'STATIC_READY_INITIAL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
