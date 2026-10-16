"""Fit a prescribed source preload with original forward physics, no integration."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import argparse,json
from scipy.optimize import least_squares
from physics import *

def run(args):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial)
    mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    q0=d.qpos.copy();qa=m.jnt_qposadr[m.actuator_trnid[:,0]];n=args.start;count=20-n
    lp=json.loads(Path(args.wrench).read_text());record=next(x for x in lp['records'] if x['feasible']);torque=np.array(record['motor_torques_Nm'])
    root_count=2 if args.root_yz else 0;setup_count=count+root_count
    x0=np.r_[np.zeros(setup_count),d.actuator_force.copy() if args.warm_control else torque];stage='force'
    def assign(x):
        d.qpos[:]=q0
        if root_count:d.qpos[1:3]+=x[:root_count]
        d.qpos[qa[n:]]+=x[root_count:setup_count];d.ctrl[:]=d.qpos[qa]+x[setup_count:]/m.actuator_gainprm[:,0]
        d.qvel[:]=0.;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d)
    def residual(x):
        assign(x)
        if stage=='acceleration':return np.r_[d.qacc*np.r_[np.full(6,5.),np.ones(20)],.0001*x[:setup_count]]
        mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM)
        return np.r_[(mass@d.qacc)/np.r_[np.ones(3),np.full(3,.1),np.full(20,.05)],.0001*x[:setup_count]]
    assign(x0);before=dict(max_qacc=float(abs(d.qacc).max()),root_qacc=d.qacc[:6].tolist())
    lower=np.r_[np.full(root_count,-.001),np.full(count,-np.radians(1.5)),np.full(20,-2.30)];upper=-lower;scales=np.r_[np.full(root_count,.001),np.full(count,.01),np.full(20,.5)]
    steps=np.r_[np.full(root_count,1e-6),np.full(count,1e-5),np.full(20,.001)]
    def jacobian(x):
        columns=[]
        for j,h in enumerate(steps):
            hi=x.copy();lo=x.copy();hi[j]=min(upper[j],x[j]+h);lo[j]=max(lower[j],x[j]-h)
            columns.append((residual(hi)-residual(lo))/(hi[j]-lo[j]))
        return np.column_stack(columns)
    fit=least_squares(residual,x0,jac=jacobian,bounds=(lower,upper),x_scale=scales,max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-8)
    stage='acceleration';steps*=.1
    fit=least_squares(residual,fit.x,jac=jacobian,bounds=(lower,upper),x_scale=scales,max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-8);assign(fit.x)
    state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=nodes(d).copy())
    np.savez_compressed(out/'INITIAL_TRACE.npz',alpha=np.array([0.]),qpos=d.qpos[None,:])
    report=dict(simulation_steps=0,initial_accepted=False,original_physical_enableflags=int(m.opt.enableflags),source_start=n,before=before,
        qacc_max=float(abs(d.qacc).max()),qacc_norm=float(np.linalg.norm(d.qacc)),root_qacc=d.qacc[:6].tolist(),
        root_yz_setup_adjustment_m=fit.x[:root_count].tolist(),source_hinge_adjustment_deg=np.degrees(fit.x[root_count:setup_count]).tolist(),torques_Nm=d.actuator_force.tolist(),torque_max_Nm=float(abs(d.actuator_force).max()),
        fore_and_root_unchanged=bool(np.array_equal(d.qpos[:7],q0[:7]) and np.array_equal(d.qpos[qa[:n]],q0[qa[:n]])),
        optimizer_success=bool(fit.success),optimizer_message=fit.message,
        scope='Prescribed preload fit only. Finite residual reported; physical stability still requires an untrimmed source hold.')
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--wrench',required=True);ap.add_argument('--out',required=True);ap.add_argument('--start',type=int,default=8);ap.add_argument('--root-yz',action='store_true');ap.add_argument('--warm-control',action='store_true');run(ap.parse_args())
