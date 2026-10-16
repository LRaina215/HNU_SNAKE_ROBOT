"""Prescribe posterior source contact in a candidate setup, never integrate."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import argparse,json
from scipy.optimize import least_squares
from physics import *

def run(args):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(MODEL));m.opt.enableflags|=64;d=mujoco.MjData(m)
    z=np.load(args.initial);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]];q0=d.qpos.copy();ctrl0=d.ctrl.copy();head=m.body('head').id;n=args.start
    geoms=[int(m.body_geomadr[head+j]) for j in range(21)];source=m.geom('pipe_straight_01').id;selected=np.arange(n,20)
    targets=[j for j in range(n+1,21) if j%2==0]
    mujoco.mj_kinematics(m,d);old=nodes(d).copy();oldC=.5*(old[:-1]+old[1:])
    def assign(v):d.qpos[:]=q0;d.qpos[qa[selected]]+=v;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    def distance(a,b):return min(.02,float(mujoco.mj_geomDistance(m,d,a,b,.02,None)))
    def residual(v):
        assign(v);p=nodes(d);c=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(c[n:,1:]-[0,1],axis=1)
        r=[np.array([distance(source,geoms[j])+.0005 for j in targets])/.0003,
           (c[n:,0]-oldC[n:,0])/.006,(c[n:,1:]-oldC[n:,1:]).ravel()/.025,v/.15,
           np.minimum(rho-.065,0)/.001,np.maximum(rho-.13,0)/.001]
        for j in range(n,21):r.append(np.array([min(0.,distance(source,geoms[j])+.0015)/.0003]))
        return np.concatenate(r)
    bound=np.radians(8);lower=np.maximum(-bound,-np.radians(88)-q0[qa[selected]]);upper=np.minimum(bound,np.radians(88)-q0[qa[selected]])
    fit=least_squares(residual,np.zeros(len(selected)),bounds=(lower,upper),diff_step=1e-5,max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-8);assign(fit.x);pose=d.qpos.copy()
    physical=mujoco.MjModel.from_xml_path(str(MODEL));actual=mujoco.MjData(physical);actual.qpos[:]=pose;actual.qvel[:]=0.;actual.ctrl[:]=pose[qa]+ctrl0-q0[qa];mujoco.mj_forward(physical,actual)
    state=np.empty(mujoco.mj_stateSize(physical,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(physical,actual,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=pose,ctrl=actual.ctrl.copy(),points=nodes(actual).copy())
    # A one-pose trace permits the existing static-wrench audit, without a path.
    np.savez_compressed(out/'INITIAL_TRACE.npz',alpha=np.array([0.]),qpos=pose[None,:])
    report=dict(simulation_steps=0,prescribed_setup_only=True,initial_accepted=False,source_start=n,
        fore_and_root_unchanged=bool(np.array_equal(pose[:7],q0[:7]) and np.array_equal(pose[qa[:n]],q0[qa[:n]])),
        posterior_setup_delta_deg=np.degrees(fit.x).tolist(),targets=targets,target_gaps_mm=[distance(source,geoms[j])*1000 for j in targets],
        optimizer_success=bool(fit.success),maximum_posterior_setup_delta_deg=float(abs(fit.x).max()*180/np.pi))
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--out',required=True);ap.add_argument('--start',type=int,default=13);run(ap.parse_args())
