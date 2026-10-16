"""Offline rough-prior front wrapping feasibility; original scene only evaluates."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,argparse
from physics import *
from rough_planner import RoughPlanner
from geometry import Geometry
from post_capture_curve import PostCaptureCurve
from contact_sensor import read_sensor

def run(truth_axis_diagnostic=False,initial_file=None,release_file=None,out=None,contact_replay=False):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    out=BASE if out is None else Path(out).resolve();initial_file=BASE/'GIVEN_INITIAL.npz' if initial_file is None else Path(initial_file).resolve()
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(initial_file);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    geometry=Geometry([1.1,.2,1.],[0.,1.,0.],.0565) if truth_axis_diagnostic else Geometry.from_dict(reg['conditions']['DIRECTION'])
    plan=RoughPlanner(geometry,d.qpos.copy(),d.ctrl.copy(),reg['kinematic_solver']);release=np.load(release_file if release_file else BASE.parent/'DIRECT_CONTACT_PATH.npz' if truth_axis_diagnostic else BASE/'ROUGH_T_RELEASE.npz');plan.q=release['qpos'][-1,plan.qa].copy();plan.observe(d.qpos.copy(),reg['direct_extension']['wrap_hinges'])
    rows=[];qrows=[];Grows=[]
    for beta in np.linspace(0,1,81):
        if contact_replay:
            plan.fk(plan.q);d.qpos[:]=plan.d.qpos;d.ctrl[:]=d.qpos[plan.qa];d.qvel[:]=0.;mujoco.mj_forward(m,d);plan.observe(d.qpos.copy(),reg['direct_extension']['wrap_hinges']);plan.observe_contacts(read_sensor(m,d)['contacts'])
        q,G,metrics=plan.solve(1.,float(max(beta,1e-12)),max_iterations=100);C=plan.fk(q);d.qpos[:]=plan.d.qpos;mujoco.mj_forward(m,d);depth=np.zeros(4)
        head=m.body('head').id
        for ct in d.contact:
            bs=[int(m.geom_bodyid[g]) for g in [ct.geom1,ct.geom2]];robot=[head<=b<head+21 for b in bs]
            if all(robot):depth[2]=max(depth[2],-ct.dist)
            elif sum(robot)==1:
                k=1 if robot[1] else 0;name=m.geom([ct.geom1,ct.geom2][1-k]).name
                if name=='floor':depth[3]=max(depth[3],-ct.dist)
                elif name and name.startswith('pipe_'):depth[0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1]=max(depth[0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1],-ct.dist)
        n=reg['direct_extension']['wrap_hinges'];metrics.update(beta=float(beta),joint_peak_deg=float(abs(q).max()*180/np.pi),source_node_motion_m=float(np.linalg.norm(C[n:]-plan.given[n:],axis=1).max()),original_depths_mm=(depth*1000).tolist())
        qrows.append(d.qpos.copy());Grows.append(G);rows.append(metrics);print(json.dumps(metrics),flush=True)
    prefix=('CONTACT_REPLAY_' if contact_replay else '')+('DIAGNOSTIC_TRUE_AXIS' if truth_axis_diagnostic else 'ROUGH')
    np.savez_compressed(out/f'{prefix}_T_WRAP.npz',beta=np.linspace(0,1,81),qpos=qrows,G=Grows)
    report=dict(simulation_steps=0,passed=all(x['accepted'] and x['head_error_m']<.005 and max(x['original_depths_mm'])<3 for x in rows),rows=rows,
        truth_axis_counterfactual_only=truth_axis_diagnostic,offline_contact_replay=contact_replay,controller_uses_this_truth_geometry_or_path=False,
        scope='Offline mechanical counterfactual only, separating prior error from fixed-source wrap obstruction. No estimator/dynamics success claim.' if truth_axis_diagnostic else 'Offline coarse-prior wrap planning only. No reliable-geometry, actual acquisition or dynamics claim.')
    report['initial_file']=str(initial_file);(out/f'{prefix}_WRAP_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic-true-axis',action='store_true');ap.add_argument('--contact-replay',action='store_true');ap.add_argument('--initial');ap.add_argument('--release');ap.add_argument('--out');args=ap.parse_args();run(args.diagnostic_true_axis,args.initial,args.release,args.out,args.contact_replay)
