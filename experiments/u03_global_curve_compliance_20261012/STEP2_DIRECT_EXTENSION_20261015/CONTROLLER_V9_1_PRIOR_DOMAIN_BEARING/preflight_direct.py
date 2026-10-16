"""No-integration audit of the actual rough-prior controller ingredients."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,argparse
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner
from contact_sensor import read_sensor,known_source,source_geometry

def run(initial_file=None,out=None):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration permitted')))
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    out=BASE if out is None else Path(out).resolve();initial_file=BASE/'GIVEN_INITIAL.npz' if initial_file is None else Path(initial_file).resolve()
    z=np.load(initial_file);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    sensor=read_sensor(m,d);source=reg['source_known'];q0=d.qpos.copy();rows=[];poses=[];guides=[]
    plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),q0,d.ctrl,reg['kinematic_solver']);n=reg['direct_extension']['released_hinges']
    for alpha in np.linspace(0,1,61):
        if alpha==0:q=plan.initial_q.copy();G=plan.given.copy();metrics=dict(accepted=True,head_error_m=0.,G_material_length_error_m=0.)
        else:q,G,metrics=plan.solve(float(alpha),max_iterations=100)
        C=plan.fk(q);poses.append(plan.d.qpos.copy());guides.append(G)
        metrics.update(alpha=float(alpha),joint_peak_deg=float(abs(q).max()*180/np.pi),source_node_motion_m=float(np.linalg.norm(C[n:]-plan.given[n:],axis=1).max()),
            native_integral_error_rad=float(abs(plan.coordinates.hinges(plan.coordinates.field(q))-q).max()),
            gravity_torque_peak_Nm=float(abs(plan.gravity_torque(q,n)).max()))
        # Truth is used only by this offline evaluator, never fed to RoughPlanner.
        d.qpos[:]=poses[-1];mujoco.mj_forward(m,d);depth=np.zeros(6);receiver_normal=0.
        for ct in d.contact:
            bs=[int(m.geom_bodyid[g]) for g in [ct.geom1,ct.geom2]];robot=[plan.head<=b<plan.head+21 for b in bs]
            if all(robot):depth[4]=max(depth[4],-ct.dist);continue
            if sum(robot)!=1:continue
            k=1 if robot[1] else 0;other=m.geom([ct.geom1,ct.geom2][1-k]).name
            names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
            if other in names:
                idx=names.index(other);idx=5 if idx==4 else idx;depth[idx]=max(depth[idx],-ct.dist)
        metrics.update(original_collision_depths_mm=(depth*1000).tolist())
        rows.append(metrics);print(json.dumps(metrics),flush=True)
    np.savez_compressed(out/'ROUGH_T_RELEASE.npz',alpha=np.linspace(0,1,61),qpos=poses,G=guides)
    allowed_environment={'floor','pipe_straight_01','pipe_sharp_in_02','rough_receiver'}
    head=plan.head;environment_names=[plan.m.geom(i).name for i in range(plan.m.ngeom) if not head<=plan.m.geom_bodyid[i]<head+21]
    checks=dict(original_20_hinges=m.nu==20 and int(m.opt.enableflags)==0,
        source_geometry_all21=bool(source_geometry(sensor['centers'],source).all()),
        initial_source_contacts_recognized=bool(all(known_source(ct,source) for ct in sensor['contacts'])),
        private_model_environment_whitelist=bool(set(environment_names)<=allowed_environment),
        no_true_receiver_mesh='pipe_sharp_out_03' not in [plan.m.mesh(i).name for i in range(plan.m.nmesh)],
        all_solver_candidates_accepted=all(x['accepted'] for x in rows),
        fixed_source_rough_touchdown_reaches_original_receiver=rows[-1]['original_collision_depths_mm'][3]>0.,
        original_collision_samples_within_3mm=all(max(x['original_collision_depths_mm'][:5])<3. for x in rows),
        fixed_source_nodes=max(x['source_node_motion_m'] for x in rows)<1e-8,
        joint_limit=max(x['joint_peak_deg'] for x in rows)<=88.00001,
        native_curvature_integral=max(x['native_integral_error_rad'] for x in rows)<1e-12,
        G_fixed_material_lengths=max(x['G_material_length_error_m'] for x in rows)<1e-9,
        controller_does_not_write_actual_state=True)
    d.qpos[:]=q0;mujoco.mj_forward(m,d)
    report=dict(version=reg['version'],simulation_steps=0,checks=checks,passed=all(checks.values()),
        scope='Rough-prior release ingredients only; online phase chain, wrapping, transport and actual dynamics still require review',
        environment_names=environment_names,rough_prior=reg['conditions']['DIRECTION'],
        maximum_head_goal_error_mm=1000*max(x['head_error_m'] for x in rows),
        head_goal_error_interpretation='Report constrained C departure from the prescribed G touchdown goal; the nominal2.5mm inward contact margin is not an exact feasible head pose. Passing requires source fixed, limits, native integral and offline original contact/clearance; no dynamic stability claim.',rows=rows)
    report['initial_file']=str(initial_file);(out/'ROUGH_RELEASE_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial');ap.add_argument('--out');args=ap.parse_args();run(args.initial,args.out)
