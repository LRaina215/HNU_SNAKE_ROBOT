"""Offline contact-wrench feasibility; not measured force or dynamic success."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,argparse,numpy as np,mujoco
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m);head=m.body('head').id
ap=argparse.ArgumentParser();ap.add_argument('--trace',default='FIXED_SOURCE_REACH_M8.npz');ap.add_argument('--output',default='SOURCE_WRENCH_FEASIBILITY.json');ap.add_argument('--all',action='store_true');args=ap.parse_args()
trace=np.load(P/args.trace);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];motor=m.jnt_dofadr[m.actuator_trnid[:,0]]
records=[]
for row in (range(len(trace['qpos'])) if args.all else [0,len(trace['qpos'])//2,len(trace['qpos'])-1]):
 d.qpos[:]=trace['qpos'][row];d.qvel[:]=0.;mujoco.mj_fwdPosition(m,d);gravity=np.zeros(m.nv)
 for j in range(21):
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,head+j);gravity-=jp.T@(m.body_mass[head+j]*m.opt.gravity)
 for retained_only in [True,False]:
  blocks=[];contacts=[];cones=[]
  for ct in d.contact:
   gs=[ct.geom1,ct.geom2];robot=[head<=m.geom_bodyid[g]<head+21 for g in gs]
   if sum(robot)!=1:continue
   j=1 if robot[1] else 0;material=int(m.geom_bodyid[gs[j]])-head;other=m.geom(gs[1-j]).name
   if other not in ['pipe_straight_01','pipe_sharp_in_02'] or retained_only and material<8:continue
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,ct.pos,head+material);basis=(1 if j==1 else -1)*ct.frame.reshape(3,3).T
   blocks.append(np.column_stack([jp.T@basis,jr.T@basis]));contacts.append(dict(material=material,point=ct.pos.tolist(),friction=ct.friction.tolist()));cones.append(ct.friction.copy())
  item=dict(alpha=float(trace['alpha'][row]),contact_scope='fixed M8-M20 only' if retained_only else 'all remaining source contacts; receiver excluded',contacts=contacts)
  if not blocks:item.update(feasible=False,reason='no source contacts');records.append(item);continue
  directions=[np.zeros(5)]
  for axis in range(5):
   for sign in [-1.,1.]:v=np.zeros(5);v[axis]=sign;directions.append(v)
  for axis1 in range(5):
   for axis2 in range(axis1+1,5):
    for sign1 in [-1.,1.]:
     for sign2 in [-1.,1.]:v=np.zeros(5);v[axis1]=sign1/np.sqrt(2);v[axis2]=sign2/np.sqrt(2);directions.append(v)
  for angle in np.linspace(0,2*np.pi,16,endpoint=False):v=np.zeros(5);v[:2]=[np.cos(angle),np.sin(angle)];directions.append(v)
  directions=np.array(directions);D=len(directions);wrench_bases=[np.vstack([np.ones(D),directions.T*friction[:,None]]) for friction in cones]
  K=np.column_stack([block@basis for block,basis in zip(blocks,wrench_bases)]);C=len(blocks);obj=np.ones(C*D);A=[];b=[]
  limit=2.30
  A.extend(K[motor]);b.extend(gravity[motor]+limit);A.extend(-K[motor]);b.extend(-gravity[motor]+limit)
  result=linprog(obj,A_ub=np.array(A),b_ub=np.array(b),A_eq=K[:6],b_eq=gravity[:6],bounds=(0,None),method='highs')
  item.update(feasible=bool(result.success),solver_message=result.message,motor_torque_bound_Nm=limit)
  if result.success:
   force=result.x;normal=np.array([force[c*D:(c+1)*D].sum() for c in range(C)]);wrenches=[wrench_bases[c]@force[c*D:(c+1)*D] for c in range(C)];torque=gravity[motor]-K[motor]@force;item.update(minimum_sum_normal_N=float(normal.sum()),normals_N=normal.tolist(),local_contact_wrenches=[w.tolist() for w in wrenches],
    motor_torques_Nm=torque.tolist(),motor_torque_peak_Nm=float(abs(torque).max()),root_balance_max_abs=float(abs(K[:6]@force-gravity[:6]).max()))
  records.append(item)
out=dict(scope='Static source contact-wrench feasibility using original scene contact set and conic combinations of valid5D elliptic-friction directions, jointly coupling tangential force/torsion/rolling; not realized contact loads or dynamic stability',
 original_physical_enableflags=int(m.opt.enableflags),simulation_steps=0,software_torque_bound_Nm=2.30,physical_actuator_limit_Nm=2.355,
 caveat='Forces require compatible elastic/contact preload; source holding controller must realize them with small compliance and be verified dynamically',records=records)
(P/args.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
