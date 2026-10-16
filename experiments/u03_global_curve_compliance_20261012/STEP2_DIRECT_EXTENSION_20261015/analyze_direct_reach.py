"""Offline fixed-source native-hinge reach/mesh-clearance proof; no dynamics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
import numpy as np,mujoco
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
ap=argparse.ArgumentParser();ap.add_argument('--release',type=int,default=8);args=ap.parse_args();N=args.release
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m);initial=np.load(P/'CANDIDATE_INITIAL.npz')
original_enableflags=int(m.opt.enableflags)
# Geometry-query model only. The original physical-run model/XML/flags stay unchanged.
# Legacy libccd signed-distance queries can return false deep penetration away from contact.
m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
q0=d.qpos.copy();head=m.body('head').id;anchor=head+N;qa=m.jnt_qposadr[m.actuator_trnid[:,0]];va=m.jnt_dofadr[m.actuator_trnid[:,0]][:N];joint_ids=m.actuator_trnid[:N,0]
anchorP=d.xpos[anchor].copy();anchorR=d.xmat[anchor].reshape(3,3).copy();rootR=Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_matrix();original_nodes=nodes(d).copy()
robot_geoms=[int(m.body_geomadr[head+i]) for i in range(21)];pipes=[m.geom(x).id for x in ['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04']]
pairs=[(robot_geoms[i],g,'source' if k in [0,1] else 'target' if k==3 else 'elbow',i) for i in range(N) for k,g in enumerate(pipes)]
pairs += [(robot_geoms[i],robot_geoms[j],'self',i) for i in range(N) for j in range(i+2,21)]
def skew(v):return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
def fk(q):
 d.qpos[:]=q0;d.qpos[qa[:N]]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
 correction=anchorR@d.xmat[anchor].reshape(3,3).T;translation=anchorP-correction@d.xpos[anchor]
 d.qpos[:3]=correction@q0[:3]+translation;d.qpos[3:7]=Rotation.from_matrix(correction@rootR).as_quat()[[3,0,1,2]]
 mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
def point_J(point,body,jpa,jra):
 jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,point,body)
 return jp[:,va]-jpa+skew(point-anchorP)@jra,jr[:,va]-jra
def distance(a,b):
 if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+.025:return .025,np.zeros(6)
 near=np.zeros(6)
 # 3.2.7 witnesses follow the narrow-phase type order for mixed geom types.
 if m.geom_type[a]>m.geom_type[b]:
  gap=mujoco.mj_geomDistance(m,d,b,a,.025,near);near=np.r_[near[3:],near[:3]]
 else:gap=mujoco.mj_geomDistance(m,d,a,b,.025,near)
 return float(gap),near
fk(q0[qa[:N]]);initial_dist=np.array([distance(a,b)[0] for a,b,_,_ in pairs]);start_center=.5*(nodes(d)[0]+nodes(d)[1]);start_T=d.xmat[head].reshape(3,3)[:,2].copy()
contact_angle=np.radians(-165.);surface_normal=np.array([np.cos(contact_angle),0.,np.sin(contact_angle)]);target_axis=np.array([0.,1.,0.]);target_tip=np.array([1.1,.20,1.])+.0565*surface_normal
final_T=np.array([.90,0.,.435]);final_T/=np.linalg.norm(final_T)
times=np.array([0.,.45,.75,1.]);position_knots=np.array([nodes(d)[0],[1.04,.10,.93],[1.045,.175,.967],target_tip]);T_knots=np.array([(1-t)*start_T+t*final_T for t in times]);T_knots/=np.linalg.norm(T_knots,axis=1)[:,None]
outward=np.r_[0.,start_center[1],start_center[2]-1.];outward/=np.linalg.norm(outward)
path=CubicSpline(times,position_knots,axis=0,bc_type=((1,np.array([0.,.20,.05])),(1,np.array([0.,.06,.10]))));tangent_path=CubicSpline(times,T_knots,axis=0)
cache={};phase=0.;previous=q0[qa[:N]].copy()
def evaluate(q):
 if cache.get('phase')==phase and 'q' in cache and np.array_equal(q,cache['q']):return cache['r'],cache['J']
 fk(q);p=nodes(d);center=.5*(p[0]+p[1]);T=d.xmat[head].reshape(3,3)[:,2];jpa=np.zeros((3,m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(m,d,jpa,jra,anchorP,anchor);jpa=jpa[:,va];jra=jra[:,va]
 jp,jr=point_J(p[0],head,jpa,jra);goal=path(phase);wanted=tangent_path(phase);wanted/=np.linalg.norm(wanted)
 residual=[(p[0]-goal)/.002,5*(T-wanted),.005*(q-previous)];jacobian=[jp/.002,-5*skew(T)@jr,.005*np.eye(N)];gaps=[];required=[];gap_J=[]
 for index,(a,b,kind,material) in enumerate(pairs):
  gap,near=distance(a,b);gaps.append(gap)
  if kind=='source':limit=min(-.0008 if material==0 else .0005,initial_dist[index]-.00005)
  elif kind=='self':limit=min(.001,initial_dist[index]-.00005)
  elif kind=='target' and material==0:limit=.002*(1-phase**8)-.00025*phase**8
  else:limit=.0015
  required.append(limit);violation=min(0.,gap-limit);residual.append(np.array([violation/.001]))
  if gap<.0249 and abs(gap)>1e-9:
   normal=(near[:3]-near[3:])/gap;Ja,_=point_J(near[:3],int(m.geom_bodyid[a]),jpa,jra)
   if kind=='self':Jb,_=point_J(near[3:],int(m.geom_bodyid[b]),jpa,jra)
   else:Jb=np.zeros_like(Ja)
   derivative=normal@(Ja-Jb);gap_J.append(derivative);jacobian.append((derivative/.001)[None,:] if violation<0 else np.zeros((1,N)))
  else:gap_J.append(np.zeros(N));jacobian.append(np.zeros((1,N)))
 # Conservative gravity moments of all distal released bodies; no help from front contacts assumed.
 torques=[];torque_J=[]
 for j,jid in enumerate(joint_ids):
  axis=d.xaxis[jid].copy();pivot=d.xanchor[jid].copy();pivot_J,angular_J=point_J(pivot,head+j+1,jpa,jra);axis_J=-skew(axis)@angular_J;moment=np.zeros(3);moment_J=np.zeros((3,N))
  for i in range(j+1):
   mass_force=m.body_mass[head+i]*m.opt.gravity;com=d.xipos[head+i];com_J,_=point_J(com,head+i,jpa,jra)
   moment+=np.cross(com-pivot,mass_force);moment_J+=np.cross((com_J-pivot_J).T,mass_force).T
  torque=-axis@moment;derivative=-moment@axis_J-axis@moment_J;torques.append(torque);torque_J.append(derivative)
 torques=np.array(torques);torque_J=np.array(torque_J);excess=np.maximum(abs(torques)-2.15,0.)
 residual.append(excess/.20);jacobian.append(np.where(excess[:,None]>0,np.sign(torques)[:,None]*torque_J/.20,0.))
 r=np.concatenate(residual);J=np.vstack(jacobian);cache.update(q=q.copy(),phase=phase,r=r,J=J,points=p.copy(),center=center.copy(),T=T.copy(),gaps=np.array(gaps),gap_J=np.array(gap_J),required=np.array(required),gravity_torques=torques.copy())
 return r,J
rows=[];qrows=[];poserows=[];gapsrows=[];fitfail=[]
distance_gradient_errors=[]
for pair_index in [1,5]:
 a,b,kind,material=pairs[pair_index];q=q0[qa[:N]].copy();fk(q);gap,near=distance(a,b)
 jpa=np.zeros((3,m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(m,d,jpa,jra,anchorP,anchor)
 analytic=((near[:3]-near[3:])/gap)@point_J(near[:3],int(m.geom_bodyid[a]),jpa[:,va],jra[:,va])[0]
 numerical=[]
 for j in range(N):
  plus=q.copy();plus[j]+=1e-6;fk(plus);hi=distance(a,b)[0];minus=q.copy();minus[j]-=1e-6;fk(minus);lo=distance(a,b)[0];numerical.append((hi-lo)/2e-6)
 distance_gradient_errors.append(float(np.max(abs(analytic-numerical))))
assert max(distance_gradient_errors)<1e-3,('Distance witness derivative failed',distance_gradient_errors)
fk(q0[qa[:N]]);cache.clear()
for phase in np.linspace(0,1,61):
 if phase==0:q=previous.copy();fit=None;evaluate(q)
 else:
  scale=np.radians(8.);lo=np.maximum(-1.,(-np.radians(88)-previous)/scale);hi=np.minimum(1.,(np.radians(88)-previous)/scale)
  def objective(u):
   r,J=evaluate(previous+scale*u);return .5*r@r
  def gradient(u):
   r,J=evaluate(previous+scale*u);return scale*J.T@r
  def clearance(u,jac=False):
   evaluate(previous+scale*u);return scale*cache['gap_J']/.001 if jac else (cache['gaps']-cache['required'])/.001
  fit=minimize(objective,np.zeros(N),jac=gradient,bounds=list(zip(lo,hi)),method='SLSQP',constraints=[dict(type='ineq',fun=clearance,jac=lambda u:clearance(u,True))],options=dict(maxiter=100,ftol=1e-9,disp=False))
  q=previous+scale*fit.x;evaluate(q)
  if np.max(cache['required']-cache['gaps'])>.00005:q=previous.copy();evaluate(q)
 wanted=tangent_path(phase);wanted/=np.linalg.norm(wanted);gaps=cache['gaps'];group={k:float(min(gaps[i] for i,pair in enumerate(pairs) if pair[2]==k)) for k in ['source','target','elbow','self']}
 hold=cache['points'][N:]-original_nodes[N:];row=dict(alpha=float(phase),head_position_error_mm=float(np.linalg.norm(cache['points'][0]-path(phase))*1000),
  head_tangent_error_deg=float(np.degrees(np.arccos(np.clip(cache['T']@wanted,-1,1)))),held_source_node_max_motion_mm=float(np.max(np.linalg.norm(hold,axis=1))*1000),
  held_source_axis_X_max_motion_mm=float(np.max(abs(hold[:,0]))*1000),joint_peak_deg=float(np.max(abs(q))*180/np.pi),
  conservative_fore_gravity_torque_peak_Nm=float(abs(cache['gravity_torques']).max()),minimum_gap_mm={k:v*1000 for k,v in group.items()},
  max_required_clearance_violation_mm=float(np.maximum(cache['required']-gaps,0.).max()*1000),optimizer_success=True if fit is None else bool(fit.success),nfev=0 if fit is None else int(fit.nfev))
 rows.append(row);qrows.append(d.qpos.copy());poserows.append(cache['points'].copy());gapsrows.append(gaps.copy());previous=q.copy()
 print(json.dumps(row),flush=True)
np.savez_compressed(P/f'FIXED_SOURCE_REACH_M{N}.npz',alpha=np.linspace(0,1,61),qpos=np.array(qrows),points=np.array(poserows),gaps=np.array(gapsrows),initial_qpos=q0,held_material_start=N)
report=dict(role='Offline prescribed-pose reachability and clearance analysis, no dynamic success claim',initial_pose='CANDIDATE_INITIAL.npz',release_materials=list(range(N)),held_source_materials=list(range(N,21)),
 released_backbone_length_m=N*.0645,retained_backbone_length_m=(21-N)*.0645,source_translation_or_rolling_applied=False,all_held_source_poses_fixed=True,
 scene_geometry_used_only_for_offline_setup_and_clearance=True,live_rough_receiver_priors_not_overwritten=True,simulation_steps=0,
 path_waypoints_m=position_knots.tolist(),position_goal_reference='M0 front node, not body center',desired_head_tip_target_m=target_tip.tolist(),desired_approach_tangent=final_T.tolist(),
 maximum_head_position_error_mm=max(r['head_position_error_mm'] for r in rows),maximum_head_tangent_error_deg=max(r['head_tangent_error_deg'] for r in rows),
 maximum_held_source_motion_mm=max(r['held_source_node_max_motion_mm'] for r in rows),maximum_joint_deg=max(r['joint_peak_deg'] for r in rows),
 maximum_conservative_fore_gravity_torque_Nm=max(r['conservative_fore_gravity_torque_peak_Nm'] for r in rows),maximum_clearance_violation_mm=max(r['max_required_clearance_violation_mm'] for r in rows),
 final_minimum_gap_mm=rows[-1]['minimum_gap_mm'],mesh_checks='Original robot cylinder/meshes against original4 pipe components and nonadjacent robot bodies; signed closest distance at61 pose samples',
 continuous_between_samples_clearance_not_yet_certified=True,original_physical_enableflags=original_enableflags,
 private_distance_enableflags=int(m.opt.enableflags),distance_query_pipeline='NativeCCD in nonintegrated geometry-query model only; original collision pipeline must separately validate saved poses',
 distance_pipeline_source='https://github.com/google-deepmind/mujoco/issues/2688',distance_gradient_error=distance_gradient_errors,rows=rows)
(P/f'FIXED_SOURCE_REACH_M{N}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
