"""Adaptive native-hinge sweep bounds, with original collision cross-check."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,numpy as np,mujoco
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
modelpath=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml';m=mujoco.MjModel.from_xml_path(str(modelpath));physical=mujoco.MjModel.from_xml_path(str(modelpath));m.opt.enableflags|=64;d=mujoco.MjData(m);actual=mujoco.MjData(physical)
z=np.load(P/'DIRECT_CONTACT_PATH.npz');initial=np.load(P/'GIVEN_INITIAL.npz');q0=initial['qpos'];qa=m.jnt_qposadr[m.actuator_trnid[:,0]];head=m.body('head').id;anchor=head+8;d.qpos[:]=q0;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
anchorP=d.xpos[anchor].copy();anchorR=d.xmat[anchor].reshape(3,3).copy();rootR=Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_matrix();reference=nodes(d).copy()
geoms=[int(m.body_geomadr[head+j]) for j in range(21)];pipes=[m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04']]
pairs=[(geoms[i],g,i,-1,'source' if k<2 else 'outlet' if k==2 else 'receiver') for i in range(8) for k,g in enumerate(pipes)]
pairs += [(geoms[i],geoms[j],i,j,'self') for i in range(8) for j in range(i+2,21)]
cache={};maximum_original_depth=np.zeros(6);worst_original_contact=[None]*6;max_hold=0.;first_original_receiver=None;worst_gaps={k:float('inf') for k in ['source','outlet','receiver','self']};start=time.time()
def evaluate(alpha):
 global max_hold,first_original_receiver
 key=round(float(alpha),14)
 if key in cache:return cache[key]
 q=np.array([np.interp(alpha,z['alpha'],z['qpos'][:,j]) for j in qa[:8]]);d.qpos[:]=q0;d.qpos[qa[:8]]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
 turn=anchorR@d.xmat[anchor].reshape(3,3).T;translation=anchorP-turn@d.xpos[anchor];d.qpos[:3]=turn@q0[:3]+translation;d.qpos[3:7]=Rotation.from_matrix(turn@rootR).as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
 gaps=[]
 for a,b,i,j,kind in pairs:
  if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+.03:gap=.03
  else:gap=min(.03,float(mujoco.mj_geomDistance(m,d,a,b,.03,None)))
  gaps.append(gap);worst_gaps[kind]=min(worst_gaps[kind],gap)
 actual.qpos[:]=d.qpos;mujoco.mj_fwdPosition(physical,actual)
 max_hold=max(max_hold,float(np.max(np.linalg.norm(nodes(actual)[8:]-reference[8:],axis=1))))
 for ct in actual.contact:
  gs=[ct.geom1,ct.geom2];robot=[head<=physical.geom_bodyid[g]<head+21 for g in gs]
  if all(robot):kind=4
  elif sum(robot)==1:
   other=physical.geom(gs[0] if not robot[0] else gs[1]).name
   kind=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','self','floor'].index(other) if other in ['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor'] else None
  else:kind=None
  if kind is not None:
   if -ct.dist>maximum_original_depth[kind]:
    robot_materials=[int(physical.geom_bodyid[g])-head for g in gs if head<=physical.geom_bodyid[g]<head+21]
    native=min(.03,float(mujoco.mj_geomDistance(m,d,int(gs[0]),int(gs[1]),.03,None)))
    worst_original_contact[kind]=dict(alpha=float(alpha),material=robot_materials,depth_mm=-float(ct.dist)*1000,native_same_pair_gap_mm=native*1000,geom_ids=[int(v) for v in gs])
   maximum_original_depth[kind]=max(maximum_original_depth[kind],-ct.dist)
   if kind==3 and ct.dist<0 and (first_original_receiver is None or alpha<first_original_receiver):first_original_receiver=float(alpha)
 row=dict(alpha=float(alpha),q=q,pos=d.geom_xpos[geoms].copy(),R=d.geom_xmat[geoms].reshape(21,3,3).copy(),gaps=np.array(gaps),points=nodes(d).copy())
 cache[key]=row
 if len(cache)%1000==0:print(json.dumps(dict(evaluated_poses=len(cache),wall_s=time.time()-start)),flush=True)
 return row
certified=[];failed=[];maximum_depth=0
def certify(a,b,depth=0):
 global maximum_depth
 maximum_depth=max(maximum_depth,depth);A=evaluate(a);B=evaluate(b);delta=abs(B['q']-A['q']);motion=np.zeros(21)
 for i in range(8):
  g=geoms[i];angle=float(np.arccos(np.clip((np.trace(A['R'][i].T@B['R'][i])-1)/2,-1,1)))
  endpoint=float(np.linalg.norm(B['pos'][i]-A['pos'][i])+m.geom_rbound[g]*angle)
  length=(8-i)*.0645+np.linalg.norm(m.geom_pos[g])+m.geom_rbound[g]
  motion[i]=endpoint+length*float(delta[i:].sum())**2/8
 lower=[];limits=[]
 for index,(_,_,i,j,kind) in enumerate(pairs):
  lower.append(max(A['gaps'][index],B['gaps'][index])-motion[i]-(motion[j] if j>=0 else 0.)-2e-6)
  if kind=='source':limit=-.0013
  elif kind=='self':limit=0.
  elif kind=='receiver' and i==0:limit=-.00035 if a>=.95-1e-12 else 0.
  else:limit=.0002
  limits.append(limit)
 lower=np.array(lower);limits=np.array(limits)
 if np.all(lower>=limits):certified.append(dict(start=a,end=b,depth=depth,minimum_bound_slack_m=float((lower-limits).min())));return
 if depth>=20 or b-a<1e-8:
  k=int(np.argmin(lower-limits));failed.append(dict(start=a,end=b,pair=pairs[k],bound_m=float(lower[k]),required_m=float(limits[k])));return
 mid=(a+b)/2;certify(a,mid,depth+1);certify(mid,b,depth+1)
for a,b in zip(z['alpha'][:-1],z['alpha'][1:]):certify(float(a),float(b))
initial_row=evaluate(0.);final=evaluate(1.);held_deviation=max_hold*1000;finalT=(final['points'][0]-final['points'][1])/.0645
physical_safety=bool(max(maximum_original_depth[:4])<=.003 and maximum_original_depth[4]<=.003 and maximum_original_depth[5]==0.)
out=dict(role='Numerical continuous-sweep bound for piecewise-linear native hinge interpolation, fixed M8 boundary; no dynamics',
 original_physics_enableflags=int(physical.opt.enableflags),private_distance_enableflags=int(m.opt.enableflags),simulation_steps=0,
 certificate_passed=not failed and physical_safety,native_geometry_sweep_passed=not failed,original_pipeline_safety_passed=physical_safety,evaluated_poses=len(cache),certified_subintervals=len(certified),maximum_subdivision_depth=maximum_depth,
 certified_alpha_measure=float(sum(v['end']-v['start'] for v in certified)),maximum_fixed_source_node_motion_mm=held_deviation,
 signed_distance_uncertainty_m=2e-6,source_fore_allowed_penetration_m=.0013,receiver_head_terminal_contact_allowed_m=.00035,
 original_collision_max_depth_component_order=['source_straight','source_inlet','receiver_outlet','receiver_straight','self','floor'],
 original_collision_max_depth_mm=(maximum_original_depth*1000).tolist(),worst_original_contacts=worst_original_contact,original_first_receiver_contact_alpha=first_original_receiver,
 minimum_evaluated_pair_gaps_mm={k:v*1000 for k,v in worst_gaps.items()},final_M0_tip_m=final['points'][0].tolist(),final_M0_tangent=finalT.tolist(),
 method='Each interval bounds rigid-geom endpoint displacement plus L*(sum|dq|)^2/8 serial-chain curvature remainder. Signed convex clearance is Lipschitz under this body motion bound; adaptive subdivisions close all gaps. Original collision pipeline cross-checked at all evaluations.',
 geometry_query_caveat='NativeCCD is used only for geometric distances because legacy MPR distance queries are unreliable. Actual physics model/XML/flags remain unchanged.',
 source='https://github.com/google-deepmind/mujoco/issues/2688',failures=failed,wall_s=time.time()-start)
(P/'DIRECT_CLEARANCE_CERTIFICATE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
