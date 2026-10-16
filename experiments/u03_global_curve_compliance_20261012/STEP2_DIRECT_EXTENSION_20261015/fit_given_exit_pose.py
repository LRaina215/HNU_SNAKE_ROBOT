"""Arrange a given all-source exit pose, not a simulated approach/resume."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No dynamics')))
modelpath=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml';m=mujoco.MjModel.from_xml_path(str(modelpath));m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_NATIVECCD);d=mujoco.MjData(m)
z=np.load(P.parent/'STEP2_FROM_SOURCE_20261015/SOURCE_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
head=m.body('head').id;N=4;anchor=head+N;qa=m.jnt_qposadr[m.actuator_trnid[:,0]];q0=d.qpos.copy();oldP=d.xpos[anchor].copy();oldR=d.xmat[anchor].reshape(3,3).copy();rootR=Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_matrix();origin=np.array([0.,0.,1.])
geoms=[int(m.body_geomadr[head+i]) for i in range(21)];pipes=[m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04']]
goal=np.array([1.015,.055,.925]);wanted=np.array([.40,.75,.52]);wanted/=np.linalg.norm(wanted);cache={}
def assign(v):
 d.qpos[:]=q0;d.qpos[qa[:N]]=v[:N];mujoco.mj_kinematics(m,d);turn=Rotation.from_rotvec([v[N],0,0]).as_matrix();targetP=origin+turn@(oldP-origin)+np.array([v[N+1],0,0]);targetR=turn@oldR
 correction=targetR@d.xmat[anchor].reshape(3,3).T;translation=targetP-correction@d.xpos[anchor];d.qpos[:3]=correction@q0[:3]+translation;d.qpos[3:7]=Rotation.from_matrix(correction@rootR).as_quat()[[3,0,1,2]]
 mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
def gap(a,b):
 if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+.025:return .025
 return float(mujoco.mj_geomDistance(m,d,a,b,.025,None))
def residual(v):
 assign(v);p=nodes(d);c=.5*(p[:-1]+p[1:]);T=d.xmat[head].reshape(3,3)[:,2];rho=np.linalg.norm(c[:N,1:]-[0,1],axis=1)
 r=[(c[0]-goal)/.003,5*(T-wanted),np.minimum(rho-.085,0)/.002,np.maximum(rho-.105,0)/.002,
    np.maximum(c[:N,0]+c[:N,1]-1.1,0)/.002]
 for i in range(N):
  for k,g in enumerate(pipes):r.append(np.array([min(0.,gap(geoms[i],g)-(-.0012 if k<2 else .003))/.001]))
  for j in range(i+2,21):r.append(np.array([min(0.,gap(geoms[i],geoms[j])+.0005)/.001]))
 return np.concatenate(r)
v0=np.r_[q0[qa[:N]],-2.8,.12];bounds=(np.r_[np.full(N,-np.radians(88)),-np.pi,-.05],np.r_[np.full(N,np.radians(88)),np.pi,.25])
fit=least_squares(residual,v0,bounds=bounds,x_scale=np.r_[np.ones(N),1.,.10],diff_step=1e-5,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-9);assign(fit.x)
pose=d.qpos.copy();physical=mujoco.MjModel.from_xml_path(str(modelpath));actual=mujoco.MjData(physical);actual.qpos[:]=pose;actual.qvel[:]=0.;actual.ctrl[:]=pose[qa];mujoco.mj_forward(physical,actual)
p=nodes(actual);c=.5*(p[:-1]+p[1:]);T=actual.xmat[head].reshape(3,3)[:,2].copy();rho=np.linalg.norm(c[:,1:]-[0,1],axis=1);contact={};depth={};normal_by_material=np.zeros(21)
for index,ct in enumerate(actual.contact):
 ids=[ct.geom1,ct.geom2];robot=[head<=physical.geom_bodyid[g]<head+21 for g in ids]
 if sum(robot)!=1:continue
 k=0 if robot[0] else 1;name=physical.geom(ids[1-k]).name;f=np.zeros(6);mujoco.mj_contactForce(physical,actual,index,f);contact[name]=contact.get(name,0.)+max(0.,f[0]);depth[name]=max(depth.get(name,0.),-ct.dist);normal_by_material[int(physical.geom_bodyid[ids[k]])-head]+=max(0.,f[0])
state=np.empty(mujoco.mj_stateSize(physical,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(physical,actual,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(P/'GIVEN_EXIT_FIT.npz',state=state,qpos=pose,ctrl=actual.ctrl.copy(),points=p)
report=dict(role='Prescribed initial arrangement only, no approach motion or dynamics',source_file='Original full-source given t0 coil, no intermediate trajectory state',
    changed_initial_front_hinges=N,initial_source_axial_setup_translation_m=float(fit.x[-1]),initial_source_roll_setup_deg=float(np.degrees(fit.x[-2])),
    head_center=c[0].tolist(),head_tip=p[0].tolist(),head_tangent=T.tolist(),head_position_goal_error_mm=float(np.linalg.norm(c[0]-goal)*1000),head_tangent_goal_error_deg=float(np.degrees(np.arccos(np.clip(T@wanted,-1,1)))),
    all21_source_radial_envelope=bool(np.all((rho>=.065)&(rho<=.13))),all21_centers_before_source_miter=bool(np.all(c[:,0]+c[:,1]<1.1)),
    source_center_radii_m=rho.tolist(),source_max_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[:,2]-1,c[:,1])))*180/np.pi),
    actual_joint_peak_deg=float(abs(pose[qa]).max()*180/np.pi),actual_front_hinge_deg=np.degrees(pose[qa[:N]]).tolist(),original_collision_contact_normal_N=contact,original_collision_depth_mm={k:v*1000 for k,v in depth.items()},
    original_physical_flags=int(physical.opt.enableflags),optimizer_nfev=int(fit.nfev),optimizer_success=bool(fit.success),simulation_steps=0,accepted_initial=False,
    note='Must still pass original-scene clearance, support/stability and stationary-source direct reach before selecting this initial')
(P/'GIVEN_EXIT_FIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
