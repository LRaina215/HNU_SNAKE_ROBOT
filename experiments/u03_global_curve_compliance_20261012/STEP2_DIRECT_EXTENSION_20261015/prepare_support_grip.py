"""Arrange source-side preload in the given pose; head/fore pose unchanged."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.optimize import least_squares
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
modelpath=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml';m=mujoco.MjModel.from_xml_path(str(modelpath));m.opt.enableflags|=64;d=mujoco.MjData(m);initial=np.load(P/'CANDIDATE_INITIAL.npz');mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
qa=m.jnt_qposadr[m.actuator_trnid[:,0]];q0=d.qpos.copy();ctrl0=d.ctrl.copy();head=m.body('head').id;geoms=[int(m.body_geomadr[head+j]) for j in range(21)];source=m.geom('pipe_straight_01').id;target_materials=[12,14,16,18,20]
oldpoints=nodes(d).copy();oldcenters=.5*(oldpoints[:-1]+oldpoints[1:]);selected=np.arange(8,20)
def assign(v):d.qpos[:]=q0;d.qpos[qa[selected]]+=v;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
def distance(a,b):return min(.02,float(mujoco.mj_geomDistance(m,d,a,b,.02,None)))
def residual(v):
 assign(v);p=nodes(d);centers=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(centers[8:,1:]-[0,1],axis=1)
 r=[np.array([distance(source,geoms[j])+.00035 for j in target_materials])/.0004,
   (centers[8:,0]-oldcenters[8:,0])/.003,(centers[8:,1:]-oldcenters[8:,1:]).ravel()/.015,v/.12,
   np.minimum(rho-.065,0)/.001,np.maximum(rho-.13,0)/.001]
 for j in range(8,21):r.append(np.array([min(0.,distance(source,geoms[j])+.0011)/.0005]))
 return np.concatenate(r)
limit=np.radians(8.);lower=np.maximum(-limit,-np.radians(88)-q0[qa[selected]]);upper=np.minimum(limit,np.radians(88)-q0[qa[selected]])
fit=least_squares(residual,np.zeros(12),bounds=(lower,upper),diff_step=1e-5,max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-9);assign(fit.x);newpose=d.qpos.copy()
physical=mujoco.MjModel.from_xml_path(str(modelpath));actual=mujoco.MjData(physical);actual.qpos[:]=newpose;actual.qvel[:]=0.;actual.ctrl[:]=newpose[qa]+(ctrl0-q0[qa]);mujoco.mj_forward(physical,actual)
state=np.empty(mujoco.mj_stateSize(physical,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(physical,actual,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(P/'SUPPORT_GIVEN_INITIAL.npz',state=state,qpos=newpose,ctrl=actual.ctrl.copy(),points=nodes(actual).copy())
trace=np.load(P/'FIXED_SOURCE_REACH_M8.npz');poses=trace['qpos'].copy();poses[:,qa[8:]]=newpose[qa[8:]];allpoints=[]
for pose in poses:actual.qpos[:]=pose;mujoco.mj_kinematics(physical,actual);allpoints.append(nodes(actual).copy())
np.savez_compressed(P/'SUPPORT_FIXED_SOURCE_REACH_M8.npz',alpha=trace['alpha'],qpos=poses,points=np.array(allpoints),initial_qpos=newpose,held_material_start=8)
report=dict(role='Given source grip arrangement only, not dynamic formation',simulation_steps=0,source_geometry_used_only=True,
 head_pose_unchanged=bool(np.array_equal(newpose[:7],q0[:7]) and np.array_equal(newpose[qa[:8]],q0[qa[:8]])),
 posterior_hinge_setup_delta_deg=np.degrees(fit.x).tolist(),maximum_posterior_setup_delta_deg=float(np.max(abs(fit.x))*180/np.pi),
 source_contact_targets=target_materials,target_gaps_mm=[distance(source,geoms[j])*1000 for j in target_materials],optimizer_success=bool(fit.success),optimizer_nfev=int(fit.nfev),
 accepted_initial=False,note='Support wrench feasibility, original collision/clearance and static stability still required; original head reach sequence retained because q0..7 and M8 pose unchanged')
(P/'SUPPORT_GIVEN_INITIAL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
