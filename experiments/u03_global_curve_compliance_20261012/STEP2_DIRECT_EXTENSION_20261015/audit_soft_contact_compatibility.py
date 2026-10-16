"""Read-only source-only soft-contact check against the controller's rigid tasks."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
from source_wrench_hold import SourceWrenchHold
from contact_sensor import read_sensor,known_source
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');ref=int(np.argmin(abs(z['t']-2.1)));records=[]
for target in [2.1,5.,10.,13.9]:
    j=int(np.argmin(abs(z['t']-target)));plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),z['qpos'][0],z['ctrl'][0],reg['kinematic_solver']);hold=SourceWrenchHold(plan,reg['source_hold'],reg['source_known']);m=plan.legacy_m;d=mujoco.MjData(m);head=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    def install(index):
        d.qpos[:]=z['qpos'][index];d.qvel[:]=z['qvel'][index];d.ctrl[:]=z['ctrl'][index];mujoco.mj_forward(m,d);return read_sensor(m,d)
    sensor=install(ref);anchor=max((ct for ct in sensor['contacts'] if ct['material']>=8 and known_source(ct,reg['source_known'])),key=lambda ct:ct['normal_N'])['material'];plan.observe(d.qpos.copy(),anchor);hold.update(sensor,z['geometric_native_command'][ref],8,.025,float(z['t'][ref]))
    sensor=install(j);anchor=max((ct for ct in sensor['contacts'] if ct['material']>=8 and known_source(ct,reg['source_known'])),key=lambda ct:ct['normal_N'])['material'];plan.observe(d.qpos.copy(),anchor)
    for a in hold.anchors.values():a['last_t']=float(z['t'][j]-.025)
    previous=d.qpos.copy();mujoco.mj_integratePos(m,previous,d.qvel,-.025);hold.previous_qpos=previous;torque,info=hold.update(sensor,z['geometric_native_command'][j],8,.025,float(z['t'][j]));qacc_model=info['source_hold_requested_qacc'].copy()
    d.ctrl[:]=d.qpos[qa]+torque/127.;mujoco.mj_forward(m,d);qacc_soft=d.qacc.copy();acc_errors=[]
    for body in range(head+8,head+21):
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,body);acc_errors.append(np.linalg.norm(jp@(qacc_soft-qacc_model)))
    row=dict(t=float(z['t'][j]),controller_compatibility_valid=bool(info['source_hold_qp_valid']),torque_delivery_error_Nm=float(abs(d.actuator_force-torque).max()),maximum_source_body_acceleration_prediction_error_m_s2=float(max(acc_errors)),generalized_acceleration_prediction_error=float(abs(qacc_soft-qacc_model).max()),actual_model_enableflags=int(m.opt.enableflags),actual_source_cone=int(m.opt.cone),impratio=float(m.opt.impratio),noslip_iterations=int(m.opt.noslip_iterations))
    # At the fixed pose, ask original soft inverse dynamics for zero velocity
    # and acceleration. Joint torques cannot supply the free-root residual.
    d.qvel[:]=0.;d.qacc[:]=0.;mujoco.mj_inverse(m,d);position=d.qpos.copy();rotated=position.copy();O=np.array([0.,0.,1.]);eps=1e-6;R=Rotation.from_rotvec([eps,0,0]);rotated[:3]=O+R.apply(position[:3]-O);rotated[3:7]=(R*Rotation.from_quat(position[3:7][[1,2,3,0]])).as_quat()[[3,0,1,2]];mode=np.empty(m.nv);mujoco.mj_differentiatePos(m,mode,eps,position,rotated)
    friction=[];normal_moment=0.
    for k,ct in enumerate(d.contact):
        gs=[int(ct.geom1),int(ct.geom2)];robot=[head<=m.geom_bodyid[g]<head+21 for g in gs]
        if sum(robot)!=1:continue
        side=int(robot[1]);force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force);friction.extend(force[1:]);world=(1 if side else -1)*ct.frame.reshape(3,3).T@np.r_[force[0],0.,0.];normal_moment+=np.cross(ct.pos-O,world)[0]
    mass=m.body_mass[head:head+21];gravity_moment=float(sum(np.cross(d.xipos[head+i]-O,mass[i]*m.opt.gravity)[0] for i in range(21)));row.update(zero_velocity_acceleration_inverse_root_roll_residual_Nm=float(mode@d.qfrc_inverse),zero_velocity_inverse_friction_peak=float(abs(np.asarray(friction)).max(initial=0.)),normal_contact_moment_about_source_X_Nm=float(normal_moment),gravity_moment_about_source_X_Nm=gravity_moment,actuator_virtual_work_on_rigid_source_roll_mode=float(abs(mode[m.jnt_dofadr[m.actuator_trnid[:,0]]]).max()),minimum_constraint_regularizer=float(d.efc_R.min()),maximum_constraint_regularizer=float(d.efc_R.max()))
    records.append(row)
out=dict(simulation_steps=0,scope='Original-flag source-only model, original robot and known source; no true receiver geometry or dynamics rollout. Compare exact delivered controller torque through actual soft contact forward dynamics; fixed-pose zero-velocity inverse is a necessary check at that pose only.',records=records,source='https://mujoco.readthedocs.io/en/3.2.7/computation/#parameters',limits='Nonzero inverse root-roll residual rules out zero-velocity/zero-acceleration equilibrium at that exact pose with joint actuation only. It does not prove all reachable fore/body shapes or the given placement cannot meet the finite12mm stationary-source tolerance. Soft contact/model mismatch is established here; its share in the whole observed failure remains unisolated.')
(C/'SOFT_CONTACT_COMPATIBILITY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
