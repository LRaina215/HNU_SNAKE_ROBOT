"""One nonzero saved-state command: independent world momentum and fore-chain checks."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/(sys.argv[1] if len(sys.argv)>1 else 'CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS');C=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
from contact_sensor import read_sensor,known_source
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);initial=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);control=DirectController(m,d,r,'DIRECTION',True);plan=control.planner;z=np.load(C/'trace.npz');ref=int(np.argmin(abs(z['t']-2.1)));j=int(np.argmin(abs(z['t']-15.)))
def install(index):
    d.qpos[:]=z['qpos'][index];d.qvel[:]=z['qvel'][index];d.ctrl[:]=z['ctrl'][index];mujoco.mj_forward(m,d)
install(ref);sensor=read_sensor(m,d);plan.observe(d.qpos.copy(),13);plan.fk(sensor['joint_actual']);stableP=plan.d.xpos[plan.head:plan.head+21].copy();stableR=plan.d.xmat[plan.head:plan.head+21].reshape(21,3,3).copy();_,initial_info=control.source_hold.update(sensor,z['geometric_native_command'][ref],8,.025,float(z['t'][ref]))
install(j);sensor=read_sensor(m,d);control.stable_source_P=stableP;control.stable_source_R=stableR;control.source_ready=True;control.first_release_t=float(z['t'][ref]);control.phase='front_release';control.alpha=float(z['alpha'][j]);control.last_t=float(z['t'][j]-.025);control.raw=z['raw_native_command'][j].copy();control.last_geo=z['geometric_native_command'][j].copy();plan.q=control.last_geo.copy()
# Install persistent known-source anchors, not a continuous rollout.
for anchor in control.source_hold.anchors.values():anchor['last_t']=float(z['t'][j]-.025)
previous=d.qpos.copy();mujoco.mj_integratePos(m,previous,d.qvel,-.025);control.source_hold.previous_qpos=previous
before=[d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()];given=plan.given.copy();raw,info=control.command(d,float(z['t'][j]));plan.fk(sensor['joint_actual']);pm=plan.m;pd=plan.d;position=pd.qpos.copy();qv=z['qvel'][j];qa=info['source_hold_requested_qacc'];ids=np.arange(plan.head,plan.head+21);mass=pm.body_mass[ids];com=np.sum(mass[:,None]*pd.xipos[ids],axis=0)/mass.sum();gravity_force=(mass[:,None]*pm.opt.gravity).sum(0);gravity_moment=np.cross(pd.xipos[ids]-com,mass[:,None]*pm.opt.gravity).sum(0);states=[]
for body,weight in zip(ids,mass):
    jp=np.zeros((3,pm.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(pm,pd,jp,jr,int(body));R=pd.ximat[body].reshape(3,3);states.append((jp,jr,pd.xipos[body].copy(),R@np.diag(pm.body_inertia[body])@R.T,weight))
h=1e-4;mujoco.mj_integratePos(pm,pd.qpos,qv,h);mujoco.mj_kinematics(pm,pd);mujoco.mj_comPos(pm,pd);inertial_force=np.zeros(3);inertial_moment=np.zeros(3)
for body,(jp,jr,point,I,weight) in zip(ids,states):
    jp1=np.zeros((3,pm.nv));jr1=np.zeros_like(jp1);mujoco.mj_jacBodyCom(pm,pd,jp1,jr1,int(body));a=jp@qa+(jp1-jp)@qv/h;alpha=jr@qa+(jr1-jr)@qv/h;omega=jr@qv;inertial_force+=weight*a;inertial_moment+=np.cross(point-com,weight*a)+I@alpha+np.cross(omega,I@omega)
pd.qpos[:]=position;mujoco.mj_kinematics(pm,pd);mujoco.mj_comPos(pm,pd);best={}
for index,ct in enumerate(sensor['contacts']):
    i=ct['material']
    if i>=8 and known_source(ct,r['source_known']) and (i not in best or ct['normal_N']>sensor['contacts'][best[i]]['normal_N']):best[i]=index
chosen=set(best.values());force=np.zeros(3);moment=np.zeros(3)
for index,ct in enumerate(sensor['contacts']):
    f=info['source_hold_allocated_force_N'][ct['material']] if index in chosen else ct['world_force_N'];point=ct['point']-.5*ct['penetration_m']*ct['normal'];force+=f;moment+=np.cross(point-com,f)
force_error=force+gravity_force-inertial_force;moment_error=moment+gravity_moment-inertial_moment;preload=plan.coordinates.hinges(info['preload_curvature_field']);expected=plan.gravity_torque(info['geometric_native_command'],8)/127.
checks=dict(initial_allocator_valid=bool(initial_info['source_hold_qp_valid']),nonzero_contact_position_error=np.linalg.norm(info['source_hold_contact_position_error_m'],axis=1).max()>.001,active_allocator_valid=bool(info['source_hold_qp_valid']),robot_mass_correct=abs(mass.sum()-2.856)<1e-10,world_linear_momentum_balance=abs(force_error).max()<1e-4,world_angular_momentum_balance=abs(moment_error).max()<1e-4,full_inverse_dynamics=abs(info['source_hold_dynamics_residual']).max()<1e-4,front_actuation_used_as_observed_input=abs(info['source_hold_total_predicted_torque_Nm'][:8]-sensor['joint_torque'][:8]).max()<1e-5,front_gravity_preload_unchanged=abs(preload[:8]-expected[:8]).max()<1e-10,posterior_torque_bound=abs(info['source_hold_total_predicted_torque_Nm'][8:]).max()<=2.300001,no_actual_state_or_ctrl_write=all(np.array_equal(a,b) for a,b in zip(before,[d.qpos,d.qvel,d.ctrl])),G_given_input_unchanged=np.array_equal(given,plan.given),native_integral=abs(plan.coordinates.hinges(info['curvature_field'])-raw).max()<1e-12,source_feed_roll_zero=info['S']==0 and info['source_commanded_feed_m']==0 and info['source_commanded_roll_rad']==0,original_2mm_command_bound=info['command_maximum_C_geometry_step_m']<=.002+1e-10)
out=dict(version=r['version'],simulation_steps=0,scope='One15s saved state with persistent2.1s source-anchor context installed; does not reproduce or prove dynamics',passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},independent_world_force_residual_N=force_error.tolist(),independent_world_moment_residual_Nm=moment_error.tolist(),contact_jacobian_rank=int(info['source_hold_contact_rank']),contact_projection_acceleration_residual_m_s2=info['source_hold_contact_projection_residual_m_s2'],posterior_torque_peak_Nm=float(abs(info['source_hold_total_predicted_torque_Nm'][8:]).max()),maximum_contact_position_error_mm=float(np.linalg.norm(info['source_hold_contact_position_error_m'],axis=1).max()*1000),QP_iterations=info['source_hold_solver_iterations'],maximum_requested_generalized_acceleration=float(abs(qa).max()),physical_flags=int(m.opt.enableflags))
(B/'DYNAMIC_COMMAND_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
