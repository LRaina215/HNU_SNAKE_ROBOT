"""One nonzero source-pose-error command and independent world wrench sum."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V4_1_ROBOT_MASS_HOLD';C=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
from contact_sensor import read_sensor,known_source
import source_wrench_hold
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
solution={};original_solver=source_wrench_hold.minimize
def capture_solution(*args,**kwargs):
    result=original_solver(*args,**kwargs);solution['x']=result.x.copy();return result
source_wrench_hold.minimize=capture_solution
reg=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);initial=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);control=DirectController(m,d,reg,'DIRECTION',True);z=np.load(C/'trace.npz');ref=int(np.argmin(abs(z['t']-2.1)));j=int(np.argmin(abs(z['t']-15.)));plan=control.planner
plan.observe(z['qpos'][ref],13);plan.fk(z['qpos'][ref,plan.qa]);stableP=plan.d.xpos[plan.head:plan.head+21].copy();stableR=plan.d.xmat[plan.head:plan.head+21].reshape(21,3,3).copy()
control.source_hold.reference_q=z['qpos'][ref,plan.qa].copy();control.source_hold.reference_P=stableP.copy();control.stable_source_P=stableP;control.stable_source_R=stableR;control.source_ready=True;control.first_release_t=float(z['t'][ref]);control.phase='front_release';control.alpha=float(z['alpha'][j]);control.last_t=float(z['t'][j]-.025);control.raw=z['raw_native_command'][j].copy();control.last_geo=z['geometric_native_command'][j].copy();plan.q=control.last_geo.copy()
d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.ctrl[:]=z['ctrl'][j];mujoco.mj_forward(m,d);sensor=read_sensor(m,d);before=[d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()];given=plan.given.copy();raw,info=control.command(d,float(z['t'][j]));plan.fk(sensor['joint_actual'])
ids=np.arange(plan.head,plan.head+21);mass=plan.m.body_mass[ids];total_mass=mass.sum();com=np.sum(mass[:,None]*plan.d.xipos[ids],axis=0)/total_mass;gravity_force=(mass[:,None]*plan.m.opt.gravity).sum(0);gravity_moment=np.cross(plan.d.xipos[ids]-com,mass[:,None]*plan.m.opt.gravity).sum(0)
force=np.zeros(3);moment=np.zeros(3);selected=[ct for ct in sensor['contacts'] if ct['material']>=8 and known_source(ct,reg['source_known'])]
for i,ct in enumerate(selected):
    normal=ct['normal'];e=np.array([1.,0.,0.]);e-=normal*(normal@e)
    if np.linalg.norm(e)<1e-6:e=np.cross(normal,[0.,1.,0.])
    e/=np.linalg.norm(e);basis=np.column_stack([normal,e,np.cross(normal,e)]);f=basis@solution['x'][3*i:3*i+3];force+=f;moment+=np.cross(ct['point']-com,f)
force_residual=force+gravity_force-info['source_hold_corrective_force_N'];moment_residual=moment+gravity_moment-info['source_hold_corrective_moment_Nm'];W=total_mass*np.linalg.norm(plan.m.opt.gravity)
checks=dict(nonzero_pose_error=abs(info['source_hold_polar_error_rad'])>.01,active_allocator_valid=bool(info['source_hold_qp_valid']),world_force_balance=abs(force_residual).max()<1e-6,world_moment_balance=abs(moment_residual).max()<1e-6,robot_mass_only=abs(total_mass-2.856)<1e-10,force_cap_correct=abs(W*.1-2.801736)<1e-8,normal_cap_correct=abs(W*4-112.06944)<1e-8,no_actual_state_or_ctrl_write=all(np.array_equal(a,b) for a,b in zip(before,[d.qpos,d.qvel,d.ctrl])),immutable_given_G_input=np.array_equal(given,plan.given),native_integral=abs(plan.coordinates.hinges(info['curvature_field'])-raw).max()<1e-12,no_S_roll=info['S']==0 and info['source_commanded_feed_m']==0 and info['source_commanded_roll_rad']==0,command_step_inside_original2mm=info['command_maximum_C_geometry_step_m']<=.002+1e-10,predicted_torque_under2p3=abs(info['source_hold_total_predicted_torque_Nm']).max()<=2.30+1e-6)
out=dict(version=reg['version'],simulation_steps=0,scope='One command at preserved15s actual state with stable2.1s robot reference; explicit internal context installation, no rollout',checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),robot_mass_kg=float(total_mass),robot_weight_N=float(W),maximum_corrective_force_N=float(W*.1),maximum_sum_normal_N=float(W*4),source_polar_error_deg=float(np.degrees(info['source_hold_polar_error_rad'])),source_axial_error_mm=info['source_hold_axial_error_m']*1000,corrective_force_N=info['source_hold_corrective_force_N'].tolist(),corrective_moment_Nm=info['source_hold_corrective_moment_Nm'].tolist(),independent_world_force_residual_N=force_residual.tolist(),independent_world_moment_residual_Nm=moment_residual.tolist(),predicted_motor_torque_peak_Nm=float(abs(info['source_hold_total_predicted_torque_Nm']).max()),actual_command_step_geometry_mm=info['command_maximum_C_geometry_step_m']*1000,physical_enableflags=int(m.opt.enableflags))
(B/'NONZERO_POSE_WRENCH_COMMAND_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
