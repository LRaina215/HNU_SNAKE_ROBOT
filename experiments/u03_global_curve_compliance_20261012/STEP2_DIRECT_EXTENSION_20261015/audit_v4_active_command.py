"""One saved source-stable state exercises the complete new local command path."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/(sys.argv[1] if len(sys.argv)>1 else 'CONTROLLER_V4_1_ROBOT_MASS_HOLD');C=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
reg=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);initial=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);control=DirectController(m,d,reg,'DIRECTION',True);z=np.load(C/'trace.npz');j=int(np.argmin(abs(z['t']-2.1)));d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.ctrl[:]=z['ctrl'][j];mujoco.mj_forward(m,d);head=m.body('head').id
# Install the already-preserved source-stable event to inspect ONE command;
# this is not a replay of the hold, a new initial, or a dynamics rollout.
control.source_ready=True;control.first_release_t=float(z['t'][j]);control.phase='front_release';control.last_t=float(z['t'][j]-.025);control.stable_source_P=d.xpos[head:head+21].copy();control.stable_source_R=d.xmat[head:head+21].reshape(21,3,3).copy()
before=[d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()];given=control.planner.given.copy();raw,info=control.command(d,float(z['t'][j]));checks=dict(active_allocator_valid=bool(info['source_hold_qp_valid']),no_actual_state_or_ctrl_write=all(np.array_equal(a,b) for a,b in zip(before,[d.qpos,d.qvel,d.ctrl])),immutable_given_G_input=np.array_equal(given,control.planner.given),native_integral=np.max(abs(control.planner.coordinates.hinges(info['curvature_field'])-raw))<1e-12,source_feed_roll_zero=info['S']==0 and info['source_commanded_feed_m']==0 and info['source_commanded_roll_rad']==0,command_step_inside_original2mm=info['command_maximum_C_geometry_step_m']<=.002+1e-10,predicted_torque_under2p3=float(abs(info['source_hold_total_predicted_torque_Nm']).max())<=2.30+1e-6)
out=dict(version=reg['version'],simulation_steps=0,scope='One command at preserved2.1s source-stable state; hold event deliberately installed for code-path audit, not new t0 or dynamics',checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),command_native_delta_peak_deg=float(abs(raw-before[2]).max()*180/np.pi),source_hold_predicted_peak_torque_Nm=float(abs(info['source_hold_total_predicted_torque_Nm']).max()),source_hold_QP_iterations=info['source_hold_solver_iterations'],physical_flags=int(m.opt.enableflags))
(B/'ACTIVE_SOURCE_COMMAND_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
