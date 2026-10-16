"""Freeze candidate data for review, without authorizing any dynamics."""
from pathlib import Path
import sys,json,shutil,numpy as np,mujoco
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m);z=np.load(P/'STATIC_READY_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
qa=m.jnt_qposadr[m.actuator_trnid[:,0]];pose=d.qpos.copy();p=nodes(d);c=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(c[:,1:]-[0,1],axis=1);h=m.body('head').id
normal=np.zeros((4,21));world=np.zeros((4,21,3));depth=np.zeros(6);names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04']
for index,ct in enumerate(d.contact):
 gs=[ct.geom1,ct.geom2];robot=[h<=m.geom_bodyid[g]<h+21 for g in gs]
 if all(robot):depth[4]=max(depth[4],-ct.dist);continue
 if sum(robot)!=1:continue
 j=1 if robot[1] else 0;other=m.geom(gs[1-j]).name
 if other not in names:
  if other=='floor':depth[5]=max(depth[5],-ct.dist)
  continue
 k=names.index(other);material=int(m.geom_bodyid[gs[j]])-h;f=np.zeros(6);mujoco.mj_contactForce(m,d,index,f);normal[k,material]+=max(0.,f[0]);world[k,material]+=(1 if j==1 else -1)*ct.frame.reshape(3,3).T@f[:3];depth[k]=max(depth[k],-ct.dist)
source=(rho>=.065)&(rho<=.13)&(c[:,0]>-.2)&(c[:,0]+c[:,1]<1.1)
proof=dict(version='DIRECT_GIVEN_INITIAL_REVIEW_CANDIDATE_V1',role='Prescribed all-source initial, never a dynamic/intermediate state',dynamics_authorized=False,independent_initial_review_passed=False,
 original_20_hinges_and_physics=True,original_collision_enableflags=int(m.opt.enableflags),simulation_steps=0,
 source_branch_definition='Original source straight plus source inlet miter segment. Straight ends X=.95m; source inlet ends on X+Y=1.1m. Receiver outlet/straight are excluded.',
 all21_source_branch_envelope=bool(source.all()),source_straight_center_count=int((c[:,0]<.95).sum()),source_inlet_center_count=int((c[:,0]>=.95).sum()),
 all21_source_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[:,2]-1,c[:,1])))*180/np.pi),retained_M8to20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(c[8:,2]-1,c[8:,1])))*180/np.pi),
 head_M0_center_m=c[0].tolist(),head_M0_front_node_m=p[0].tolist(),head_initial_tangent=((p[0]-p[1])/.0645).tolist(),
 source_axis=[1.,0.,0.],source_axis_point_m=[0.,0.,1.],source_radius_m=.0565,receiver_axis=[0.,1.,0.],receiver_entry_axis_point_m=[1.1,.20,1.],
 source_center_radius_m=rho.tolist(),source_center_X_m=c[:,0].tolist(),all_robot_centers_before_receiver_straight=bool(np.max(c[:,1])<.15),
 component_order=names,initial_component_normal_N=normal.sum(1).tolist(),initial_component_vertical_N=world[:,:,2].sum(1).tolist(),
 receiver_outlet_and_straight_normal_N=float(normal[2:].sum()),initial_depths_pipe_self_floor_mm=(depth*1000).tolist(),
 joint_peak_deg=float(np.max(abs(d.qpos[qa]))*180/np.pi),initial_torque_peak_Nm=float(abs(d.actuator_force).max()),qvel_norm=float(np.linalg.norm(d.qvel)),
 initial_qacc_max=float(abs(d.qacc).max()),exact_static_equilibrium=False,logged_source_stability_gate_still_required=True,
 release_materials=list(range(8)),retained_source_materials=list(range(8,21)),front_release_length_m=8*.0645,retained_source_length_m=13*.0645,
 source_preacquisition_feed_and_roll_zero=True,control_phase_requirement='Settle -> front release/contact/probe/wrap with stationary source support -> after capture whole-body feed and rolling. Alpha and post-capture S must be distinct.',
 acceptance_status='Geometry/support review candidate; no full-action or dynamic-stability success claim')
shutil.copy2(P/'STATIC_READY_INITIAL.npz',P/'GIVEN_INITIAL.npz')
trace=np.load(P/'FIXED_SOURCE_REACH_M8.npz');poses=trace['qpos'].copy();poses[:,qa[8:]]=pose[qa[8:]];points=[]
for row in poses:d.qpos[:]=row;mujoco.mj_kinematics(m,d);points.append(nodes(d).copy())
np.savez_compressed(P/'DIRECT_CONTACT_PATH.npz',alpha=trace['alpha'],qpos=poses,points=np.array(points),initial_qpos=pose,held_material_start=8)
(P/'GIVEN_INITIAL_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
