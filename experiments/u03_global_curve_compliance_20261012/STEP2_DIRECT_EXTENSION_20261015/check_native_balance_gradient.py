"""One saved actual pose/contact directional check, zero dynamics steps."""
from pathlib import Path
import sys,json,numpy as np
P=Path(__file__).resolve().parent;B=P/sys.argv[1];C=P/'CONTROLLER_V8_1_NATIVE_CONTINUOUS_RELEASE/T_UPDATE';sys.path.insert(0,str(B))
from physics import *
from native_robot import NativeRobot
from source_balance import SourceBalance
from contact_sensor import read_sensor,known_source
r=json.loads((B/'REGISTERED.json').read_text());v=json.loads((C/'LIVE_STATE.json').read_text());records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open()];record=min(records,key=lambda x:abs(x['t']-v['t']))
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
bal=SourceBalance(m,r['source_known'],r['source_balance']);bal.baseline=bal.measure(d,read_sensor(m,d));d.qpos[:]=v['qpos'];mujoco.mj_forward(m,d)
sensor=read_sensor(m,d);sensor['joint_ctrl']=np.asarray(v['raw']);sensor['joint_torque']=np.asarray(v['torque']);sensor['contacts']=record['contacts']
for ct in sensor['contacts']:
    for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
offset,info=bal.update(d,sensor,.025,True,None,v['t']);g=info['source_balance_gradient_Nm_rad'];direction=g/max(np.linalg.norm(g),1e-12)
robot=NativeRobot(m,d);robot.observe(d,info['source_balance_anchor_material']);q=d.qpos[robot.qa].copy();origin=np.r_[0.,r['source_known']['center_yz_m']];external=[]
for ct in sensor['contacts']:
    if known_source(ct,r['source_known']):continue
    body=robot.head+ct['material'];local=d.xmat[body].reshape(3,3).T@(ct['point']-d.xpos[body]);external.append((body,local,ct['world_force_N']))
def value(command):
    robot.fk(command);pd=robot.d
    moment=sum(np.cross(pd.xipos[robot.head+j]-origin,m.body_mass[robot.head+j]*m.opt.gravity)[0] for j in range(21))
    for body,local,force in external:moment+=np.cross(pd.xpos[body]+pd.xmat[body].reshape(3,3)@local-origin,force)[0]
    return float(moment)
epsilon=1e-6;fd=(value(q+epsilon*direction)-value(q-epsilon*direction))/(2*epsilon);analytic=float(g@direction);error=abs(fd-analytic)
out=dict(simulation_steps=0,saved_actual_pose_t_s=v['t'],saved_contact_t_s=record['t'],input_source=str(C),contact_forces='Actual saved unlabeled loads, not regenerated private forward forces',analytic_directional_Nm_rad=analytic,finite_difference_Nm_rad=fd,absolute_error_Nm_rad=error,source_anchor=info['source_balance_anchor_material'],current_external_moment_Nm=info['source_balance_external_X_moment_Nm'],baseline_moment_Nm=bal.baseline,rear_gradient_Nm_rad=g[12:].tolist(),first_rate_deg_s=np.degrees(info['source_balance_rate_rad_s'][12:]).tolist(),passed=bool(error<1e-5 and np.linalg.norm(g)>1e-5))
(B/'SOURCE_BALANCE_GRADIENT_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));assert out['passed']
