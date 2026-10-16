from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from snake_control.suspended_supply_transfer import nodes
from snake_control.cmu_window_gait import cylinder
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
INITIAL=ROOT/'experiments/cmu_native_underup_20261012/U03/initial.npz'

def snapshot(m,d):
    mujoco.mj_forward(m,d);p=nodes(d);c=(p[:-1]+p[1:])/2;h=m.body('head').id
    forces=np.zeros((2,21));force_world=np.zeros((2,21,3));depth=np.zeros(4)
    for i,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];bs=[int(m.geom_bodyid[g]) for g in gs];robot=[h<=b<h+21 for b in bs];names=[m.geom(g).name or '' for g in gs]
        kind=2 if all(robot) else 3 if 'floor' in names and any(robot) else None
        for j,name in enumerate(names):
            if name.startswith('pipe_') and robot[1-j]:
                kind=0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);forces[kind,bs[1-j]-h]+=max(0,f[0]);force_world[kind,bs[1-j]-h]+=(1 if 1-j==1 else -1)*ct.frame.reshape(3,3).T@f[:3]
        if kind is not None:depth[kind]=max(depth[kind],-float(ct.dist))
    return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),centers=c,points=p,forces=forces,force_world=force_world,depth=depth,torque=d.actuator_force.copy())
