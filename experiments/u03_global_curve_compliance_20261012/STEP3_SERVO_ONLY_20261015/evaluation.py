"""Scene truth ONLY for logging, original safety and offline acceptance."""
from physics import *
def snapshot(m,d):
    mujoco.mj_forward(m,d);p=nodes(d);c=(p[:-1]+p[1:])/2;h=m.body('head').id
    forces=np.zeros((2,21));force_world=np.zeros((2,21,3));depth=np.zeros(4);split_normal=np.zeros((4,21));split_world=np.zeros((4,21,3));split_depth=np.zeros(4)
    for i,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];bs=[int(m.geom_bodyid[g]) for g in gs];robot=[h<=b<h+21 for b in bs];names=[m.geom(g).name or '' for g in gs]
        kind=2 if all(robot) else 3 if 'floor' in names and any(robot) else None
        for j,name in enumerate(names):
            if name.startswith('pipe_') and robot[1-j]:
                kind=0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);forces[kind,bs[1-j]-h]+=max(0,f[0]);force_world[kind,bs[1-j]-h]+=(1 if 1-j==1 else -1)*ct.frame.reshape(3,3).T@f[:3]
                component=['pipe_straight_01','pipe_sharp_in_02','pipe_straight_04','pipe_sharp_out_03'].index(name)
                split_depth[component]=max(split_depth[component],-float(ct.dist))
                split_normal[component,bs[1-j]-h]+=max(0.,f[0]);split_world[component,bs[1-j]-h]+=(1 if 1-j==1 else -1)*ct.frame.reshape(3,3).T@f[:3]
        if kind is not None:depth[kind]=max(depth[kind],-float(ct.dist))
    tangent=p[:-1]-p[1:];tangent/=np.linalg.norm(tangent,axis=1)[:,None];spin=[]
    for j in range(21):
        velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,h+j,velocity,0);spin.append(velocity[:3]@tangent[j])
    return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),centers=c,points=p,forces=forces,force_world=force_world,depth=depth,torque=d.actuator_force.copy(),
        eval_pipe_normal_N=split_normal,eval_pipe_world_N=split_world,eval_pipe_depth_m=split_depth,actual_axial_spin_rad_s=np.array(spin))

