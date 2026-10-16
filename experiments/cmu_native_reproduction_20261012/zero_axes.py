from native_run import *
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[7:]=0;mujoco.mj_forward(m,d)
print('nodes',nodes(d)[:6]);print('axis',[d.joint('joint'+str(i)).xaxis.tolist() for i in range(1,7)])
print('body R',[d.xmat[m.body('head').id+i].reshape(3,3).tolist() for i in range(3)])
