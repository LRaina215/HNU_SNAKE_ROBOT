from run import *
m=mujoco.MjModel.from_xml_string(make_model());d=mujoco.MjData(m);d.qpos[3]=1
for phase in np.linspace(0,2*np.pi,13):
    d.qpos[7:]=helix(1.42,.23089076,phase);mujoco.mj_forward(m,d)
    a,o,r,p,rms=cylinder(centers(m,d))
    print(round(phase,3),round(r,4),round(p,4),round(rms,4))
print('Head virtual full helix registered at carrying module M4; geometry diagnostic only')
for polar in [0,np.pi/2,np.pi,3*np.pi/2]:
    src,s=initial(m,1.42,.23089076,0,4,-.18,polar)
    d.qpos[:]=src.qpos;d.qpos[7:]=helix(1.42,.23089076,np.pi/2);mujoco.mj_forward(m,d)
    a,o,r,p,rms=cylinder(centers(m,d));b=m.body('M5').id
    R=src.xmat[b].reshape(3,3)@d.xmat[b].reshape(3,3).T
    print('polar',polar,'M4axis',src.xmat[b].reshape(3,3)[:,0],'head helix axis',R@a)
print('two-window direct geometry with only rigid carrying-frame registration')
src,s=initial(m,1.42,.23089076,0,4,-.18,0)
for nt in [0,1,2,3,4,5,6]:
    d.qpos[:]=src.qpos;d.qpos[7:],L=window(1.42,.23089076,np.pi/2,nt,4)
    mujoco.mj_forward(m,d);b=m.body(f'M{int(nt)+1}').id
    R=src.xmat[b].reshape(3,3)@d.xmat[b].reshape(3,3).T
    cc=(centers(m,d)-d.xpos[b])@R.T+src.xpos[b]
    print(nt,'head',np.round(cc[0],3),'carrying',np.round(src.xpos[b],3))
