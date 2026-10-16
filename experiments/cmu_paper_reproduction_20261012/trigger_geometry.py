from run import *
from initial_audit import distances
m=mujoco.MjModel.from_xml_string(make_model());d=mujoco.MjData(m)
best=[]
for phase in np.linspace(0,2*np.pi,49)[:-1]:
    src,s=initial(m,1.42,.23089076,phase,4,-.18,0)
    b=m.body('M3').id;ax=src.xmat[b].reshape(3,3)[:,0]
    polar=-np.pi/2-np.arctan2(ax[2],ax[1])
    src,s=initial(m,1.42,.23089076,phase,4,-.18,polar)
    d.qpos[:]=src.qpos;d.qpos[7:]=helix(1.42,.23089076,phase+np.pi/2);mujoco.mj_forward(m,d)
    a,o,r,p,e=cylinder(centers(m,d));b=m.body('M4').id;R=src.xmat[b].reshape(3,3)@d.xmat[b].reshape(3,3).T;aa=R@a
    ax=src.xmat[m.body('M3').id].reshape(3,3)[:,0]
    best.append(dict(phase=float(phase),polar=float(polar),selected_axis=ax.tolist(),selected_center=centers(m,src)[4].tolist(),virtual_head_axis=aa.tolist(),cost=float(abs(ax[0])+abs(aa[0])+abs(aa[2]))))
best.sort(key=lambda x:x['cost']);(BASE/'trigger_geometry_scan.json').write_text(json.dumps(best,indent=2))
for x in best[:6]:print(x)
