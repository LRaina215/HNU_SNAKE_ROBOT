"""Robot pose + unlabeled environment contacts. No environment names/meshes."""
from physics import np,mujoco,nodes

def read_sensor(m,d):
    p=nodes(d);h=m.body('head').id; contacts=[]
    for k,ct in enumerate(d.contact):
        b=[int(m.geom_bodyid[g])-h for g in [ct.geom1,ct.geom2]]
        robot=[0<=v<21 for v in b]
        if sum(robot)!=1:continue
        j=int(robot[1]);force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
        if force[0]<.02:continue
        sign=1 if j==1 else -1
        normal=sign*ct.frame[:3].copy();world_force=sign*ct.frame.reshape(3,3).T@force[:3]
        # Contact midpoint +/- signed gap/2 -> environment surface estimate.
        point=ct.pos-.5*float(ct.dist)*normal
        contacts.append(dict(material=b[j],point=point,normal=normal,world_force_N=world_force,normal_N=float(max(0.,force[0])),penetration_m=max(0.,-float(ct.dist))))
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    return dict(points=p,centers=.5*(p[:-1]+p[1:]),contacts=contacts,
        joint_actual=d.qpos[qa].copy(),joint_ctrl=d.ctrl.copy(),joint_torque=d.actuator_force.copy())

def known_source(contact,source):
    p=contact['point']; n=contact['normal'];r=np.linalg.norm(p[1:]-source['center_yz_m'])
    lo,hi=source['axial_bounds_m']; wall=lo-.005<=p[0]<=hi+.005 and abs(r-source['radius_m'])<.006 and abs(n[0])<.45
    if 'miter_plane_sum_xy_m' in source:
        # Source inlet is a known source-side cylinder cut by X+Y=constant.
        plane=source['miter_plane_sum_xy_m'];wall=wall and p[0]+p[1]<=plane+.006
        cap=abs(p[0]+p[1]-plane)<.005 and r<source['radius_m']+.005 and abs(n@np.array([1.,1.,0.])/np.sqrt(2))>.65
    else:cap=abs(p[0]-hi)<.005 and r<source['radius_m']+.005 and abs(n[0])>.65
    # At a known cylinder/cap edge the collision normal mixes both surface normals.
    # Classify the known seam disk by its position, with the existing tolerances.
    seam='straight_end_x_m' in source and abs(p[0]-source['straight_end_x_m'])<.005 and r<source['radius_m']+.005
    return bool(wall or cap or seam)

def source_geometry(centers,source):
    c=np.asarray(centers);rho=np.linalg.norm(c[:,1:]-source['center_yz_m'],axis=1)
    mask=(abs(rho-(source['radius_m']+.036))<.031)&(c[:,0]>source['axial_bounds_m'][0])&(c[:,0]<source['axial_bounds_m'][1])
    if 'miter_plane_sum_xy_m' in source:mask&=c[:,0]+c[:,1]<source['miter_plane_sum_xy_m']+.005
    return mask

def classified_observation(sensor,geometry,source):
    # 0 known-source-consistent, 1 receiver-cylinder-candidate-consistent,
    # 2 bridge/unknown. A non-source contact is not automatically receiver load.
    f=np.zeros((3,21));depth=np.zeros((3,21));normal=np.zeros((3,21,3));point=np.zeros((3,21,3));classes=[]
    for ct in sensor['contacts']:
        source_match=known_source(ct,source);axial,rad,rho=geometry.coordinates(ct['point'])
        radial_alignment=float(rad@ct['normal']/max(float(rho),1e-9));axis_dot=abs(float(geometry.axis@ct['normal']))
        receiver_match=bool(not source_match and -.05<axial<1. and abs(rho-geometry.radius)<.012 and axis_dot<.25 and radial_alignment>.90)
        kind=0 if source_match else 1 if receiver_match else 2
        classes.append(dict(estimated_class=kind,source_consistent=source_match,receiver_candidate_consistent=receiver_match,
            estimated_axial_m=float(axial),surface_residual_m=float(rho-geometry.radius),normal_axis_abs_dot=axis_dot,normal_radial_dot=radial_alignment))
        i=ct['material'];f[kind,i]+=ct['normal_N']
        if ct['penetration_m']>=depth[kind,i]:
            depth[kind,i]=ct['penetration_m'];normal[kind,i]=ct['normal'];point[kind,i]=ct['point']
    return dict(sensor,forces=f,body_depth=depth,contact_normal=normal,contact_point=point,contact_estimated_classes=classes,
        receiver_qualified_material=f[1]>.02,unknown_contact_material=f[2]>.02)

