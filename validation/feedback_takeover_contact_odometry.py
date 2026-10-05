"""Encoder-only pipe transport under hypothetical sticking CAD contacts.

Unlike rigid point-cloud alignment, the instantaneous material contact point
includes link rotation. This assumption may fail under slip or wrong contact
assignment and must be evaluated before being used to release support.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def transport_contacts(query, before, after, points, estimate, load, dt=.02, full_target=False, normal_only=False):
    c=np.asarray(estimate['corner_mm']);s=np.asarray(estimate['source']);t=np.asarray(estimate['target'])
    gaps=query.gaps(before,points,c,s,t,estimate['pipe_radius_mm'],full_target)
    m,d=query.m,query.d;mujoco.mj_comPos(m,d)
    vadr=m.jnt_dofadr[m.actuator_trnid[:,0]]
    dq=(np.asarray(after)-before)/dt
    adjacent=np.maximum(np.r_[0,abs(load)],np.r_[abs(load),0])
    rows=[];rhs=[];weights=[];candidates=[]
    for module,ids in enumerate(query.robot_geoms):
        side=1 if full_target else int(np.argmin(gaps[:,module]))
        gap=gaps[side,module]
        if gap>3. or gap<-10. or adjacent[module]<.1:continue
        slot=query.pipe_slots[side];best=None
        for geom in ids:
            segment=np.zeros(6)
            distance=mujoco.mj_geomDistance(m,d,int(geom),slot,.2,segment)
            if best is None or distance<best[0]:best=(distance,segment,int(geom))
        _,segment,geom=best;jp=np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,jp,None,segment[:3],int(m.geom_bodyid[geom]))
        if normal_only:
            normal=segment[:3]-segment[3:]
            if np.linalg.norm(normal)<1e-10:continue
            normal/=np.linalg.norm(normal)
            rows.append(normal[None,:]@jp[:,:6]);rhs.append(np.atleast_1d(-normal@jp[:,vadr]@dq))
        else:
            rows.append(jp[:,:6]);rhs.append(-jp[:,vadr]@dq)
        weights.append(float(np.exp(-max(0,gap)/2)))
        candidates.append([side,module])
    width=1 if normal_only else 3
    if len(rows)<(4 if normal_only else 2):
        return dict(estimate),dict(candidate_count=len(rows),observable=False,residual_mps=None,twist=np.zeros(6))
    a=np.vstack(rows);b=np.concatenate(rhs);base=np.asarray(weights);weight=base.copy()
    # Balance translation and angular unknowns at a 0.1 m body scale.
    scale=np.diag([1.,1.,1.,10.,10.,10.]);twist=np.zeros(6)
    for _ in range(4):
        w=np.repeat(np.sqrt(weight),width)
        twist=scale@np.linalg.lstsq(a@scale*w[:,None],b*w,rcond=1e-5)[0]
        residual=np.linalg.norm((a@twist-b).reshape(-1,width),axis=1)
        weight=base*np.minimum(1.,.01/np.maximum(residual,1e-12))
    rotation=Rotation.from_rotvec(twist[3:]*dt).as_matrix()
    result=dict(estimate,corner_mm=rotation.T@(c-1000*twist[:3]*dt),source=rotation.T@s,target=rotation.T@t)
    return result,dict(candidate_count=len(rows),candidates=candidates,observable=bool(np.linalg.matrix_rank(a@scale)>=(4 if normal_only else 6)),
        residual_mps=float(np.sqrt(np.mean((a@twist-b)**2))),twist=twist)
