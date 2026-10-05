"""CAD proximity to hypothesised pipes; never queries the physical contacts.

MuJoCo is used as a kinematic/convex-distance library on a private model copy.
Two cylinder slots are overwritten by the supplied hypothesis. Their original
pose/size, actual contact force, solver state and floating pose are not inputs.
Near a virtual pipe does not prove a force-bearing contact with a real pipe.
"""
import copy
import numpy as np
import mujoco


class EncoderPipeProximity:
    def __init__(self,robot_model):
        self.m=copy.copy(robot_model);self.d=mujoco.MjData(self.m)
        # 3.2.7's legacy standalone distance query gives large false negative
        # distances for these long cylinders. Native GJK agrees with the saved
        # contact gap. This flag belongs ONLY to the non-stepping query copy.
        self.m.opt.enableflags|=mujoco.mjtEnableBit.mjENBL_NATIVECCD
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]
        self.d.qpos[:7]=[0,0,0,1,0,0,0]
        names=['head']+['link'+str(i) for i in range(1,20)]+['tail']
        self.robot_geoms=[]
        for name in names:
            ids=np.flatnonzero((self.m.geom_bodyid==self.m.body(name).id)&(self.m.geom_contype!=0))
            self.robot_geoms.append(ids)
        robot_ids=set(np.concatenate(self.robot_geoms))
        # Only storage slots are borrowed. All queried pipe geometry is replaced.
        self.pipe_slots=[i for i in range(self.m.ngeom)
            if i not in robot_ids and self.m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER][:2]
        if len(self.pipe_slots)!=2:raise ValueError('Two cylinder storage slots required')

    def gaps(self,measured,observed_points,corner,source,target,radius,target_full_line=False):
        m,d=self.m,self.d;d.qpos[self.qadr]=measured;mujoco.mj_kinematics(m,d)
        own=1000*np.vstack([d.site('s_head').xpos,
            *[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
        observed=np.asarray(observed_points)
        # Express the hypothesis in the arbitrary encoder FK frame. At the
        # final unprivileged stage observed_points must also come from encoders.
        x=own-own.mean(axis=0);y=observed-observed.mean(axis=0)
        u,_,vt=np.linalg.svd(x.T@y)
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@correction@u.T
        translation=observed.mean(axis=0)-rotation@own.mean(axis=0)
        c=rotation.T@(corner-translation)/1000
        result=[]
        for slot,axis,sign in zip(self.pipe_slots,[source,target],[-1,1]):
            a=rotation.T@axis
            h=np.eye(3)[np.argmin(abs(a))];h-=h@a*a;h/=np.linalg.norm(h)
            m.geom_size[slot]=[radius/1000,1.,0.]
            d.geom_xpos[slot]=c+sign*a
            if target_full_line and sign==1:
                # Once the source controller is gone, the old junction's
                # axial coordinate is not observable. Query the target line
                # across the body rather than impose an arbitrary end cap.
                d.geom_xpos[slot]=c+((own.mean(axis=0)/1000-c)@a)*a
            d.geom_xmat[slot]=np.column_stack((h,np.cross(a,h),a)).reshape(9)
            distances=[]
            for ids in self.robot_geoms:
                distances.append(min(mujoco.mj_geomDistance(m,d,int(g),slot,.2,None) for g in ids)*1000)
            result.append(distances)
        return np.asarray(result)

    def observe(self,measured,points,corner,source,target,radius,joint_load,target_full_line=False):
        gap=self.gaps(measured,points,corner,source,target,radius,target_full_line)
        load=abs(np.asarray(joint_load))
        adjacent=np.maximum(np.r_[0,load],np.r_[load,0])
        near=gap<1.
        supported=near&(adjacent[None,:]>.1)
        # Unit-valued hypotheses fit the existing Boolean-history interface.
        # They must never be labelled normal forces or measured contacts.
        proxy=np.zeros((3,len(adjacent)));proxy[:2]=supported.astype(float)
        return proxy,dict(cad_gap_mm=gap,adjacent_load_Nm=adjacent,potential_contact=near,
            support_hypothesis=supported,not_measured_contact=True)
