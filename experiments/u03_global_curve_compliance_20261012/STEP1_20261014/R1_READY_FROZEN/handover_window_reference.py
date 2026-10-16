"""A12: bounded local finite-hinge correction of A11's nominal handover.

The original common material frame, curve and S/F/M stay intact. This is two
nominal geometry steps on at most eight bridge hinges, never actual-state IK.
"""
from supply_window_reference import SupplyWindowReference
import mujoco
import numpy as np
from physics import MODEL,nodes

class HandoverWindowReference(SupplyWindowReference):
    def __init__(self,points,angles):
        self.mapping_enabled=False
        super().__init__(points,angles)
        self.native_model=mujoco.MjModel.from_xml_path(str(MODEL))
        self.native_data=mujoco.MjData(self.native_model)
        self.native_qa=self.native_model.jnt_qposadr[self.native_model.actuator_trnid[:,0]]
        self.mapping_enabled=True
        self.spec.update(version=12,handover_change='two damped nominal finite-hinge steps on at most8 bridge joints within4LM of prefix midpoint, quintic spatial taper tozero at edges; source and target helix joints untouched',handover_activation='quintic S6..8 and source/target support-edge taper over1LM; fixed24deg correction budget, each step12deg max, no temporal warm start',handover_registration='nominal source nodes if >=6, otherwise last6 nodes, rigid fit solely for shape residual; no actual/root feedback',handover_carrier='all source/target window raw hinge commands and common material psi inherited exactly from A11; local correction carries no separate phase clock',handover_evidence='A11 J11/J12 nominal supports leave source before real receiver acquisition, retain actual source loads60..80s; finite-hinge shape errors precede and coexist with saturation',high_level_feedback=False)

    def native_shape(self,q,C):
        self.native_data.qpos[self.native_qa]=q
        mujoco.mj_fwdPosition(self.native_model,self.native_data)
        p=nodes(self.native_data)
        x=p[self.anchor_ids];y=C[self.anchor_ids];xc=x.mean(0);yc=y.mean(0)
        u,_,v=np.linalg.svd((x-xc).T@(y-yc));rot=u@v
        if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@v
        return (p-xc)@rot+yc

    def command(self,S):
        base,i=super().command(S)
        if not self.mapping_enabled:return base,i
        loc=i['reference_locations'];C=i['nominal_curve_points'];center=.5*(self.start_arc+self.prefix_end_arc)
        mid=.5*(loc[:-2]+loc[2:])
        width=4*.0645
        weights=np.array([self.smooth(1-abs(a-center)/width)*self.smooth((c-self.start_arc)/.0645)*self.smooth((self.target_arc-b)/.0645) for a,b,c in zip(mid,loc[:-2],loc[2:])])
        ids=np.flatnonzero(weights>0)
        source_ids=np.flatnonzero(loc<self.start_arc)
        self.anchor_ids=source_ids if len(source_ids)>=6 else np.arange(16,22)
        activation=self.smooth((S-6)/2);weights*=activation;q=base.copy();before=self.native_shape(base,C)
        if len(ids) and activation>0:
            # Only nominal positions; the posterior anchor retains the A11
            # material orientation through unchanged source hinge commands.
            count=min(22,int(ids[-1])+3);target=C[:count];reg=.30
            def residual(delta):
                candidate=base.copy();candidate[ids]+=weights[ids]*delta
                p=self.native_shape(candidate,C)
                return np.r_[((p[:count]-target)/.0645).ravel(),reg*delta]
            delta=np.zeros(len(ids));eps=1e-5
            for _ in range(2):
                f=residual(delta);jac=np.column_stack([(residual(delta+np.eye(len(ids))[j]*eps)-f)/eps for j in range(len(ids))])
                step=np.linalg.solve(jac.T@jac+.05*np.eye(len(ids)),-jac.T@f)
                delta=np.clip(delta+np.clip(step,-np.radians(12),np.radians(12)),-np.radians(24),np.radians(24))
            q[ids]+=weights[ids]*delta
        after=self.native_shape(q,C);mask=weights
        i.update(curvature_raw_command=base.copy(),local_native_delta=q-base,local_native_window=mask,nominal_native_points=after,local_native_front_before_mm=float(np.max(np.linalg.norm(before[:7]-C[:7],axis=1)))*1000,local_native_front_after_mm=float(np.max(np.linalg.norm(after[:7]-C[:7],axis=1)))*1000)
        return q,i
