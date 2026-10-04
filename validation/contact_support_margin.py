"""Optimistic static torque bound for the contacts of a recorded posture.

Solve J.T*f + S*tau = gravity with unilateral contact and box outer bounds
on MuJoCo's full friction wrench cone (including torsion/rolling friction).
The result <= 1 is NOT a support certificate. A value > 1 rules out static
equilibrium for these contacts even in this relaxed cone. No forces are
applied and no physical parameters are changed.
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog
from three_segment_review import context
from segmented_pole_experiment import write_json


class StaticSupportBound:
    def __init__(self,model):
        self.m=model
        self.d=mujoco.MjData(model)
        self.selection=np.zeros((model.nv,model.nu))
        self.selection[model.jnt_dofadr[model.actuator_trnid[:,0]],np.arange(model.nu)]=1.

    def evaluate(self,qpos,pipe_geom=None):
        m,d=self.m,self.d
        d.qpos[:]=qpos;d.qvel[:]=0.;d.ctrl[:]=0.
        mujoco.mj_forward(m,d)
        columns=[];frictions=[]
        for c in d.contact:
            if c.efc_address<0 or (pipe_geom is not None and pipe_geom not in (c.geom1,c.geom2)):
                continue
            b1,b2=m.geom_bodyid[[c.geom1,c.geom2]]
            j1=np.zeros((3,m.nv));r1=np.zeros_like(j1)
            j2=np.zeros_like(j1);r2=np.zeros_like(j1)
            mujoco.mj_jac(m,d,j1,r1,c.pos,b1);mujoco.mj_jac(m,d,j2,r2,c.pos,b2)
            frame=c.frame.reshape(3,3)
            columns.append(np.column_stack(((j2-j1).T@frame.T,(r2-r1).T@frame.T))[:,:c.dim])
            frictions.append(c.friction[:c.dim-1].copy())
        if not columns:return dict(lower_bound=None,contacts=0,feasible_relaxation=False)
        jac=np.column_stack(columns);n=jac.shape[1]
        equality=np.column_stack((jac,self.selection,np.zeros(m.nv)))
        rows=[];bounds=[];offset=0
        for block,friction in zip(columns,frictions):
            for h,mu in enumerate(friction,1):
                for sign in (-1.,1.):
                    row=np.zeros(n+m.nu+1);row[offset]=-mu;row[offset+h]=sign;rows.append(row)
            bounds.extend([(0.,None)]+[(None,None)]*len(friction));offset+=block.shape[1]
        for j in range(m.nu):
            for sign in (-1.,1.):
                row=np.zeros(n+m.nu+1);row[n+j]=sign
                row[-1]=-m.actuator_forcerange[j,1];rows.append(row)
        cost=np.zeros(n+m.nu+1);cost[-1]=1.
        result=linprog(cost,A_ub=np.asarray(rows),b_ub=np.zeros(len(rows)),
            A_eq=equality,b_eq=d.qfrc_bias-d.qfrc_passive,
            bounds=bounds+[(None,None)]*m.nu+[(0.,None)],method='highs')
        return dict(lower_bound=float(result.x[-1]) if result.success else None,
            contacts=len(columns),feasible_relaxation=bool(result.success),
            joint_torques=result.x[n:n+m.nu].tolist() if result.success else None)


def diagnose(folder,times):
    _,z,m,_,_=context(folder);model=StaticSupportBound(m)
    t=z['time']-z['time'][0];rows=[]
    for sec in times:
        i=int(np.argmin(abs(t-sec)));row=model.evaluate(z['qpos'][i])
        row.update(time_seconds=float(t[i]),source_contact_modules=np.flatnonzero(z['normal_force'][i,0]>.1).tolist(),
            target_contact_modules=np.flatnonzero(z['normal_force'][i,1]>.1).tolist())
        rows.append(row)
    report=dict(interpretation='Optimistic quasi-static lower bound, not a dynamic stability certificate.',
        includes_contact_torsion=True,friction_cone='box outer approximation',
        contact_source='MuJoCo contact geometry on a read-only data copy',rows=rows)
    write_json(folder/'static_support_bound.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--times',type=float,nargs='+',default=[30,33,34.3,34.5,34.7,35,36,38,45])
    args=p.parse_args();print(diagnose(args.folder,args.times))
