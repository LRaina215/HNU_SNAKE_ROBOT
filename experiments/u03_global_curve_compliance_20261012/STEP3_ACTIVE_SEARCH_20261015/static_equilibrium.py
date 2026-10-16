"""Quasistatic force balance of the unique given coil; no time integration."""
import numpy as np
import mujoco
from scipy.optimize import least_squares

def balance(m,d):
    q0=d.qpos.copy();ctrl0=d.ctrl.copy()
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    acceleration_refinement=False
    # Static compliant settling of this same coil. No later trajectory state.
    def assign(x):
        d.qpos[:]=q0;d.qpos[1:3]+=x[:2]
        angle=np.linalg.norm(x[2:5]);quat=np.array([1.,0.,0.,0.])
        if angle>0:mujoco.mju_axisAngle2Quat(quat,x[2:5]/angle,angle)
        mujoco.mju_mulQuat(d.qpos[3:7],quat,q0[3:7]);d.qpos[qa]+=x[5:25]
        d.ctrl[:]=d.qpos[qa]+x[25:]/m.actuator_gainprm[:,0]
        d.qvel[:]=0.;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d)
    def residual(x):
        assign(x)
        if acceleration_refinement:return np.r_[d.qacc*np.r_[np.full(6,10.),np.ones(20)],x[5:25]*.001]
        mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM)
        return np.r_[(mass@d.qacc)/np.r_[np.ones(3),np.full(23,.1)],x[5:25]*.001]
    lower=np.r_[[-.005,-.005],np.full(3,-np.radians(3)),np.full(20,-np.radians(3)),np.full(20,-2.30)]
    upper=-lower
    scales=np.r_[np.full(2,.001),np.full(23,.01),np.full(20,.5)]
    fd_fraction=.003
    def jacobian(x):
        columns=[]
        for j,h in enumerate(scales*fd_fraction):
            plus=x.copy();minus=x.copy();plus[j]=min(x[j]+h,upper[j]);minus[j]=max(x[j]-h,lower[j])
            columns.append((residual(plus)-residual(minus))/(plus[j]-minus[j]))
        return np.column_stack(columns)
    fit=least_squares(residual,np.zeros(45),jac=jacobian,bounds=(lower,upper),x_scale=scales,
        ftol=1e-11,xtol=1e-11,gtol=1e-9,max_nfev=600)
    fd_fraction=.0001
    fit=least_squares(residual,fit.x,jac=jacobian,bounds=(lower,upper),x_scale=scales,
        ftol=1e-12,xtol=1e-12,gtol=1e-9,max_nfev=1200)
    fd_fraction=.00003;acceleration_refinement=True
    fit=least_squares(residual,fit.x,jac=jacobian,bounds=(lower,upper),x_scale=scales,
        ftol=1e-12,xtol=1e-12,gtol=1e-9,max_nfev=1200)
    assign(fit.x)
    return dict(method='bounded static forward force balance of one full-source coil',joint_angles_changed=True,joint_static_deflection_rad=fit.x[5:25].tolist(),base_delta_yz_m=fit.x[:2].tolist(),
        base_rotation_vector_rad=fit.x[2:5].tolist(),servo_static_offset_rad=(d.ctrl-ctrl0).tolist(),optimizer_success=bool(fit.success),
        optimizer_message=fit.message,optimality=float(fit.optimality),nfev=int(fit.nfev),qacc_max_abs=float(np.max(abs(d.qacc))),qacc_norm=float(np.linalg.norm(d.qacc)),
        torque_max_Nm=float(np.max(abs(d.actuator_force))),simulation_steps=0)

