"""Private-model error-state EKF driven by commands and encoder angles only.

The local 1-ms dynamics linearization is propagated across a 20-ms measurement
interval. This approximation is deliberately tested before controller use;
switching contact modes can invalidate it. Neither physical pose nor contacts
are part of its API.
"""
import numpy as np
import mujoco


class DynamicsFilter:
    def __init__(self,model,q,velocity=None,joint_velocity=None,flow_linearization=False,velocity_observation=False):
        self.model=model;self.data=mujoco.MjData(model);self.nv=model.nv
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.data.qpos[:7]=[0,0,0,1,0,0,0];self.data.qpos[self.qadr]=q
        if velocity is not None:self.data.qvel[:6]=velocity
        if joint_velocity is not None:self.data.qvel[self.vadr]=joint_velocity
        mujoco.mj_forward(model,self.data)
        std=np.r_[np.full(3,.001),np.full(3,np.radians(.5)),np.full(20,np.radians(.05)),
            np.full(3,.03),np.full(3,.3),np.full(20,.2)]
        self.covariance=np.diag(std**2)
        process_std=np.r_[np.full(3,.00005),np.full(3,.0002),np.full(20,.00001),
            np.full(3,.002),np.full(3,.02),np.full(20,.01)]
        self.process=np.diag(process_std**2)
        self.measurement_variance=np.radians(.01)**2
        self.transition=np.empty((2*self.nv,2*self.nv))
        self.flow_linearization=flow_linearization
        self.scratch=mujoco.MjData(model)
        self.velocity_observation=velocity_observation;self.previous_measurement=np.asarray(q).copy()

    def discrete_transition(self,command,count):
        """Differentiate the complete measurement interval, including mode changes."""
        m=self.model;d=self.data;scratch=self.scratch
        q=d.qpos.copy();v=d.qvel.copy();warm=d.qacc_warmstart.copy();clock=d.time
        d.ctrl[:]=command;mujoco.mj_step(m,d,nstep=count)
        tangent=np.zeros(self.nv);difference=np.zeros(self.nv)
        for j in range(2*self.nv):
            scratch.qpos[:]=q;scratch.qvel[:]=v;scratch.qacc_warmstart[:]=warm;scratch.time=clock;scratch.ctrl[:]=command
            eps=1e-6
            if j<self.nv:
                tangent[:]=0.;tangent[j]=eps;mujoco.mj_integratePos(m,scratch.qpos,tangent,1.)
            else:scratch.qvel[j-self.nv]+=eps
            mujoco.mj_step(m,scratch,nstep=count)
            mujoco.mj_differentiatePos(m,difference,eps,d.qpos,scratch.qpos)
            self.transition[:self.nv,j]=difference
            self.transition[self.nv:,j]=(scratch.qvel-d.qvel)/eps
        return self.transition

    def step(self,command,measured,dt):
        model=self.model;d=self.data;d.ctrl[:]=command
        count=round(dt/model.opt.timestep)
        if self.flow_linearization:transition=self.discrete_transition(command,count)
        else:
            mujoco.mjd_transitionFD(model,d,1e-6,0,self.transition,None,None,None)
            transition=np.linalg.matrix_power(self.transition,count)
            mujoco.mj_step(model,d,nstep=count)
        predicted_covariance=transition@self.covariance@transition.T+self.process*(dt/.02)
        innovation=np.asarray(measured)-d.qpos[self.qadr]
        indices=self.vadr;noise=np.full(len(measured),self.measurement_variance)
        angle_innovation=innovation.copy()
        if self.velocity_observation:
            # Backward differences are delayed/noisy encoder information, not
            # an extra sensor. Give them a broad 0.1 rad/s uncertainty.
            derivative=(np.asarray(measured)-self.previous_measurement)/dt
            innovation=np.r_[innovation,derivative-d.qvel[self.vadr]]
            indices=np.r_[indices,self.nv+self.vadr];noise=np.r_[noise,np.full(len(measured),.1**2)]
        self.previous_measurement=np.asarray(measured).copy()
        cross=predicted_covariance[:,indices]
        measured_covariance=predicted_covariance[np.ix_(indices,indices)]+np.diag(noise)
        gain=np.linalg.solve(measured_covariance,cross.T).T
        correction=gain@innovation
        mujoco.mj_integratePos(model,d.qpos,correction[:self.nv],1.)
        d.qvel[:]+=correction[self.nv:]
        remainder=np.eye(2*self.nv);remainder[:,indices]-=gain
        self.covariance=remainder@predicted_covariance@remainder.T+(gain*noise)@gain.T
        self.covariance=(self.covariance+self.covariance.T)/2
        mujoco.mj_forward(model,d)
        return dict(innovation_rms_deg=float(np.degrees(np.sqrt(np.mean(angle_innovation**2)))),
            maximum_transition_entry=float(abs(transition).max()),maximum_state_variance=float(abs(self.covariance).max()),
            private_position_correction_mm=float(1000*np.linalg.norm(correction[:3])),
            private_rotation_correction_deg=float(np.degrees(np.linalg.norm(correction[3:6]))))
