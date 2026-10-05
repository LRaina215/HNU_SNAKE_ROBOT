"""Private-model error-state EKF driven by commands and encoder angles only.

The local 1-ms dynamics linearization is propagated across a 20-ms measurement
interval. This approximation is deliberately tested before controller use;
switching contact modes can invalidate it. Neither physical pose nor contacts
are part of its API.
"""
import numpy as np
import mujoco


class DynamicsFilter:
    def __init__(self,model,q,velocity=None,joint_velocity=None,flow_linearization=False,velocity_observation=False,bounded_update=False,covariance_bound=False):
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
        self.bounded_update=bounded_update
        self.covariance_bound=covariance_bound
        # This is the trust region of a local Gaussian belief, not a physical
        # speed/position clamp. Near-unobservable directions otherwise produce
        # unbounded linearized covariance and numerical overflow.
        self.maximum_std=np.r_[np.full(3,.05),np.full(3,.3),np.full(20,.1),
            np.full(3,.3),np.full(3,3.),np.full(20,1.)]

    def local_covariance(self,covariance):
        scale=np.minimum(1.,self.maximum_std/np.sqrt(np.maximum(np.diag(covariance),1e-20)))
        return scale[:,None]*covariance*scale[None,:]

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
        if self.covariance_bound:predicted_covariance=self.local_covariance(predicted_covariance)
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
        fraction=1.
        if self.bounded_update:
            # A large correction leaves the EKF's local tangent approximation.
            # Damping the whole gain preserves its direction and the Joseph
            # covariance update. These bounds constrain only the private belief,
            # never physical pose, command, force or model contacts.
            for part,limit in ((correction[:3],.002),(correction[3:6],np.radians(1.)),
                    (correction[self.nv:self.nv+3],.05),(correction[self.nv+3:self.nv+6],.5)):
                magnitude=np.linalg.norm(part)
                if magnitude>limit:fraction=min(fraction,limit/magnitude)
            gain*=fraction;correction*=fraction
        mujoco.mj_integratePos(model,d.qpos,correction[:self.nv],1.)
        d.qvel[:]+=correction[self.nv:]
        remainder=np.eye(2*self.nv);remainder[:,indices]-=gain
        self.covariance=remainder@predicted_covariance@remainder.T+(gain*noise)@gain.T
        self.covariance=(self.covariance+self.covariance.T)/2
        mujoco.mj_forward(model,d)
        return dict(innovation_rms_deg=float(np.degrees(np.sqrt(np.mean(angle_innovation**2)))),
            maximum_transition_entry=float(abs(transition).max()),maximum_state_variance=float(abs(self.covariance).max()),
            update_fraction=float(fraction),
            private_position_correction_mm=float(1000*np.linalg.norm(correction[:3])),
            private_rotation_correction_deg=float(np.degrees(np.linalg.norm(correction[3:6]))))


class TopologyDynamicsFilter(DynamicsFilter):
    """Select missing-arm hypotheses by prediction, using one shared belief.

All candidates start at the same previous inferred state. A new topology must
reduce joint prediction RMS by 40% for three successive observations. Physical
contact identities are not accepted by this API.
"""
    def __init__(self,models,*args,initial_topology='none',**kwargs):
        self.models=models;self.topology=initial_topology;self.pending=None;self.pending_count=0
        super().__init__(models[self.topology],*args,**kwargs)
        self.prediction_data={name:mujoco.MjData(model) for name,model in models.items()}

    def step(self,command,measured,dt):
        scores={};prior=self.data
        for name,model in self.models.items():
            trial=self.prediction_data[name]
            trial.qpos[:]=prior.qpos;trial.qvel[:]=prior.qvel;trial.qacc_warmstart[:]=prior.qacc_warmstart
            trial.time=prior.time;trial.ctrl[:]=command
            mujoco.mj_step(model,trial,nstep=round(dt/model.opt.timestep))
            scores[name]=float(np.degrees(np.sqrt(np.mean((trial.qpos[self.qadr]-measured)**2))))
        best=min(scores,key=scores.get)
        changed=None
        evidence=best!=self.topology and scores[self.topology]>.02 and scores[best]<.6*scores[self.topology]
        if evidence:
            self.pending_count=self.pending_count+1 if self.pending==best else 1;self.pending=best
        else:self.pending_count=0;self.pending=None
        if self.pending_count>=3:
            changed=[self.topology,best];self.topology=best;self.model=self.models[best]
            self.data=mujoco.MjData(self.model)
            self.data.qpos[:]=prior.qpos;self.data.qvel[:]=prior.qvel;self.data.qacc_warmstart[:]=prior.qacc_warmstart
            self.data.time=prior.time;self.scratch=mujoco.MjData(self.model)
            self.pending_count=0;self.pending=None
        info=super().step(command,measured,dt)
        return dict(info,topology=self.topology,topology_changed=changed,topology_prediction_rms_deg=scores)
