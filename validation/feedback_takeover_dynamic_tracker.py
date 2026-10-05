"""Encoder-local control adapter for response-calibrated private dynamics.

The only update inputs are measured joint angles, the preceding applied joint
command, and elapsed time. It exposes inferred pipe geometry and model support,
not physical contacts. True simulator pose/geometry are never read here.
"""
import copy
import time
import numpy as np
import mujoco
from encoder_pipe_proximity import EncoderPipeProximity
from feedback_takeover_dynamic_observer_probe import virtual_model,predict
from feedback_takeover_dynamic_filter import TopologyDynamicsFilter


class DynamicPipeTracker:
    def __init__(self,robot,robot_xml,fit,records):
        self.initial=fit['estimate'];self.fixed=copy.deepcopy(fit['estimate'])
        self.m=copy.copy(robot);self.d=mujoco.MjData(self.m);self.d.qpos[:7]=[0,0,0,1,0,0,0]
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]
        self.models={label:virtual_model(robot,fit['estimate'],fit['gravity'],robot_xml,extension)
            for label,extension in [('none',None),('source','source'),('target','target')]}
        topology=fit.get('extension') or 'none'
        q=np.asarray(records[-1]['q']);prediction=predict(self.models[topology],records,fit['initial_velocity'])
        self.observer=TopologyDynamicsFilter(self.models,q,initial_topology=topology,velocity_observation=True,bounded_update=True,covariance_bound=True)
        private=self.observer.data
        private.qpos[:]=prediction['final_qpos'];private.qvel[:]=prediction['final_qvel'];private.qpos[self.qadr]=q
        private.ctrl[:]=records[-1]['command'];mujoco.mj_forward(self.observer.model,private)
        self.pipe_slots=EncoderPipeProximity(robot).pipe_slots
        self.body_to_module={robot.body(name).id:i for i,name in enumerate(['head']+['link'+str(i) for i in range(1,20)]+['tail'])}
        self.target_only=False;self.info={};self.refresh(q)

    def points(self,angles):
        self.d.qpos[self.qadr]=angles;mujoco.mj_kinematics(self.m,self.d)
        return 1000*np.vstack([self.d.site('s_head').xpos,
            *[self.d.joint('joint'+str(i)).xanchor for i in range(1,21)],self.d.site('s_tail').xpos])

    def refresh(self,angles):
        d=self.observer.data;m=self.observer.model
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,d.qpos[3:7]);rotation=rotation.reshape(3,3)
        self.estimate=dict(self.fixed,corner_mm=rotation.T@(self.fixed['corner_mm']-1000*d.qpos[:3]),
            source=rotation.T@self.fixed['source'],target=rotation.T@self.fixed['target'])
        self.previous=self.points(angles);self.support=np.zeros((3,len(angles)+1))
        for j in range(d.ncon):
            contact=d.contact[j]
            for axis,slot in enumerate(self.pipe_slots):
                if slot not in (contact.geom1,contact.geom2):continue
                other=contact.geom2 if contact.geom1==slot else contact.geom1
                module=self.body_to_module.get(int(m.geom_bodyid[other]))
                if module is None:continue
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,j,wrench);self.support[axis,module]+=max(0.,wrench[0])
        self.info=dict(self.info,estimate=self.estimate,model_support_N=self.support.copy(),model_support_not_measured=True,
            topology=self.observer.topology)

    def update(self,angles,command,dt):
        started=time.perf_counter();self.info=self.observer.step(command,angles,dt);self.refresh(angles)
        self.info['inference_seconds']=time.perf_counter()-started
        return self.previous.copy()
