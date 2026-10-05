"""Causal encoder-frame pipe tracking for the closed-loop substitution test.

The private kinematic model's base stays at identity. Pipe axes are inferred
from shape and CAD/load envelopes; no floating pose or actual contacts enter.
Initial right-angle, comparable-radius and already-draped assumptions remain.
"""
import time
import copy
import numpy as np
import mujoco
from encoder_pipe_geometry import right_angle_hypothesis
from encoder_pipe_proximity import EncoderPipeProximity
from feedback_takeover_cad_refinement import refine,refine_target
from feedback_takeover_encoder_tracking import local_transport,transport


class EncoderPipeTracker:
    def __init__(self,model,angles,load,nominal_radius,single_target_after_release=False):
        self.m=copy.copy(model);self.d=mujoco.MjData(self.m)
        self.d.qpos[:7]=[0,0,0,1,0,0,0]
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]
        self.query=EncoderPipeProximity(model)
        self.previous=self.points(angles)
        self.initial=right_angle_hypothesis(self.previous,nominal_radius)
        if not self.initial['geometric_candidate']:
            raise ValueError('Initial encoder shape does not support the branch hypothesis')
        self.estimate=refine(self.query,angles,self.previous,self.initial,load)
        self.info=dict(initial=True,estimate=self.estimate)
        self.single_target_after_release=single_target_after_release
        self.target_only=False

    def points(self,angles):
        self.d.qpos[self.qadr]=angles;mujoco.mj_kinematics(self.m,self.d)
        return 1000*np.vstack([self.d.site('s_head').xpos,
            *[self.d.joint('joint'+str(i)).xanchor for i in range(1,21)],self.d.site('s_tail').xpos])

    def update(self,angles,load,refine_now,source_active=True):
        started=time.perf_counter();points=self.points(angles)
        self.target_only=self.single_target_after_release and not source_active
        if self.target_only:
            predicted=transport(self.previous[:8],points[:8],self.estimate)
            self.estimate=refine_target(self.query,angles,points,predicted,load) if refine_now else predicted
        else:
            predicted=local_transport(self.previous,points,self.estimate)
            self.estimate=refine(self.query,angles,points,predicted,load,fixed_radius=True) if refine_now else predicted
        self.previous=points.copy()
        self.info=dict(estimate=self.estimate,cad_refined=refine_now,target_only=self.target_only,
            inference_seconds=time.perf_counter()-started)
        return points
