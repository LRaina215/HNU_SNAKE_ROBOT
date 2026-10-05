"""Causal encoder-frame pipe tracking for the closed-loop substitution test.

The private kinematic model's base stays at identity. Pipe axes are inferred
from shape and CAD/load envelopes; no floating pose or actual contacts enter.
Initial right-angle, comparable-radius and already-draped assumptions remain.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import copy
import numpy as np
import mujoco
from snake_control.encoder_pipe_geometry import right_angle_hypothesis
from snake_control.encoder_pipe_proximity import EncoderPipeProximity
from feedback_takeover_cad_refinement import refine,refine_target
from feedback_takeover_encoder_tracking import local_transport,transport


class EncoderPipeTracker:
    def __init__(self,model,angles,load,nominal_radius,single_target_after_release=False,whole_coil_registration=False,source_cohort_confidence=False):
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
        self.whole_coil_registration=whole_coil_registration
        self.source_cohort_confidence=source_cohort_confidence
        self.source_weak_seconds=0.
        self.source_unobserved=False
        self.source_candidate_count=None

    def points(self,angles):
        self.d.qpos[self.qadr]=angles;mujoco.mj_kinematics(self.m,self.d)
        return 1000*np.vstack([self.d.site('s_head').xpos,
            *[self.d.joint('joint'+str(i)).xanchor for i in range(1,21)],self.d.site('s_tail').xpos])

    def update(self,angles,load,refine_now,source_active=True,front_point_count=8,source_weights=None):
        started=time.perf_counter();points=self.points(angles)
        self.target_only=self.single_target_after_release and not source_active
        predicted=None if self.target_only or self.source_unobserved else local_transport(self.previous,points,self.estimate,front_point_count)
        if self.source_cohort_confidence and not self.source_unobserved and not self.target_only and refine_now:
            # Count evidence only on the still-commanded rear cohort. Middle
            # links passing the old junction cannot keep its axis observable.
            gap=self.query.gaps(angles,points,predicted['corner_mm'],predicted['source'],predicted['target'],predicted['pipe_radius_mm'])[0]
            adjacent=np.maximum(np.r_[0,abs(load)],np.r_[abs(load),0])
            weights=np.asarray(source_weights)
            rear=np.maximum(np.r_[0,weights],np.r_[weights,0])>.5
            self.source_candidate_count=int(np.count_nonzero((gap<5.)&(adjacent>.1)&rear))
            self.source_weak_seconds=self.source_weak_seconds+.2 if self.source_candidate_count<3 else 0.
            self.source_unobserved=self.source_weak_seconds>=.6-1e-9
        if self.target_only or self.source_unobserved:
            # The head's free ends change shape at final handoff. Once rear
            # control is gone, the whole coil supplies a more stable frame
            # prediction than assuming its first eight points move rigidly.
            whole=self.whole_coil_registration and self.target_only
            before=self.previous if whole else self.previous[:front_point_count]
            after=points if whole else points[:front_point_count]
            predicted=transport(before,after,self.estimate)
            self.estimate=refine_target(self.query,angles,points,predicted,load,self.target_only) if refine_now else predicted
        else:
            self.estimate=refine(self.query,angles,points,predicted,load,fixed_radius=True) if refine_now else predicted
        self.previous=points.copy()
        self.info=dict(estimate=self.estimate,cad_refined=refine_now,target_only=self.target_only,
            front_point_count=front_point_count,
            source_candidates=self.source_candidate_count,source_unobserved=self.source_unobserved,
            source_weak_seconds=self.source_weak_seconds,
            inference_seconds=time.perf_counter()-started)
        return points
