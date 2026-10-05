"""Contact-turn controllers accept copied observations, never MuJoCo objects.

Only the known-direction feasibility policy is enabled until its stage-1 gate.
Its oracle is explicitly privileged; it cannot be called an autonomous policy.
"""
from dataclasses import dataclass
import math

import numpy as np

from snake_control.local_bend_backbone import LocalBendBackbone


@dataclass(frozen=True)
class JointObservation:
    time_s: float
    sample_unix_ms: float
    received_unix_ms: float
    q: tuple
    velocity: tuple
    applied_target: tuple
    torque_nm: tuple
    applied_command_id: int


@dataclass(frozen=True)
class KnownDirectionOracle:
    head_near_joint: bool
    target_contact_modules: tuple
    source_contact_modules: tuple
    target_wrap_rad: float
    tail_in_target: bool
    direction_rad: float
    ground_contact: bool


class KnownDirectionTransfer:
    """Nominal field first; advance the window only after new support appears."""
    def __init__(self,helix,reference,phase,front=4,width_mm=64.5,turn_deg=60.,release=1.):
        self.helix=helix
        self.reference=np.array(reference,float)
        self.initial_phase=self.phase=float(phase)
        self.front=int(front)
        self.width=float(width_mm)
        self.maximum_turn=math.radians(turn_deg)
        self.release=float(release)
        self.mu=front*64.5/2
        self.turn=0.
        self.direction=None
        self.state='approach'
        self.state_start=0.
        self.records=[]

    def compute(self,observation,oracle,dt=.05):
        if not isinstance(observation,JointObservation) or not isinstance(oracle,KnownDirectionOracle):
            raise TypeError('copied observation and explicit oracle required')
        if oracle.ground_contact:
            self.state='failed_ground'
        if self.state=='approach' and oracle.head_near_joint:
            self.state='turn';self.state_start=observation.time_s
            self.direction=float(oracle.direction_rad)
        if self.state=='turn':
            self.turn=min(self.maximum_turn,self.turn+math.radians(10)*dt)
            if len(set(oracle.target_contact_modules))>=3 and oracle.target_wrap_rad>=math.pi/2:
                self.state='transfer';self.state_start=observation.time_s
        if self.state=='transfer':
            # Stop moving the bend if its target support is lost. This is a
            # privileged feasibility reference, not contact-blind perception.
            if len(set(oracle.target_contact_modules))>=3 and oracle.target_wrap_rad>=math.pi/2:
                self.mu=min(21*64.5-self.width,self.mu+12.*dt)
            if oracle.tail_in_target:
                self.state='target_rolling';self.state_start=observation.time_s
        if self.state in ('approach','transfer','target_rolling'):
            self.phase-=.75*dt
        shape=LocalBendBackbone(self.helix,np.full(21,64.5),self.mu,self.width,self.turn,
                                 self.direction or 0.,self.release,self.reference,self.initial_phase)
        q=shape.joint_angles(self.phase)
        self.records.append(dict(time_s=observation.time_s,state=self.state,phase=self.phase,
                                 mu_mm=self.mu,turn_deg=math.degrees(self.turn),direction_rad=self.direction,
                                 target_contact_modules=list(oracle.target_contact_modules),
                                 target_wrap_rad=oracle.target_wrap_rad,
                                 source_contact_modules=list(oracle.source_contact_modules)))
        return q,shape
