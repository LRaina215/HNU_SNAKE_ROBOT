"""Prescribed target-only opening, insertion, closing; no world feedback.

The action coordinate is separate from material feed. Geometry is fitted
offline to an arc-length-preserving variable-radius target helix, while the
old source and transition downstream of link10 are retained.
"""
import numpy as np
from pathlib import Path
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from snake_control.local_material_reference import BoundaryMaterialCurve


class TargetOpenCloseReference:
    opening_mm = 35.
    insert_mm = 80.625
    close_start = opening_mm + insert_mm
    close_end = close_start + opening_mm
    action_end = close_end + insert_mm

    def __init__(self, path, original_path):
        self.original = BoundaryMaterialCurve(original_path)
        with np.load(path) as z:
            self.coordinates = z['action_mm'].copy()
            self.states = z['states'].copy()
        self.rotations = Rotation.from_rotvec(self.states[:, 3:6])
        self.orientation = Slerp(self.coordinates, self.rotations)
        # Private nominal FK only: keeps the source/bridge anchor fixed during
        # interpolation. Independent root SLERP plus joint interpolation would
        # otherwise move it and create spurious source-side penetrations.
        model_path=Path(__file__).resolve().parents[1]/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
        self.model=mujoco.MjModel.from_xml_path(str(model_path))
        self.data=mujoco.MjData(self.model)
        self.qadr=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.anchor=self.model.body('link10').id
        if self.coordinates[0] != 0 or self.coordinates[-1] != self.action_end:
            raise ValueError('Complete open-insert-close geometry required')

    def __getattr__(self, name):
        return getattr(self.original, name)

    @classmethod
    def material_feed(cls, action):
        if action <= cls.close_end:
            return float(np.clip(action-cls.opening_mm, 0., cls.insert_mm))
        return float(cls.insert_mm + action-cls.close_end)

    @classmethod
    def opening(cls, action):
        return float(np.clip(min(action, cls.close_end-action), 0., cls.opening_mm))

    def segment(self, action):
        if not 0 <= action <= self.action_end:
            raise ValueError('Action outside checked open-close interval')
        i = min(int(np.searchsorted(self.coordinates, action, side='right')-1), len(self.coordinates)-2)
        i = max(i, 0)
        span = self.coordinates[i+1]-self.coordinates[i]
        u = (action-self.coordinates[i])/span
        return i, span, u, u*u*(3-2*u)

    def state(self, action):
        feed=self.material_feed(action)
        old=self.original.state(feed)
        if self.opening(action)==0.:
            return old
        i, span, u, h = self.segment(action)
        x = self.states[i] + h*(self.states[i+1]-self.states[i])
        x[16:]=old[16:]  # Exactly preserve original J11..J20 for every coordinate.
        self.data.qpos[:3]=old[:3]/1000
        quat=Rotation.from_rotvec(old[3:6]).as_quat()
        self.data.qpos[3:7]=quat[[3,0,1,2]]
        self.data.qpos[self.qadr]=old[6:]
        mujoco.mj_kinematics(self.model,self.data)
        position=self.data.xpos[self.anchor].copy()
        orientation=self.data.xmat[self.anchor].reshape(3,3).copy()
        self.data.qpos[self.qadr]=x[6:]
        mujoco.mj_kinematics(self.model,self.data)
        rotate=orientation@self.data.xmat[self.anchor].reshape(3,3).T
        x[:3]=(position+rotate@(old[:3]/1000-self.data.xpos[self.anchor]))*1000
        x[3:6]=(Rotation.from_matrix(rotate)*Rotation.from_rotvec(old[3:6])).as_rotvec()
        return x

    def joint_angles(self, action, roll_rad=0.):
        if roll_rad != 0:
            raise ValueError('No additional rolling')
        return self.state(action)[6:]

    def nominal_root(self, action):
        x = self.state(action)
        return x[:3]-np.array([1100., 0., 1000.]), Rotation.from_rotvec(x[3:6]).as_matrix()

    def feed_tangent(self, action):
        # Preserve the original holding projector at both exact closed ends.
        if action == 0. or action == self.action_end:
            return self.original.feed_tangent(self.material_feed(action))
        i, span, u, h = self.segment(action)
        return (self.states[i+1, 6:]-self.states[i, 6:])/span

    def joint_derivative(self, action):
        i, span, u, h = self.segment(action)
        return 6*u*(1-u)*(self.states[i+1, 6:]-self.states[i, 6:])/span
