"""Simulation prototype: retain spatial bends instead of imposing one pitch.

Lengths are mm, angles rad. Between neighbouring link centres the tangent
follows the shortest rotation between the two measured link directions. Its
parallel-transported (Bishop) bending components are constant on that span.
The same field drives Yamada integration and continuous reconstruction.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation


class SpatialBishopBackbone:
    def __init__(self, points, links):
        points = np.asarray(points, float)
        self.links = np.asarray(links, float)
        tangents = np.diff(points, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        self.bounds = np.r_[0., np.cumsum(self.links)-self.links/2, self.links.sum()]
        self.field = np.zeros((len(self.links)+1, 2))
        tangent = tangents[0]
        normal = np.eye(3)[np.argmin(abs(tangent))]
        normal -= normal @ tangent * tangent
        normal /= np.linalg.norm(normal)
        second = np.cross(tangent, normal)
        self.start_frame = np.column_stack([tangent, normal, second])
        self.start_point = points[0].copy()
        self.mid_frames = []
        for j, (before, after) in enumerate(zip(tangents[:-1], tangents[1:])):
            cross = np.cross(before, after)
            sine = np.linalg.norm(cross)
            angle = math.atan2(sine, np.clip(before @ after, -1., 1.))
            rotvec = cross * (angle/sine) if sine > 1e-12 else np.zeros(3)
            span = self.bounds[j+2]-self.bounds[j+1]
            derivative = np.cross(rotvec/span, before)
            self.field[j+1] = [derivative @ normal, derivative @ second]
            frame = np.column_stack([before, normal, second])
            self.mid_frames.append(Rotation.from_rotvec(rotvec/2).as_matrix() @ frame)
            rotation = Rotation.from_rotvec(rotvec).as_matrix()
            normal, second = rotation @ normal, rotation @ second
        self.mid_frames = np.asarray(self.mid_frames)

    def integral(self, lo, hi, field=None):
        field = self.field if field is None else np.asarray(field)
        lengths = np.maximum(0., np.minimum(hi, self.bounds[1:])-np.maximum(lo, self.bounds[:-1]))
        return lengths @ field

    def joint_angles(self, phase=0., field=None):
        joints = np.r_[0., np.cumsum(self.links)]
        b = np.array([self.integral(lo, hi, field) for lo, hi in zip(joints[:-2], joints[2:])])
        co, sn = math.cos(phase), math.sin(phase)
        return -np.where(np.arange(len(b)) % 2, b[:, 0]*co-b[:, 1]*sn, b[:, 0]*sn+b[:, 1]*co)

    def material_field(self, reference):
        """Fix the Bishop gauge using physical alternating yaw/pitch axes.

        Only a common frame angle is fitted: no per-joint residual is added.
        For actual robot FK, this reproduces the physical joint angles.
        """
        basis = np.column_stack([self.joint_angles(), self.joint_angles(math.pi/2)])
        weights = np.linalg.lstsq(basis, reference, rcond=None)[0]
        phase = math.atan2(weights[1], weights[0])
        co, sn = math.cos(phase), math.sin(phase)
        rotation = np.array([[co, sn], [-sn, co]])
        return self.field @ rotation, rotation

    def section_delta(self, rows, gains, material_rotation):
        """Small local signed-turn correction, preserving the spatial bend.

        This prototype uses delta_gamma=K*gamma on each local cross section.
        The normal-plane part of rotation about its pipe axis is added to the
        existing Bishop field; its tangential twist is excluded. A reversed
        local bend is retained, not replaced with a globally chosen winding.
        This is an incremental 3D extension, not the paper's full A-D rebuild.
        """
        delta = np.zeros_like(self.field)
        for j, (row, gain, frame) in enumerate(zip(rows, gains, self.mid_frames)):
            axis, gamma = row[4], row[2]
            span = self.bounds[j+2]-self.bounds[j+1]
            derivative = np.cross(axis*gain*gamma/span, frame[:, 0])
            delta[j+1] = derivative @ frame[:, 1:]
        return delta @ material_rotation

    def sample(self, field=None, phase=0., step_mm=5., material_rotation=None):
        """Integrate the commanded Bishop field using exact span rotations."""
        field = self.field if field is None else np.asarray(field)
        co, sn = math.cos(phase), math.sin(phase)
        field = field @ np.array([[co, sn], [-sn, co]])
        point = self.start_point.copy();frame = self.start_frame.copy()
        if material_rotation is not None:
            frame[:, 1:] = frame[:, 1:] @ material_rotation
        points = [point.copy()];arc = [0.]
        for lo, hi, b in zip(self.bounds[:-1], self.bounds[1:], field):
            spin = np.array([0., -b[1], b[0]])
            rate = np.linalg.norm(spin)
            count = max(1, math.ceil((hi-lo)/step_mm));h = (hi-lo)/count
            if rate > 1e-12:
                axis = spin/rate
                cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
                integral = h*np.eye(3)+(1-math.cos(rate*h))/rate*cross+(h-math.sin(rate*h)/rate)*(cross@cross)
                rotation = Rotation.from_rotvec(spin*h).as_matrix()
            else:
                integral = h*np.eye(3);rotation = np.eye(3)
            for k in range(count):
                point += frame @ integral[:, 0]
                frame = frame @ rotation
                points.append(point.copy());arc.append(lo+(k+1)*h)
        return dict(s_mm=arc, points_mm=points)


class SpatialShapeController:
    """Experimental feedback around captured spatial shape and initial preload.

    A captured target-minus-measurement field retains the initial servo preload;
    it rotates with total material phase and does not fade. Unlike the previous
    temporary switch residual, both fields come from full spatial robot FK.
    Whether retaining that preload remains suitable is a physical test, not a
    claim of stability. Local compliance adds signed projected turn K*gamma.
    """
    def __init__(self, target_bone, target_angles, measured_bone, measured_angles):
        self.nominal, self.rotation = target_bone.material_field(target_angles)
        observed, _ = measured_bone.material_field(measured_angles)
        self.preload = self.nominal-observed

    def feedback_field(self, measured_bone, measured_angles, rows, gains, total_phase, step_phase):
        observed, rotation = measured_bone.material_field(measured_angles)
        correction = measured_bone.section_delta(rows, gains, rotation)
        def rotate(field, phase):
            co, sn = math.cos(phase), math.sin(phase)
            return field @ np.array([[co, sn], [-sn, co]])
        return rotate(observed+correction, step_phase)+rotate(self.preload, total_phase)
