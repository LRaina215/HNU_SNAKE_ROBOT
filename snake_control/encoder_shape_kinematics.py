"""Module-centre Jacobians from joint encoders and known robot geometry.

The floating head is fixed at an arbitrary identity pose in this kinematic
copy. No live free-body pose, contacts, forces or simulator velocities enter.
MuJoCo supplies the model's kinematics only; global slipping remains unseen.
"""
import numpy as np
import mujoco


class EncoderShapeKinematics:
    def __init__(self, model, reference_module=0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        joints = model.actuator_trnid[:, 0]
        self.qadr = model.jnt_qposadr[joints]
        self.vadr = model.jnt_dofadr[joints]
        self.bodies = [model.body('head').id]+[model.body('link'+str(i)).id for i in range(1, 20)]+[model.body('tail').id]
        self.reference_module = reference_module

    def jacobians(self, measured):
        m, d = self.model, self.data
        d.qpos[self.qadr] = measured
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        points = np.vstack([d.site('s_head').xpos, *[d.joint('joint'+str(i)).xanchor for i in range(1, 21)], d.site('s_tail').xpos])
        centers = (points[:-1]+points[1:])/2
        reference = self.reference_module
        ref_jac = np.zeros((3, m.nv)); ref_rot = np.zeros_like(ref_jac)
        mujoco.mj_jac(m, d, ref_jac, ref_rot, centers[reference], self.bodies[reference])
        rotation = d.xmat[self.bodies[reference]].reshape(3, 3)
        result = []
        for module, (center, body) in enumerate(zip(centers, self.bodies)):
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, center, body)
            offset = center-centers[reference]
            skew = np.array([[0., -offset[2], offset[1]], [offset[2], 0., -offset[0]], [-offset[1], offset[0], 0.]])
            relative = rotation.T @ (jac-ref_jac+skew@ref_rot)
            # Only the downstream body is regulated relative to this frame.
            # Upstream rolling is retained; its rigid motion cancels from the
            # downstream Jacobians instead of receiving a braking command.
            result.append(relative[:, self.vadr] if module >= reference else np.zeros((3, len(self.vadr))))
        return np.asarray(result)
