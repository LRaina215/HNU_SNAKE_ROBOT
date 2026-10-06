"""钩挂回拉：冻结当前身体形状、整身滚动的关节目标。长度 mm，角度 rad。

整身滚动 = 身体形状在空间中不变，各连杆绕自身纵轴（连杆 z 轴）转过同一角度 psi。
交替单自由度关节无法精确实现任意形状的滚动，因此对每个 psi 求最小二乘：
全部节点位置保持原形状，各连杆姿态等于原姿态再绕自身 z 轴转 psi，根位姿一并求解。
后部（与源管接触、产生滚动推进）的滚动约束权重高于悬空部分。
结果离线做成 psi → 关节角 表，控制时插值使用。
"""
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REAR_ROLL_WEIGHT_MM_PER_RAD = 60.
FREE_ROLL_WEIGHT_MM_PER_RAD = 20.
CONTINUITY_WEIGHT_MM_PER_RAD = 2.


def node_points(model, data):
    return 1000 * np.vstack([data.site('s_head').xpos,
                             *[data.joint('joint' + str(i)).xanchor for i in range(1, 21)],
                             data.site('s_tail').xpos])


class RollSolver:
    def __init__(self, model, root_pos_mm, root_quat_wxyz, joints, rear_modules,
                 rear_weight=REAR_ROLL_WEIGHT_MM_PER_RAD, free_weight=FREE_ROLL_WEIGHT_MM_PER_RAD):
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.head = model.body('head').id
        self.weights = np.full(21, float(free_weight))
        self.weights[np.asarray(rear_modules)] = rear_weight
        self.low = np.full(20, -np.pi / 2 + 1e-3)
        self.high = -self.low
        x0 = np.r_[root_pos_mm, Rotation.from_quat(np.asarray(root_quat_wxyz)[[1, 2, 3, 0]]).as_rotvec(), joints]
        self.p0, self.frames0 = self.forward(x0)
        self.x0 = x0

    def forward(self, x):
        d = self.d
        d.qpos[:3] = x[:3] / 1000
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[self.qadr] = x[6:]
        mujoco.mj_kinematics(self.m, d)
        frames = d.xmat[self.head:self.head + 21].reshape(-1, 3, 3).copy()
        return node_points(self.m, d), frames

    def residual(self, x, psi, previous):
        p, frames = self.forward(x)
        roll = Rotation.from_rotvec([0., 0., psi])
        errors = [(p - self.p0).ravel()]
        want = Rotation.from_matrix(self.frames0) * roll
        errors.append((self.weights[:, None] * (want.inv() * Rotation.from_matrix(frames)).as_rotvec()).ravel())
        errors.append(CONTINUITY_WEIGHT_MM_PER_RAD * (x[6:] - previous[6:]))
        return np.concatenate(errors)


class SplitRollSolver(RollSolver):
    """前部绕自身纵轴滚动并随预测的螺旋运动移动，后部保持不动，过渡段节点自由。

    front_modules 滚动 psi；front 节点按 front_screw(psi) 移动（目标管上的理想不打滑螺旋运动）；
    rear 节点与姿态保持；transition 节点不约束位置，只受关节连续性约束。
    """

    def __init__(self, model, root_pos_mm, root_quat_wxyz, joints, front_modules, rear_modules, front_screw):
        super().__init__(model, root_pos_mm, root_quat_wxyz, joints, rear_modules, REAR_ROLL_WEIGHT_MM_PER_RAD, 0.)
        self.front = np.asarray(front_modules)
        self.rear_modules = np.asarray(rear_modules)
        self.front_nodes = np.arange(self.front.min(), self.front.max() + 2)
        self.rear_nodes = np.arange(self.rear_modules.min(), self.rear_modules.max() + 2)
        self.front_screw = front_screw

    def residual(self, x, psi, previous):
        p, frames = self.forward(x)
        moved, rotation = self.front_screw(psi, self.p0[self.front_nodes])
        errors = [(p[self.front_nodes] - moved).ravel(), (p[self.rear_nodes] - self.p0[self.rear_nodes]).ravel()]
        want_front = rotation * Rotation.from_matrix(self.frames0[self.front]) * Rotation.from_rotvec([0., 0., psi])
        errors.append((REAR_ROLL_WEIGHT_MM_PER_RAD * (want_front.inv() * Rotation.from_matrix(frames[self.front])).as_rotvec()).ravel())
        want_rear = Rotation.from_matrix(self.frames0[self.rear_modules])
        errors.append((REAR_ROLL_WEIGHT_MM_PER_RAD * (want_rear.inv() * Rotation.from_matrix(frames[self.rear_modules])).as_rotvec()).ravel())
        errors.append(CONTINUITY_WEIGHT_MM_PER_RAD * (x[6:] - previous[6:]))
        return np.concatenate(errors)

    def table(self, psis):
        x = self.x0.copy()
        rows = []
        for psi in psis:
            low = np.r_[np.full(6, -np.inf), self.low]
            high = np.r_[np.full(6, np.inf), self.high]
            solution = least_squares(self.residual, np.clip(x, low, high), args=(psi, x), bounds=(low, high),
                                     x_scale=np.r_[np.full(3, 10.), np.full(3, .1), np.full(20, .1)],
                                     diff_step=1e-6, xtol=1e-10, ftol=1e-10, max_nfev=400)
            x = solution.x
            p, frames = self.forward(x)
            rolls = [float((Rotation.from_matrix(self.frames0[k]).inv() * Rotation.from_matrix(frames[k])).as_rotvec()[2])
                     for k in range(21)]
            rows.append(dict(psi=float(psi), x=x.copy(), point_error_mm=np.linalg.norm(p - self.p0, axis=1),
                             link_roll_rad=np.array(rolls)))
        return rows


class RollTable:
    """psi → 关节目标；psi 为后部材料滚动角，超出表范围时拒绝。"""

    def __init__(self, psi, joints):
        self.psi = np.asarray(psi, dtype=float)
        self.interpolator = PchipInterpolator(self.psi, np.asarray(joints, dtype=float), axis=0)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['psi_rad'], data['joint_rad'])

    def __call__(self, psi):
        if not self.psi[0] - 1e-12 <= psi <= self.psi[-1] + 1e-12:
            raise ValueError('roll phase outside admitted table')
        return self.interpolator(psi)
