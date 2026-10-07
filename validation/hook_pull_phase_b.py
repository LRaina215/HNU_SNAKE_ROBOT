"""阶段B历史实现：几何点目标拟合，当前仅供无步进接口复核。

实际 run 调用 plan_step，不调用含连杆自转约束的 solve_pulse。
plan_step 使用恢复时的世界配准先验和私有运动学 data；进展/包绕/接触是
世界真值诊断，不是编码器可观测的全局滑移。后部供料读取仿真执行器力矩，
尚无实机负载映射。模型参数与 10°/s 目标限制保留。
加载后速度及 ctrl 预载现已保留，但零运动拟合目标仍趋向实测角；有效的
几何增量到已执行目标映射尚未定义。DYNAMICS_ADMITTED=False 禁止运行。
判据（运行前登记，同后续执行方案）：
  B2 前部沿目标管净前进 ≥25 mm 且后部沿源管净前进 ≥25 mm；≥1 个 M≥8 模块
     落到目标管斜切区以外（垂足 y≥60 mm）且接触力 >0.1 N 保持到结束；
  B3 停止送身后保持 10 s：末 2 s 模块位移 ≤6.45 mm、目标管 ≥3 接触、包绕 >180°；
  C  21 模块全部离开源管（本脚本运行窗口内评估，整身 rolling 延续另列）；
  通用：不触地、无非相邻自碰、包绕 ≥180°、穿管 ≤3 mm、关节 ≤88°。
录像：全程离屏渲染 mp4（大文件只留本地）。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import argparse

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from curve_guided_geometry import MODEL, CORNER, points
from snake_control.hook_pull_motion import RollSolver, SplitRollSolver
from snake_control.hook_pull_review_support import (
    R95_MM, PITCH_MM, L_TURN_MM, ROT_PER_MM, AX_PER_MM,
    restore_saved_state, axis_progress_mm, continuous_target)

BASE = ROOT / 'experiments/hook_pull_20261006'
OUT = BASE / 'phase_b'
AXIS_T = np.array([1100., 0., 1000.])   # mm（节点坐标为毫米）
R_B = 29.   # 外壳半径 mm
TOUCH = 84.
AXIS_S = np.array([0., 0., 1000.])
DYNAMICS_ADMITTED = False  # State loading/units alone do not admit an action.
RATE = np.radians(10.)
PULSE_MM = 8.
JAM_TORQUE = 1.6
JAM_TICKS = 8
GAIN_MIN_MM = 1.
MAX_PULSES = 30
MAX_RUN_S = 150.
STEP_MM = 2.   # 每个规划步的弧长
TICK_STEPS, TICK = 50, .05   # 每个控制tick的物理步数与时长
CRITERION = dict(B2_front_y_min_mm=25., B2_rear_x_min_mm=25., B2_contact_N=.1,
                 B3='保持10s：末2s模块位移≤6.45mm、目标管≥3接触、包绕>180°',
                 C='21模块全部离开源管',
                 gates='不触地、无非相邻自碰、包绕≥180°、穿管≤3mm、关节≤88°（安全停止）',
                 pulse='实际为逐tick几何拟合，未调用SplitRollSolver；跟踪≤10°/s',
                 planning_info='初态世界配准先验+仿真qfrc_actuator；世界位移用于诊断停滞/评分，非编码器滑移测量')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


class Runner:
    def __init__(self, start, record=True, case=None):
        self.start = start
        self.case = Path(case) if case is not None else OUT / f'{start}_dyn'
        # A disabled, non-recording load probe may reuse its diagnostic folder.
        # A dynamics/video case always remains exclusive.
        self.case.mkdir(parents=True, exist_ok=case is not None and not record and not DYNAMICS_ADMITTED)
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.head = self.m.body('head').id
        self.floor = self.m.geom('floor').id
        self.pipe_source = {self.m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']}
        self.pipe_target = {self.m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}
        self.start_controller_history = restore_saved_state(self.m, self.d, BASE / start / 'final_state.npz')
        self.t0_state = float(self.d.time)
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.rows = []
        self.renderer = mujoco.Renderer(self.m, height=480, width=720) if record else None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [1.06, .12, .99]
        self.camera.distance = 1.15
        self.camera.azimuth = 150
        self.camera.elevation = -22
        self.video = cv2.VideoWriter(str(self.case / 'run_1x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1020, 480)) if record else None
        self.free_floor = {}
        self.pulse_end = 'FRONT'

    def t(self):
        return float(self.d.time) - self.t0_state

    def nodes(self):
        return points(self.m, self.d)

    def centers(self):
        p = self.nodes()
        return (p[:-1] + p[1:]) / 2

    def jac(self, p, module):
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, np.asarray(p, float), self.head + module)
        return self.jacp.copy()

    def closest(self, k, pipe):
        a, b = self.nodes()[k], self.nodes()[k + 1]
        p0, axis = (AXIS_T, np.array([0., 1., 0.])) if pipe == 'target' else (AXIS_S, np.array([1., 0., 0.]))
        u, w = b - a, a - p0
        u_p, w_p = u - (u @ axis) * axis, w - (w @ axis) * axis
        t = float(np.clip(-(w_p @ u_p) / (u_p @ u_p), 0., 1.)) if u_p @ u_p > 1e-12 else .5
        x = a + t * u
        foot = p0 + ((x - p0) @ axis) * axis
        v = x - foot
        return x, foot, float(np.linalg.norm(v)), v / float(np.linalg.norm(v))

    def in_miter(self, foot, pipe):
        return foot[1] < 60. if pipe == 'target' else foot[0] > 1040.

    def wrap_deg(self):
        C = self.centers()
        uv = (C[:9] - AXIS_T)[:, [0, 2]][::-1]
        a = np.arctan2(uv[:, 1], uv[:, 0])
        return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))

    def transition_torque(self):
        return float(np.mean(np.abs(self.d.qfrc_actuator[self.vadr][7:12])))

    def module_forces(self):
        f = np.zeros((2, 21))
        for index, c in enumerate(self.d.contact):
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            for pipe, body in [(c.geom1, b2), (c.geom2, b1)]:
                if body < self.head:
                    continue
                group = 0 if pipe in self.pipe_source else 1 if pipe in self.pipe_target and c.pos[1] * 1000 > 75. else None
                if group is not None:
                    fv = np.zeros(6)
                    mujoco.mj_contactForce(self.m, self.d, index, fv)
                    f[group, body - self.head] += max(0., fv[0])
        return f

    def danger(self):
        deepest, ground, selfhit = 0., False, False
        for c in self.d.contact:
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            ground |= (c.geom1 == self.floor and b2 >= self.head) or (c.geom2 == self.floor and b1 >= self.head)
            # Same 0.5 mm threshold as hook_pull_step4_rolling.touching.
            selfhit |= b1 >= self.head and b2 >= self.head and abs(b1 - b2) > 1 and c.dist < -0.5e-3
            if (c.geom1 in self.pipe_source | self.pipe_target and b2 >= self.head) or (c.geom2 in self.pipe_source | self.pipe_target and b1 >= self.head):
                deepest = min(deepest, float(c.dist))
        return deepest * 1000, bool(ground), bool(selfhit)

    def screw(self, pipe, u_mm):
        """推进端沿管螺旋前进 u_mm 的刚体变换；旋向按实际缠绕（θ 随前进方向的实测符号）。"""
        if pipe == 'target':
            return (Rotation.from_rotvec([0., -ROT_PER_MM * u_mm, 0.]), AXIS_T,
                    np.array([0., AX_PER_MM * u_mm, 0.]))
        return (Rotation.from_rotvec([-ROT_PER_MM * u_mm, 0., 0.]), AXIS_S,
                np.array([AX_PER_MM * u_mm, 0., 0.]))

    def solve_pulse(self, end, u_mm):
        """从当前实测状态解脉冲终态关节角。返回 (q_end, 诊断) 或 (None, 原因)。"""
        d = self.d
        if end == 'FRONT':
            def screw(psi, pts):
                rot, axis_point, shift = self.screw('target', psi / ROT_PER_MM)
                return (rot.apply(pts - axis_point) + axis_point + shift, rot)
            front_modules, rear_modules = list(range(0, 8)), list(range(10, 21))
        else:
            def screw(psi, pts):
                rot, axis_point, shift = self.screw('source', psi / ROT_PER_MM)
                return (rot.apply(pts - axis_point) + axis_point + shift, rot)
            front_modules, rear_modules = list(range(11, 21)), list(range(0, 9))
        solver = SplitRollSolver(self.m, d.qpos[:3] * 1000, d.qpos[3:7], d.qpos[self.qadr],
                                 front_modules, rear_modules, screw)
        target_psi = u_mm * ROT_PER_MM
        rows = solver.table(np.array([0., target_psi]))
        x = rows[-1]['x']
        p, _ = solver.forward(x)
        moved, _ = screw(rows[-1]['psi'], solver.p0[solver.front_nodes])
        front_err = float(np.linalg.norm(p[solver.front_nodes] - moved, axis=1).max())
        rear_err = float(np.linalg.norm(p[solver.rear_nodes] - solver.p0[solver.rear_nodes], axis=1).max())
        joint = float(np.degrees(abs(x[6:])).max())
        if front_err > 8. or rear_err > 8. or joint > 88.:
            return None, dict(front_err=round(front_err, 2), rear_err=round(rear_err, 2), joint_deg=round(joint, 1))
        return x, dict(front_err=round(front_err, 2), rear_err=round(rear_err, 2), joint_deg=round(joint, 1))

    def frame(self, tag):
        if self.renderer is None:
            return
        self.renderer.update_scene(self.d, camera=self.camera)
        frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        panel = np.full((480, 300, 3), 30, np.uint8)
        scale, origin = .9, np.array([150, 240])
        cv2.circle(panel, tuple(origin), int(55 * scale), (90, 160, 90), -1)
        cv2.circle(panel, tuple(origin), int(84 * scale), (90, 160, 90), 1)
        uv = (self.centers()[:12] - CORNER)[:, [0, 2]]
        pix = (origin + uv * [scale, -scale]).astype(int)
        for a, b in zip(pix[:-1], pix[1:]):
            cv2.line(panel, tuple(a), tuple(b), (0, 200, 255), 2)
        fm = self.module_forces()
        for k, p in enumerate(pix):
            cv2.circle(panel, tuple(p), 4, (0, 0, 255) if fm[1, k] > .1 else (255, 255, 255), -1)
        cv2.putText(panel, 'target-pipe section M0-M11', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(panel, f't={self.t():6.1f}s {tag}', (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        cv2.putText(panel, f'trans |torque| {self.transition_torque():4.2f} Nm', (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(panel, f'wrap {self.wrap_deg():5.1f} deg', (8, 95), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(panel, 'red = touching target', (8, 455), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 255), 1)
        self.video.write(np.hstack([frame, panel]))

    def plan_step(self, u_f, u_r, rear_feed):
        """v2 式单步规划：前部沿自身钩形螺旋推进 2 mm 弧长；后部按供料节奏推进。"""
        # Removed unused E/e contact rows: they never entered this residual.
        # Evaluate candidate root/joints on private kinematic data, not self.d.
        tgt = self.p0_plan.copy()
        axial_f = u_f / (abs(self.kappa) * self.arc_per_rad)
        rot_y = -self.kappa * axial_f
        rotF = Rotation.from_rotvec([0., rot_y, 0.])
        tgt[:9] = rotF.apply(self.p0_plan[:9] - AXIS_T) + AXIS_T + np.array([0., axial_f, 0.])
        rotR = Rotation.from_rotvec([-u_r * ROT_PER_MM, 0., 0.])
        tgt[10:] = rotR.apply(self.p0_plan[10:] - AXIS_S) + AXIS_S + np.array([u_r * AX_PER_MM, 0., 0.])
        for n in range(10, 22):
            fade = float(np.clip((self.arc_before[n] - u_r) / 48. + 1., 0., 1.))
            self.w_plan[n] = 60. * fade
        self.w_plan[9] = 15.

        def residual(xx, prev):
            p, _ = self.planner.forward(xx)
            return np.concatenate([(self.w_plan[:, None] * (p - tgt)).ravel(), 20. * (xx[6:] - prev[6:])])

        low = np.r_[np.full(6, -np.inf), np.full(20, -np.pi / 2 + 1e-3)]
        high = np.r_[np.full(6, np.inf), np.full(20, np.pi / 2 - 1e-3)]
        sol = least_squares(residual, self.x_plan, bounds=(low, high), args=(self.x_plan,),
                            x_scale=np.r_[np.full(3, 10.), np.full(3, .1), np.full(20, .1)],
                            diff_step=1e-6, xtol=1e-10, ftol=1e-10, max_nfev=300)
        self.x_plan = sol.x
        return sol.x

    def planner_contacts(self, centers):
        out = []
        for k in range(21):
            rt = float(np.hypot(centers[k, 0] - AXIS_T[0], centers[k, 2] - AXIS_T[2]))
            rs = float(np.hypot(centers[k, 1] - AXIS_S[1], centers[k, 2] - AXIS_S[2]))
            if 79.5 <= rt <= 92. and centers[k, 1] >= -20.:
                out.append((k, 'target'))
            elif 79.5 <= rs <= 92. and centers[k, 0] <= 1112.:
                out.append((k, 'source'))
        return out

    def run(self, max_run_s=MAX_RUN_S):
        if not DYNAMICS_ADMITTED:
            raise RuntimeError('Review only: zero-motion preload mapping and executable rolling/scoring admission are unresolved; do not step physics')
        start_nodes = self.nodes()
        start_front_y = float(((start_nodes[:-1] + start_nodes[1:]) / 2)[:8, 1].mean())
        start_rear_x = float(((start_nodes[:-1] + start_nodes[1:]) / 2)[13:21, 0].mean())
        C0 = self.centers()
        joints0 = self.d.qpos[self.qadr].copy()
        solver = RollSolver(self.m, self.d.qpos[:3] * 1000, self.d.qpos[3:7], joints0, list(range(10, 21)))
        self.planner = solver
        self.p0_plan = solver.p0.copy()
        self.x_plan = solver.x0.copy()
        th = np.arctan2(C0[:9, 2] - 1000., C0[:9, 0] - 1100.)
        y = C0[:9, 1]
        order = np.argsort(y)
        th_u = np.unwrap(th[order])
        self.kappa = float(np.polyfit(y[order], th_u, 1)[0])
        self.arc_per_rad = float(8 * 64.5 / max(abs(th_u[-1] - th_u[0]), 1e-6))
        self.w_plan = np.full(22, 15.)
        self.w_plan[:9] = 60.
        self.w_plan[10:] = 60.
        self.arc_before = {n: (n - 10) * 64.5 + 32.25 for n in range(10, 22)}
        self.rho_planner = {}
        print(f'{self.start}: 规划 kappa={self.kappa:.5f} arc/rad={self.arc_per_rad:.1f}', flush=True)
        events, ledger = [], []
        u_f = u_r = 0.
        rear_every = 4
        stop = 'time_cap'
        tick = 0
        last_print = 0.
        progress_win = []
        while self.t() < max_run_s:
            tick += 1
            rear_feed = (tick % rear_every == 0) and self.transition_torque() <= JAM_TORQUE
            q_ref = self.plan_step(u_f, u_r, rear_feed)
            if rear_feed:
                u_r += STEP_MM
            u_f += STEP_MM
            self.d.ctrl[:] = continuous_target(self.d.ctrl, self.x_plan[6:], RATE, TICK)
            for _ in range(TICK_STEPS):
                mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            centers = self.centers()
            front_y, rear_x = axis_progress_mm(centers, start_front_y, start_rear_x)
            tq = self.transition_torque()
            deepest, ground, selfhit = self.danger()
            f = self.module_forces()
            q_deg = float(np.degrees(abs(self.d.qpos[self.qadr])).max())
            crossed = {e.get('module') for e in events if e.get('event') == 'cross_onto_target'}
            for k in range(8, 21):
                rt = float(np.hypot(centers[k, 0] - AXIS_T[0], centers[k, 2] - AXIS_T[2]))
                if rt <= 92. and centers[k, 1] >= 60. and f[1, k] > .1 and k not in crossed:
                    events.append(dict(t=self.t(), event='cross_onto_target', module=k))
                    print(f'  t={self.t():.1f}s M{k} 落上目标管!', flush=True)
            n_source = int(np.count_nonzero(f[0] > .1))
            self.rows.append(dict(t=self.t(), front_y_mm=front_y, rear_x_mm=rear_x,
                                  max_joint_deg=q_deg, wrap_deg=self.wrap_deg(), transition_torque_Nm=tq,
                                  target_contacts=int(np.count_nonzero(f[1] > .1)), source_contacts=n_source,
                                  deepest_mm=deepest, u_front_mm=u_f, u_rear_mm=u_r))
            if self.t() - last_print >= 2.:
                last_print = self.t()
                print(f"t={self.t():6.1f}s front+Y {front_y:7.1f} rear+X {rear_x:7.1f} mm wrap {self.wrap_deg():5.1f} deg "
                      f"|q| {q_deg:5.1f} trans {tq:4.2f} Nm T{int(np.count_nonzero(f[1] > .1))}/S{n_source} "
                      f"u_f={u_f:.0f} u_r={u_r:.0f}", flush=True)
            if tick % 5 == 0:
                self.frame('rolling')
            failed = []
            if ground or selfhit:
                failed.append('ground_or_self_contact')
            if deepest < -3.:
                failed.append('pipe_penetration')
            if self.wrap_deg() < 180.:
                failed.append('front_wrap_below_180')
            if q_deg > 88.:
                failed.append(f'joint_limit J{int(np.argmax(abs(self.d.qpos[self.qadr]))) + 1}')
            progress_win.append(front_y)
            if len(progress_win) > 20:
                win = progress_win[-20:]
                if abs(win[-1] - win[0]) < 0.5 and u_f > 20.:
                    failed.append('front_stalled')
            if failed:
                stop = ','.join(failed)
                break
        hold_start = self.t()
        ctrl_hold = self.d.ctrl.copy()
        hold_rows = []
        while self.t() < hold_start + 10.:
            self.d.ctrl[:] = ctrl_hold
            for _ in range(50):
                mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            f = self.module_forces()
            centers = self.centers()
            front_y, rear_x = axis_progress_mm(centers, start_front_y, start_rear_x)
            hold_rows.append(dict(t=self.t(), front_y_mm=front_y, rear_x_mm=rear_x,
                                  target_contacts=int(np.count_nonzero(f[1] > .1)),
                                  source_contacts=int(np.count_nonzero(f[0] > .1)),
                                  wrap_deg=self.wrap_deg()))
            self.frame('hold')
        self.frame('final')
        if self.video is not None:
            self.video.release()
        if self.renderer is not None:
            self.renderer.close()
        np.savez_compressed(self.case / 'trace.npz',
                            **{k: np.array([r[k] for r in self.rows]) for k in self.rows[0]})
        np.savez_compressed(self.case / 'hold.npz',
                            **{k: np.array([r[k] for r in hold_rows]) for k in hold_rows[0]})
        write(self.case / 'events.json', events)
        result = self.evaluate(events, hold_rows, stop, start_front_y, start_rear_x)
        write(self.case / 'result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('criterion',)},
                         ensure_ascii=False, indent=1), flush=True)
        return result

    def evaluate(self, events, hold_rows, stop, sfy, srx):
        centers = self.centers()
        front_y, rear_x = axis_progress_mm(centers, sfy, srx)
        f = self.module_forces()
        wrap = self.wrap_deg()
        n_target = int(np.count_nonzero(f[1] > .1))
        n_source = int(np.count_nonzero(f[0] > .1))
        crossed = sorted({e.get('module') for e in events if e.get('event') == 'cross_onto_target'})
        b2 = dict(front_y_mm=front_y, rear_x_mm=rear_x, crossed_modules=crossed,
                  crossed_keep=[k for k in crossed if f[1, k] > CRITERION['B2_contact_N']],
                  passed=bool(front_y >= CRITERION['B2_front_y_min_mm'] and rear_x >= CRITERION['B2_rear_x_min_mm']
                              and crossed and all(f[1, k] > CRITERION['B2_contact_N'] for k in crossed)))
        hold = dict(target_contacts_end=n_target, wrap_end_deg=wrap, source_contacts_end=n_source)
        if hold_rows:
            w2 = hold_rows[-max(1, int(2 / 0.05)):]
            p2 = np.array([[r['front_y_mm'], r['rear_x_mm']] for r in w2])
            hold['last2s_drift_mm'] = float(np.abs(p2 - p2[0]).max()) if len(p2) else None
            hold['target_contacts_last1s'] = int(min(r['target_contacts'] for r in hold_rows[-20:]))
        b3_passed = bool(hold.get('last2s_drift_mm') is not None and hold['last2s_drift_mm'] <= 6.45
                         and hold.get('target_contacts_last1s', 0) >= 3 and wrap > 180)
        off_source = n_source == 0
        return dict(start=self.start, criterion=CRITERION, stop_reason=stop,
                    pulses=[], events=events[:60], diagnostic_only=True,
                    scoring_complete=False, dynamics_admitted=DYNAMICS_ADMITTED,
                    final=dict(front_y_mm=front_y, rear_x_mm=rear_x, target_contacts=n_target,
                               source_contacts=n_source, wrap_deg=wrap,
                               transition_torque_Nm=self.transition_torque()),
                    B2=b2, B3=dict(**hold, passed=b3_passed),
                    C=dict(all_off_source=off_source, passed=bool(off_source and crossed),
                           note='整身 rolling 延续未单独运行' if not off_source else '全部离开源管'),
                    verdict='UNVALIDATED_SCORING',
                    missing_scoring='full-module drift, sustained contact history and failure-stop-qualified holding')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', nargs='+', default=['CLOSE1B', 'CLOSE2B'])
    parser.add_argument('--max-run-s', type=float, default=MAX_RUN_S)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if not DYNAMICS_ADMITTED:
        raise RuntimeError('Review only: no dynamics admission; preserve previous results')
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for start in args.starts:
        runner = Runner(start, record=not args.no_video)
        summary[start] = runner.run(max_run_s=args.max_run_s)
    write(OUT / 'phase_b_summary.json', summary)


if __name__ == '__main__':
    main()
