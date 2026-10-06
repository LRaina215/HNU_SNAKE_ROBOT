"""第4步滚动约束版可行性（离线微分运动学，不推进物理时间）。

问题：从 CLOSE1B 抱稳状态出发，在接触模块持续贴管且不打滑的条件下，是否存在连续的关节运动，
使前部沿目标管远离接头（+Y）、后部沿源管靠近接头（+X）？能走多远，被什么挡住？

接触模型：每个接触模块与管为点接触，接触点（连杆轴线上离管轴最近的点向管轴方向偏移体半径）
的瞬时速度为零：J_c·qvel = 0（法向另加贴管半径的漂移校正）。连杆可绕过接触点的任意轴转动，
滚动、翻滚、自旋都允许，打滑不允许。root 不受驱动，速度由约束决定。
在约束零空间内用加权最小二乘找同时满足“前部 +Y、后部 +X”的运动，并加关节正则、限位回推、
自由模块离管余量；每步关节变化 ≤0.2°，用 mj_integratePos 积分。
接触切换：自由模块与目标管/源管真实几何发生接触即落管（加入约束）；源管接触点走到接头斜切区
（x>1040 mm）或目标管接触点退到斜切区（y<60 mm）即释放。
只检查运动学，不检查受力（重力、力矩上限、摩擦锥都不在模型里）。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
import mujoco
import numpy as np
from scipy.linalg import null_space

from curve_guided_geometry import MODEL

BASE = ROOT / 'experiments/hook_pull_20261006'
OUT = BASE / 'step4_rolling'
START = BASE / 'CLOSE1B/final_state.npz'
R_B = .029
PIPES = {'target': (np.array([1.1, 0., 1.]), np.array([0., 1., 0.])),
         'source': (np.array([0., 0., 1.]), np.array([1., 0., 0.]))}
MITER = {'target': lambda foot: foot[1] < .060, 'source': lambda foot: foot[0] > 1.040}
DQ_MAX = np.radians(.2)
LIMIT_SOFT, LIMIT_GATE = np.radians(80.), np.radians(88.)
K_R = 5.
# 层级权重（统一量纲：进度 mm/s，关节 rad/s）：进度近似硬约束 ≫ 离管余量 > 限位回推 ≫ 关节最小化
W_PROG, W_CLEAR, W_LIMIT, LAMBDA = 100., 10., 1., .05
K_LIMIT, CLEAR_MARGIN, K_CLEAR = .5, .006, 2.
MAX_STEPS, REPORT_EVERY = 8000, 250
FREE_FLOOR = {}
# 新落管模块的接触模型：noslip=加入不打滑等式约束（运行前登记的模型）；
# frictionless=只防穿管、可滑可离（变体：假设新落管时法向力很小）。起始承力接触始终不打滑。
LANDING = 'noslip'
# 运行前登记的判定标准（不得事后修改）
CRITERION = dict(front_y_min_mm=25., rear_x_min_mm=25.,
                 gates='关节 ≤88°、无非相邻自碰、不触地、穿管 ≤3 mm、前部包绕 ≥180°、两管各至少 1 个接触、未停滞')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


class Model:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.head = self.m.body('head').id
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.pipe_geoms = {'source': {self.m.geom(x).id for x in ['pipe_straight_01', 'pipe_sharp_in_02']},
                           'target': {self.m.geom(x).id for x in ['pipe_sharp_out_03', 'pipe_straight_04']}}
        self.floor = self.m.geom('floor').id
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

    def update(self):
        mujoco.mj_fwdPosition(self.m, self.d)

    def nodes(self):
        d = self.d
        return np.vstack([d.site('s_head').xpos, *[d.joint('joint' + str(i)).xanchor for i in range(1, 21)],
                          d.site('s_tail').xpos])

    def jac(self, point, module):
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, np.asarray(point, float), self.head + module)
        return self.jacp.copy(), self.jacr.copy()

    def touching(self):
        """真实几何下的 (模块, 管) 接触与最深穿透（m）；另报触地与非相邻自碰。"""
        pairs, deepest, ground, selfhit = {}, 0., False, False
        for c in self.d.contact:
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            ground |= (c.geom1 == self.floor and b2 >= self.head) or (c.geom2 == self.floor and b1 >= self.head)
            selfhit |= b1 >= self.head and b2 >= self.head and abs(b1 - b2) > 1 and c.dist < -5e-4
            for geom, body in [(c.geom1, b2), (c.geom2, b1)]:
                if body < self.head:
                    continue
                for name, ids in self.pipe_geoms.items():
                    if geom in ids:
                        key = (body - self.head, name)
                        pairs[key] = min(pairs.get(key, 1.), float(c.dist))
                        deepest = min(deepest, float(c.dist))
        return pairs, deepest, bool(ground), bool(selfhit)


def closest(a, b, pipe):
    """连杆轴线段 a-b 上离管轴最近的点 x、其在管轴上的垂足 foot。"""
    p0, axis = PIPES[pipe]
    u, w = b - a, a - p0
    u_p, w_p = u - (u @ axis) * axis, w - (w @ axis) * axis
    t = float(np.clip(-(w_p @ u_p) / (u_p @ u_p), 0., 1.)) if u_p @ u_p > 1e-12 else .5
    x = a + t * u
    foot = p0 + ((x - p0) @ axis) * axis
    return x, foot


def radial(x, foot):
    v = x - foot
    r = float(np.linalg.norm(v))
    return r, v / r


def wrap_deg(nodes):
    centers = (nodes[:-1] + nodes[1:]) / 2
    uv = (centers[:9] - PIPES['target'][0])[:, [0, 2]][::-1]
    a = np.arctan2(uv[:, 1], uv[:, 0])
    return float(np.degrees(abs(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def priority_solve(Aeq, beq, P, p, nv, reg_index):
    """一级：等式约束精确满足；二级：进度在其零空间内尽量满足；三级：关节速度最小。"""
    if len(Aeq):
        particular = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
        N = null_space(Aeq, rcond=1e-9)
    else:
        particular, N = np.zeros(nv), np.eye(nv)
    if N.shape[1] == 0:
        return particular
    base = particular
    N_rest = N
    if len(P):
        M1 = P @ N
        z1 = np.linalg.pinv(M1, rcond=1e-6) @ (p - P @ particular)
        base = particular + N @ z1
        N2 = null_space(M1, rcond=1e-6)
        N_rest = N @ N2 if N2.shape[1] else np.zeros((nv, 0))
    if N_rest.shape[1]:
        R = np.zeros((len(reg_index), nv))
        R[np.arange(len(reg_index)), reg_index] = 1.
        w = np.linalg.lstsq(R @ N_rest, -R @ base, rcond=None)[0]
        base = base + N_rest @ w
    return base


def solve_active(model, contacts, rho0, target):
    """接触不打滑为硬约束；限位、离管、自碰为不等式（主动集，只阻止继续变坏）。target=(前+Y, 后+X) mm/s。"""
    nodes = model.nodes()
    centers = (nodes[:-1] + nodes[1:]) / 2
    nv = model.m.nv
    E, e = [], []
    for (k, pipe) in contacts:
        x, foot = closest(nodes[k], nodes[k + 1], pipe)
        r, n = radial(x, foot)
        E.append(model.jac(x - R_B * n, k)[0])
        e.append(-K_R * (r - rho0[(k, pipe)]) * n)
    E = np.vstack(E) if E else np.zeros((0, nv))
    e = np.concatenate(e) if len(e) else np.zeros(0)
    front = sorted({k for k, p in contacts if p == 'target'})
    rear = sorted({k for k, p in contacts if p == 'source'})
    P, p = [], []
    if front and target[0] is not None:
        P.append(1000. * np.mean([model.jac(centers[k], k)[0][1] for k in front], axis=0))
        p.append(target[0])
    if rear and target[1] is not None:
        P.append(1000. * np.mean([model.jac(centers[k], k)[0][0] for k in rear], axis=0))
        p.append(target[1])
    P, p = np.asarray(P).reshape(-1, nv), np.asarray(p)
    G, h, tags = [], [], []
    q = model.d.qpos[model.qadr]
    for j in range(20):
        if abs(q[j]) > LIMIT_SOFT:
            row = np.zeros(nv)
            row[model.vadr[j]] = -np.sign(q[j])
            G.append(row)
            h.append(0.)
            tags.append(f'limit J{j + 1}')
    in_contact = {k for k, _ in contacts}
    for k in range(21):
        if k in in_contact:
            continue
        for pipe in PIPES:
            x, foot = closest(nodes[k], nodes[k + 1], pipe)
            if MITER[pipe](foot):
                continue
            r, n = radial(x, foot)
            floor = min(.084, FREE_FLOOR.setdefault((k, pipe), r)) - .0005
            if r < floor + CLEAR_MARGIN:
                # 只禁止继续往管里走（以起始半径为底，壳体网格比 29 mm 圆柱略瘦），不要求弹开
                G.append(n @ model.jac(x, k)[0])
                h.append(-K_R * max(r - floor, 0.))
                tags.append(f'clear M{k}-{pipe}')
    for i in range(21):
        for j in range(i + 2, 21):
            dvec = centers[i] - centers[j]
            dist = float(np.linalg.norm(dvec))
            if dist < .062:
                u = dvec / dist
                G.append(u @ (model.jac(centers[i], i)[0] - model.jac(centers[j], j)[0]))
                h.append(-K_R * (dist - .058))
                tags.append(f'self M{i}-M{j}')
    G, h = np.asarray(G).reshape(-1, nv), np.asarray(h)
    active = []
    for _ in range(30):
        Aeq = np.vstack([E, G[active]]) if active else E
        beq = np.concatenate([e, h[active]]) if active else e
        qvel = priority_solve(Aeq, beq, P, p, nv, model.vadr)
        slack = G @ qvel - h if len(G) else np.zeros(0)
        candidates = [i for i in range(len(G)) if i not in active and slack[i] < -1e-9]
        if not candidates:
            break
        active.append(min(candidates, key=lambda i: slack[i]))
    achieved = P @ qvel if len(P) else np.zeros(0)
    return qvel, dict(active=[tags[i] for i in active], achieved=achieved.tolist(),
                      requested=p.tolist(), contact_residual=float(np.linalg.norm(E @ qvel - e)) if len(E) else 0.)


def solve_step(model, contacts, rho0, mode):
    """mode: (front_target, rear_target)，各为期望的前部 +Y、后部 +X 速度（无量纲）。"""
    nodes = model.nodes()
    nv = model.m.nv
    rows, rhs = [], []
    for (k, pipe) in contacts:
        x, foot = closest(nodes[k], nodes[k + 1], pipe)
        r, n = radial(x, foot)
        jp, _ = model.jac(x - R_B * n, k)
        rows.append(jp)
        rhs.append(-K_R * (r - rho0[(k, pipe)]) * n)
    Jc = np.vstack(rows) if rows else np.zeros((0, nv))
    bc = np.concatenate(rhs) if rhs else np.zeros(0)
    particular = np.linalg.lstsq(Jc, bc, rcond=None)[0] if rows else np.zeros(nv)
    N = null_space(Jc, rcond=1e-9) if rows else np.eye(nv)

    A, b = [], []
    front = sorted({k for k, p in contacts if p == 'target'})
    rear = sorted({k for k, p in contacts if p == 'source'})
    centers = (nodes[:-1] + nodes[1:]) / 2
    # 进度（mm/s）：目标量为期望速度，量纲与关节项统一后用大权重近似硬约束
    if front and mode[0] is not None:
        row = 1000. * np.mean([model.jac(centers[k], k)[0][1] for k in front], axis=0)
        A.append(W_PROG * row)
        b.append(W_PROG * mode[0])
    if rear and mode[1] is not None:
        row = 1000. * np.mean([model.jac(centers[k], k)[0][0] for k in rear], axis=0)
        A.append(W_PROG * row)
        b.append(W_PROG * mode[1])
    reg = np.zeros((20, nv))
    reg[np.arange(20), model.vadr] = LAMBDA
    A.extend(reg)
    b.extend(np.zeros(20))
    q = model.d.qpos[model.qadr]
    for j in np.flatnonzero(abs(q) > LIMIT_SOFT):
        row = np.zeros(nv)
        row[model.vadr[j]] = W_LIMIT
        A.append(row)
        b.append(-W_LIMIT * K_LIMIT * (abs(q[j]) - LIMIT_SOFT) * np.sign(q[j]))
    in_contact = {k for k, _ in contacts}
    for k in range(21):
        if k in in_contact:
            continue
        for pipe in PIPES:
            x, foot = closest(nodes[k], nodes[k + 1], pipe)
            if MITER[pipe](foot):
                continue
            r, n = radial(x, foot)
            limit = .084 + CLEAR_MARGIN
            if r < limit:
                # 离管余量（mm/s）：不足时要求向外的速度
                A.append(W_CLEAR * 1000. * (n @ model.jac(x, k)[0]))
                b.append(W_CLEAR * K_CLEAR * 1000. * (limit - r))
    A, b = np.asarray(A), np.asarray(b)
    n_prog = int(front and mode[0] is not None) + int(rear and mode[1] is not None)
    # 任务优先级：一级=进度（在接触零空间内精确满足），二级=限位/离管/关节最小化（在一级零空间内）
    Ap, bp = A[:n_prog] / W_PROG, b[:n_prog] / W_PROG
    As, bs = A[n_prog:], b[n_prog:]
    if n_prog:
        M1 = Ap @ N
        z1 = np.linalg.pinv(M1, rcond=1e-6) @ (bp - Ap @ particular)
        N2 = null_space(M1, rcond=1e-6)
        prog_residual = float(np.linalg.norm(M1 @ z1 - (bp - Ap @ particular)))
    else:
        z1, N2, prog_residual = np.zeros(N.shape[1]), np.eye(N.shape[1]), 0.
    if N2.shape[1]:
        base = particular + N @ z1
        w = np.linalg.lstsq(As @ N @ N2, bs - As @ base, rcond=None)[0]
        z = z1 + N2 @ w
    else:
        z = z1
    qvel = particular + N @ z
    residual = float(np.linalg.norm(Jc @ qvel - bc)) if rows else 0.
    return qvel, residual + prog_residual, N.shape[1], Jc


def progress_rates(model, contacts, qvel):
    nodes = model.nodes()
    centers = (nodes[:-1] + nodes[1:]) / 2
    front = sorted({k for k, p in contacts if p == 'target'})
    rear = sorted({k for k, p in contacts if p == 'source'})
    fy = float(np.mean([model.jac(centers[k], k)[0][1] @ qvel for k in front])) if front else 0.
    rx = float(np.mean([model.jac(centers[k], k)[0][0] @ qvel for k in rear])) if rear else 0.
    joint = float(np.abs(qvel[model.vadr]).max())
    return fy, rx, joint


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = Model()
    with np.load(START) as state:
        mujoco.mj_setState(model.m, model.d, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    model.d.qvel[:] = 0
    model.update()
    pairs, _, _, _ = model.touching()
    contacts = sorted(k for k, dist in pairs.items() if dist < 1e-3)
    nodes = model.nodes()
    rho0 = {}
    for (k, pipe) in contacts:
        x, foot = closest(nodes[k], nodes[k + 1], pipe)
        rho0[(k, pipe)] = radial(x, foot)[0]
    start_contacts = [f'M{k}-{p}' for k, p in contacts]
    print('起始接触：', start_contacts, flush=True)

    # ---- Q1：起点瞬时可行方向（接触不打滑为硬约束，进度目标 1 mm/s） ----
    instant = {}
    for name, mode in [('both', (1., 1.)), ('front_only', (1., None)), ('rear_only', (None, 1.)),
                       ('front_hold_rear', (1., 0.)), ('rear_hold_front', (0., 1.))]:
        qvel, info = solve_active(model, contacts, rho0, mode)
        fy, rx, jmax = progress_rates(model, contacts, qvel)
        per_deg = (1. / np.degrees(jmax)) if jmax > 1e-12 else 0.
        instant[name] = dict(front_y_mm_s=fy * 1000, rear_x_mm_s=rx * 1000,
                             max_joint_deg_s=float(np.degrees(jmax)),
                             front_y_mm_per_joint_deg=fy * 1000 * per_deg, rear_x_mm_per_joint_deg=rx * 1000 * per_deg,
                             active_inequalities=info['active'], contact_residual=info['contact_residual'])
        print(f'瞬时 {name:16s}: 前部 +Y {fy * 1000:+6.3f} mm/s 后部 +X {rx * 1000:+6.3f} mm/s '
              f'所需最大关节速度 {np.degrees(jmax):6.2f}°/s 起作用的不等式 {info["active"]}', flush=True)

    # ---- Q2：积分 ----
    roll = np.zeros(21)
    start_nodes = model.nodes()
    start_front_y = float(((start_nodes[:-1] + start_nodes[1:]) / 2)[:8, 1].mean())
    start_rear_x = float(((start_nodes[:-1] + start_nodes[1:]) / 2)[13:21, 0].mean())
    log, events = [], []
    landed = set()
    stop = 'step_cap'
    window = []
    for step in range(MAX_STEPS):
        qvel, info = solve_active(model, contacts, rho0, (1., 1.))
        res, dim = info['contact_residual'], len(info['active'])
        jmax = float(np.abs(qvel[model.vadr]).max())
        if jmax < 1e-12:
            stop = 'no_motion'
            break
        dt = DQ_MAX / jmax
        # 材料滚动：各模块角速度在自身纵轴上的分量
        for k in range(21):
            t = model.d.xmat[model.head + k].reshape(3, 3)[:, 2]
            _, jr = model.jac(model.d.xpos[model.head + k], k)
            roll[k] += float(t @ (jr @ qvel)) * dt
        qpos = model.d.qpos.copy()
        mujoco.mj_integratePos(model.m, qpos, qvel, dt)
        model.d.qpos[:] = qpos
        model.update()
        nodes = model.nodes()
        centers = (nodes[:-1] + nodes[1:]) / 2
        pairs, deepest, ground, selfhit = model.touching()
        q = model.d.qpos[model.qadr]
        # 接触切换
        for key, dist in pairs.items():
            if key not in contacts and key not in landed and dist < 3e-4:
                k, pipe = key
                x, foot = closest(nodes[k], nodes[k + 1], pipe)
                if not MITER[pipe](foot):
                    landed.add(key)
                    events.append(dict(step=step, event='land', module=k, pipe=pipe, model=LANDING))
                    if LANDING == 'noslip':
                        contacts.append(key)
                        rho0[key] = radial(x, foot)[0]
        for key in list(contacts):
            k, pipe = key
            x, foot = closest(nodes[k], nodes[k + 1], pipe)
            if MITER[pipe](foot):
                contacts.remove(key)
                events.append(dict(step=step, event='release_at_miter', module=k, pipe=pipe))
        contacts.sort()
        front_y = float(centers[:8, 1].mean()) - start_front_y
        rear_x = float(centers[13:21, 0].mean()) - start_rear_x
        wrap = wrap_deg(nodes)
        row = dict(step=step, front_y_mm=front_y * 1000, rear_x_mm=rear_x * 1000, max_joint_deg=float(np.degrees(abs(q)).max()),
                   deepest_mm=deepest * 1000, wrap_deg=wrap, n_contacts=len(contacts),
                   n_target=sum(p == 'target' for _, p in contacts), n_source=sum(p == 'source' for _, p in contacts),
                   n_active=dim, active=info['active'], achieved_mm_s=info['achieved'], constraint_residual=res,
                   step_mm_per_joint_deg=[a * dt / np.degrees(DQ_MAX) for a in info['achieved']],
                   roll_front_deg=float(np.degrees(roll[:8].mean())), roll_rear_deg=float(np.degrees(roll[13:21].mean())))
        log.append(row)
        window.append((front_y, rear_x))
        if len(window) > 300:
            window.pop(0)
        failed = []
        if np.abs(q).max() > LIMIT_GATE:
            failed.append(f'joint_limit J{int(np.argmax(np.abs(q))) + 1}')
        if selfhit:
            failed.append('nonadjacent_self_contact')
        if ground:
            failed.append('ground')
        if deepest < -3e-3:
            failed.append('pipe_penetration')
        if wrap < 180.:
            failed.append('front_wrap_below_180')
        if not any(p == 'source' for _, p in contacts):
            failed.append('no_source_contact')
        if not any(p == 'target' for _, p in contacts):
            failed.append('no_target_contact')
        if len(window) == 300 and (window[-1][0] - window[0][0]) < 2e-4 and (window[-1][1] - window[0][1]) < 2e-4:
            failed.append('stall')
        if failed:
            stop = ','.join(failed)
            break
        if step % REPORT_EVERY == 0:
            print(f"step {step:5d}: 前部 +Y {row['front_y_mm']:6.1f} mm 后部 +X {row['rear_x_mm']:6.1f} mm "
                  f"|q|max {row['max_joint_deg']:5.1f}° 包绕 {wrap:5.1f}° 接触 T{row['n_target']}/S{row['n_source']} "
                  f"材料滚动 前{row['roll_front_deg']:6.1f}° 后{row['roll_rear_deg']:6.1f}° 最深 {row['deepest_mm']:5.2f} mm "
                  f"达成 {np.round(info['achieved'], 3).tolist()} 起作用 {info['active']}", flush=True)

    last = log[-1] if log else {}
    peak_front = max((r['front_y_mm'] for r in log), default=0.)
    peak_rear = max((r['rear_x_mm'] for r in log), default=0.)
    passed = bool(last and last['front_y_mm'] >= CRITERION['front_y_min_mm']
                  and last['rear_x_mm'] >= CRITERION['rear_x_min_mm'])
    result = dict(
        question='接触模块持续贴管且不打滑时，前部 +Y（远离接头）、后部 +X（靠近接头）能否同时连续进行',
        landing_model=LANDING, criterion=CRITERION, verdict='ROLLING_FEASIBLE' if passed else 'NOT_ESTABLISHED',
        peak_front_y_mm=peak_front, peak_rear_x_mm=peak_rear,
        start_state='CLOSE1B final', start_contacts=start_contacts, instantaneous=instant,
        integration=dict(steps=len(log), stop_reason=stop,
                         front_y_advance_mm=last.get('front_y_mm'), rear_x_advance_mm=last.get('rear_x_mm'),
                         final_max_joint_deg=last.get('max_joint_deg'), final_wrap_deg=last.get('wrap_deg'),
                         final_contacts=[f'M{k}-{p}' for k, p in contacts],
                         roll_front_deg=last.get('roll_front_deg'), roll_rear_deg=last.get('roll_rear_deg'),
                         active_at_stop=last.get('active'), achieved_at_stop_mm_s=last.get('achieved_mm_s'),
                         max_constraint_residual=max((r['constraint_residual'] for r in log), default=0.)),
        events=events,
        model_limits='只检查运动学：重力、力矩上限、摩擦锥、法向力正负都不在模型里；接触为双边点接触')
    write(OUT / f'step4_rolling_landing_{LANDING}.json', result)
    write(OUT / f'step4_rolling_log_landing_{LANDING}.json', log[::10] + ([log[-1]] if log else []))
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, ensure_ascii=False, indent=1), flush=True)
    print('事件：', events[:20], flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--landing', choices=('noslip', 'frictionless'), default='noslip')
    LANDING = parser.parse_args().landing
    main()
