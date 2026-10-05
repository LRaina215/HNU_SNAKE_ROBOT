"""Find static support torques with MuJoCo's full elliptic wrench cones.

Read-only posture diagnosis. A feasible static solution is not a dynamic
stability guarantee; failure of the numerical solver proves no impossibility.
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog, minimize
from three_segment_review import context
from segmented_pole_experiment import write_json


def solve(model, qpos):
    m = model
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    blocks, spans, scales = [], [], []
    count = 0
    for c in d.contact:
        if c.efc_address < 0:
            continue
        j1 = np.zeros((3, m.nv)); r1 = np.zeros_like(j1)
        j2 = np.zeros_like(j1); r2 = np.zeros_like(j1)
        mujoco.mj_jac(m, d, j1, r1, c.pos, m.geom_bodyid[c.geom1])
        mujoco.mj_jac(m, d, j2, r2, c.pos, m.geom_bodyid[c.geom2])
        frame = c.frame.reshape(3, 3)
        scale = np.r_[1., c.friction[:c.dim-1]]
        block = np.column_stack(((j2-j1).T @ frame.T, (r2-r1).T @ frame.T))[:, :c.dim]
        blocks.append(block * scale)
        spans.append((count, count+c.dim)); scales.extend(scale)
        count += c.dim
    if not blocks:
        return dict(feasible_solution_found=False, reason='no active contacts')
    selection = np.zeros((m.nv, m.nu))
    selection[m.jnt_dofadr[m.actuator_trnid[:, 0]], np.arange(m.nu)] = m.actuator_forcerange[:, 1]
    equality = np.column_stack((*blocks, selection, np.zeros(m.nv)))
    rhs = d.qfrc_bias-d.qfrc_passive
    size = count+m.nu+1
    objective = np.zeros(size); objective[-1] = 1.
    bounds = [(None, None)] * size
    box = []
    for lo, hi in spans:
        bounds[lo] = (0., None)
        for j in range(lo+1, hi):
            for sign in (-1., 1.):
                row = np.zeros(size); row[lo] = -1.; row[j] = sign; box.append(row)
    torque_rows = []
    for j in range(m.nu):
        for sign in (-1., 1.):
            row = np.zeros(size); row[count+j] = sign; row[-1] = -1.
            torque_rows.append(row)
    bounds[-1] = (0., None)
    box = np.asarray(box+torque_rows)
    relaxed = linprog(objective, A_ub=box, b_ub=np.zeros(len(box)), A_eq=equality,
                      b_eq=rhs, bounds=bounds, method='highs')
    if not relaxed.success:
        return dict(feasible_solution_found=False, reason=relaxed.message)
    torque_rows = np.asarray(torque_rows)

    def cone(x):
        return np.array([x[lo]-np.linalg.norm(x[lo+1:hi]) for lo, hi in spans])

    def cone_jac(x):
        jac = np.zeros((len(spans), size))
        for k, (lo, hi) in enumerate(spans):
            jac[k, lo] = 1.
            norm = np.linalg.norm(x[lo+1:hi])
            if norm > 1e-12:
                jac[k, lo+1:hi] = -x[lo+1:hi]/norm
        return jac

    result = minimize(lambda x: float(x[-1]), relaxed.x, jac=lambda x: objective,
        bounds=bounds, method='SLSQP', constraints=[
            dict(type='eq', fun=lambda x: equality @ x-rhs, jac=lambda x: equality),
            dict(type='ineq', fun=cone, jac=cone_jac),
            dict(type='ineq', fun=lambda x: -torque_rows @ x, jac=lambda x: -torque_rows)],
        options=dict(ftol=1e-10, maxiter=1000))
    residual = float(np.max(abs(equality @ result.x-rhs)))
    violation = float(max(0., -np.min(cone(result.x)), np.max(torque_rows @ result.x)))
    valid = bool(result.success and residual < 1e-7 and violation < 1e-7)
    return dict(feasible_solution_found=valid,
        within_actuator_capacity=bool(valid and result.x[-1] <= 1.+1e-7),
        relaxed_torque_lower_bound=float(relaxed.fun),
        solver_success=bool(result.success), message=result.message,
        maximum_normalized_torque=float(result.x[-1]), static_equality_residual=residual,
        cone_or_torque_constraint_violation=violation,
        joint_torques=(result.x[count:count+m.nu]*m.actuator_forcerange[:, 1]).tolist(),
        contact_wrenches=(result.x[:count]*scales).tolist(), contacts=len(spans))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path); p.add_argument('--times', type=float, nargs='+', required=True)
    args = p.parse_args()
    _, z, m, _, _ = context(args.folder)
    time = z['time']-z['time'][0]
    rows = []
    for seconds in args.times:
        i = int(np.argmin(abs(time-seconds)))
        row = solve(m, z['qpos'][i]); row['time_seconds'] = float(time[i]); rows.append(row)
    report = dict(friction_cone='elliptic, full MuJoCo contact dimensions',
        method='convex static wrench problem, SLSQP initialized by box relaxation', rows=rows,
        limitation='Static existence at recorded geometry only; not a dynamic stability or controller guarantee.')
    write_json(args.folder/'elliptic_support_check.json', report)
    for row in rows:
        print({k: v for k, v in row.items() if k not in ('joint_torques', 'contact_wrenches')})
