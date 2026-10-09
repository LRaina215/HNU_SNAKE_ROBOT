"""Units and lossless state loading for the reviewed A-prime/B branch only."""
import mujoco
import numpy as np

R95_MM, PITCH_MM = 95., 220.
L_TURN_MM = float(np.hypot(2 * np.pi * R95_MM, PITCH_MM))
ROT_PER_MM = 2 * np.pi / L_TURN_MM  # rad per mm of helical arc
AX_PER_MM = PITCH_MM / L_TURN_MM   # mm axial advance per mm of arc


def restore_saved_state(model, data, path):
    """Restore all INTEGRATION fields; never replace ctrl by measured qpos.

    The separate ctrl/time fields are consistency checks, not replacement values.
    psi is the saved CLOSE controller history, not a new rolling command.
    mj_forward computes diagnostics without advancing physical time.
    """
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    with np.load(path) as saved:
        state = saved['integration_state'].copy()
        if state.shape != (mujoco.mj_stateSize(model, spec),) or not np.isfinite(state).all():
            raise ValueError('Saved integration state does not match this model')
        ctrl = saved['ctrl'].copy()
        history = {key: float(saved[key]) for key in ('psi_rad', 'sim_time') if key in saved}
    mujoco.mj_setState(model, data, state, spec)
    if ctrl.shape != data.ctrl.shape or not np.array_equal(data.ctrl, ctrl):
        raise ValueError('Separate saved ctrl disagrees with integration_state')
    if 'sim_time' in history and history['sim_time'] != float(data.time):
        raise ValueError('Separate saved time disagrees with integration_state')
    qpos, qvel, targets = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    warmstart = data.qacc_warmstart.copy()
    mujoco.mj_forward(model, data)
    # Forward diagnosis refreshes the solver warmstart in MuJoCo 3.2.7.
    # Loading happens before this runner's first physical step; retain the
    # saved solver history too, rather than only preserving qpos/qvel/ctrl.
    data.qacc_warmstart[:] = warmstart
    if not all(np.array_equal(a, b) for a, b in
               ((qpos, data.qpos), (qvel, data.qvel), (targets, data.ctrl))):
        raise RuntimeError('Forward evaluation changed restored qpos/qvel/ctrl')
    return history


def axis_progress_mm(centers_mm, start_front_y_mm, start_rear_x_mm):
    """Both point coordinates and the resulting net displacements are mm."""
    centers = np.asarray(centers_mm)
    return (float(centers[:8, 1].mean() - start_front_y_mm),
            float(centers[13:21, 0].mean() - start_rear_x_mm))


def continuous_target(last_sent, desired, rate_rad_s, dt_s):
    """Rate limit from the executed target; this does not preserve preload.

    In particular desired=measured qpos can unload the restored clamp even
    though every individual command satisfies the target-rate limit.
    """
    last_sent, desired = np.asarray(last_sent), np.asarray(desired)
    if last_sent.shape != desired.shape or not np.isfinite(desired).all():
        raise ValueError('Finite joint targets of matching shape required')
    if np.any(abs(desired) > np.pi / 2):
        raise ValueError('Desired target exceeds existing joint range')
    return last_sent + np.clip(desired - last_sent, -rate_rad_s * dt_s, rate_rad_s * dt_s)
