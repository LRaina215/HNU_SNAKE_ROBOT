"""Paused MuJoCo snapshots for the original Adaptive Pole controller.

Store the compiled model and integration state, including warm-start forces.
JSON holds controller state; no pickle and no live network/session restoration.
"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pole_compliance import ComplianceController, ComplianceParameters, PiecewiseBackbone, TargetSegment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / 'validation/results/manual-captures'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def controller_state(controller):
    if controller is None:
        return None
    from local_pitch_runtime import InteractivePitchController
    if isinstance(controller, InteractivePitchController):
        fields = ('requested', 'standard_return', 'phase_offset', 'started', 'status', 'key',
                  'ignore_feedback_risk', 'alpha_state', 'transition_basis', 'transition_elapsed',
                  'standard_elapsed', 'ready', 'geometry', 'last_sample_ms')
        values = {name: getattr(controller, name) for name in fields}
        for name in ('radius_state', 'radius_age'):
            values[name] = [float(x) if np.isfinite(x) else None for x in getattr(controller, name)]
        return dict(kind='interactive_local_pitch',
                    original=controller_state(controller._initial_controller), runtime=values)
    if type(controller).__name__ != 'ComplianceController':
        raise ValueError('当前保存功能仅支持原 Adaptive_Pole_Climbing_Mujoco 入口')
    bone = controller.backbone
    state = dict(parameters=asdict(controller.parameters), gain_is_tuple=isinstance(controller.parameters.gain, tuple),
                 expected_lengths=controller.expected_lengths,
                 phase_offset=controller.phase_offset, started=controller.started,
                 status=controller.status, key=controller.key,
                 ignore_feedback_risk=controller.ignore_feedback_risk, backbone=None)
    if bone is not None:
        if type(bone) is not PiecewiseBackbone:
            raise ValueError('当前保存功能不支持此目标曲线类型')
        state['backbone'] = dict(segments=[asdict(s) for s in bone.segments], links=bone.links,
                                pitch=bone.pitch, start_xy=bone.start_xy,
                                tangent_xy=[np.cos(bone.heading), np.sin(bone.heading)],
                                axis=bone.axis, start_z=bone.start_z, axial_sign=bone.axial_sign)
    return state


def restore_controller(state):
    if state is None:
        return None
    if state.get('kind') == 'interactive_local_pitch':
        from local_pitch_runtime import InteractivePitchController
        from local_pitch_geometry import VariablePitchBackbone
        from pole_compliance import build_backbone
        controller = InteractivePitchController(restore_controller(state['original']))
        for name, value in state['runtime'].items():
            if name in ('alpha_state', 'transition_basis', 'radius_state', 'radius_age') and value is not None:
                if name == 'radius_age':
                    value = [np.inf if x is None else x for x in value]
                value = np.asarray(value, dtype=float)
            if name == 'key' and value is not None:
                value = tuple(value)
            setattr(controller, name, value)
        if controller.geometry is not None:
            alpha = controller.alpha_state
            p = controller.parameters
            controller.backbone = (build_backbone(controller.geometry, ComplianceParameters(p.gain, float(alpha[0]), p.inside))
                if np.ptp(alpha) < 1e-14 else VariablePitchBackbone(controller.geometry, p, alpha,
                                                                 np.r_[0., np.cumsum(controller.expected_lengths)]))
        return controller
    parameters = dict(state['parameters'])
    if state['gain_is_tuple']:
        parameters['gain'] = tuple(parameters['gain'])
    controller = ComplianceController(ComplianceParameters(**parameters),
                                      state['expected_lengths'], state['ignore_feedback_risk'])
    for name in ('phase_offset', 'started', 'status'):
        setattr(controller, name, state[name])
    controller.key = tuple(state['key']) if state['key'] is not None else None
    if state['backbone'] is not None:
        args = dict(state['backbone'])
        args['segments'] = [TargetSegment(**segment) for segment in args['segments']]
        controller.backbone = PiecewiseBackbone(**args)
    return controller


def save_checkpoint(model, data, settings, controller, geometry, source_model, directory=None):
    """Call only on the physics thread at an idle single-step boundary."""
    serialized_controller = controller_state(controller)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    folder = Path(directory) if directory is not None else DEFAULT_DIRECTORY / (
        f'{stamp}_{Path(source_model).stem}_step{settings["completed"]}')
    folder.mkdir(parents=True, exist_ok=False)
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_type))
    mujoco.mj_getState(model, data, state, state_type)
    mujoco.mj_saveModel(model, str(folder / 'model.mjb'))
    np.savez_compressed(folder / 'state.npz', integration_state=state,
                        qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy())
    metadata = dict(format_version=1, mujoco_version=mujoco.__version__, state_type=int(state_type),
                    captured_at=stamp, source_model=str(Path(source_model).resolve()),
                    sim_time=float(data.time), settings=settings, controller=serialized_controller,
                    geometry_at_capture=geometry,
                    hashes={name: sha(folder / name) for name in ('model.mjb', 'state.npz')},
                    source_hashes={name: sha(ROOT / 'codes' / name) for name in
                                   ('Adaptive_Pole_Climbing_Mujoco.py', 'pole_compliance.py',
                                    'mujoco_checkpoint.py')})
    (folder / 'checkpoint.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                     default=json_value, allow_nan=False), encoding='utf-8')
    return folder


def load_checkpoint(directory, verify_hashes=True):
    folder = Path(directory).resolve()
    metadata = json.loads((folder / 'checkpoint.json').read_text(encoding='utf-8'))
    if metadata['format_version'] != 1 or metadata['mujoco_version'] != mujoco.__version__:
        raise ValueError(f'请使用保存时的 MuJoCo {metadata["mujoco_version"]} 恢复此状态')
    for name, expected in (metadata['hashes'].items() if verify_hashes else ()):
        if sha(folder / name) != expected:
            raise ValueError(f'保存文件校验失败：{name}')
    model = mujoco.MjModel.from_binary_path(str(folder / 'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(folder / 'state.npz', allow_pickle=False) as saved:
        state = saved['integration_state'].copy()
    mujoco.mj_setState(model, data, state, metadata['state_type'])
    mujoco.mj_forward(model, data)
    # Forward refreshes geometry but also updates the solver warm start. Keep
    # the original warm start so subsequent integration matches the saved run.
    mujoco.mj_setState(model, data, state, metadata['state_type'])
    return model, data, metadata, restore_controller(metadata['controller'])
