"""Gated contact-turn research. Simulation only; commands use the existing EXE relay.

Stage 0 never substitutes wall-clock waiting for physical holding. Raw snapshots
are taken after mj_forward at the recorded qpos/time, at 100 physical Hz.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'visualizer')]
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
from mujoco_execution_bridge import MujocoExecutionBridge
from python_client import SnakeControlClient

BASE = ROOT / 'experiments/contact_turn_20261003'


class ProbeLimit(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def straight_model(out):
    """Independent horizontal negative control; preserve robot and dynamics."""
    source = ROOT / 'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
    tree = ET.parse(source)
    root = tree.getroot()
    original = mujoco.MjModel.from_xml_path(str(source))
    line = original.numeric('pipe_centerline').data.reshape(-1, 4)
    axis = line[1, :3] - line[0, :3]
    axis /= np.linalg.norm(axis)
    origin = line[0, :3] - axis * np.dot(line[0, :3], axis)
    template = next(e for e in root.iter('geom') if e.get('name', '').startswith('pipe'))
    properties = {k: template.get(k) for k in ('friction', 'condim', 'group', 'material', 'rgba')}
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('name', '').startswith('pipe'):
                parent.remove(child)
    for mesh in root.findall('asset/mesh'):
        if mesh.get('file'):
            mesh.set('file', str((source.parent / mesh.get('file')).resolve()))
    ET.SubElement(root.find('worldbody'), 'geom', name='pipe_source', type='cylinder',
                  size=str(line[0, 3]), fromto=' '.join(map(str, np.r_[origin-.2*axis, origin+2.6*axis])),
                  **properties)
    for num in root.findall('custom/numeric'):
        if num.get('name') in ('pipe_centerline', 'feature_span', 'pipe_unused_branch'):
            root.find('custom').remove(num)
    ET.SubElement(root.find('custom'), 'numeric', name='pipe_centerline',
                  data=' '.join(map(str, np.r_[origin, line[0, 3], origin+2.6*axis, line[0, 3]])))
    path = out / 'horizontal_straight.xml'
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf8', xml_declaration=True)
    derived = mujoco.MjModel.from_xml_path(str(path))
    for name in ('body_mass', 'body_inertia', 'jnt_axis', 'jnt_range', 'dof_damping',
                 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange'):
        a,b = getattr(original,name),getattr(derived,name)
        if name.startswith('body_'):
            a,b = a[original.body('head').id:],b[derived.body('head').id:]
        assert np.array_equal(a,b), name
    assert np.array_equal(original.key_qpos, derived.key_qpos)
    return path, origin, axis


class Simulation:
    def __init__(self, model_path, directory, initial_phase=0.):
        assert mujoco.__version__ == '3.3.0'
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.model = m = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = d = mujoco.MjData(m)
        assert m.nu == 20 and m.opt.timestep == .001
        self.qadr = np.array([int(m.joint(f'joint{i}').qposadr[0]) for i in range(1, 21)])
        self.vadr = np.array([int(m.joint(f'joint{i}').dofadr[0]) for i in range(1, 21)])
        self.jids = np.array([m.joint(f'joint{i}').id for i in range(1, 21)])
        assert np.array_equal(m.actuator_trnid[:, 0], self.jids)
        mujoco.mj_resetDataKeyframe(m, d, m.key('experiment_start').id)
        line = m.numeric('pipe_centerline').data.reshape(-1, 4)
        self.axis = (line[1, :3]-line[0, :3]) / np.linalg.norm(line[1, :3]-line[0, :3])
        self.origin = line[0, :3] - self.axis*np.dot(line[0, :3], self.axis)
        if initial_phase:
            nominal = HelixBackbone.from_radius_pitch(81.5, 220.)
            basis = nominal.joint_angles(20, 64.5, 0.)
            initial_kappa = nominal.kappa * np.dot(d.qpos[self.qadr], basis)/np.dot(basis,basis)
            initial_helix = HelixBackbone(initial_kappa, nominal.tau)
            d.qpos[self.qadr] = initial_helix.joint_angles(20,64.5,initial_phase)
            mujoco.mj_forward(m,d)
            pts = np.vstack([d.joint(f'joint{i}').xanchor for i in range(1,21)])
            original_center = pts.mean(0)
            tangents = np.diff(pts,axis=0)
            tangents /= np.linalg.norm(tangents,axis=1)[:,None]
            _,_,vt = np.linalg.svd(tangents-tangents.mean(0))
            estimated_axis = vt[-1]
            if estimated_axis@(pts[0]-pts[-1])<0:
                estimated_axis = -estimated_axis
            rot = Rotation.align_vectors([self.axis],[estimated_axis])[0]
            aligned = rot.apply(pts-original_center)
            perpendicular = np.array([0.,1.,0.])
            other = np.cross(self.axis,perpendicular)
            xy = aligned@np.column_stack([perpendicular,other])
            circle = least_squares(lambda x:np.linalg.norm(xy-x[:2],axis=1)-x[2], [0.,0.,.096]).x
            shift = self.origin+self.axis*.5 - perpendicular*circle[0]-other*circle[1]
            d.qpos[:3] = rot.apply(d.qpos[:3]-original_center)+shift
            d.qpos[3:7] = (rot*Rotation.from_quat(d.qpos[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        d.ctrl[:] = d.qpos[self.qadr]
        mujoco.mj_forward(m, d)
        self.initial_qpos = d.qpos.copy()
        self.records = []
        self.commands = []
        self.probe_monitor = False
        self.probe_high_runs = np.zeros(20,int)
        self.stage = 'initial'
        self.phase = initial_phase
        self.app = self.bridge = self.client = None
        self.wall_start = time.monotonic()
        self.snapshot()
        appdir = directory/'app'
        appdir.mkdir()
        exe = appdir/'SnakeRobotVisualizer.exe'
        shutil.copy2(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', exe)
        shutil.copy2(ROOT/'validation/fixtures/app-state.json', appdir/'app-state.json')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.old_port = os.environ.get('SNAKE_VISUALIZER_PORT')
        os.environ['SNAKE_VISUALIZER_PORT'] = str(self.port)
        env = os.environ.copy()
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        self.app = subprocess.Popen([str(exe)], cwd=appdir, env=env, startupinfo=startup)
        self.bridge = MujocoExecutionBridge(m, d)
        self.bridge.period = .001
        self.bridge.record_commands = True
        self.client = SnakeControlClient(port=self.port, timeout=.5)
        deadline = time.monotonic()+20
        while True:
            self.bridge.publish(d)
            try:
                g = self.client.get_geometry()
                if g.get('mode') == 'Mujoco' and g['target_state'].get('mujoco_executor_ready'):
                    break
            except (OSError, RuntimeError, KeyError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('isolated EXE did not become ready')
            time.sleep(.01)
        self.geometry = g
        self.command_id = 0

    def snapshot(self):
        m, d = self.model, self.data
        # mj_step leaves cached forces at its pre-integration state. Recompute
        # all dynamics here so angles, velocities and torque belong to d.time.
        mujoco.mj_forward(m, d)
        points = np.vstack([d.site('s_head').xpos, *[d.joint(f'joint{i}').xanchor for i in range(1,21)], d.site('s_tail').xpos])
        q = d.qpos[self.qadr].copy()
        expected = np.clip(m.actuator_gainprm[:,0]*(d.ctrl-q), m.actuator_forcerange[:,0], m.actuator_forcerange[:,1])
        robot_bodies = set(range(m.body('head').id, m.nbody))
        floor = m.geom('floor').id
        contact_modules = set()
        ground = self_contact = False
        normal = np.zeros(21)
        for contact_index,c in enumerate(d.contact):
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in robot_bodies, b2 in robot_bodies
            ground |= bool((c.geom1 == floor and r2) or (c.geom2 == floor and r1))
            self_contact |= bool(r1 and r2 and abs(b1-b2)>1)
            if r1 != r2 and (m.geom(c.geom2 if r1 else c.geom1).name or '').startswith('pipe'):
                module = (b1 if r1 else b2)-m.body('head').id
                contact_modules.add(module)
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, contact_index, f)
                normal[module] += f[0]
        self.records.append(dict(sim_time=float(d.time), wall_unix_ms=time.time()*1000,
                                 qpos=d.qpos.copy(), qvel=d.qvel.copy(), q=q, velocity=d.qvel[self.vadr].copy(),
                                 ctrl=d.ctrl.copy(), torque=d.qfrc_actuator[self.vadr].copy(),
                                 actuator_force=d.actuator_force.copy(), expected_torque=expected,
                                 points=points, stage=self.stage, phase=self.phase,
                                 applied_command_id=self.bridge.applied_command_id if self.bridge else 0,
                                 ground=ground, self_contact=self_contact,
                                 contact_modules=np.array([i in contact_modules for i in range(21)]), normal_force=normal))

    def command(self, target):
        target = np.asarray(target)
        if target.shape != (20,) or not np.isfinite(target).all():
            raise ValueError('invalid target')
        if np.any(target < self.model.jnt_range[self.jids,0]) or np.any(target > self.model.jnt_range[self.jids,1]):
            raise ValueError('target outside joint range; no hard clipping')
        captured_after = time.time()*1000
        self.bridge.publish(self.data)
        deadline = time.monotonic()+3
        while True:
            geometry = self.client.get_geometry()
            if geometry['sample_unix_ms'] >= captured_after and geometry.get('age_ms',1e9)<200:
                break
            self.bridge.publish(self.data)
            if time.monotonic()>deadline:
                raise RuntimeError('no fresh feedback after physical advance')
            time.sleep(.005)
        self.command_id += 1
        started = time.monotonic()
        record = dict(command_id=self.command_id, computed_sim_time=float(self.data.time),
                      submitted_unix_ms=time.time()*1000, stage=self.stage, target_rad=target.tolist(),
                      source_sequence=geometry['sequence'], applied=False)
        self.commands.append(record)
        ack = self.client.submit_targets(geometry, np.degrees(target*self.bridge.signs), self.command_id,
                                         execution='mujoco', valid_for_ms=500)
        record['ack'] = ack
        while not self.bridge.apply_received(self.data, expected_targets=target):
            self.bridge.publish(self.data)
            if time.monotonic()-started > 3:
                raise RuntimeError('target relay timeout: '+self.bridge.last_error)
            time.sleep(.002)
        record.update(applied=True, applied_sim_time=float(self.data.time),
                      applied_unix_ms=time.time()*1000, delivery_id=self.bridge.applied_delivery_id,
                      relay_wall_s=time.monotonic()-started)

    def advance(self, duration):
        steps = round(duration/self.model.opt.timestep)
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if (i+1)%10 == 0:
                self.snapshot()
                if self.probe_monitor:
                    row=self.records[-1]
                    self.probe_high_runs=np.where(abs(row['torque'])>2.355*.9,self.probe_high_runs+1,0)
                    if self.probe_high_runs.max()>=20:
                        raise ProbeLimit('90 percent torque for 0.2 physical seconds')
                    if row['ground'] or row['self_contact']:
                        raise ProbeLimit('support or self-collision gate')
        self.bridge.publish(self.data)

    def close(self):
        if self.client:
            self.client.close()
        if self.bridge:
            self.bridge.close()
        if self.app and self.app.poll() is None:
            self.app.terminate()
            self.app.wait(timeout=10)
        if hasattr(self, 'old_port'):
            if self.old_port is None:
                os.environ.pop('SNAKE_VISUALIZER_PORT', None)
            else:
                os.environ['SNAKE_VISUALIZER_PORT'] = self.old_port
        if self.records:
            np.savez_compressed(self.directory/'trace.npz', **{k:np.asarray([r[k] for r in self.records]) for k in self.records[0]})
        write_json(self.directory/'commands.json', self.commands)


def stage0_trial(model_path, directory, phase, radius, ramp_s, relax_fraction=None):
    sim = None
    result = dict(stage=0, initial_command_phase_rad=phase,
                  radius_mm=radius, pitch_mm=220., ramp_s=ramp_s,
                  initialization_relaxation_fraction=relax_fraction, outcome='engineering_failure')
    try:
        sim = Simulation(model_path, directory, phase)
        target = HelixBackbone.from_radius_pitch(radius, 220.).joint_angles(20, 64.5, phase)
        initial = sim.data.ctrl.copy()
        sim.stage = 'initialization'
        if ramp_s:
            for weight in np.linspace(0, 1, round(ramp_s/.05)+1)[1:]:
                sim.command(initial*(1-weight)+target*weight)
                sim.advance(.05)
        else:
            sim.command(target)
        sim.advance(2.)
        if relax_fraction is not None:
            # Common initialization repair: one measured-angle-based relaxation
            # of pre-tension. This is NOT an online force-feedback controller.
            # It is frozen for all later methods, no friction/actuator changes.
            sim.stage = 'common_pretension_relaxation'
            before = sim.data.ctrl.copy()
            measured = sim.data.qpos[sim.qadr].copy()
            kp = sim.model.actuator_gainprm[:,0]
            bound = 2.355*relax_fraction/kp
            target = measured+np.clip(before-measured,-bound,bound)
            result['pre_relax_target_rad'] = before.tolist()
            result['relaxed_target_rad'] = target.tolist()
            for weight in np.linspace(0,1,11)[1:]:
                sim.command(before*(1-weight)+target*weight);sim.advance(.05)
            sim.advance(2.)
        sim.stage = 'hold'
        start = sim.data.time
        p0 = sim.records[-1]['points'].mean(axis=0)
        sim.advance(10.)
        rows = [r for r in sim.records if r['sim_time']>start+1e-8]
        displacement = [(r['points'].mean(axis=0)-p0)@sim.axis for r in rows]
        torque = np.array([r['torque'] for r in rows])
        result.update(hold_start_s=float(start), hold_duration_s=10.,
                      hold_net_slip_mm=float(abs(displacement[-1])*1000),
                      hold_max_slip_mm=float(max(abs(np.array(displacement)))*1000),
                      ground_contact=any(r['ground'] for r in rows),
                      self_contact=any(r['self_contact'] for r in rows),
                      torque_rms_nm=float(np.sqrt(np.mean(torque**2))),
                      saturation_fraction=float(np.mean(abs(torque)>=2.355*.999)),
                      hold_max_torque_nm=float(abs(torque).max()),
                      minimum_contact_modules=int(min(sum(r['contact_modules']) for r in rows)),
                      mapping_formula_error_nm=float(max(np.max(abs(r['expected_torque']-r['torque'])) for r in sim.records)),
                      actuator_mapping_error_nm=float(max(np.max(abs(r['actuator_force']-r['torque'])) for r in sim.records)),
                      max_link_length_error_mm=float(max(np.max(abs(np.linalg.norm(np.diff(r['points'],axis=0),axis=1)*1000-64.5)) for r in sim.records)))
        result['hold_passed'] = result['hold_net_slip_mm']<=20 and not result['ground_contact'] and not result['self_contact']
        result['outcome'] = 'hold_passed' if result['hold_passed'] else 'hold_failed'
        # Sustained high torque means no probe margin under the prescribed gate.
        high = abs(torque)>2.355*.9
        runs = np.zeros(20, int)
        maximum = 0
        for row in high:
            runs = np.where(row, runs+1, 0)
            maximum = max(maximum, int(runs.max()))
        result['maximum_high_load_duration_s'] = maximum*.01
        result['probe_margin'] = maximum < 20
        if result['hold_passed'] and result['probe_margin']:
            sim.stage = 'probe'
            baseline_q = sim.data.qpos[sim.qadr].copy()
            probes = []
            for axis in range(2):
                for sign in (1., -1.):
                    offset = np.zeros(20)
                    offset[axis:4:2] = sign*math.radians(.5)
                    reached=0.;reason=None
                    probe_start=len(sim.records)
                    sim.probe_high_runs[:]=0
                    sim.probe_monitor=True
                    try:
                        for weight in np.linspace(0,1,11)[1:]:
                            sim.command(target+offset*weight)
                            reached=float(weight)
                            sim.advance(.05)
                        sim.advance(.25)
                    except ProbeLimit as error:
                        reason=str(error)
                    finally:
                        sim.probe_monitor=False
                    sim.stage='probe_return'
                    for weight in np.linspace(reached,0,11)[1:]:
                        sim.command(target+offset*weight); sim.advance(.05)
                    sim.advance(.5)
                    residual=np.degrees(sim.data.qpos[sim.qadr]-baseline_q)
                    recovered=bool(abs(residual).max()<=.1 and not any(r['ground'] or r['self_contact'] for r in sim.records[probe_start:]))
                    probes.append(dict(axis=axis,sign=sign,peak_requested_deg=.5,reached_deg=.5*reached,
                                       stop_reason=reason,increment_limit_deg=.25,
                                       residual_deg=residual.tolist(),recovered=recovered))
                    sim.stage='probe'
                    if not recovered:
                        break
                if probes and not probes[-1]['recovered']:
                    break
            result['probes'] = probes
            result['probe_recovery_passed']=len(probes)==4 and all(x['recovered'] for x in probes)
        result['physical_end_s'] = float(sim.data.time)
        result['initial_qpos'] = sim.initial_qpos.tolist()
    except Exception as error:
        result['error'] = repr(error)
        if not directory.exists():
            directory.mkdir(parents=True)
    finally:
        if sim:
            sim.close()
        write_json(directory/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_qpos','probes','pre_relax_target_rad','relaxed_target_rad')}, ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage0', action='store_true')
    p.add_argument('--round', type=int, choices=(0,1,2), default=0)
    p.add_argument('--out', type=Path, default=BASE)
    p.add_argument('--attempt', default='initial')
    args = p.parse_args()
    if not args.stage0:
        p.error('select --stage0; later stages require gates')
    args.out.mkdir(parents=True, exist_ok=True)
    models = args.out/'models';models.mkdir(exist_ok=True)
    model_path, origin, axis = straight_model(models)
    settings = [(81.5,0.,None), (81.5,0.,.7), (81.5,0.,.5)]
    radius, ramp, relax = settings[args.round]
    manifest = dict(actual_date='2026-10-04', experiment_plan_date='2026-10-03', stage=0,
                    python=sys.executable, python_version=sys.version.split()[0], mujoco=mujoco.__version__,
                    numpy=np.__version__, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    source_hash=sha(__file__), model_hash=sha(model_path), exe_hash=sha(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe'),
                    asset_hash={name:sha(ROOT/'mujoco_model/asset'/name) for name in ('c_link_H.stl','c_link_V.stl')},
                    source_axis=axis.tolist(),source_origin=origin.tolist(),record_hz=100,decision_hz=20,
                    dynamics_changed=False, real_hardware=False, parameter_round=args.round,radius_mm=radius,ramp_s=ramp,
                    common_initialization_relaxation_fraction=relax)
    label=f'stage0_round{args.round}_{args.attempt}'
    manifest_path=args.out/f'manifest_{label}.json'
    if manifest_path.exists():
        raise RuntimeError('refusing to overwrite manifest')
    write_json(manifest_path, manifest)
    results=[]
    for i, phase in enumerate((0.,2*math.pi/3,4*math.pi/3)):
        directory=args.out/label/f'phase{i}'
        if (directory/'result.json').exists():
            raise RuntimeError('refusing to overwrite recorded attempt')
        results.append(stage0_trial(model_path,directory,phase,radius,ramp,relax))
    write_json(args.out/f'{label}.json', dict(results=results,
               support_gate=all(r.get('hold_passed',False) for r in results),
               probe_gate=all(r.get('hold_passed',False) and r.get('probe_margin',False) and r.get('probe_recovery_passed',False) for r in results)))


if __name__ == '__main__':
    main()
