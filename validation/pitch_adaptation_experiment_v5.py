"""Isolated Python -> EXE -> MuJoCo pitch trials (20 joints, no real hardware).

Default is the eight mandatory A/D training pilots, seeds=1, k=0.
Full comparisons require a saved training freeze; --main refuses an absent freeze.
The simulation budget is physical seconds, independent of EXE waiting time.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,as_completed

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'visualizer'),str(ROOT/'validation')]
SCENES=('pipe_straight_small','pipe_straight_large','pipe_taper_horizontal','real_s2l')
ALPHA=math.atan2(220.,2*np.pi*81.5)


def model_path(scene):
    return ROOT/'mujoco_model'/('Snake_Robot_o20_Real_Pipe_S2L.xml' if scene=='real_s2l' else
                              f'Snake_Robot_o20_Pipe_{scene[5:]}.xml')


def run_gait():
    import mujoco
    import threading
    from snake_control import adaptive_debug
    from snake_control import pole_compliance
    import Adaptive_Pole_Climbing_Mujoco as gait
    from snake_control.local_pitch_controller_v5 import LocalPitchController
    LocalPitchController.method=os.environ['PITCH_METHOD']
    LocalPitchController.exponent=float(os.environ.get('PITCH_K','0'))
    LocalPitchController.fixed_alpha=float(os.environ.get('PITCH_FIXED_ALPHA',str(ALPHA)))
    LocalPitchController.records=[]
    # Reuse exactly the gait's common design helix; this is an initialization
    # input, not an online pipe-radius estimate or adaptive-policy input.
    setup_model=mujoco.MjModel.from_xml_path(os.environ['SNAKE_MUJOCO_MODEL'])
    design=setup_model.numeric('experiment_helix_mm').data
    LocalPitchController.initial_backbone=gait.HelixBackbone.from_radius_pitch(float(design[0]),float(design[1]))
    pole_compliance.ComplianceController=LocalPitchController
    gait.DEBUG=False;gait.COMPLIANCE_ENABLED=True
    measurements=adaptive_debug.measurements
    raw_progress=adaptive_debug.pipe_progress
    def precise_progress(model,data):
        values=raw_progress(model,data)
        print('PITCH_PROGRESS '+json.dumps(dict(sim_time=float(data.time),progress=list(map(float,values)))),flush=True)
        return values
    adaptive_debug.pipe_progress=precise_progress
    physics=[];last_time=None
    physical_clock={"time_s":0.}
    LocalPitchController.control_clock=staticmethod(lambda:physical_clock["time_s"])
    def measured(model,data):
        nonlocal last_time
        if last_time != data.time:
            last_time=float(data.time)
            physical_clock["time_s"]=last_time
            floor=model.geom('floor').id
            robot_geoms={i for i in range(model.ngeom) if model.geom_bodyid[i] and
                         not (model.geom(i).name or '').startswith('pipe')}
            ground=any((c.geom1==floor or c.geom2==floor) and
                       (c.geom2 if c.geom1==floor else c.geom1) in robot_geoms
                       for c in data.contact)
            progress=raw_progress(model,data)
            physics.append(dict(sim_time=last_time,actuator_force=data.actuator_force.copy().tolist(),
                                ground_contact=bool(ground),progress=list(map(float,progress))))
        return measurements(model,data)
    adaptive_debug.measurements=measured
    threading.Thread(target=gait.uart_listener,daemon=True).start()
    try:
        gait.main()
    finally:
        path=Path(os.environ['PITCH_CONTROL_LOG'])
        with gzip.open(path,'wt',encoding='utf-8') as f:
            json.dump(dict(control=LocalPitchController.records,physics=physics),f,ensure_ascii=False)


def hidden_start(exe,appdir,env):
    kwargs={}
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        kwargs['startupinfo']=startup
    return subprocess.Popen([str(exe)],cwd=appdir,env=env,**kwargs)


def trial(scene,method,seed,k,alpha,out,budget=120.):
    import mujoco
    from python_client import SnakeControlClient
    label=f'{scene}__{method}__s{seed}__k{k:g}__a{math.degrees(alpha):.4f}'
    directory=out/label;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        return json.loads((directory/'result.json').read_text(encoding='utf-8'))
    model=mujoco.MjModel.from_xml_path(str(model_path(scene)))
    assert model.nu==20 and mujoco.__version__=='3.3.0'
    try: feature=float(model.numeric('feature_span').data[1])
    except KeyError: feature=float(model.numeric('pipe_profile').data[3])
    # The root of the private app directory contains no real-servo profile.
    appdir=directory/'app';appdir.mkdir(exist_ok=True)
    exe=appdir/'SnakeRobotVisualizer.exe'
    shutil.copy2(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe',exe)
    shutil.copy2(ROOT/'validation/fixtures/app-state.json',appdir/'app-state.json')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    env=os.environ.copy()
    for name in ('SNAKE_HELIX_RADIUS_MM','SNAKE_HELIX_PITCH_MM','SNAKE_COMPLIANCE_GAIN'):
        env.pop(name,None)
    env.update(SNAKE_VISUALIZER_PORT=str(port),SNAKE_MUJOCO_HEADLESS='1',SNAKE_SINGLE_STEP='1',
               SNAKE_MUJOCO_MODEL=str(model_path(scene)),SNAKE_CONTROL_VIA_VISUALIZER='1',
               SNAKE_VISUALIZER_ENABLED='1',SNAKE_GEOMETRY_EXCHANGE='1',SNAKE_TRIAL_SEED=str(seed),
               SNAKE_MUJOCO_TRACE=str(directory/'trace.npz'),SNAKE_MUJOCO_MAX_SECONDS='1800',
               PITCH_METHOD=method,PITCH_K=str(k),PITCH_FIXED_ALPHA=str(alpha),
               PITCH_CONTROL_LOG=str(directory/'control.json.gz'),PYTHONIOENCODING='utf-8')
    app=child=None;client=SnakeControlClient(port=port)
    log=directory/'execution.log';done=0
    record=dict(scene=scene,method=method,seed=seed,k=k,alpha_deg=math.degrees(alpha),joints=20,
                movement_budget_s=budget,hold_budget_s=10.,outcome='not_started',movement_steps=0,
                initialization_s=2.2,hold_max_retreat_m=.02,goal_feature_m=feature,status=[],rejections=0)
    record['canceled_feedback_retries']=0
    record['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (model_path(scene),ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe',
                       ROOT/'snake_control/local_pitch_controller_v5.py',ROOT/'snake_control/local_pitch_feedback.py',ROOT/'snake_control/local_pitch_geometry.py',
                       ROOT/'snake_control/pole_compliance.py',ROOT/'codes/Adaptive_Pole_Climbing_Mujoco.py',
                       Path(__file__).resolve())}
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as handle:
        def read():return log.read_text(encoding='utf-8',errors='replace')
        def send(command):
            child.stdin.write(command+'\n');child.stdin.flush()
        def wait_text(pattern,count=1,timeout=12.):
            deadline=time.monotonic()+timeout
            while read().count(pattern)<count:
                if child.poll() is not None or time.monotonic()>deadline:return False
                time.sleep(.025)
            return True
        def step(command):
            nonlocal done
            # Retry only explicitly canceled deliveries: these advance neither
            # physical time nor roll phase. Never hide a controller rejection.
            for attempt in range(6):
                before=read().count('未执行：');canceled=read().count('本步取消：')
                send(command);deadline=time.monotonic()+10
                while f'完成第 {done+1} 步' not in read():
                    if read().count('未执行：')>before:
                        record['rejections']+=1
                        raise RuntimeError(read().split('未执行：')[-1].split('\n')[0])
                    if read().count('本步取消：')>canceled:
                        record['canceled_feedback_retries']+=1;time.sleep(.25);break
                    if child.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('communication or step timeout: '+read()[-400:])
                    time.sleep(.025)
                else:
                    done+=1;return
            raise RuntimeError('Six consecutive feedback/relay cancellations without a physical step')
        def status():
            count=read().count('RADIUS ');state_count=read().count('STATUS ');send('status')
            if not wait_text('RADIUS ',count+1):raise RuntimeError('status timeout')
            if not wait_text('STATUS ',state_count+1):raise RuntimeError('status timeout')
            text=read();radius=text.split('RADIUS ')[-1].split('\n')[0];state=text.split('STATUS ')[-1].split('\n')[0]
            values=re.search(r'最后/最前=([\d.-]+)/([\d.-]+)/([\d.-]+)',radius)
            row=dict(sim_time=float(re.search(r'sim=([\d.]+)s',state).group(1)),
                     mean=float(values[1]),tail=float(values[2]),head=float(values[3]),
                     contacts=int(re.search(r'管道接触=(\d+)',state).group(1)),
                     height=float(re.search(r'关节平均高度=([\d.-]+)m',state).group(1)),
                     tracking_deg=float(re.search(r'最大跟踪误差=([\d.]+)deg',state).group(1)))
            exact=json.loads(text.split('PITCH_PROGRESS ')[-1].split('\n')[0])
            if abs(exact['sim_time']-row['sim_time'])>.001:
                raise RuntimeError('Precise progress and status times differ')
            row.update(sim_time=exact['sim_time'],mean=exact['progress'][0],tail=exact['progress'][1],head=exact['progress'][2])
            record['status'].append(row)
            geometry=client.get_geometry()
            with gzip.open(directory/f'feedback_{done:04d}.json.gz','wt',encoding='utf-8') as f:json.dump(geometry,f,ensure_ascii=False)
            return row
        try:
            app=hidden_start(exe,appdir,env)
            deadline=time.monotonic()+12
            while True:
                try:client.get_state();break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--run'],
                                   cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=handle,stderr=handle,text=True)
            if not wait_text('MUJOCO_RELAY 已连接'):raise RuntimeError('EXE relay not connected')
            step('apply')
            # Common native-target stabilization before any A/B/C/D controller update.
            for _ in range(10):step('hold')
            initial=status();record['initial']=initial
            # Common production geometry readiness; physical time stays frozen.
            ready_deadline=time.monotonic()+12.
            while True:
                ready=client.get_geometry()
                if ready.get('feedback_usable') and ready['curve']['valid'] and time.time()*1000-ready['sample_unix_ms']<=500:
                    break
                if time.monotonic()>ready_deadline:raise RuntimeError('Common geometry readiness timeout')
                time.sleep(.025)
            record['takeover_ready_sim_time']=ready.get('sim_time')
            if initial['contacts']==0 or initial['height']<.12:
                record['outcome']='initialization_failure'
            else:
                record['outcome']='time_budget'
                rolling=[];holding=False;hold_start=None;hold_mean=None;passed_mean=None
                while True:
                    step('hold' if holding else 'back')
                    if not holding:record['movement_steps']+=1
                    if done%5!=1:continue
                    current=status();elapsed=current['sim_time']-initial['sim_time']
                    if current['height']<.12:
                        record['outcome']='ground_collision';break
                    if current['contacts']==0:
                        record['outcome']='support_loss';break
                    if holding:
                        if not scene.startswith('pipe_straight') and current['tail']<=feature:
                            record['outcome']='hold_region_lost';break
                        if hold_mean-current['mean']>.02:
                            record['outcome']='hold_slip';break
                        if current['sim_time']-hold_start>=10.-1e-8:
                            record['outcome']='success';break
                        continue
                    rolling.append(current)
                    if scene.startswith('pipe_straight'):
                        reached=current['mean']-initial['mean']>=.5
                    else:
                        if current['tail']>feature and passed_mean is None:passed_mean=current['mean']
                        reached=passed_mean is not None and current['tail']>feature and current['mean']-passed_mean>=.2
                    if reached:
                        holding=True;hold_start=current['sim_time'];hold_mean=current['mean']
                        record.update(hold_start_s=hold_start,hold_start_mean_m=hold_mean)
                        continue
                    old=next((r for r in reversed(rolling) if current['sim_time']-r['sim_time']>=10.-1e-8),None)
                    if old and current['mean']-old['mean']<.01:
                        record['outcome']='no_progress_10s';break
                    if elapsed>=budget-1e-8:break
        except Exception as error:
            record['outcome']='target_rejection' if record['rejections'] else 'communication_failure'
            record['error']=str(error)
        finally:
            if child is not None and child.poll() is None:
                try:send('quit');child.stdin.close();child.wait(timeout=30)
                except (OSError,subprocess.TimeoutExpired):child.terminate();child.wait(timeout=10)
            client.close()
            if app is not None and app.poll() is None:app.terminate();app.wait(timeout=10)
    record['wall_s']=time.monotonic()-start
    if (directory/'trace.npz').exists():
        with np.load(directory/'trace.npz') as raw:
            record['physical_s']=float(raw['sim_time'][-1]);record['completed_steps']=int(raw['completed_steps'][-1])
            arrays={key:raw[key] for key in raw.files}
        compressed=directory/'trace_compressed.npz'
        np.savez_compressed(compressed,**arrays)
        compressed.replace(directory/'trace.npz')
    if (directory/'control.json.gz').exists():
        with gzip.open(directory/'control.json.gz','rt',encoding='utf-8') as f:extra=json.load(f)
        controls=extra['control'];physics=extra['physics']
        record['control_updates']=len(controls)
        record['degraded_update_fraction']=float(np.mean([r['degraded'] for r in controls])) if controls else None
        record['compute_ms_max']=max((r['compute_ms'] for r in controls),default=None)
        record['any_ground_contact']=any(r['ground_contact'] for r in physics)
        ground=[r for r in physics if r['ground_contact']]
        if ground:
            record['outcome']='initialization_failure' if ground[0]['sim_time']<=2.2+1e-8 else 'ground_collision'
        if physics:
            times=np.array([r['sim_time'] for r in physics]);forces=np.array([r['actuator_force'] for r in physics])
            dt=np.diff(np.r_[0.,times]);record['torque_rms_nm']=float(np.sqrt(np.average(np.mean(forces**2,axis=1),weights=dt)))
            record['torque_squared_integral']=float(np.sum(np.sum(forces**2,axis=1)*dt))
            record['saturation_fraction']=float(np.average(np.mean(abs(forces)>=2.355*.999,axis=1),weights=dt))
        if controls:
            record['radius_valid_fraction']=float(np.mean([r['radius_valid'] for r in controls]))
            record['alpha_range_deg']=[float(np.degrees(min(min(r['alpha_rad']) for r in controls))),
                                       float(np.degrees(max(max(r['alpha_rad']) for r in controls)))]
    record['success']=record['outcome']=='success'
    record['controller_version']=5
    (directory/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in ('status','initial')},ensure_ascii=False),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--scenes',nargs='+',choices=SCENES,default=list(SCENES))
    parser.add_argument('--methods',nargs='+',choices=('A','B','C','D'),default=['A','D'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[1])
    parser.add_argument('--k',type=float,default=0.)
    parser.add_argument('--alpha-deg',type=float,default=math.degrees(ALPHA))
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/pitch_adaptation_20261003/repair_v5/pilots')
    parser.add_argument('--main',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.run:return run_gait()
    if args.main:
        raise RuntimeError('Formal trials require completed training and a verified freeze; this entry is currently training-only.')
    args.out.mkdir(parents=True,exist_ok=True)
    settings=dict(stage='training',joints=20,scenes=args.scenes,methods=args.methods,seeds=args.seeds,k=args.k,
                  alpha_deg=args.alpha_deg,movement_s=120.,hold_s=10.,step_rad=.15,step_s=.2,gain=[.01,.02],
                  initialization='scene native apply then 10 native holds (2.2 physical s); common moving-helix 2s takeover; no nominal fallback',
                  limits=dict(alpha_deg=[8,30],alpha_step_deg=1,group_target_step_deg=30),
                  fallback='hold previous radius; fixed alpha after 0.5 physical s; C retains last global median',
                  timing='physical clock for filtering and fallback; wall clock for transport freshness',
                  estimator=dict(sampler='measured link centres',window=7,minimum_coverage_deg=60,maximum_condition=100,
                                 residual_limit='max(2mm,5% radius)',spatial_sigma_links=1,time_gain=.2))
    (args.out/'pilot_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    trials=[(scene,method,seed) for scene in args.scenes for method in args.methods for seed in args.seeds]
    np.random.default_rng(20261003).shuffle(trials)
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(trial,scene,method,seed,args.k,math.radians(args.alpha_deg),args.out) for scene,method,seed in trials]
        for future in as_completed(futures):
            results.append(future.result())
            (args.out/'pilot_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'pilot_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
