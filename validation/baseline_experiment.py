"""基线实验：现有步态（顺应性 / 标准 Yamada 螺旋）在全部实物参数管道场景上的重复攀爬。

用法：
  python validation/baseline_experiment.py               运行全部（已完成的试验自动跳过，可中断后续跑）
  python validation/baseline_experiment.py --analyze     只根据已保存的试验数据重新汇总、作图
  python validation/baseline_experiment.py --scenes pipe_bend90 --trials 1 --max-steps 20 --out <目录>   小规模试跑

每次试验：独立 EXE（隔离端口）+ 无界面 MuJoCo，完整走 Python → EXE → MuJoCo 链路；
Ctrl+U 应用场景预设 R/P 后连续 Ctrl+H（管外上行），每 STATUS_EVERY 步读一次 Ctrl+K 状态。
结束条件：整条身体越过特征（成功）、跌落、连续无法执行（停滞）或达到步数上限。
重复试验用 SNAKE_TRIAL_SEED 做可复现扰动：管道摩擦 ×U(0.8, 1.2)，初始姿态绕起始管轴随机转动。
原始轨迹分析后压缩保存到 experiments/baseline_<日期>/trials/，汇总表与图在同一目录。
"""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
OUT = ROOT / 'experiments/baseline_20260930'
WORK = ROOT / 'validation/results/baseline'  # 原始日志、轨迹与各 EXE 工作目录（不入库）
SCENES = ['real_s2l'] + [f'pipe_{name}' for name in (
    'bend45', 'bend90', 'bend90_tight', 'offset', 'taper_bend', 'bend_taper', 'flange10', 'flange25', 'flange40',
    'flange55', 'flange_pair', 'eccentric', 'tee', 'incline30', 'slip_band')]
METHODS = ('compliance', 'nominal')
TRIALS = 5
MAX_STEPS = 450
STATUS_EVERY = 10
STEP_TIMEOUT_S, STEP_RETRIES = 8., 6
FALL_HEIGHT_M = .12  # 关节平均高度低于此值视为跌落到地面
WORKERS = 10
BASE_PORT = 19100
OVERLOAD_HOLD_S = 2.  # 连续顶在力矩上限超过 2 s 记为一次持续堵转（参照舵机保护时间 200×10 ms）


def model_path(scene):
    name = {'real_s2l': 'Snake_Robot_o26_Real_Pipe_S2L.xml'}.get(scene, f'Snake_Robot_o26_Pipe_{scene[5:]}.xml')
    return ROOT / 'mujoco_model' / name


def feature_end(model):
    """整条身体须越过的沿管位置（m，从地面起）。real_s2l 没有标记，取异径接头上端。"""
    try:
        return float(model.numeric('feature_span').data[1])
    except KeyError:
        sys.path.insert(0, str(ROOT / 'tools'))
        import build_real_pipe_scene as real
        return real.Z_TAPER + real.TAPER_LENGTH


class Trial:
    def __init__(self, scene, method, seed, max_steps):
        self.scene, self.method, self.seed, self.max_steps = scene, method, seed, max_steps
        self.label = f'{scene}__{method}__s{seed}'


def run_trial(trial, port, appdir, out):
    """驱动一次试验；返回控制过程记录（轨迹由子进程写入 npz）。"""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path(trial.scene)))
    goal = feature_end(model)
    log = WORK / 'logs' / f'{trial.label}.log'
    trace = WORK / 'traces' / f'{trial.label}.npz'
    log.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(SNAKE_VISUALIZER_PORT=str(port), SNAKE_MUJOCO_HEADLESS='1', SNAKE_SINGLE_STEP='1',
               SNAKE_MUJOCO_MODEL=str(model_path(trial.scene)), SNAKE_CONTROL_VIA_VISUALIZER='1',
               SNAKE_VISUALIZER_ENABLED='1', SNAKE_GEOMETRY_EXCHANGE='1', SNAKE_TRIAL_SEED=str(trial.seed),
               SNAKE_MUJOCO_MAX_SECONDS=str(trial.max_steps * 12 + 300), SNAKE_MUJOCO_TRACE=str(trace),
               PYTHONIOENCODING='utf-8')
    env.pop('SNAKE_COMPLIANCE_GAIN', None)  # 使用场景 XML 自带的增益
    args = [sys.executable, '-X', 'utf8', str(ROOT / 'validation/adaptive_pole_single_step.py'), '--run']
    if trial.method == 'nominal':
        args.append('--nominal')
    record = dict(scene=trial.scene, method=trial.method, seed=trial.seed, goal_m=goal, outcome='budget',
                  steps=0, retries=0, status=[], wall_s=0.)
    started = time.monotonic()
    with log.open('w', encoding='utf-8') as handle:
        child = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=handle, stderr=handle, text=True)
        text = lambda: log.read_text(encoding='utf-8', errors='replace')

        def send(command):
            child.stdin.write(command + '\n')
            child.stdin.flush()

        def wait_for(pattern, timeout, count=1):
            end = time.monotonic() + timeout
            while text().count(pattern) < count:
                if child.poll() is not None or time.monotonic() > end:
                    return False
                time.sleep(.03)
            return True

        try:
            if not wait_for('MUJOCO_RELAY 已连接', 20):
                raise RuntimeError('执行端未连接\n' + text()[-800:])
            time.sleep(.5)
            send('apply')
            if not wait_for('完成第 1 步', 20):
                raise RuntimeError('Ctrl+U 未执行\n' + text()[-800:])
            done = 1
            while record['steps'] < trial.max_steps:
                for attempt in range(STEP_RETRIES):
                    send('back')  # 管外 P>0：Ctrl+H 上行
                    if wait_for(f'完成第 {done + 1} 步', STEP_TIMEOUT_S):
                        done += 1
                        record['steps'] += 1
                        break
                    record['retries'] += 1
                    time.sleep(.5)
                else:
                    record['outcome'] = 'stalled'
                    break
                if record['steps'] % STATUS_EVERY == 0 or record['steps'] == trial.max_steps:
                    count = text().count('RADIUS ')
                    send('status')
                    if wait_for('RADIUS ', 10, count + 1):
                        line = text().split('RADIUS ')[-1].split('\n')[0]
                        status = text().split('STATUS ')[-1].split('\n')[0]
                        mean, last, first = map(float, re.search(r'最后/最前=([\d.-]+)/([\d.-]+)/([\d.-]+)', line).groups())
                        height = float(re.search(r'关节平均高度=([\d.-]+)m', status).group(1))
                        record['status'].append(dict(step=record['steps'], mean=mean, last=last, first=first, height=height))
                        if last > goal:
                            record['outcome'] = 'success'
                            break
                        if height < FALL_HEIGHT_M:
                            record['outcome'] = 'fell'
                            break
        finally:
            if child.poll() is None:
                try:
                    send('quit')
                except OSError:
                    pass
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
    record['wall_s'] = time.monotonic() - started
    record['log_rejections'] = text().count('未执行')
    return record, trace


def analyze_trace(trial, record, trace):
    """把原始轨迹压缩为逐步数据，并计算本次试验的指标。"""
    import mujoco
    from adaptive_debug import centerline_nearest
    model = mujoco.MjModel.from_xml_path(str(model_path(trial.scene)))
    data = mujoco.MjData(model)
    raw = np.load(trace)
    completed = raw['completed_steps']
    ends = np.flatnonzero(np.diff(np.r_[0, completed]) == 1)  # 每一步（含 Ctrl+U）结束时刻
    nu = model.nu
    qpos = raw['qpos']
    progress, height, head = [], [], []
    for row in ends:
        data.qpos[:] = qpos[row]
        mujoco.mj_kinematics(model, data)
        points = np.array([data.joint(f'joint{i}').xanchor for i in range(1, nu + 1)])
        nearest = centerline_nearest(model, points)
        progress.append(points[:, 2] if nearest is None else nearest[2])
        height.append(points[:, 2])
        head.append(data.site('s_head').xpos[2])
    progress, height, head = np.array(progress), np.array(height), np.array(head)
    moving = raw['moving'].astype(bool)
    q, ctrl = qpos[:, 7:7 + nu], raw['ctrl']
    kp = model.actuator_gainprm[:, 0]
    limit = model.actuator_forcerange[:, 1]
    torque = np.clip(kp * (ctrl - q), -limit, limit)[moving]
    saturated = np.abs(torque) >= limit * .999
    dt = 10 * model.opt.timestep  # 运动中每条记录对应 10 个物理步
    longest = np.zeros(nu)
    run = np.zeros(nu)
    events = np.zeros(nu, dtype=int)
    for row in saturated:
        run = np.where(row, run + dt, 0.)
        events += (run >= OVERLOAD_HOLD_S) & (run - dt < OVERLOAD_HOLD_S)
        longest = np.maximum(longest, run)
    mean_progress = progress.mean(axis=1)
    step_moves = np.diff(mean_progress)
    success_step = None
    passed = np.flatnonzero(progress.min(axis=1) > record['goal_m'])
    if len(passed):
        success_step = int(passed[0])
    steps_rows = completed[moving]
    per_step_sat = np.array([saturated[steps_rows == k].mean() if np.any(steps_rows == k) else np.nan
                             for k in range(int(completed.max()) + 1)])
    metrics = dict(
        # 轨迹判定优先：状态查询间隔内已越过特征但随后跌落的，仍按越过时刻记为成功
        outcome='success' if success_step is not None else record['outcome'],
        success=success_step is not None,
        steps=int(len(ends) - 1),
        steps_to_success=success_step,
        start_progress_m=float(mean_progress[0]),
        max_rise_m=float(mean_progress.max() - mean_progress[0]),
        final_rise_m=float(mean_progress[-1] - mean_progress[0]),
        max_mean_progress_m=float(mean_progress.max()),
        max_tail_progress_m=float(progress.min(axis=1).max()),
        max_head_progress_m=float(progress.max(axis=1).max()),
        max_mean_height_m=float(height.mean(axis=1).max()),
        max_joint_height_m=float(height.max()),
        max_head_height_m=float(head.max()),
        rise_per_step_mm=float(1000 * (mean_progress.max() - mean_progress[0]) / max(1, int(np.argmax(mean_progress)))),
        slip_back_m=float(-step_moves[step_moves < 0].sum()),
        saturation_fraction=float(saturated.mean()),
        worst_joint_saturation_fraction=float(saturated.mean(axis=0).max()),
        worst_joint=int(np.argmax(saturated.mean(axis=0)) + 1),
        longest_saturation_s=float(longest.max()),
        overload_events=int(events.sum()),
        overload_joints=int(np.count_nonzero(events)),
        rms_torque_nm=float(np.sqrt((torque ** 2).mean())),
        peak_mean_abs_torque_nm=float(np.abs(torque).mean(axis=0).max()),
        heat_proxy_n2m2s=float((torque ** 2).sum() * dt),
        max_tracking_error_deg=float(np.degrees(raw['tracking_error_rad'][moving].max())),
        mean_pipe_contacts=float(raw['pipe_contacts'][ends].mean()),
        compliance_step_fraction=float(np.mean([raw['compliance_active'][completed == k].any()
                                                for k in range(1, int(completed.max()) + 1)])) if completed.max() > 0 else 0.,
        retries=record['retries'],
        rejections=record['log_rejections'],
        wall_s=round(record['wall_s'], 1),
    )
    trials = OUT / 'trials'
    trials.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trials / f'{trial.label}.npz',
                        step_qpos=qpos[ends].astype(np.float32), step_ctrl=ctrl[ends].astype(np.float32),
                        joint_progress_m=progress.astype(np.float32), joint_height_m=height.astype(np.float32),
                        head_height_m=head.astype(np.float32), step_saturation_fraction=per_step_sat.astype(np.float32),
                        joint_saturation_fraction=saturated.mean(axis=0).astype(np.float32),
                        joint_longest_saturation_s=longest.astype(np.float32),
                        joint_rms_torque_nm=np.sqrt((torque ** 2).mean(axis=0)).astype(np.float32))
    result = dict(scene=trial.scene, method=trial.method, seed=trial.seed, goal_m=record['goal_m'],
                  runner_outcome=record['outcome'], status=record['status'], **metrics)
    (trials / f'{trial.label}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def worker(port, jobs, results, lock):
    appdir = WORK / f'app{port}'
    appdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'visualizer/windows_app/bin/SnakeRobotVisualizer.exe', appdir / 'SnakeRobotVisualizer.exe')
    shutil.copy2(ROOT / 'validation/fixtures/app-state.json', appdir / 'app-state.json')
    env = os.environ.copy()
    env['SNAKE_VISUALIZER_PORT'] = str(port)
    app = subprocess.Popen([str(appdir / 'SnakeRobotVisualizer.exe')], cwd=appdir, env=env)
    time.sleep(1.5)
    try:
        while True:
            try:
                trial = jobs.get_nowait()
            except queue.Empty:
                return
            for attempt in range(2):  # EXE 偶发未连接时重启该试验一次
                try:
                    record, trace = run_trial(trial, port, appdir, OUT)
                    result = analyze_trace(trial, record, trace)
                    trace.unlink(missing_ok=True)
                    Path(str(trace) + '.received.json').unlink(missing_ok=True)
                    break
                except Exception as error:  # 记录后重试，不中断其他试验
                    result = dict(scene=trial.scene, method=trial.method, seed=trial.seed, error=str(error)[-500:])
                    if app.poll() is not None:
                        app = subprocess.Popen([str(appdir / 'SnakeRobotVisualizer.exe')], cwd=appdir, env=env)
                        time.sleep(1.5)
            with lock:
                results.append(result)
                print(f"[{len(results)}] {trial.label} {result.get('outcome', 'ERROR')} "
                      f"rise={result.get('max_rise_m', float('nan')):.3f}m steps={result.get('steps')} "
                      f"sat={result.get('saturation_fraction', float('nan')):.2f} {result.get('error', '')}", flush=True)
    finally:
        if app.poll() is None:
            app.terminate()
            app.wait(timeout=10)


def summarize():
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((OUT / 'trials').glob('*.json'))]
    rows = [r for r in rows if 'error' not in r]
    trial_fields = ['scene', 'method', 'seed', 'outcome', 'success', 'steps', 'steps_to_success', 'goal_m',
                    'start_progress_m', 'max_rise_m', 'final_rise_m', 'max_mean_progress_m', 'max_tail_progress_m',
                    'max_head_progress_m', 'max_mean_height_m', 'max_joint_height_m', 'max_head_height_m',
                    'rise_per_step_mm', 'slip_back_m', 'saturation_fraction', 'worst_joint_saturation_fraction',
                    'worst_joint', 'longest_saturation_s', 'overload_events', 'overload_joints', 'rms_torque_nm',
                    'peak_mean_abs_torque_nm', 'heat_proxy_n2m2s', 'max_tracking_error_deg', 'mean_pipe_contacts',
                    'compliance_step_fraction', 'retries', 'rejections', 'wall_s']
    with (OUT / 'trials.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, trial_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (SCENES.index(r['scene']), r['method'], r['seed'])))
    summary = []
    for scene in SCENES:
        for method in METHODS:
            group = [r for r in rows if r['scene'] == scene and r['method'] == method]
            if not group:
                continue
            values = lambda key: np.array([r[key] for r in group if r[key] is not None], dtype=float)
            stat = lambda key: (float(values(key).mean()) if len(values(key)) else None,
                                float(values(key).std()) if len(values(key)) else None)
            outcomes = [r['outcome'] for r in group]
            item = dict(scene=scene, method=method, trials=len(group), success=sum(r['success'] for r in group),
                        fell=outcomes.count('fell'), stalled=outcomes.count('stalled'), budget=outcomes.count('budget'))
            for key in ('max_rise_m', 'steps_to_success', 'rise_per_step_mm', 'slip_back_m', 'saturation_fraction',
                        'worst_joint_saturation_fraction', 'longest_saturation_s', 'overload_events', 'rms_torque_nm',
                        'heat_proxy_n2m2s', 'max_tracking_error_deg', 'mean_pipe_contacts'):
                item[f'{key}_mean'], item[f'{key}_std'] = stat(key)
            item['max_rise_m_max'] = float(values('max_rise_m').max())
            summary.append(item)
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, list(summary[0]))
        writer.writeheader()
        writer.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in summary])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(summary)
    return summary


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    scenes = [s for s in SCENES if any(x['scene'] == s for x in summary)]
    get = lambda scene, method, key: next((x[key] for x in summary if x['scene'] == scene and x['method'] == method), np.nan)
    x = np.arange(len(scenes))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for offset, method, color in ((-.2, 'compliance', '#2b7bba'), (.2, 'nominal', '#d9772b')):
        label = {'compliance': '顺应性 K=0.01/0.02', 'nominal': '标准 Yamada 螺旋'}[method]
        rate = [get(s, method, 'success') / get(s, method, 'trials') for s in scenes]
        axes[0].bar(x + offset, rate, .4, color=color, label=label)
        axes[1].bar(x + offset, [get(s, method, 'max_rise_m_mean') for s in scenes], .4,
                    yerr=[get(s, method, 'max_rise_m_std') for s in scenes], color=color, capsize=2)
        axes[2].bar(x + offset, [100 * get(s, method, 'saturation_fraction_mean') for s in scenes], .4,
                    yerr=[100 * get(s, method, 'saturation_fraction_std') for s in scenes], color=color, capsize=2)
    axes[0].set_ylabel('成功率')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    axes[1].set_ylabel('最大沿管上升 (m)')
    axes[2].set_ylabel('顶到力矩上限的时间 (%)')
    axes[2].set_xticks(x, [s.replace('pipe_', '') for s in scenes], rotation=40, ha='right')
    for ax in axes:
        ax.grid(axis='y', alpha=.3)
    fig.suptitle('基线：现有步态在实物参数管道场景上的重复攀爬（每组 5 次）')
    fig.tight_layout()
    fig.savefig(OUT / 'summary.png', dpi=150)
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--scenes', nargs='*', default=SCENES)
    parser.add_argument('--methods', nargs='*', default=list(METHODS))
    parser.add_argument('--trials', type=int, default=TRIALS)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if not args.analyze:
        jobs = queue.Queue()
        for scene in args.scenes:
            for method in args.methods:
                for seed in range(1, args.trials + 1):
                    trial = Trial(scene, method, seed, args.max_steps)
                    if not (OUT / 'trials' / f'{trial.label}.json').exists():
                        jobs.put(trial)
        print(f'待运行 {jobs.qsize()} 次试验，{args.workers} 个并行', flush=True)
        results, lock = [], threading.Lock()
        threads = [threading.Thread(target=worker, args=(BASE_PORT + i, jobs, results, lock))
                   for i in range(min(args.workers, jobs.qsize()))]
        for thread in threads:
            thread.start()
            time.sleep(2)
        for thread in threads:
            thread.join()
        errors = [r for r in results if 'error' in r]
        if errors:
            (OUT / 'errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding='utf-8')
    for row in summarize():
        print(row['scene'], row['method'], f"{row['success']}/{row['trials']}",
              f"rise={row['max_rise_m_mean']:.3f}±{row['max_rise_m_std']:.3f}", f"sat={row['saturation_fraction_mean']:.2f}")


if __name__ == '__main__':
    main()
