"""弯曲轴顺应性对比实验：与基线（experiments/baseline_20260930）相同的场景、种子、判定和指标。

新增两个步态（codes/Adaptive_Pole_Climbing_CurvedAxis_Mujoco.py）：
  curved_local  局部轴 + 局部截面（第 1、2 层，消融）
  curved        局部轴 + 局部截面 + 螺距角调制（第 1–3 层，完整方法）
基线两种步态不重跑，汇总时直接读取基线的逐次结果。
用法：
  python validation/curved_axis_experiment.py              运行全部（已完成的试验跳过）
  python validation/curved_axis_experiment.py --analyze    只重新汇总
  python validation/curved_axis_experiment.py --scenes pipe_bend90_tight --methods curved --trials 1 --out <目录>
"""
import argparse
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_experiment as be

BASELINE = be.OUT
ENTRY = str(be.ROOT / 'codes/Adaptive_Pole_Climbing_CurvedAxis_Mujoco.py')
be.LAUNCHERS.update(curved_local=[ENTRY, '--variant', 'local'], curved=[ENTRY, '--variant', 'curved'])
be.METHOD_STYLE.update(compliance=('原方法：直轴顺应性', '#2b7bba'), curved_local=('弯曲轴：局部轴+局部截面', '#7fbf7b'),
                       curved=('弯曲轴：+螺距角调制（完整）', '#1b7837'))
be.METHODS = ('compliance', 'nominal', 'curved_local', 'curved')
be.TITLE = '弯曲轴顺应性与基线对比（实物参数管道场景，每组 5 次）'
be.OUT = be.ROOT / 'experiments/curved_axis_20261001'
be.WORK = be.ROOT / 'validation/results/curved-axis-exp'
NEW_METHODS = ('curved_local', 'curved')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--scenes', nargs='*', default=be.SCENES)
    parser.add_argument('--methods', nargs='*', default=list(NEW_METHODS))
    parser.add_argument('--trials', type=int, default=be.TRIALS)
    parser.add_argument('--max-steps', type=int, default=be.MAX_STEPS)
    parser.add_argument('--workers', type=int, default=be.WORKERS)
    parser.add_argument('--out', type=Path, default=be.OUT)
    args = parser.parse_args()
    be.OUT = args.out.resolve()
    if not args.analyze:
        jobs = queue.Queue()
        for scene in args.scenes:
            for method in args.methods:
                for seed in range(1, args.trials + 1):
                    trial = be.Trial(scene, method, seed, args.max_steps)
                    if not (be.OUT / 'trials' / f'{trial.label}.json').exists():
                        jobs.put(trial)
        print(f'待运行 {jobs.qsize()} 次试验，{args.workers} 个并行', flush=True)
        results, lock = [], threading.Lock()
        threads = [threading.Thread(target=be.worker, args=(be.BASE_PORT + 40 + i, jobs, results, lock))
                   for i in range(min(args.workers, jobs.qsize()))]
        for thread in threads:
            thread.start()
            time.sleep(2)
        for thread in threads:
            thread.join()
        errors = [r for r in results if 'error' in r]
        if errors:
            print(f'{len(errors)} 次试验出错，未记入结果：', *[e['error'][-200:] for e in errors[:3]], sep='\n')
    for row in be.summarize(extra_dirs=[BASELINE]):
        print(row['scene'], row['method'], f"{row['success']}/{row['trials']}",
              f"rise={row['max_rise_m_mean']:.3f}±{row['max_rise_m_std']:.3f}", f"sat={row['saturation_fraction_mean']:.2f}")


if __name__ == '__main__':
    main()
