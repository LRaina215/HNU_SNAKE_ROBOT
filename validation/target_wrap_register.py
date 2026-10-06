"""Register round2 from existing offline diagnosis; no physics evolution."""
from pathlib import Path
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'experiments/curve_guided_transfer_20261006'
OLD = PARENT / 'stage_continuation_20261006'
BASE = PARENT / 'target_wrap_round2'


def main():
    for name in ['baseline_timeline.json', 'baseline_summary.json', 'baseline_diagnosis.png']:
        if not (BASE / name).exists():
            raise RuntimeError('Completed diagnosis required: ' + name)
    with (PARENT / 'trial_ledger.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if len(rows) != 11 or (BASE / 'revision_registry.json').exists():
        raise RuntimeError('Reconcile counts or preserve existing registration')
    inputs = BASE / 'inputs'
    inputs.mkdir()
    for name in ['reference_ready.npz', 'command_schedule.npz', 'path_gate.json', 'interface_check.json']:
        shutil.copy2(OLD / 'local_path_boundary_speed3' / name, inputs / name)
    for name in ['initial_boundary_pose.npz', 'initial_boundary_pose_perturb.npz']:
        shutil.copy2(OLD / name, BASE / name)
    with (BASE / 'trial_ledger.csv').open('w', encoding='utf8', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    registry = dict(status='ACTIVE', prior_dynamics=11, new_dynamics=0, total_dynamics=11,
        new_budget=8, total_budget=19, formal_revisions=0, maximum_formal_revisions=2,
        freeze_trials_reserved=2, oracle_axis='target only', oracle_maximum_pairs=1,
        oracle_formal_success_allowed=False, full_V3_path='NOT_READY',
        freeze_perturbation='root x +1mm before first step; same held qpos/ACK ctrl, qvel zero',
        further_dynamics_authorized=True)
    (BASE / 'revision_registry.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (BASE / 'manifest.md').write_text('''# 第二轮运行前登记

2026-10-06。依据审阅会话内用户明确授权及第二轮任务书。历史11次保留；新增最多8次、累计最多19次，最多两轮正式机制修订，保留2次冻结复核空间。现有baseline记录是本轮此前已完成的只读诊断，物理步0；本次不重复执行。旧V3完整路径仍NOT_READY。

## 基线证据与假设

前部在5秒已形成目标支撑，并非首次抓管前就全面卷紧；原承重模块为M0/M6/M7。目标方向误差从约12度增加到t50的32度，后于此阶段前部方位净跨度降到终点112度；名义前部跨度仍约292度。极角跨度只是开链描述量，不是包围拓扑或力学稳定证明。M9末态参考外壳径向间隙约8.31mm，实际约15.43mm，精确目标接触始终0N；J9/J10权重0。原承重模块接触减少与轴误差、受力迁移并存，尚不能写唯一原因。

主要待检验解释：目标轴估计不能维持已包绕身体的正确方向，同时有限身体单元进入目标区域后，材料窗口仍将其排除。没有充分证据优先采用局部张开；更大半径可能扩大已有间隙。不能默认提高K解决。

## 先运行方向诊断

只替换目标管轴在头部坐标系中的方向 a_t=R_head.T*[0,1,0]，标为ORACLE_AXIS_DIAGNOSTIC_TARGET。原管轴、点云检查、局部窗口、投影、增益、恢复项、初态、参考和时序保持原R2。原管方向误差未替换，所以本观察只能判断目标方向单独是否足够，不能排除两侧误差耦合。保持失败则不做其B。真值版本永不计正式通过或冻结复核。

固定原66.8秒观察时长、3mm/s、d0..161.25、3秒中停/5秒末停，额外rolling0。控制器成熟接入时间45.25秒及旧登记轴先验连续继承。两侧物理、执行器和通信确认机制不变。

## 候选与复核边界

正式候选在诊断后登记；默认仅20关节编码器、已确认命令及历史、初始几何与模型。若修订在d=0对任意观测和状态完全等同原保持分支，可复用原66.8秒C0；否则先运行匹配C0。不能以首条目标相同代替分支等价。

冻结差异预登记：根部世界x增加1mm，首次步进前完成，其他保存初态及ACK目标相同，速度初置0；需要匹配该初态的C0。运行后不重置位姿或速度、不加支撑。原M8/M9、实际按序接触、相对C0额外推进、中停及末端最后2秒全模块位移≤6.45mm和原保持收敛标准均不变。

全部物理演化都计次，失败同计。世界位姿及接触仅诊断、计分、停止；只有上述目标轴诊断例外。新目录独立保存，不覆盖原报告、源码、准入，无SHA、全仓审计、实机连接或无依据扫参。
''', encoding='utf8')
    print('Registered round2: new 0/8 dynamics; prior 11; target-axis-only diagnostic.')


if __name__ == '__main__':
    main()
