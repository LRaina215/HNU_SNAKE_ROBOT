"""Finish a failed-gate report only after all five verification traces audit."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from contact_turn_experiment import BASE,write_json,sha
from contact_turn_stage1_audit import audit


def main():
    gate=json.loads((BASE/'stage1_release_gate.json').read_text(encoding='utf8'))
    assert gate['total']==5 and len(gate['results'])==5 and not gate['passed']
    rows=[]
    for item in gate['results']:
        row=audit(ROOT/item['directory']);rows.append(row)
        print(json.dumps(dict(audited=row['label'],outcome=row['outcome'])),flush=True)
    write_json(BASE/'stage1_verification_independent_audit.json',dict(passed=True,rows=rows))
    training=json.loads((BASE/'stage1_training_independent_audit.json').read_text(encoding='utf8'))['rows']
    stage0=json.loads((BASE/'stage0_round2_clearance.json').read_text(encoding='utf8'))
    freeze=json.loads((BASE/'stage1_parameter_freeze.json').read_text(encoding='utf8'))
    output=[]
    for group,values in [('training',training),('verification',rows)]:
        for row in values:
            output.append({k:row[k] for k in ('label','outcome','success','actual_target_contact_modules',
                    'ground_contact','self_contact','tracking_max_deg','torque_rms_nm','saturation_fraction')})
            output[-1]['group']=group
    with (BASE/'stage1_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=output[0]);writer.writeheader();writer.writerows(output)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for group,values,marker in [('training',training,'o'),('verification',rows,'x')]:
        axes[0].scatter(np.arange(len(values)),[x['tracking_max_deg'] for x in values],label=group,marker=marker)
        axes[1].scatter(np.arange(len(values)),[x['actual_target_contact_modules'] for x in values],label=group,marker=marker)
    axes[0].set_ylabel('Maximum tracking error during motion (deg)');axes[1].set_ylabel('Maximum actual target-pipe contact modules')
    for ax in axes:ax.set_xlabel('Trial index within group');ax.legend();ax.grid(alpha=.2)
    fig.savefig(BASE/'stage1_summary.png',dpi=170);plt.close(fig)
    slip=[x['hold_net_slip_mm'] for x in stage0['results']]
    rms=[x['torque_rms_nm'] for x in stage0['results']]
    peaks=[x['hold_max_torque_nm'] for x in stage0['results']]
    rejected=sum(bool(p['stop_reason']) for r in stage0['results'] for p in r['probes'])
    residual=max(abs(x) for r in stage0['results'] for p in r['probes'] for x in p['residual_deg'])
    report=f'''# 局部负载主动探测与急弯跨越：阶段验证报告

执行日期：2026-10-04。方案日期：2026-10-03。结论停在阶段1：已知方向的运动可行性未通过，不放行阶段2、3。

## 结果与结论边界

- **已确认：** 阶段0最终有效批次三个相位均通过10秒真实物理抱管与试探回退。局部弯曲的数值/模型几何检查通过。
- **已确认：** 水平L、20关节、已知目标方向的名义局部弯曲控制，12组训练成功0/12；冻结参数后五个不同相位/位置复核成功{gate['successes']}/5，未达到4/5门槛。训练和复核均没有真实目标管接触；独立评分按目标管碰撞体及接触位置确认。
- **合理推测：** 暂停rolling后的局部变形没有建立新接触，当前运动过程缺少能完成接近、释放、重包绕的动作；理想后部锚定FK选择方向不能保证真实接触动力学下的头部轨迹。负载/跟踪误差也提示名义形状未被完全执行。这些是解释候选，尚未通过因果消融证明。
- **未验证：** 逐关节反馈能否识别分支方向；主动相对被动是否有优势；融合在线顺应规则后的转移；完整自主触发/识别/转移；T型、竖直工况、连续rolling及噪声/延迟泛化。

这不是“扭矩识别障碍方向不可行”的结论，也不是“20关节必然不能过弯”的证明。它是所实现的名义局部弯曲形状族、方向选择器和运动调度在本轮条件下未通过的负面结果。遵照方案前一阶段未通过就停止后续批量试验的要求，不输出虚构混淆矩阵或自主成功率。

## 实现与固定条件

新增 `snake_control/local_bend_backbone.py`、`snake_control/contact_turn_controller.py`、`validation/contact_turn_experiment.py`，以及独立数值检查、训练、审计、复核、回放和报告入口。新控制器使用复制的 `JointObservation`；真值只经明确命名的 `KnownDirectionOracle`进入K辅助组。本轮未启用P/A控制器。

两分量Bishop弯曲场同时生成三维目标曲线及Yamada积分。局部高斯窗口减弱基础螺旋，再加入平面弯曲；这是受2016文献启发的新参数化，**不是2016原文EKF复现，也不是原2023控制器**。初始化后的名义参考通过统一弯曲场校正后严格匹配整组已执行目标，不叠加一套独立关节覆盖路径。μ仅在至少3个目标接触模块及净包绕≥π/2时允许后移，本轮没有满足此条件。

环境：Python 3.9.21、MuJoCo 3.3.0、NumPy 1.23.5；21个质量模块、20关节，64.5 mm间距、kp=127、gear=1、力矩范围±2.355 N·m、1 ms物理步长、真实重力。未调整摩擦/力矩/质量，未焊接尾部或使用隐形支撑。直管负对照由水平L派生，保持同高度、朝向、材质及机器人动力学。

目标仍经Python→隔离EXE→MuJoCo转发。每次等待目标实际应用后才推进物理时间，保存计算/提交/应用时刻及编号；这是受物理时钟调度的仿真，**不证明实时20 Hz通信性能**。原始状态100 Hz、决策20 Hz；`mj_forward`在采样qpos/qvel/ctrl上重算动力学，使力矩与角度属于同一物理时刻。网络等待不计作抱管时间，也不恢复qpos。EXE私有工作副本不入库，未操作实机。

当前比例执行器的力矩是角误差的确定性截断变换。本轮没有比较角度特征和增加力矩特征，也没有把仿真力矩换算为实机电流。

## 阶段0：最终有效批次

初始相位0、2π/3、4π/3，统一预缠绕半径102 mm；初始间隙5.699、7.391、3.925 mm，初始接触数均0。预设R=81.5 mm、P=220 mm，共同初始化后基于实测角作一次预紧释放，将目标误差限为0.5×力矩上限/kp；随后目标固定。该规则仅是共同初始化，**不作为在线顺应控制**。

| 相位 | 10秒净滑移/mm | 力矩RMS/N·m | 保持最大力矩/N·m |
| --- | ---: | ---: | ---: |
| 0 | {slip[0]:.6f} | {rms[0]:.6f} | {peaks[0]:.6f} |
| 2π/3 | {slip[1]:.6f} | {rms[1]:.6f} | {peaks[1]:.6f} |
| 4π/3 | {slip[2]:.6f} | {rms[2]:.6f} | {peaks[2]:.6f} |

全部无触地、无活动的非相邻自碰撞。12次正反试探中{rejected}次触发90%力矩持续0.2 s门槛并停止继续增加；均真实渐进回退，最大回退残差{residual:.6f}°，低于预声明的0.1°近姿态检查。目标峰值0.5°，记录实际达到幅度，未通过增大峰值绕过拒绝。没有将被拒绝试探写成完整幅度成功。

独立逐帧重算最终三个状态：100 Hz时间步严格递增，每次保持1000帧，关节与速度地址、力矩公式、执行器映射、FK位置与记录一致。详见 `stage0_independent_audit.json`。原位等待UI的时间未作为物理保持时间。

## 数值与几何检查

零转弯退化误差0；初始化参考形状及独立积分误差均4.44×10⁻¹⁶ rad；Bishop标架正交误差2.81×10⁻¹¹；2 mm折线采样弧长相对误差1.68×10⁻⁵。参数微小扰动连续，最大目标变化0.000205°。

4/6关节、64.5/129 mm宽度、15–90°转弯，8方向×3相位的480组模型几何检查，最大目标≤75.425°，无活动的非相邻自碰撞。模型相邻壳体碰撞由原铰链模型过滤，**这些检查不证明实物壳体/铰链间隙**。实际FK在四个方向确有不同的头部位移，但后部刚性锚定只用于离线方向选择，不能作为动力学支撑保证。

## 阶段1：训练、冻结及复核

训练候选在运行前写入 `stage1_training_plan.json`：前部4/6关节、窗口64.5/129 mm、总弯曲45/60/90°，共12组；初相位0，最大预算60 s。机器人先沿旧管rolling，头部距离接头约80 mm时暂停相位，以10°/s渐增局部弯曲。若15 s没有新的推进/目标支撑进展则结束，所有失败保留。

独立审计全部训练决策状态的目标接触归属，并重算FK、初态无接触、力矩公式及命令编号。12组均无触地、无活动非相邻自碰撞；运动阶段最大跟踪误差约9.32–19.22°。目标角没有硬裁剪以制造几何通过。

训练成功均0；按预声明规则选取前部{freeze['front']}关节、宽度{freeze['width_mm']} mm、转弯{freeze['turn_deg']}°用于必需的复核，**这不表示已经找到有效参数**。五个测试状态在训练前确定，相位0、72、144、216、288°，入管位置偏移−20、0、+20、−10、+10 mm；偏移只在第一次物理步前设置，此后不重置姿态。全部使用冻结的名义形状与调度，没有据测试调参。

最终{gate['successes']}/5，门槛失败；未完成有效包绕、尾部通过、目标管继续推进0.2 m以及随后10秒保持，不能以头部接近接头、低力矩或投影包绕角代替。详见 `stage1_release_gate.json`、`stage1_verification_independent_audit.json`和 `stage1_summary.csv`。

交付补充审计见 `delivery_independent_audit.json`：17次转管共48,232个100 Hz保存姿态重新计算碰撞体归属，目标管接触均为零；有效20次试验中54,575个命令后采样点的ctrl与已应用命令目标一致。冻结源码、复核入口及每次复核所用冻结文件哈希一致。该审计只重算轨迹，不推进物理时间。数据入口、代表性视频及逐文件哈希见数据目录的 `README.md`、`delivery_manifest.json`。

## 工程探索与排除项

保存了初始控制器/采样器的全部中断和探索。早期反馈过期由物理推进后的旧快照造成，修复为取得新鲜帧再提交。原第一相位生成路径在第三相位有初态穿透；旧批次不能作为阶段0正式证据。首版试探遗漏持续负载停止检查，补齐后重跑；旧试探仅作不合规探索。首个运动评分曾用横向位置误判旧管侧面接触，已改为真实目标管碰撞体及接触区，并重新训练和独立审计。

共同预紧只进行了70%、50%两轮有界修订；相位初态穿透、反馈过期与保护遗漏属于实验输入/执行器错误，单独修复并保留旧数据。没有把这些失败删除或混入有效统计。旧探索中缺少完整源码快照的尝试仅保留哈希/轨迹，不能作为完整可复现结论；最终有效批次及训练冻结源码已保存到 `source_snapshots/`。

## 回放与复现

数据：`experiments/contact_turn_20261003/`；实际运行日期为10月4日，保留方案约定的目录名。每次含 `trace.npz`、`commands.json`、结果、控制状态与必要真值诊断；训练/复核有manifest及冻结参数。`trace.npz`可用统一回放脚本生成视频，视频重算运动学但不推进物理。

```powershell
$turnPython = "$env:TEMP\\snake-mujoco330\\Scripts\\python.exe"
# 核对已保存的最终阶段0和局部几何；不启动运动
& $turnPython -X utf8 validation/contact_turn_checks.py
# 在新目录名下重复阶段0，拒绝覆盖已有尝试
& $turnPython -X utf8 validation/contact_turn_experiment.py --stage0 --round 2 --attempt reproduce_1
# 单独重复冻结名义参数的水平L试验
& $turnPython -X utf8 validation/contact_turn_experiment.py --stage1 --front {freeze['front']} --width {freeze['width_mm']} --turn {freeze['turn_deg']} --phase 0 --attempt reproduce_1
# 审计/冻结已有12项；复核入口验证源码冻结并跳过已有结果
& $turnPython -X utf8 validation/contact_turn_stage1_audit.py
& $turnPython -X utf8 validation/contact_turn_verify.py --run
& $turnPython -X utf8 validation/contact_turn_report.py
# 对任一次保存轨迹生成回放（替换成实际case路径）
& $turnPython -X utf8 validation/contact_turn_replay.py --case experiments/contact_turn_20261003/stage0_round2_clearance/phase0 --video
```

下一研究动作应先改善已知方向运动参考：检验接近距离、解旋/前送/重包绕动作以及基于真实反馈的转向与支撑保持。扩大动作结构或恢复rolling需要新版本、有界训练和新的冻结；本轮没有继续搜索或借真值将名义失败包装成自主成功。
'''
    (ROOT/'docs/局部负载主动探测与急弯跨越/局部负载主动探测与急弯跨越_阶段验证报告.md').write_text(report,encoding='utf8')
    write_json(BASE/'completion_audit.json',dict(stage0_proven=True,geometry_proven=True,
                 stage1_training_proven=True,stage1_verification_proven=True,stage1_passed=False,
                 allowed_stop_condition_satisfied=True,stage2_tested=False,stage3_tested=False,
                 report_sha256=sha(ROOT/'docs/局部负载主动探测与急弯跨越/局部负载主动探测与急弯跨越_阶段验证报告.md'),
                 verification_audit_sha256=sha(BASE/'stage1_verification_independent_audit.json')))


if __name__=='__main__':main()
