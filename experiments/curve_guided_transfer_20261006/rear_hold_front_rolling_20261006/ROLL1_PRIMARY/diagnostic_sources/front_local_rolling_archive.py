"""Archive real ROLL1/ROLL2 outcomes, never runs physics or fitting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import csv,json,shutil,re,ast
import numpy as np
from front_local_rolling_geometry import BASE,PARENT,write
from snake_control.front_local_rolling import FrontLocalRollingReference


def read(path):return json.loads(path.read_text(encoding='utf8'))
def ledger(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    first=read(BASE/'ROLL1_PRIMARY/result.json');second=read(BASE/'ROLL2_PRIMARY/result.json')
    old=ledger(BASE/'historical_17_trial_ledger.csv');allrows=ledger(PARENT/'trial_ledger.csv');new=ledger(BASE/'trial_ledger.csv')
    if allrows[:17]!=old or allrows[17:]!=new or len(allrows)!=19 or len(new)!=2:raise RuntimeError('Expected preserved17+2 actual runs')
    if not second['combination_pass'] or second['forward_roll_grasp_pass']:raise RuntimeError('Archive only this partial-success outcome')
    # The raw result keeps its internal combination_pass. The current ledger
    # names both outcomes, with the original recorded CSV retained alongside.
    for path in [PARENT/'trial_ledger.csv',BASE/'trial_ledger.csv']:
        backup=BASE/('before_goal_outcome_'+('total' if path.parent==PARENT else 'local')+'_trial_ledger.csv')
        if not backup.exists():shutil.copy2(path,backup)
        rows=ledger(path);fields=list(rows[0]);rows[-1]['outcome']='local_roll_hold_pass_forward_fail'
        with path.open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for name in ['ROLL1_PRIMARY','ROLL2_PRIMARY']:
        case=BASE/name;z=np.load(case/'trace.npz');ref=FrontLocalRollingReference(case/'reference.npz',case/'original_reference.npz')
        ts=[0,5,float(z['sim_time'][-1])]
        with (case/'actual_key_joint_postures.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f);writer.writerow(['actual_time_s','stage']+['q_J'+str(i) for i in range(1,21)]+['ACK_J'+str(i) for i in range(1,21)]+['retained_J'+str(i) for i in range(1,21)])
            for time_s in ts:
                i=int(abs(z['sim_time']-time_s).argmin());h=z['ctrl'][i]-ref.joint_angles(float(z['front_phase_rad'][i]))
                writer.writerow([float(z['sim_time'][i]),str(z['stage'][i])]+np.degrees(np.r_[z['q'][i],z['ctrl'][i],h]).tolist())
        dest=case/'diagnostic_sources';dest.mkdir(exist_ok=True)
        for file in ['front_local_rolling_diagnostics.py','front_local_rolling_archive.py']:shutil.copy2(ROOT/'validation'/file,dest/file)
    if not (BASE/'progress_report_before_archive.md').exists():shutil.copy2(BASE/'result_report.md',BASE/'progress_report_before_archive.md')
    motion=read(BASE/'ROLL2_PRIMARY/diagnostic_metrics.json');review=read(BASE/'direction_revision_roll2/fixed_reference_review.json')
    report=f'''# 后部保持与前部局部 rolling：滚动保持通过，向前运动未通过

2026-10-06。水平两管L型、20关节MuJoCo，实机由用户另行安排。**本轮新增2次、累计19次，停止当前候选；剩余2次不使用。** 原始无扰动R2起点上，ROLL2完成局部滚动、后部门槛内保持及10秒停后抓持；实际前部净退回，未实现“向前滚并抱稳”。不开展扰动C0、FREEZE或第三个候选。完整V3仍为NOT_READY。

## 实际结果

| 实测项目 | ROLL1首试验 | ROLL2唯一方向/连接修订 |
| --- | --- | --- |
| 总物理时长 | 23.65秒，中止 | 27.00秒，完成 |
| 入段5秒保持 | 通过 | 通过 |
| 后部M13–M20最大位移 | 6.46466 mm，超过6.45 mm | 4.02528 mm，门槛内 |
| 后部最大源管周向变化 | 2.25904° | 2.40385°，小于5° |
| 后部源侧接触模块 | 全程至少4，停止帧5 | 全程至少4，末帧5 |
| 停后保持 | 未到达 | 10秒通过 |
| 末2秒最大模块漂移 | 动作中止，不能作为停后保持 | 0.04818 mm |
| 末5秒最大模块速度 | 动作中止 | 0.06639 mm/s，低于5 mm/s |
| 六个前部截面实际转动，末1秒均值 | −0.819°至−5.459°，未达门槛 | +10.181°至+18.339°，全部达到7.5° |
| 实际前部M0–M6轴向位移中位数，末1秒评分 | −5.13931 mm | −6.62677 mm，向前门槛>+1 mm未通过 |
| 末1秒目标管在前部投影区域内 | 100%，但没有停后窗口 | 100%；至少3个前部持续接触模块 |
| 最大关节跟踪误差 | 0.96128° | 0.90737° |
| 最大连接段跟踪误差 | 0.71866° | 0.74777° |
| 力矩饱和 | 无 | 无，峰值2.01125 Nm |
| 隔离EXE命令应用确认 | 473/473 | 540/540 |

ROLL1最先触发后部实际位移停止，动作与10秒末保持尚未完成；不能把中途未达到完整转动量说成完整动作已经执行。源侧后部法向力约59.54→68.61 N，接触没有丢失。模块沿源管负X整体位移并伴有形变；**仅模块中心位移不能唯一识别精确接触面的滚动/滑动机制**，不认定松开、摩擦不足或驱动力不足是唯一原因。

ROLL2的后部4.025 mm是验收门槛内保持，不能写成世界位置完全不动。前部停后抓持发生在**前后两管共同支撑**的条件下，未验证前部独立承重，也未做方向/初态扰动复核，不宣称鲁棒。没有对照证明新版控制优于旧版。

ROLL2实际位移时序显示前部中位轴向位移曾达到+{motion['actual_front_median_axial_peak_mm']:.5f} mm，随后退回，最大过程退回{motion['actual_front_median_axial_maximum_return_mm']:.5f} mm，末帧−6.61831 mm；正式末1秒评分为−6.62677 mm。不能用短暂向前、相位增加、名义+Y或被动位移代替净前进成功。六个真实身体截面角、实际模块位置及其时序均来自原始物理轨迹。

## 修订与依据

固定后部J13–J20主动名义角，d=0、后部主动rolling=0，原顺应继续。前部J1–J6局部滚转，J7–J12用同一有限链协调连接；R95/P220、周向±5°及轴向±32.25 mm构造范围保留。后部不锁世界位姿，不主动放松。私有link12名义锚定只计算参考，从未应用到物理根。运行开始后无位置/姿态/速度/控制状态重置、触地、非相邻自碰撞或附加外力支撑。

首参考的名义轴向协调为负，实际前部也退回，是选择唯一方向修订的明确依据。ROLL2反转管相对材料截面转动目标及曲率引导符号，并协调连接到名义+Y；没有增加夹紧量、增益或驱动能力，也未减小控制相位范围。单纯反转后半段仍退回的中间拟合保留，随后把相对起点的正向下限纳入同一次拟合。它不保证逐步单调前进。

控制参数仍0→15°，**不等于身体实际滚了15°**。ROLL2固定参考终点六个名义截面转动约11.436°–12.688°，精确15°±3°拟合未通过；旧精度失败记录保留。真实末态转动约10.2°–18.3°，其大小与上述名义量分开。名义前部净+Y中位数1.50 mm、过程最大正向5.98522 mm、最大退回4.98522 mm；实际仍净退回，说明当前参考与控制组合尚未兑现真实前进目标。关节跟踪误差小也不能据此认定空间运动方向正确。

保留ACK可执行形变状态：每次确认后`h = ACK目标 − 当时名义角`，后续以`名义角 + h − 编码器角`计算跟踪与局部顺应调整，恢复时间0.5秒。K仍0.01/0.02，±90°、10°/s、2.355 Nm及模型质量/摩擦/重力不变。控制仅使用编码器、ACK/历史及已登记初始几何；世界位姿、管相对材料角与精确接触只用于离线拟合、诊断、计分和危险停止，未加入控制反馈。

## 离线准入的明确取舍

ROLL1完成旧准入并按限速重定时为5+33.95+10=48.95秒；23.65秒中止。ROLL2同一方向修订累计数值用时2400.003秒后固定唯一参考，不再搜索。所有终点节点已经生成，但旧裕度/线性方向要求仍有69个失败样本，**旧准入没有通过**。30分钟上限与额外10分钟收尾均保存，不把求解器未收敛或微小裕度残差写成物理不可能。

审阅在物理运行前明确调整离线准入：0.05 mm保留为拟合目标，独立全样本代理间隙须非负（仅1e−8 mm浮点精度），所有实体接触dist须非负；精确起点与M11源侧原正间隙兼容保留。方向准入改为相对起点名义中位+Y全程非负（0.001 mm精度）、终点仍>1 mm，未强制每点达到`1.5 mm×参数/15°`。M11近不变代理行比较精度1e−7 mm及方向数值精度修复都明示，不能声称离线几何数字全部不变。**真实7.5°滚转、>1 mm前进、后部6.45 mm/5°、接触及末保持判据全部不变。**

固定参考经0.05°、所有节点及所有50毫秒指令独立检查，共539个唯一姿态：无实体/代理穿透，最小代理间隙0.000065386 mm（继承原起点），长度误差≤4.84e−13 mm，后部名义节点误差≤1.37e−12 mm，后部名义角差0，最大关节69.78791°，最大名义速率5.42766°/s。5秒入段+12秒动作+10秒停后保持共27秒。检查是有限采样，不是连续区间的数学证明。两个版本因可执行速率要求而采用不同动作时长，不能从两次结果唯一分离方向、连接与时序的作用。

零相位完整委托原R2，保持任意合法编码器/ACK历史的分支等价，复用已有66.8秒C0；初态/qpos和零分支目标差均为0。这只证明起始保持分支，不代替非零rolling和停后实际验证。五段始终指身体参考曲线。

## 证据与交付

- ROLL1：[结果](ROLL1_PRIMARY/result.json)、[原始轨迹](ROLL1_PRIMARY/trace.npz)、[控制/h/A–D](ROLL1_PRIMARY/control.json)、[精确接触](ROLL1_PRIMARY/exact_contacts.json)、[应用确认](ROLL1_PRIMARY/commands.json)、[正常速度实际回放](ROLL1_PRIMARY/replay_1x.mp4)、[实际关键20关节姿态](ROLL1_PRIMARY/actual_key_joint_postures.csv)。
- ROLL2：[结果](ROLL2_PRIMARY/result.json)、[原始轨迹](ROLL2_PRIMARY/trace.npz)、[实际位移/转动时序](ROLL2_PRIMARY/actual_motion_timeline.json)、[控制/h/A–D](ROLL2_PRIMARY/control.json)、[精确接触](ROLL2_PRIMARY/exact_contacts.json)、[应用确认](ROLL2_PRIMARY/commands.json)、[正常速度实际回放](ROLL2_PRIMARY/replay_1x.mp4)、[末帧](ROLL2_PRIMARY/final_frame.png)、[诊断图](ROLL2_PRIMARY/actual_diagnostics.png)、[实际关键20关节姿态](ROLL2_PRIMARY/actual_key_joint_postures.csv)、[运行时源码](ROLL2_PRIMARY/sources/)。
- 固定参考：[拟合](direction_revision_roll2/reference_frozen_at_2400.npz)、[名义20关节姿态](direction_revision_roll2/key_joint_postures.csv)、[准入取舍](direction_revision_roll2/admission_revision.json)、[独立检查](direction_revision_roll2/fixed_reference_review.json)、[旧69样本失败](direction_revision_roll2/fixed_reference_old_admission_failures.json)、[接口](direction_revision_roll2/interface_check.json)、[名义时序](direction_revision_roll2/nominal_motion_timeline.json)。
- 代码：[控制器](../../../snake_control/front_local_rolling.py)、[实际入口](../../../validation/front_local_rolling_experiment.py)、[固定参考检查](../../../validation/front_local_rolling_fixed_admission.py)、[回放/诊断](../../../validation/front_local_rolling_diagnostics.py)。
- 当前状态：[active_registry](active_registry.json)、[本轮台账](trial_ledger.csv)、[总台账](../trial_ledger.csv)。早期[STOPPED](revision_registry.json)、[离线历史报告](offline_history_report.md)、600秒数值收尾、1800/2400秒记录、V1/V2/V3及所有17次历史不覆盖。

回放只重建真实qpos，20fps/1倍速，不播放名义根运动或补造末端窗口；含0秒与实际终点，末帧额外显示0.05秒。新实验在独立目录；没有F、RL、其他管型、实机连接、SHA或全仓审计。

**当前候选停止。** 已确认单一起点、双管支撑下的局部滚动保持能力；向前运动机制尚未实现。缺项是可兑现真实前进的空间运动与支撑协调，原因仍需证据，不认定过渡段、摩擦或观测能力为唯一原因。本轮不追加修补或方向相关复核，不自动进入B、完整路径或前部独立承重。
'''
    (BASE/'result_report.md').write_text(report,encoding='utf8')
    registry=read(BASE/'active_registry.json')
    if not (BASE/'before_archive_active_registry.json').exists():write(BASE/'before_archive_active_registry.json',registry)
    old_registry=read(BASE/'before_archive_active_registry.json')
    registry['roll1_registration']={k:old_registry[k] for k in ['total_s','active_s','nominal_actual_material_twist_deg','geometry_start_compatibility']}
    registry.pop('nominal_actual_material_twist_deg',None)
    registry.update(status='STOPPED_FORWARD_OBJECTIVE_NOT_MET',new_dynamics=2,total_dynamics=19,remaining_unused_trials=2,
        further_dynamics_authorized=False,local_roll_and_hold='PASS_single_original_initial_dual_pipe_support',
        forward_roll_and_grasp='FAIL_actual_front_net_negativeY',direction_revision_status='PHYSICALLY_RUN',
        freeze='NOT_RUN_forward_gate_not_met',matched_perturbed_C0='NOT_RUN_forward_gate_not_met',robustness='NOT_TESTED',
        front_independent_load_bearing='NOT_TESTED',old_V3_path='NOT_READY',last_case='ROLL2_PRIMARY',
        last_outcome='local_roll_hold_pass_forward_fail',old_direction_admission_pass=False,
        final_admission_revision='direction_revision_roll2/admission_revision.json',report='result_report.md',
        total_s=review['total_s'],active_s=review['active_s'],nominal_material_twist_deg=review['nominal_director_change_deg'],
        direction_geometry_goal='nominal front median+Y nonnegative(.001mm precision) throughout, endpoint>1mm; not monotonic',
        geometry_start_compatibility='exact saved start, nonnegative proxy1e-8mm precision, actual entity penetration hard fail, M11 original positive gap comparison1e-7mm; .05mm is fit goal',
        original_direction_fit_goal='1.5mm*parameter/15deg; old69 failures preserved',actual_front_median_axial_mm=-6.626772491090151)
    write(BASE/'active_registry.json',registry)
    handoff=ROOT/'交接记录.md';s=handoff.read_text(encoding='utf8')
    s=re.sub(r'### 曲线引导与顺应反馈跨管：[^\n]*', '### 曲线引导与顺应反馈跨管：局部滚动保持通过，实际向前未通过并停止',s,count=1)
    a=s.index('- **范围与最新结论：**');b=s.index('\n- **候选与几何：**',a)
    s=s[:a]+'''- **范围与最新结论：** 水平两管L形、20关节MuJoCo，实机由用户安排。历史17次保留；新授权后部保持+前部局部rolling本轮新增2/4次、累计19/21次，停止当前候选，剩余2次不使用。ROLL1于23.65秒后部最大位移6.46466mm先停止、前部净退回；唯一方向/连接修订ROLL2完成27秒，局部滚动+后部保持+10秒停后抓持通过，但前部实际净Y中位数−6.62677mm，向前滚并抱稳未实现。后部最大位移4.02528mm属门槛内保持，不能说世界位置不动。成功限原始无扰动起点、双管共同支撑；未验证前部独立承重或鲁棒性，不进行扰动C0/FREEZE/新候选。完整V3仍NOT_READY，当前以active_registry和总台账为准。''' +s[b:]
    a=s.index('- **后部保持+前部局部rolling实现与边界：**');b=s.find('\n### ',a)
    if b<0:b=s.find('\n## ',a)
    if b<0:b=len(s)
    s=s[:a]+'''- **后部保持+前部局部rolling实现与边界：** front_local_rolling.py固定J13–J20主动名义角、d=0和后部主动rolling=0，原顺应保留；前部J1–J6、连接J7–J12有限链配合。R95/P220、±5°周向、±32.25mm轴向范围、K0.01/0.02、0.5秒恢复及±90°/10°s/2.355Nm不变。私有link12锚定从未作用到物理根，控制仅编码器/ACK/历史和初始先验。参数0→15°不是实际滚15°；ROLL2名义末截面11.436–12.688°，精确15±3°拟合失败保留。名义净+Y1.5mm、峰值5.985mm、过程最多退回4.985mm，不保证单调前进。
- **局部rolling实际结果：** ROLL1后部模块负X位移伴形变、源侧始终≥4接触、停止帧5，不能唯一认定接触面滑动/摩擦不足。ROLL2后部最大4.025mm/2.404°、源侧≥4接触，停后10秒通过，末2秒漂移0.04818mm、末5秒速度0.06639mm/s、末目标前部≥3接触且管轴在投影区域；真实六截面转动10.181–18.339°。前部曾+1.489mm后退回，最大过程退回8.166mm、末1秒净−6.62677mm，向前目标未通过。最大跟踪0.907°、无饱和/触地/自碰撞/额外外力/运行后重置，累计1013条EXE命令全部应用确认；无对照不宣称控制优势，独立承重及方向/扰动复核未做。
- **离线准入取舍与交付：** 同一方向修订累计2400.003秒后固定参考，旧69个裕度/线性+Y样本失败保留，未声称旧准入通过。审阅于物理前将0.05mm保留为拟合目标，准入改为代理非负/实体无穿透，精确起点/M11基线兼容保留；方向全程相对起点名义中位非负(.001mm精度)、终点>1mm，实际7.5°/1mm/6.45mm/5°及末保持门槛不变。539个唯一姿态及全部指令通过，长度/后部固定/±90°成立，名义峰值5.428°s，5+12+10=27秒；非连续数学证明。完整零相位R2/ACK等价复用已有66.8秒C0。见[最终报告](experiments/curve_guided_transfer_20261006/rear_hold_front_rolling_20261006/result_report.md)与[ROLL2正常速度回放](experiments/curve_guided_transfer_20261006/rear_hold_front_rolling_20261006/ROLL2_PRIMARY/replay_1x.mp4)，含ROLL1/ROLL2真实轨迹、接触、h/A–D、20关节关键实态、固定参考和源码。当前停止，旧STOPPED/TIME_CAP及17次原记录保留；无新增B/F/RL/其他管型/实机/SHA/全仓审计，不据此否定用户方法或唯一归因过渡段/摩擦。
''' +s[b:]
    handoff.write_text(s,encoding='utf8')
    sources=sorted((ROOT/'validation').glob('front_local_rolling*.py'))+[ROOT/'snake_control/front_local_rolling.py']
    for source in sources:ast.parse(source.read_text(encoding='utf8'))
    dest=BASE/'final_sources'
    for source in sources:
        target=dest/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    links=re.findall(r'\]\(([^)]+)\)',report);missing=[]
    for link in links:
        if not (BASE/link).resolve().exists():missing.append(link)
    if missing:raise RuntimeError('Missing report evidence: '+str(missing))
    write(BASE/'delivery_check.json',dict(physics_steps_added=0,historical17_rows_unchanged=True,total_suffix_equals_local_ledger=True,
        new_dynamics=2,total_dynamics=19,remaining_unused_trials=2,no_additional_runs_authorized=True,
        local_roll_hold_pass=True,forward_roll_grasp_pass=False,old_V3_path='NOT_READY',robustness_not_tested=True,
        normal_speed_replays=['ROLL1_PRIMARY/replay_1x.mp4','ROLL2_PRIMARY/replay_1x.mp4'],
        report_links_checked=len(links),missing_links=missing,necessary_source_syntax_checked=True,no_SHA=True))
    print(json.dumps(dict(new_dynamics=2,total_dynamics=19,local_roll_hold=True,forward=False,links=len(links),status=registry['status'])),flush=True)


if __name__=='__main__':main()
