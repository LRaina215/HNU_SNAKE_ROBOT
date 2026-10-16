# -*- coding: utf-8 -*-
"""Build the report and completion status from saved evidence (stdlib only)."""
from pathlib import Path
import json

base=Path(__file__).resolve().parent
comparison=json.loads((base/'COMPARISON.json').read_text(encoding='utf-8'))
review=json.loads((base/'FINAL_REVIEW.json').read_text(encoding='utf-8'))
rows=comparison['results']; by_name={r['case']:r for r in rows}
def link(label, path): return f'[{label}](<{path.resolve().as_posix()}>)'
def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

text='''# STEP1 R1 首轮结果与验收边界

2026-10-14。已完成登记的12条正式结果（11次新动力学运行、1条复用A16），并完成执行端核查。三种工况四版均通过整身跨管与后续持续滚动。所有调速比例始终为1，故本轮没有验证调速干预、停送恢复或调速与调形的协同收益。调形减少目标侧侵入与累计力矩饱和时间，但仍增加源侧侵入。首轮执行和报告完成；反馈效果验证尚不完整，等待用户验收，不进入STEP2。

## 范围、登记与冻结

给定原U03初态，准确已知有限L管，允许仿真真值辅助。保留原20关节、21杆、接触参数、伺服、0.001秒物理步、235秒运行时间及原停止判据。判定要求整身进入目标有限直段后持续至少50秒、每杆沿目标轴前进至少40毫米，并核对每杆同向自轴滚转至少90°。本轮没有重新运行A系列或旧轻微脉冲，没有初态形成、主动搜索、未知管道估计、实机或去真值实验。

OPEN关闭高层反馈；SPEED仅开启共同进度反馈；SHAPE仅开启曲线调形；JOINT同时开启二者。NOM无外载，RECEIVE_LAG在96–102秒向head施加世界−Y方向8N，SOURCE_DRAG在108–114秒向tail施加世界−X方向6N。载荷按物理时间作用于指定body质心，每次开发试验均6000个物理步，冲量分别为[0,−48,0]、[−36,0,0] N·s（浮点舍入忽略）。工况名称表达登记意图；实际载荷没有制造足以触发调速的持续滞后。

最终运行冻结是R1_READY_FROZEN。未启动的R1_FROZEN为早期草案，不是本轮运行依据。正式运行前修正了预测/观测有效区间不一致和不存在的runner复制项；随后三次新增名义运行通过才派发八次开发运行。结果出现后没有改控制器、提高载荷或补扫参数。运行环境统一为MuJoCo3.2.7、NumPy1.23.5、SciPy1.13.1，OPENBLAS/MKL/OMP线程各1。新运行冻结文件、模型及初态均直接核对一致，未做SHA校验。

'''
text += '设计与冻结依据：'+', '.join([link('DESIGN',base/'DESIGN.md'),link('REGISTERED',base/'REGISTERED.json'),link('最终冻结登记',base/'R1_READY_FROZEN/REGISTERED.json'),link('开发派发核查',base/'R1_READY_FROZEN/development_dispatch.json'),link('名义启动门槛',base/'nominal_gate.json')])+'.\n\n'
text += '''## 观察器与控制接口

材料编号始终固定。实际接触支撑角色与几何进度图分开，包含仍在源端承载的M20；目标杆自身周期性失去接触时，只要仍在已确认目标图中，继续累计真实轴向位移。源图累计ΔX/a_source，目标图累计ΔY/a_target；无有效图的自由过渡段保留未知量，不补名义位移、不当作零速度。首次目标预测锚点不重置累计材料历史。预测F和实际进展只在同一有效时间掩码上比较，缺测预测量另存。

共同S驱动F、M、滚转及名义角色；调形按物理时间更新，代码接口允许S暂停时继续调形。常规减速要求接收滞后且低进展，或源端高载荷、低进展且跟踪受阻；高力矩单独不触发。所有运行实际都没有减速或暂停，因此“暂停时调形与恢复”仅有实现，尚无本轮动力学证据。

接收径向调形仍为±2毫米、速率±0.25毫米/秒。新增源出口γ曲率模态只作用于材料节点15以后，真实最深源接触法向及原曲率映射Jacobian选择向外方向，再走原交替关节积分；首14个关节积分保持。γ幅值上限0.4/米，标量速率上限0.08/(米·秒)。方向随时间改变，γ乘方向的系数向量速率不等同于γ标量速率，已单独记录。重建局部C保持单位弧长；离散弦长最大误差0.1733毫米，物理杆长未变。

私有固定姿态FK的向外扰动支持方向选择，但不能预测接触后的动力学收益。早期key_state_source_direction.json仅是被后续方案替代的探索记录；正式准备与解释依据为PREPARATION.json。观察器恢复了有效源/目标图中的连续材料历史，并没有恢复自由过渡段全部实际行程，也没有证明实际身体严格跟踪世界曲线。

## 十二条正式结果

下表“后段”从各自整身稳定转移时刻至234.9秒；因此不能用不同后段的累计Y直接比较运输速度。侵入是仿真接触几何深度诊断，不是硬件损伤量。饱和为0.1秒保存点估计的全部关节累计时间。

|工况/版本|整身转移(s)|后段(s)|每杆最小Y(mm)|每杆最小滚转(°)|饱和(关节·s)|源/目标最大侵入(mm)|动作判定|
|---|---:|---:|---:|---:|---:|---:|---|
'''
for r in rows:
    d=r['max_depth_mm']
    text+=f"|{r['case']}|{r['full_stable_time_s']:.1f}|{r['final_full_target_s']:.1f}|{r['minimum_each_module_post_Y_mm']:.2f}|{r['post_minimum_abs_rod_spin_deg']:.2f}|{r['sampled_torque_saturated_joint_time_s']:.1f}|{d[0]:.3f}/{d[1]:.3f}|通过|\n"
text+='''
所有运行至235秒结束，无地面/自碰撞侵入。各版都超过原50秒/40毫米及同向90°滚转要求。N_OPEN_REUSE的动力学、模型和判定来自既有A16，新增视频只是保存状态的当前参考叠加渲染；不计为新运行。

## 公共150–234.9秒窗口

所有版本在150秒前已完成整身转移。下表给出相同84.9秒中的21杆最小/平均轴向位移及最小自轴滚转；SPEED与OPEN、JOINT与SHAPE各自完全一致，合并显示。

|工况/版本|最小Y(mm)|平均Y(mm)|最小滚转(°)|
|---|---:|---:|---:|
'''
for r in review['results']:
    if r['case'] in ['N_OPEN_REUSE','N_SHAPE','L_OPEN','L_SHAPE','D_OPEN','D_SHAPE']:
        text+=f"|{r['case']}|{r['common_min_Y_mm']:.3f}|{r['common_mean_Y_mm']:.3f}|{r['common_min_abs_spin_deg']:.3f}|\n"
text+='''
名义、接收载荷、源拖拽中，SHAPE相对OPEN的平均Y只增加约0.201、0.440、0.412毫米，属于本轮小差异，不支持明显提高持续运输速度的结论。每杆滚转由保存的真实qpos/qvel和本杆方向计算，不把控制器滚转相位当作实际滚转。

## 贡献与未验证部分

**已确认事实：** 三工况SPEED与OPEN的qpos、qvel、ctrl、raw_command、S、chi、centers全程逐值相同；JOINT与SHAPE也全程相同。所有调速比例最小值与均值均为1，名义S损失为0，调速原因代码为0。最大匹配接收滞后：OPEN/SHAPE名义12.68/19.50毫米、接收载荷19.98/32.23毫米、源拖拽13.53/22.07毫米，均低于40毫米死区；源端联合门槛也未请求减速。SOURCE_DRAG的OPEN最大缺测预测17.44毫米另存，不混入实际滞后。

据此，本轮不能判断调速是否有效，不能宣称联合控制具有协同收益，也不能把本轮相同轨迹外推成“调速一般无用”。预登记固定慢速对照仅在收益主要来自减速时启动；此次从未减速，固定比例1会复制OPEN，因此未增加固定慢速对照。没有固定调形或去源模态消融，不能归因于实时适应的唯一贡献或新增源模态的独立贡献。

**调形的收益与成本：** 名义SHAPE提前0.5秒完成，累计饱和少20.9关节·秒，源侵入增加0.716毫米，目标侵入减少0.762毫米。接收载荷工况提前0.3秒、饱和少17.0关节·秒，源侵入增加0.343毫米、目标减少0.723毫米。源拖拽工况提前0.2秒、饱和少18.5关节·秒，源侵入增加0.256毫米、目标减少0.346毫米。最长单关节连续饱和22.899秒，较OPEN的22.699秒没有改善。动作均通过，不能据此认定所有接触代价改善。

新增源模态γ最大值仅约0.0566–0.0703/米。名义源最大侵入从旧接收调形B1_NOM_V2的1.90420毫米降为1.88689毫米，仅少0.01732毫米，仍明显高于A16的1.17090毫米。这是观察到的小变化；没有独立消融，不能认定新增源模态已解决源出口问题。

**合理推测：** 当前载荷方向和时段确实影响了完成时刻与源接触，但强度/持续机制没有形成调速门槛要求的持续输送受阻。源尾耦合仍可能使接收调形收益转移为源端成本，现有结果与该解释相容。它们都不是已经验证的因果结论。本轮不通过事后提高载荷制造调速效果。

**验收判断：** 给定初态与真值辅助下的跨管动作已复核成功；观察器和分离的控制接口已完成。首轮尚不能完成“调速干预有效、停送恢复有效、联合协调有额外收益”的验收。更严格曲线跟踪、源模态独立收益、普遍鲁棒性和完整研究方案均未完成。

## 可复核材料与连续视频

每次新运行保存trace.npz（真实状态、命令、曲线、角色、进度与反馈）、material_paths.csv、material_observer.csv、final_state/integration_state和控制器checkpoint。当前G窗口为青色，当前局部C为品红；实际源支撑橙、目标直接支撑绿、已确认目标包络无自身直接支撑浅绿、桥/双侧支撑另色。角色来自接触和几何诊断，不能把名义窗口当实际支撑。

所有视频2350帧、10 fps、235秒，首/中/末帧解码通过，连续双视角1倍速。执行端看过三工况110/230秒叠加蒙太奇；八次开发运行在施载前与各自已获监督独立复核的名义轨迹逐值相同，保留下侧获取路径。动作、沿轴推进和同向滚转另按真实数据判定。旧global_local_actual.png为旧分析图，当前参考解释使用global_local_actual_current.png及视频。

|正式结果|视频|当前曲线/实际身体|真实轨迹|
|---|---|---|---|
'''
for r in rows:
    name=r['case']; case=base/name
    physical=base.parent/'A16' if name=='N_OPEN_REUSE' else case
    plot=base/'NOMINAL_actual_overlay_montage.png' if name=='N_OPEN_REUSE' else case/'global_local_actual_current.png'
    text+=f"|{name}|{link('1倍速双视角',case/'actual_side_oblique_1x.mp4')}|{link('当前G/C/实际',plot)}|{link('trace',physical/'trace.npz')}|\n"
text+='\n汇总与核查：'+', '.join([link('COMPARISON.json',base/'COMPARISON.json'),link('COMPARISON.csv',base/'COMPARISON.csv'),link('FINAL_REVIEW',base/'FINAL_REVIEW.json'),link('准备检查',base/'PREPARATION.json'),link('名义独立复核',base.parent/'SUPERVISION/STEP1_NOMINAL_independent_outcome_20261014.json')])+'.\n\n'
text+='''监督会话已独立复核三次新增名义运行；截至本报告生成，尚未发现本轮八次开发运行的独立监督结论文件，开发结论目前为执行端保存数据核查。旧B_HELDOUT独立结论只属于旧四次轻微脉冲，不能替代本轮复核。

本轮没有待运行试验。已停在用户验收点，STEP2未授权、未实施；若用户要求继续解决上述未验证部分，应先明确下一轮工况与验收目标，再登记新轮，不把当前首轮结果改写为完整闭环成功。
'''
(base/'RESULTS.md').write_text(text,encoding='utf-8')
status=dict(stage='STEP1',state='first_round_report_complete_awaiting_user_acceptance',paused=True,
    formal_results_in_round=12,new_dynamics_runs=11,reused_formal_results=['A16'],completed_cases=[r['case'] for r in rows],running=[],
    nominal_passed=True,all12_action_passed=True,first_round_execution_complete=True,report_complete=True,
    speed_implemented=True,speed_triggered=False,pause_recovery_dynamics_validated=False,joint_extra_benefit_validated=False,
    feedback_effect_validation_complete=False,step1_user_accepted=False,step2_authorized=False,
    fixed_slow_required=False,fixed_slow_reason=comparison['conditional_controls']['reason'],source_shape_benefit_isolated=False,
    summary='All12 action passes; all speed ratios1; shape has small benefits and higher source intrusion; full feedback validation incomplete',
    report='RESULTS.md',comparison='COMPARISON.json',final_review='FINAL_REVIEW.json',freeze='R1_READY_FROZEN',pending_registered_tests=[])
write_json(base/'CURRENT_STATUS.json',status)
root_path=base.parent/'CURRENT_STATUS.json'; root=json.loads(root_path.read_text(encoding='utf-8'))
root.update(state=status['state'],paused=True,pause_reason='STEP1 first-round report complete; explicit user acceptance required before STEP2',
    running_cases=[],new_runs_queued=False,pending_registered_tests=[],current_batch_complete=True,
    stage_B_complete=False,stage_B_execution_complete=False,step1_complete=False,
    step1_first_round_execution_complete=True,step1_first_round_report_complete=True,
    step1_feedback_effect_validation_complete=False,step1_action_validation_passed=True,
    step1_user_accepted=False,step2_authorized=False,step1_speed_triggered=False,
    latest_candidate='STEP1_R1_READY_FROZEN',latest_summary=status['summary'],
    latest_nominal_verdict='all_three_new_nominals_independently_passed_speed_inactive_source_cost_remains',
    next='Wait for explicit user acceptance; do not implement STEP2 or run further experiments',
    targeted_mismatch_development_completed=True,full_validation_matrix_complete=False,
    remaining_full_plan=['User acceptance of STEP1 report','Effective shared-speed intervention and pause/recovery evidence remain absent','Source-shape independent contribution unresolved','STEP2 and later stages require user authorization'],
    step1_formal_completed=[r['case'] for r in rows],step1_summary='STEP1_20261014/RESULTS.md',
    step1_development_independent_review_received=False)
for evidence in ['STEP1_20261014/RESULTS.md','STEP1_20261014/COMPARISON.json','STEP1_20261014/FINAL_REVIEW.json']:
    if evidence not in root['evidence']:root['evidence'].append(evidence)
write_json(root_path,root)
print('Report and status written; awaiting user acceptance; no further trials.')
