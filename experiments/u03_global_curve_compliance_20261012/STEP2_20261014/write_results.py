# -*- coding: utf-8 -*-
"""Assemble the final report from checked saved data; no physics."""
from pathlib import Path
import json
base=Path(__file__).resolve().parent;c=json.loads((base/'COMPARISON.json').read_text(encoding='utf-8'))
c['results']=[json.loads((base/r['case']/'metrics.json').read_text(encoding='utf-8')) for r in c['results']]
c['independent_review']='../SUPERVISION/STEP2_R2_independent_outcome_20261015.json'
c['independent_all_four_actions_passed']=True
c['new_pipe_scenes_authorized_for_future_preregistered_validation']=True
c['new_scenes_added_in_current_batch']=False
(base/'COMPARISON.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
rows=c['results'];by_name={r['case']:r for r in rows}
def link(label,path):return f'[{label}](<{path.resolve().as_posix()}>)'
def time(value):return '—' if value is None else f'{value:.3f}'
def change(value):return '未能比较' if value is None else f'{value:+.3f}'
passed={key:sum(r['action_success'] for r in rows if r['write_global']==key) for key in [False,True]}
text=f'''# STEP2 R2：接触修正全局参考的四次开发对照

2026-10-14开始，2026-10-15完成本轮材料。最终四次对照已执行并保存；更新G版动作通过{passed[True]}/2，冻结G版通过{passed[False]}/2。两版均使用相同粗略先验、无标签接触传感、估计器、小幅试探、局部顺应、推进规则、完整初态和物理条件，只有G写回权限不同。本报告为执行端核查，监督独立结论另列。

本轮目标是给定接触初态后的参考定型。初态已有部分包绕，不是未接触前自主发现，也不是从旧U03初态连续完成的未知几何端到端流程。步骤1调速未触发、停送恢复/协同未验证和源侧成本问题仍保留。本轮四次与报告完成后暂停，未启动主动搜索、初态自主形成、实机或去真值。

## 版本、初态和冻结

监督纠正到达时四次R1已启动。它们在48.702–54.902秒经requested_stop正常保存；没有按成功或失败筛选。R2修正了非源接触混入接收载荷，以及七次提议全异常时拒绝记录未初始化的问题。四次旧运行连同轨迹、视频、状态、日志和源码全部保留。总动力学启动数是8（4次中止旧版本、4次最终R2），不能称为只有4次启动。先验、初态、估计/更新数值增益、频率、物理、伺服和原验收没有根据旧轨迹调参。

给定初态从N_SHAPE的90秒保存qpos/qvel/ctrl构造，其他状态从新MjData初始化，再保存完整mjSTATE_INTEGRATION。它不是旧历史90秒的完整检查点。配对初态文件完全相同；该瞬间真实源侧直接接触力为0，目标/接头承载约74N，不能描述为源侧始终承载。原后部命令和有限支撑规则保留，后续源接触另按实际记录。

起始S为14.158890201993934。统一新运行235秒，固定0.012Hz使参考到旧约230.7秒相位，避免给定90秒构型后再用0.02Hz推进到旧325秒并越过原有限管末端。保持20关节/21杆、原模型、0.001秒物理步、伺服及力矩限幅、20°/秒控制限速和88°控制限幅。两版都用同一较慢频率；不与旧0.02Hz直接归因比较，也不声称填补步骤1调速贡献缺口。

方向例先验轴偏4°；位置/半径例入口X+6毫米、Z+4毫米、半径+2毫米。两例初态均有可触及的新环境候选接触。最终冻结为R2_READY_FROZEN，登记和输入核查在启动前完成。准确几何的四个保存S检查中，新G材料节点与旧链完全相同、关节命令最大差约1.6e−14rad；这只是原映射复用核查。

'''
text+='依据：'+', '.join([link('设计',base/'DESIGN.md'),link('登记',base/'REGISTERED.json'),link('冻结',base/'R2_READY_FROZEN/FREEZE.json'),link('派发核查',base/'DISPATCH.json'),link('监督纠正检查',base/'R2_CORRECTION_CHECKS.json'),link('R1排除与完整保留',base/'R1_SUPERSEDED_STARTS/README.md')])+'.\n\n'
text+='''## 控制输入和几何可辨识范围

控制只读取机体位姿/关节/受载真值及无环境标签的接触点、法向、法向力与signed gap。真实目标轴、半径、有限边界和环境目标/弯头标签只用于独立评价与原安全停止，没有回流到估计、曲线、局部反馈或材料观察。源管几何已知。旧目标生成构造器不在运行链里；目标曲线、接收径向方向、接收门和材料轴向投影均由Geometry输入定义。

接触分为“已知源一致”“接收圆柱候选一致”“桥/不明”三组。非源不等于接收：接触点相对当前候选的表面距离、轴向位置、法向轴向分量与径向法向一致性共同判断。不明载荷单独保存，不进入接收反馈；图中绿色等都是估计角色，不能当作场景真值标签。材料接管另用真实环境独立验收。

估计每秒使用最近12秒，要求不同点/法向样本、至少两个材料、空间跨度与足够法向多样性，并要求两独立曲率方向均已发出。圆柱轴/横向中心/半径候选由接触点和法向联合拟合，接头不一致观测保留为不明或低置信证据。窗口和边界预登记，信息不足时不写G。轴向入口和有限长度不能从圆柱侧面唯一识别；入口沿轴仍用先验，保留40毫米不确定性，规划范围不是已估计的真实管长。数值置信和不确定度是残差/条件数诊断，不是校准后的统计区间。

首次新接触后，材料0–8的两项Bishop曲率依次作14秒有符号正弦试探，幅值0.18/米。所有试探与回退都经曲线积分和真实伺服执行。正常滚动/推进也提供信息；没有去试探消融，不能把估计收敛唯一归因于小试探。OBSERVATION_PREPARATION是旧准确几何轨迹回读，probe时序只是接口标记，不能当本轮真实试探或在线更新收益。

## 四次动作判定

动作要求全21杆进入原有限目标直段，源清空，之后至少50秒、每杆真实轴向前进40毫米且同向自轴旋转90°，并满足原轴向、关节、力矩、地面/自碰与停止要求。下面位移/滚转从各自最终持续目标区间起算；不直接用这些不同区间的累计量比较速度。

|版本|动作|整身稳定开始(s)|后段(s)|每杆最小轴向(mm)|每杆最小自轴滚转(°)|源/目标最大侵入(mm)|累计饱和(关节·s)|
|---|---|---:|---:|---:|---:|---:|---:|
'''
for r in rows:
    d=r['source_target_self_floor_max_depth_mm'];text+=f"|{r['case']}|{'通过' if r['action_success'] else r['failure_classification']}|{time(r['full_stable_time_s'])}|{r['final_full_target_s']:.1f}|{r['minimum_each_post_axial_mm']:.2f}|{r['minimum_each_abs_post_spin_deg']:.2f}|{d[0]:.3f}/{d[1]:.3f}|{r['torque_saturated_joint_time_s']:.1f}|\n"
text+='\n实际停止原因：'+', '.join(f"{r['case']}={r['stop']}，{r['sim_time_s']:.3f}秒" for r in rows)+'.\n\n'
text+='''## 接触、候选、G和实际运动的时间顺序

|版本|首次合格接触(s)|首个可靠候选(s)|首次G写回(s)|写回次数|最终估计轴误差(°)|最终估计轴线横向误差(mm)|最终估计半径误差(mm)|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for r in rows:text+=f"|{r['case']}|{time(r['first_contact_t'])}|{time(r['first_reliable_geometry_t'])}|{time(r['first_G_write_t'])}|{r['G_write_count']}|{r['final_estimated_axis_error_deg']:.4f}|{r['final_estimated_centerline_error_mm']:.4f}|{r['final_estimated_radius_error_mm']:.4f}|\n"
text+='''
|配对|首次保存命令分离(s)|首次保存真实状态分离(s)|写回前逐值一致|同S推进与试探参数|
|---|---:|---:|---|---|
'''
for p in c['pair_checks']:text+=f"|{p['frozen']}/{p['updated']}|{time(p['first_saved_raw_command_difference_t'])}|{time(p['first_saved_actual_state_difference_t'])}|{p['pre_write_all_exact']}|S差={p['entire_final_S_max_difference']:.3g}，probe差={p['probe_parameter_max_difference']:.3g}|\n"
text+='\n更新版接触→可靠候选→有依据写回的顺序核查：'+', '.join(f"{r['case']}={r['causal_contact_geometry_write_order_passed']}" for r in rows if r['write_global'])+'. 写回前qpos/qvel/ctrl/raw_command/centers/S/probe及估计量均直接比较；写回后真实逐杆运动与接管时刻另见每次material_events/material_paths。初态已有部分包绕，这里评价的是后续材料的包绕与接管，不把给定初态接触当自主发现。\n\n'
text+='''## 写回G的效果与代价

|版本|最终G轴误差(°)|最终G轴线横向误差(mm)|最终G半径误差(mm)|实际身体轴与最终G夹角(°)|实际拟合半径/规划半径(mm)|实际螺距/规划螺距(mm)|
|---|---:|---:|---:|---:|---:|---:|
'''
for r in rows:text+=f"|{r['case']}|{r['final_G_axis_error_deg']:.4f}|{r['final_G_centerline_error_mm']:.4f}|{r['final_G_radius_error_mm']:.4f}|{r['end_actual_vs_final_G_axis_deg']:.3f}|{r['end_body_fit_radius_mm']:.2f}/{r['planned_helix_radius_mm']:.2f}|{r['end_body_fit_pitch_mm']:.2f}/{r['planned_pitch_mm']:.2f}|\n"
text+='\n配对差值按“更新−冻结”列出，完成时间负值表示提前，侵入或饱和正值表示成本增加：\n\n|工况|完成时间变化(s)|源侵入变化(mm)|目标侵入变化(mm)|饱和变化(关节·s)|\n|---|---:|---:|---:|---:|\n'
for p in c['pair_checks']:text+=f"|{p['updated']}|{change(p['completion_change_s'])}|{change(p['source_peak_change_mm'])}|{change(p['target_peak_change_mm'])}|{change(p['saturated_joint_time_change_s'])}|\n"
text+='\n公共150–234.9秒窗口的实际轴向输送：\n\n|版本|每杆最小(mm)|21杆平均(mm)|\n|---|---:|---:|\n'
for r in c['common_window']:text+=f"|{r['case']}|{r['minimum_axial_mm']:.3f}|{r['mean_axial_mm']:.3f}|\n"
text+='''
**已确认事实：** 两版运行同一估计器，冻结版即使得到可靠候选也不改G；最终G参数保持原先验。更新版将独立接触估计渐进写回，改变原曲率命令，随后产生真实状态差异。模型、完整初态及各次冻结文件直接核对，仿真环境保持MuJoCo3.2.7/NumPy1.23.5/SciPy1.13.1及单线程。所有改变均在预登记的同S命令与材料节点边界内，原伺服限速继续生效。

**解释边界：** 参考参数更接近环境不等于每项物理代价都改善，也不等于冻结版必然失败。动作通过数、完成时刻、持续输送、侵入和饱和要分别读。只有两个确定性开发条件，不是普遍鲁棒性或统计成功率。准确几何旧结果只是能力参照，不能替代此处同输入权限的配对归因。

**尚未验证：** 入口沿轴/有限长度的识别、接触前自主发现、初态形成、试探独立贡献、未知多管网、噪声/延迟及实机能力。本轮没有验证步骤1的调速、停送恢复或协同收益。源出口局部模态的独立贡献也没有新增消融。

全局参考不是从真实身体逐帧重建。原自由根、曲率积分执行链依靠接触约束，不能把几何参数收敛当严格世界曲线跟踪。每次另存同编号G/C世界误差、材料邻域限制的几何距离和实际螺距/半径/轴向诊断；最近距离较小也不能替代所有对应误差。

## 可复核交付

每次contact_evidence.jsonl保存全部无标签接触及估计分类，geometry_hypotheses.jsonl保存不同候选与置信/不确定性，global_updates.jsonl保存每次提议/写回的依据、同S前后命令与拒绝原因。trace.npz含实际qpos/qvel、命令、当前G/C、估计几何、估计角色、材料进展与局部反馈；完整末状态及controller checkpoint同时保存。

视频来自单次连续真实动力学，10 fps、1倍速、完整预算或实际停止时长；当前G青、局部C品红、估计轴黄、估计接收角色绿、桥/不明紫，原场景和真实机体均显示。冻结G指参数不写回，材料窗口仍随共同S前进。

|版本|1倍速连续双视角|接触/几何/动作事件图|当前G/C/真实身体同帧图|逐杆轨迹|逐杆真实接管事件|
|---|---|---|---|---|---|
'''
for r in rows:
    p=base/r['case'];text+=f"|{r['case']}|{link('视频',p/'actual_side_oblique_1x.mp4')}|{link('事件链',p/'contact_geometry_action.png')}|{link('同帧图',p/'current_G_C_actual.png')}|{link('material_paths',p/'material_paths.csv')}|{link('material_events',p/'material_events.csv')}|\n"
text+='\n'+', '.join([link('完整数值与配对核查',base/'COMPARISON.json'),link('结果表CSV',base/'COMPARISON.csv'),link('方向配对图',base/'T_UPDATE_paired_effect.png'),link('位置/半径配对图',base/'P_UPDATE_paired_effect.png'),link('输入隔离记录',base/'INPUT_AUDIT.json')])+'.\n\n'
text+='截至执行端交付，独立监督结论由SUPERVISION中相应STEP2文件补充；本报告不将执行端通过写成监督已验收。已停止新增试验，等待独立核查，后续主动搜索未授权。\n'
text=text.replace('本报告为执行端核查，监督独立结论另列。','监督已独立确认四次动作均通过。几何定型成立，但动作收益不一致，位置/半径组有明确退化。')
text=text.replace('所有改变均在预登记的同S命令与材料节点边界内','所有G写回均在预登记的同S命令与材料节点边界内')
text=text.replace('截至执行端交付，独立监督结论由SUPERVISION中相应STEP2文件补充；本报告不将执行端通过写成监督已验收。已停止新增试验，等待独立核查，后续主动搜索未授权。',
    '监督已独立复核四条保存轨迹、模型/初态/冻结、材料输送与自轴滚转，以及写回前后配对一致性；报告本身等待最终审阅。已停止新增试验，后续主动搜索未授权。独立依据：'+link('STEP2 R2独立验收',base.parent/'SUPERVISION/STEP2_R2_independent_outcome_20261015.json')+'。')
extra='''## 退化定位与参考对应的剩余问题

方向组写回使完成提前0.4秒、源侵入减少0.752毫米，但目标侵入增加0.385毫米、饱和增加2.9关节·秒。位置/半径组写回使完成晚9.7秒，源侵入增加0.058毫米、目标侵入增加1.218毫米、饱和增加63.6关节·秒。这是明确退化；公共后段平均输送增加6.590毫米不能抵消这些代价，也不支持整体更优。

位置/半径更新组在30秒，M12对真实目标直管达到3.453毫米峰值；这时试探已结束、源γ为零，接收局部δ约0.942毫米。G尚处在未完成修正的阶段，但该点最新拟合不合格，写回已经暂停，继续执行最近有效G；29.275秒判为不一致，30.275秒恢复合格。不能把峰值说成某一次即时写回直接造成。大于3毫米的保存点连续区间仅0.1秒，未达到原0.3秒停止条件。方向组源峰值同在39.401秒M20/源直管端部，更新版2.065毫米，冻结版2.817毫米。没有放宽停止条件或删除高代价片段。

已确认的因果分离是“G写回权限改变→命令/实际运动改变”。合理推测是几何更新、材料采样/标架相位与现有局部反馈之间的执行耦合引起瞬时压入和接管延后；仅凭这四次不能确定唯一主因，也不能把退化归咎于估计没有收敛。几何正确与动作代价低是两项不同的验证。

|版本|末帧同索引G世界RMS(mm)|同索引C世界RMS(mm)|材料邻域限制G距离RMS(mm)|
|---|---:|---:|---:|
'''
for r in rows:extra+=f"|{r['case']}|{r['last_same_index_world_G_rms_mm']:.3f}|{r['last_same_index_world_C_rms_mm']:.3f}|{r['last_order_restricted_G_distance_rms_mm']:.3f}|\n"
extra+='''
两个更新版仍有约201/220毫米同索引G世界偏差，不能称为严格世界曲线跟踪。没有每帧移动或旋转G去配合真实身体，也没有用无限制最近点替代同编号偏差。

补充材料标架诊断按原机器人关节轴约定，去除固定π翻转后，将实际和C的同材料切向偏差与法向扭转差分开。法向扭转差用最小切向对齐做姿态分解；世界G/C的位置未重新配准。该诊断不等于实际自轴转数或共同χ的误差，较大的切向/扭转差反映对应与标架跟踪尚不成立，不新增或放宽动作验收。

|版本|末帧同材料切向角RMS(°)|分解后的包裹扭转差RMS(°)|
|---|---:|---:|
'''
for r in json.loads((base/'MATERIAL_FRAME_DIAGNOSTICS.json').read_text()):extra+=f"|{r['case']}|{r['last']['tangent_angle_rms_deg']:.3f}|{r['last']['wrapped_material_twist_rms_deg']:.3f}|\n"
extra+='''
“入口”字段是规划螺旋接收站位/轴线点，不是已识别的物理管端。本轮站位约在真实直段起点后50毫米，沿轴位置仍用先验；登记40毫米仅为先验标称不确定度，不是包含真端点的保证界限。目标有限长度也未识别。旧字段actual_target_support_count只是估计接收候选计数，不能作真值杆数；字段语义单独存档。

'''
extra+='补充证据：'+', '.join([link('接触代价/站位定位',base/'COST_AND_STATION_DIAGNOSTICS.json'),link('标架诊断CSV',base/'MATERIAL_FRAME_DIAGNOSTICS.csv'),link('字段语义',base/'TRACE_FIELDS.md')])+'.\n\n'
extra+='''## 新管道授权与真正需要跨场景的问题

用户已授权在验证全局＋局部算法确有需要时，自主搭建新管道场景；可变夹角、目标轴位置/方向、管径或连接形状，保留20关节机器人并先登记场景、完整初态与配对。当前原L管四次结果及全部R1中止记录保留，本批不追加新场景或新运行。

需要跨物理场景验证的是：固定源—目标连接组织是否适用于其他夹角/位置，表面半径到离散执行螺旋的39毫米余量是否随管径失效，以及圆柱候选能否在不同接头/弯面中正确保留不确定性。这些不能只靠同一L管上改变先验证明。

位置/半径组的接管退化、材料/标架对应和200毫米量级世界偏差已经在本场景出现，首先可以用当前保存数据定位；扩大场景不能替代解释这些现有问题。冻结版也全部完成，可能有管道物理约束提供较大捕获范围的作用，但本轮没有独立分离该机制。后续新场景可以检验这种解释及方法泛化；主动搜索、初态自主形成和实机仍依原阶段顺序，不自动启动。

'''
text=text.replace('## 可复核交付',extra+'## 可复核交付')
text=text.replace('实际拟合半径/规划半径(mm)','实际杆心拟合半径/规划曲线点半径(mm)')
text=text.replace('配对差值按“更新−冻结”列出','半径的两列计算对象分别是实际杆心与规划曲线点，包含弦长采样差，不能把其差值全部当作IK误差。\n\n配对差值按“更新−冻结”列出')
text+='\n每次无标签接触点/法向与候选空间图：'+', '.join(link(r['case'],base/r['case']/'contact_points_normals_hypothesis.png') for r in rows)+'.\n'
(base/'RESULTS.md').write_text(text,encoding='utf-8')
status=dict(stage='STEP2',state='R2_execution_report_complete_awaiting_independent_review',paused=True,final_comparison_count=4,total_dynamics_starts=8,superseded_starts=4,
    completed=[r['case'] for r in rows],running=[],pending=[],action_passed=passed[True]+passed[False],updated_action_passed=passed[True],frozen_action_passed=passed[False],
    execution_complete=True,report_complete=True,independent_review_received=False,step2_complete=False,next_stage_authorized=False,freeze='R2_READY_FROZEN',summary='RESULTS.md',comparison='COMPARISON.json',
    step1_speed_limitations_resolved=False,axial_entry_identified=False,precontact_search=False)
status.update(state='R2_execution_report_complete_awaiting_report_review',independent_review_received=True,independent_all4_action_verified=True,independent_review='../SUPERVISION/STEP2_R2_independent_outcome_20261015.json',new_pipe_scenes_authorized=True,new_scenes_added_in_batch=False,geometric_settling_supported=True,action_benefit_consistent=False,position_radius_action_degradation=True,world_curve_exact_tracking_validated=False)
(base/'CURRENT_STATUS.json').write_text(json.dumps(status,indent=2))
root_path=base.parent/'CURRENT_STATUS.json';root=json.loads(root_path.read_text(encoding='utf-8'));root.update(stage='STEP2',state=status['state'],paused=True,pause_reason='STEP2 corrected four-case execution/report complete; wait for independent supervision',
    running_cases=[],pending_registered_tests=[],new_runs_queued=False,current_batch_complete=True,step2_execution_complete=True,step2_report_complete=True,step2_complete=False,step2_independent_review_received=False,
    step2_action_passed=passed[True]+passed[False],step2_updated_action_passed=passed[True],step2_frozen_action_passed=passed[False],step2_total_dynamics_starts=8,step2_superseded_starts=4,
    step2_summary='STEP2_20261014/RESULTS.md',step2_freeze='STEP2_20261014/R2_READY_FROZEN',step2_current_version='STEP2_R2_SUPERVISION_CORRECTED',step3_authorized=False,
    next='Independent review of corrected STEP2 four cases; no precontact active search or further simulations',latest_summary=f'R2 four cases complete: update{passed[True]}/2, frozen{passed[False]}/2 action passes; geometry, causal command/state changes and mixed costs in STEP2 report;8 total starts with4 interrupted R1 retained',
    supervisor_pending_corrections=[],step2_corrections_executed=['Receiver consistent/bridge unknown force and role separation','All7 proposal exceptions restore original valid G/command'],step2_correction_evidence='STEP2_20261014/R2_CORRECTION_CHECKS.json',
    full_validation_matrix_complete=False,stage_B_complete=False,active_search_in_scope=False,initial_pose_formation_in_scope=False,hardware_porting_in_scope=False,truth_reduction_in_scope=False)
root.update(state=status['state'],pause_reason='STEP2 four corrected cases and report complete; independent numeric action review passed; final report review pending',step2_independent_review_received=True,step2_independent_numeric_review_complete=True,step2_independent_review_source='SUPERVISION/STEP2_R2_independent_outcome_20261015.json',step2_geometric_settling_supported=True,step2_action_benefit_consistent=False,step2_position_radius_action_degradation=True,new_pipe_scenes_authorized=True,new_pipe_scenes_scope='Future preregistered paired algorithm validation after current stage report; keep20-joint robot and originalL data; no addition to current batch',next='Final supervision review of STEP2 report; no added runs in this batch or automatic active-search/initial-formation/hardware stage')
root.update(nominal_S_speed_unchanged=False,step1_nominal_S_speed_unchanged=True,step2_frequency_hz=.012,nominal_S_speed_scope='STEP2 paired fixed0.012Hz from given contact initial; historical STEP1 nominal0.02Hz unchanged',baseline_nominal_reused=False,step1_baseline_nominal_reused=True,step2_formal_baseline_reused=False,step2_development_not_heldout=True,new_pipe_scene_authorization_date='2026-10-15')
for path in ['STEP2_20261014/RESULTS.md','STEP2_20261014/COMPARISON.json','STEP2_20261014/R2_CORRECTION_CHECKS.json','SUPERVISION/STEP2_R2_independent_outcome_20261015.json']:
    if path not in root['evidence']:root['evidence'].append(path)
root_path.write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Final report written; all final results and4 superseded starts accounted for; paused for independent review.')
