"""Prepare local source load feedback without changing V3.3 or its actual case."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS';B=P/'CONTROLLER_V4_SOURCE_WRENCH_HOLD'
if (B/'REVIEW_SNAPSHOT').exists() or (B/'T_UPDATE').exists():raise RuntimeError('Refuse to overwrite frozen/started version')
B.mkdir(exist_ok=True)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
reg=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));reg.update(version='DIRECT_EXTENSION_V4_SOURCE_WRENCH_HOLD',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=22,prior_legacy_postwrap_starts_preserved=8,revision_reason='V3.3 actual source coil rolled circumferentially with fixed posterior targets; implement explicit local contact-load/preload feedback, preserve G and original initial/physics/guards.')
reg['source_hold']=dict(conservative_source_friction=1.,friction_facets=16,torque_bound_Nm=2.30,posterior_shape_stiffness_Nm_rad=25.,pose_frequency_hz=.5,maximum_corrective_force_weight_fraction=.1,maximum_corrective_moment_Nm=.2,maximum_sum_normal_weight_fraction=4.,torque_regularization_weight=.05,maximum_solver_iterations=60,invalid_timeout_s=.5,blend_alpha=.1,reference='Posterior measured joint state and known-source body positions at original source-stable release; no new given pose',input='Robot state, unlabeled known-source points/normals/forces, known robot masses and source geometry only',output='Local servo preload represented by same curvature modes after geometric C; no external plant force/root write or G backfill')
reg['local_feedback']['initial_preload_policy']='Preserve exact prescribed source hold until stable; then blend explicit source wrench feedforward and software posterior shape feedback in local curvature; no source feed/roll. Last preload fades15s after capture.'
reg['phase_coordinates']['alpha']='Front geometric release, first8 hinges; posterior local source load feedback allowed, source feed and material roll remain zero'
reg['prior_version']='DIRECT_EXTENSION_V3_3_GRAVITY_SOURCE_CLASS, one actual failed case preserved'
(B/'REGISTERED.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Keep a small idempotent metadata writer; old preparation utilities are archived.
(B/'register_direct.py').write_text('"""Frozen V4 registration is created by ../prepare_v4_source_hold.py; do not reset an active run."""\n',encoding='utf-8')
f=B/'direct_controller.py';s=f.read_text(encoding='utf-8').replace('from source_mode import reconstruct,SourceMode','from source_mode import reconstruct,SourceMode\nfrom source_wrench_hold import SourceWrenchHold')
before="        self.given_source_P=d.xpos[self.m.body('head').id:self.m.body('head').id+21].copy()"
s=s.replace(before,before+"\n        self.source_hold=SourceWrenchHold(self.planner,registration['source_hold'],registration['source_known']);self.last_hold_preload=self.initial_preload.copy()")
s=s.replace("tracking=bool(max(abs(sensor['joint_actual']-self.raw).max(),abs(self.raw-self.last_geo).max())<=np.radians(self.p['progress_tracking_gate_deg']))","tracking=bool(self.source_hold.valid and max(abs(sensor['joint_actual']-self.raw).max(),abs(self.raw-self.last_geo).max())<=np.radians(self.p['progress_tracking_gate_deg']))")
start=s.index('        preload=self.initial_preload.copy();blend=NativeMapping.smooth(self.alpha/.1)');end=s.index('        preload_field=',start)
s=s[:start]+'''        hold_info=self.source_hold.empty_info();preload=self.initial_preload.copy()
        if self.source_ready and self.post is None:
            torque,hold_info=self.source_hold.update(sensor,geometric,minimum,dt,t)
            if self.source_hold.invalid_age>=self.reg['source_hold']['invalid_timeout_s']:self.failure='source_wrench_allocation_invalid'
            blend=NativeMapping.smooth(self.alpha/self.reg['source_hold']['blend_alpha'])
            old_preload=self.initial_preload.copy();old_preload[minimum:]=self.planner.initial_ctrl[minimum:]-geometric[minimum:]
            desired_preload=torque/127.;desired_preload[minimum:]+=sensor['joint_actual'][minimum:]-geometric[minimum:]+self.reg['source_hold']['posterior_shape_stiffness_Nm_rad']/127.*(self.source_hold.reference_q[minimum:]-sensor['joint_actual'][minimum:])
            preload=(1-blend)*old_preload+blend*desired_preload;self.last_hold_preload=preload.copy()
        elif self.post is not None:
            preload=self.last_hold_preload*(1-NativeMapping.smooth((t-self.first_capture_t)/15.))
''' +s[end:]
s=s.replace('            **roles,**source_mode,source_ready=', '            **roles,**source_mode,**hold_info,source_ready=')
s=s.replace('The\n            # compensating local preload below keeps their total commands fixed.', 'The\n            # independent load-feedback layer below supplies their local preload.')
f.write_text(s,encoding='utf-8')
design='''# V4 源侧载荷反馈

V3.3 唯一实际例在15.125秒、alpha0.52时因源段相对给定初态12.009mm停止。实际源轴向位移较小，主要是约6度共同周向转动；固定后段关节目标未保持源侧世界位姿。四个已保存实际姿态的保守源侧力分配均可行，所需峰值1.32–1.55Nm；此前实际关节索引17持续2.355Nm饱和。这不足以证明唯一原因，也不证明分配能被伺服实现。

仅新增独立局部源侧载荷层：由机器人已知质量、关节和一个实测承载位姿重建私有机器人状态，使用无标签且与已知源管相符的后段接触点/法向/力。保守mu1的16面内接摩擦锥（原源管mu2不改），忽略接触力矩，求满足浮动基六维载荷平衡的接触力；关节目标跟随力矩计入2.30Nm软件约束，总法向力不超过4倍重量。以已测载荷为近邻目标，少量力矩正则。它不是向实际模型写入接触力。

源侧X位置及周向相位以2.1秒稳定参考闭环，0.5Hz临界阻尼，轴向校正力限0.1倍重量，周向校正力矩限0.2Nm。这是载荷分配目标，经关节伺服预载实现。后段稳定关节角作为软件25Nm/rad弹性参考，原物理伺服kp127不改；已保存5秒状态的约1.2度后段误差若直接按127恢复会单独要求2.62Nm，故软件形状刚度取25，使同等误差贡献约0.52Nm并与载荷分配共同受2.30Nm约束。alpha前0.1段平滑切入。输入没有真实接收几何，输出只进入独立局部曲率预载，不改G。原2mm运动、曲率/关节速率与幅值限制全数保留；原12mm源侧及其他物理停止条件不放宽。分配失效暂停alpha/beta，连续0.5秒无效停止。接管后保留最后预载并在原15秒窗口渐退，避免重置。

当前给定初态和真实物理不变。该版尚未积分；只申请一个物理t0 T_UPDATE诊断。先做少量保存状态重放、原生积分及入口连续性核对，再独立当前版审阅后运行，不等待整条包绕离线收敛。此前V3.2/V3.3全部失败保留。完整Step2、反馈效果与最终四例尚未验证。
'''
(B/'CONTROLLER_DESIGN.md').write_text(design,encoding='utf-8')
f=B/'prepare_review_snapshot.py';s=f.read_text(encoding='utf-8').replace("'FOCUSED_CORRECTION_AUDIT.json'","'SOURCE_HOLD_REPLAY_AUDIT.json'").replace("'../V3_2_ACTUAL_DIAGNOSTIC.md'","'../V3_3_ACTUAL_DIAGNOSTIC.md'").replace("'../CONTROLLER_V3_POSITIVE_POLAR/T_UPDATE/OUTCOME_SUMMARY.json'","'../CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE/OUTCOME_SUMMARY.json'").replace("'../CONTROLLER_V3_POSITIVE_POLAR/T_UPDATE/SOURCE_DRIFT_AND_GRAVITY_AUDIT.json'","'../CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS/T_UPDATE/SOURCE_WRENCH_DIAGNOSIS.json'")
f.write_text(s,encoding='utf-8')
print(str(B))
