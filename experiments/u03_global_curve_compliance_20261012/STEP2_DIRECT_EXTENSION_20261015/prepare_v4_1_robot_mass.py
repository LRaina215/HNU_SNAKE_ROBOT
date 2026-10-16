"""Retain rejected V4; correct only robot-mass factors in a separate revision."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V4_SOURCE_WRENCH_HOLD';B=P/'CONTROLLER_V4_1_ROBOT_MASS_HOLD';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
f=B/'source_wrench_hold.py';s=f.read_text(encoding='utf-8');s=s.replace('W=float(np.sum(m.body_mass)*9.81)','W=float(np.sum(m.body_mass[head:head+21])*np.linalg.norm(m.opt.gravity))').replace('mass=np.sum(m.body_mass)','mass=np.sum(m.body_mass[head:head+21])');f.write_text(s,encoding='utf-8')
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V4_1_ROBOT_MASS_HOLD',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,prior_version='V4 failed independent robot-mass check, zero dynamics; V3.3 actual failure preserved',revision_reason='Correct local source-hold robot mass/weight and COM normalization to head..head+20 only; fixed pipe_network mass excluded. No other controller or physical parameter change.')
r['source_hold']['mass_scope']='Only robot bodies head..head+20:2.856kg,28.01736N. Fixed pipe_network mass13.03496kg is excluded.'
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
design='''# V4.1 机器人质量因子修正

V4 独立审阅未通过：私有模型全体刚体质量15.890959kg包括固定 pipe_network 的13.034959kg；反馈/限力和COM归一化用了这个错误总量。机器人本身为2.856kg、28.01736N。重力广义载荷原本已只累加21个机器人刚体。V4未积分，失败审阅和原快照保留。

本版本仅将局部反馈质量、重量和COM归一化明确限定机器人 head..head+20；其余V4代码、参数、原初态与真实物理保持。独立当前版通过后才做一个T_UPDATE实际诊断。原V4层仍属新控制行为，其静力/QP检查不能证明实际保持、目标包绕或Step2成功。

'''+(old/'CONTROLLER_DESIGN.md').read_text(encoding='utf-8')
(B/'CONTROLLER_DESIGN.md').write_text(design,encoding='utf-8');print(str(B))
